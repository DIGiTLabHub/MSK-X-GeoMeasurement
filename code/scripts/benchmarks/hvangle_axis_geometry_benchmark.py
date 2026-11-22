#!/usr/bin/env python3
"""HVAngleEst: image -> three bone-axis masks -> explicit HVA/IMA geometry.

The checkpoint is selected on a patient-disjoint validation split using either
mean Dice or a tolerance-normalized HVA/IMA validation MAE. The test split is
evaluated once after selection. Images and masks are letterboxed together. Soft
probability-weighted PCA always returns three axes, so there is no ground-truth
fallback and no excluded test case.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from protocol_geometry_benchmark import (
    ROOT, acute_angle, hvangle_splits, measurement_metrics, seed_everything,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "benchmarks" / "hvangle_axis_geometry"
MASK_DIR = ROOT / "HVAngleEst" / "HVAngleEst" / "annotations" / "masks"


class SegmentationDataset(Dataset):
    def __init__(self, records, image_size: int, augment: bool):
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        mask = Image.open(
            MASK_DIR / Path(record.image_path).with_suffix(".png").name).convert("L")
        scale = min(self.image_size / image.width, self.image_size / image.height)
        resized_width = max(1, int(round(image.width * scale)))
        resized_height = max(1, int(round(image.height * scale)))
        left = (self.image_size - resized_width) // 2
        top = (self.image_size - resized_height) // 2
        resized_image = image.resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR)
        resized_mask = mask.resize(
            (resized_width, resized_height), Image.Resampling.NEAREST)
        image = Image.new("RGB", (self.image_size, self.image_size), color=0)
        mask = Image.new("L", (self.image_size, self.image_size), color=0)
        image.paste(resized_image, (left, top))
        mask.paste(resized_mask, (left, top))
        if self.augment and random.random() < .5:
            image, mask = TF.hflip(image), TF.hflip(mask)
            left = self.image_size - left - resized_width
        if self.augment:
            image = T.ColorJitter(.15, .15, .10, .03)(image)
        image_tensor = T.ToTensor()(image)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        metadata = torch.tensor([
            record.width, record.height, resized_width, resized_height, left, top,
        ], dtype=torch.float32)
        return image_tensor, mask_tensor, torch.from_numpy(record.points_norm.copy()), \
            torch.from_numpy(record.targets.copy()), metadata, record.identifier


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class AxisUNet(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        base = base_channels
        self.encoder1 = DoubleConv(3, base)
        self.encoder2 = DoubleConv(base, base * 2)
        self.encoder3 = DoubleConv(base * 2, base * 4)
        self.encoder4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.decoder4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.decoder3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.decoder2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.decoder1 = DoubleConv(base * 2, base)
        self.output = nn.Conv2d(base, 4, 1)

    def forward(self, inputs):
        level1 = self.encoder1(inputs)
        level2 = self.encoder2(self.pool(level1))
        level3 = self.encoder3(self.pool(level2))
        level4 = self.encoder4(self.pool(level3))
        bottleneck = self.bottleneck(self.pool(level4))
        decoded = self.decoder4(torch.cat([self.up4(bottleneck), level4], 1))
        decoded = self.decoder3(torch.cat([self.up3(decoded), level3], 1))
        decoded = self.decoder2(torch.cat([self.up2(decoded), level2], 1))
        decoded = self.decoder1(torch.cat([self.up1(decoded), level1], 1))
        return self.output(decoded)


def probability_axis(probability: np.ndarray, width: float, height: float,
                     power: float = 4.0, quantile: float = .95,
                     hard_support: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fit an axial direction to a class probability map with weighted PCA."""
    used_soft_fallback = hard_support is None or int(hard_support.sum()) < 5
    if used_soft_fallback:
        cutoff = np.quantile(probability, quantile)
        weights = np.where(probability >= cutoff, probability ** power, 0.0).reshape(-1)
    else:
        weights = hard_support.astype(np.float64).reshape(-1)
    map_height, map_width = probability.shape
    grid_y, grid_x = np.mgrid[:map_height, :map_width]
    coordinates = np.stack([
        (grid_x + .5) / map_width * width,
        (grid_y + .5) / map_height * height,
    ], axis=-1).reshape(-1, 2)
    total = max(float(weights.sum()), 1e-12)
    center = (coordinates * weights[:, None]).sum(0) / total
    centered = coordinates - center
    covariance = (centered.T * weights) @ centered / total
    return center, np.linalg.eigh(covariance)[1][:, -1], used_soft_fallback


def dice_statistics(prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intersection, denominator = np.zeros(3, float), np.zeros(3, float)
    for class_index in range(1, 4):
        predicted_class, true_class = prediction == class_index, truth == class_index
        intersection[class_index - 1] = 2 * np.logical_and(predicted_class, true_class).sum()
        denominator[class_index - 1] = predicted_class.sum() + true_class.sum()
    return intersection, denominator


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Binary morphological skeleton used for line-aware clDice reporting."""
    current = np.asarray(mask, bool)
    skeleton = np.zeros_like(current)
    structure = ndimage.generate_binary_structure(2, 1)
    while current.any():
        opened = ndimage.binary_opening(current, structure=structure)
        skeleton |= current & ~opened
        current = ndimage.binary_erosion(current, structure=structure)
    return skeleton


def cldice_counts(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return topology-precision/sensitivity numerators and denominators."""
    counts = np.zeros((3, 4), float)
    for class_index in range(1, 4):
        predicted_class, true_class = prediction == class_index, truth == class_index
        predicted_skeleton = morphological_skeleton(predicted_class)
        true_skeleton = morphological_skeleton(true_class)
        counts[class_index - 1] = [
            np.logical_and(predicted_skeleton, true_class).sum(), predicted_skeleton.sum(),
            np.logical_and(true_skeleton, predicted_class).sum(), true_skeleton.sum(),
        ]
    return counts


def axis_angular_error(predicted_axis: np.ndarray, true_line: np.ndarray) -> float:
    true_axis = true_line[1] - true_line[0]
    cosine = abs(float(np.dot(predicted_axis, true_axis))) / max(
        float(np.linalg.norm(predicted_axis) * np.linalg.norm(true_axis)), 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def evaluate(model, loader, device, class_weights, axis_power: float,
             axis_quantile: float, include_predictions: bool,
             hard_mask_first: bool = False) -> dict:
    model.eval()
    total_loss, batches = 0.0, 0
    intersections, denominators = np.zeros(3), np.zeros(3)
    topology_counts = np.zeros((3, 4), float)
    axis_errors = [[], [], []]
    hva_truth, ima_truth, hva_prediction, ima_prediction, identifiers = [], [], [], [], []
    predicted_axes_all = []
    fallback_count = 0
    with torch.no_grad():
        for images, masks, points, targets, metadata, batch_ids in loader:
            logits = model(images.to(device))
            total_loss += F.cross_entropy(logits, masks.to(device), weight=class_weights).item()
            batches += 1
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            hard_prediction = probabilities.argmax(1)
            masks_np, points_np = masks.numpy(), points.numpy()
            targets_np, metadata_np = targets.numpy(), metadata.numpy()
            for batch_index in range(len(images)):
                inter, denom = dice_statistics(hard_prediction[batch_index], masks_np[batch_index])
                intersections += inter
                denominators += denom
                if include_predictions:
                    topology_counts += cldice_counts(
                        hard_prediction[batch_index], masks_np[batch_index])
                width, height, content_width, content_height, left, top = metadata_np[batch_index]
                left, top = int(left), int(top)
                content_width, content_height = int(content_width), int(content_height)
                row_slice = slice(top, top + content_height)
                col_slice = slice(left, left + content_width)
                fitted_axes = [
                    probability_axis(
                                     probabilities[batch_index, class_index, row_slice, col_slice],
                                     width, height,
                                     axis_power, axis_quantile,
                                     (hard_prediction[batch_index, row_slice, col_slice] == class_index)
                                     if hard_mask_first else None)
                    for class_index in (1, 2, 3)
                ]
                centers = np.stack([item[0] for item in fitted_axes])
                axes = np.stack([item[1] for item in fitted_axes])
                fallback_count += sum(bool(item[2]) for item in fitted_axes)
                true_points_px = points_np[batch_index] * np.asarray([width, height])
                for axis_index in range(3):
                    axis_errors[axis_index].append(axis_angular_error(
                        axes[axis_index], true_points_px[2 * axis_index:2 * axis_index + 2]))
                hva_prediction.append(acute_angle(
                    np.stack([np.zeros(2), axes[0]]), np.stack([np.zeros(2), axes[1]])))
                ima_prediction.append(acute_angle(
                    np.stack([np.zeros(2), axes[1]]), np.stack([np.zeros(2), axes[2]])))
                hva_truth.append(float(targets_np[batch_index, 0]))
                ima_truth.append(float(targets_np[batch_index, 1]))
                identifiers.append(batch_ids[batch_index])
                predicted_axes_all.append({
                    "centers_px": centers.tolist(),
                    "directions": axes.tolist(),
                })
    class_dice = intersections / np.maximum(denominators, 1e-12)
    result = {
        "segmentation": {
            "great_toe_dice": float(class_dice[0]),
            "first_metatarsal_dice": float(class_dice[1]),
            "second_metatarsal_dice": float(class_dice[2]),
            "mean_dice": float(class_dice.mean()),
            "cross_entropy": total_loss / max(batches, 1),
        },
        "axis": {
            "great_toe_angular_mae_deg": float(np.mean(axis_errors[0])),
            "first_metatarsal_angular_mae_deg": float(np.mean(axis_errors[1])),
            "second_metatarsal_angular_mae_deg": float(np.mean(axis_errors[2])),
            "mean_angular_mae_deg": float(np.mean(axis_errors)),
            "fit_probability_power": axis_power,
            "fit_probability_quantile": axis_quantile,
            "hard_mask_first": hard_mask_first,
            "soft_fallback_axes": int(fallback_count),
        },
        "measurements": {
            "hva": measurement_metrics(hva_truth, hva_prediction, 5.0),
            "ima": measurement_metrics(ima_truth, ima_prediction, 3.0),
        },
    }
    if include_predictions:
        topology_precision = topology_counts[:, 0] / np.maximum(topology_counts[:, 1], 1e-12)
        topology_sensitivity = topology_counts[:, 2] / np.maximum(topology_counts[:, 3], 1e-12)
        class_cldice = (2 * topology_precision * topology_sensitivity /
                        np.maximum(topology_precision + topology_sensitivity, 1e-12))
        result["segmentation"].update({
            "great_toe_cldice": float(class_cldice[0]),
            "first_metatarsal_cldice": float(class_cldice[1]),
            "second_metatarsal_cldice": float(class_cldice[2]),
            "mean_cldice": float(class_cldice.mean()),
        })
        result["identifiers"] = identifiers
        result["predicted_axis_directions"] = predicted_axes_all
    return result


def train_seed(splits, args, seed: int, device) -> dict:
    seed_everything(seed)
    datasets = {
        split: SegmentationDataset(records, args.image_size, augment=split == "train")
        for split, records in splits.items()
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(dataset, args.batch_size, shuffle=split == "train",
                          num_workers=args.num_workers, pin_memory=True,
                          generator=generator if split == "train" else None)
        for split, dataset in datasets.items()
    }
    model = AxisUNet(args.base_channels).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    class_weights = torch.tensor([.1, 1.0, 1.0, 1.0], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    checkpoint = args.output / f"hvangle_axis_seed{seed}.pt"
    best_score, best_epoch, stale = -float("inf"), 0, 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss = 0.0
        for images, masks, _, _, _, _ in loaders["train"]:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss = F.cross_entropy(model(images), masks, weight=class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            training_loss += loss.item()
        scheduler.step()
        validation = evaluate(model, loaders["val"], device, class_weights,
                              args.axis_power, args.axis_quantile, False)
        validation_dice = validation["segmentation"]["mean_dice"]
        if args.checkpoint_metric == "dice":
            selection_score = validation_dice
        else:
            # Normalize by the predeclared target tolerances so HVA and IMA
            # contribute on comparable clinical scales. Test labels are never
            # used for selection.
            selection_score = -0.5 * (
                validation["measurements"]["hva"]["mae"] / 5.0
                + validation["measurements"]["ima"]["mae"] / 3.0
            )
        if selection_score > best_score + 1e-6:
            best_score, best_epoch, stale = selection_score, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        print(json.dumps({
            "dataset": "hvangle", "seed": seed, "epoch": epoch,
            "train_loss": training_loss / max(len(loaders["train"]), 1),
            "val_mean_dice": validation_dice,
            "val_hva_mae": validation["measurements"]["hva"]["mae"],
            "val_ima_mae": validation["measurements"]["ima"]["mae"],
            "checkpoint_metric": args.checkpoint_metric,
            "selection_score": selection_score,
            "best_epoch": best_epoch,
        }), flush=True)
        if stale >= args.patience:
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    validation = evaluate(model, loaders["val"], device, class_weights,
                          args.axis_power, args.axis_quantile, False)
    test = evaluate(model, loaders["test"], device, class_weights,
                    args.axis_power, args.axis_quantile, True)
    return {
        "dataset": "hvangle",
        "seed": seed,
        "architecture": "U-Net three-axis segmentation plus weighted-PCA geometry",
        "parameters": parameters,
        "trainable_parameters": parameters,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "checkpoint_metric": args.checkpoint_metric,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.time() - started,
        "split_sizes": {split: len(records) for split, records in splits.items()},
        "validation": validation,
        "test": test,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--axis-power", type=float, default=4.0)
    parser.add_argument("--axis-quantile", type=float, default=.95)
    parser.add_argument("--checkpoint-metric", choices=("dice", "measurement"),
                        default="dice")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The full benchmark requires CUDA")
    splits = hvangle_splits()
    result_path = args.output / "results.json"
    results = json.loads(result_path.read_text()) if result_path.is_file() else []
    completed = {int(result["seed"]) for result in results}
    for seed in args.seeds:
        if seed in completed:
            print(f"Skipping completed seed {seed}", flush=True)
            continue
        results.append(train_seed(splits, args, seed, device))
        result_path.write_text(json.dumps(results, indent=2))
        print(f"Saved seed {seed} to {result_path}", flush=True)


if __name__ == "__main__":
    main()
