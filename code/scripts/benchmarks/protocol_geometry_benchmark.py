#!/usr/bin/env python3
"""Protocol-aware landmark-to-measurement benchmark.

This benchmark implements the constructive path proposed by the paper:

    radiograph -> spatial landmarks/axes -> deterministic clinical geometry

The test labels are never used for checkpoint selection or as an inference-time
fallback.  AASCE uses the persisted controlled split from
``cross_model_benchmark.py``; HVAngleEst uses a fixed patient-grouped split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF


from workspace_paths import ROOT, split_manifest
DEFAULT_OUTPUT = ROOT / "outputs" / "benchmarks" / "protocol_geometry"
VERSIONED_DEPS = Path(__file__).with_name(f".deps-py{sys.version_info.major}{sys.version_info.minor}")
LOCAL_DEPS = VERSIONED_DEPS if VERSIONED_DEPS.is_dir() else Path(__file__).with_name(".deps")
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
os.environ.setdefault("TORCH_HOME", str(ROOT / "outputs" / "model_cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / "outputs" / "model_cache" / "huggingface"))


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def icc_2_1(y_true, y_pred) -> float:
    """ICC(2,1) computational form, used descriptively for a fixed pair."""
    x = np.column_stack([np.asarray(y_true, float), np.asarray(y_pred, float)])
    n, k = x.shape
    mean_rows, mean_cols, grand = x.mean(1), x.mean(0), x.mean()
    msr = k * np.square(mean_rows - grand).sum() / (n - 1)
    msc = n * np.square(mean_cols - grand).sum() / (k - 1)
    residual = np.square(x - mean_rows[:, None] - mean_cols + grand).sum()
    mse = residual / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denominator) if denominator else float("nan")


def measurement_metrics(y_true, y_pred, tolerance: float) -> dict:
    truth, pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    absolute_error, difference = np.abs(pred - truth), pred - truth
    bias = float(difference.mean())
    sd = float(difference.std(ddof=1))
    return {
        "n": int(len(truth)),
        "mae": float(absolute_error.mean()),
        "mae_sd": float(absolute_error.std(ddof=1)),
        "icc_2_1": icc_2_1(truth, pred),
        "within_tolerance_percent": float(100 * np.mean(absolute_error <= tolerance)),
        "tolerance_deg": float(tolerance),
        "bias": bias,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "ground_truth": truth.tolist(),
        "predictions": pred.tolist(),
    }


def parse_segment(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in str(value).split(",")], np.float32).reshape(2, 2)


def acute_angle(line_a: np.ndarray, line_b: np.ndarray) -> float:
    vector_a, vector_b = line_a[1] - line_a[0], line_b[1] - line_b[0]
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cosine = abs(float(np.dot(vector_a, vector_b))) / max(float(denominator), 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def cobb_from_landmarks(points_px: np.ndarray) -> tuple[float, int, int]:
    """Apply the persisted AASCE end-plate selection and Cobb definition."""
    vertebrae = np.asarray(points_px, float).reshape(17, 4, 2)
    superior = np.degrees(np.arctan2(
        vertebrae[:, 1, 1] - vertebrae[:, 0, 1],
        vertebrae[:, 1, 0] - vertebrae[:, 0, 0],
    ))
    inferior = np.degrees(np.arctan2(
        vertebrae[:, 3, 1] - vertebrae[:, 2, 1],
        vertebrae[:, 3, 0] - vertebrae[:, 2, 0],
    ))
    upper, lower = int(np.argmax(superior)), int(np.argmin(inferior))
    if upper >= lower:
        upper, lower = 0, 16
    angle = abs(float(superior[upper] - inferior[lower]))
    if angle > 90:
        angle = 180 - angle
    return float(angle), upper, lower


def cobb_with_known_end_vertebrae(points_px: np.ndarray, upper: int, lower: int) -> float:
    vertebrae = np.asarray(points_px, float).reshape(17, 4, 2)
    top = vertebrae[upper, [0, 1]]
    bottom = vertebrae[lower, [2, 3]]
    angle_top = math.degrees(math.atan2(*(top[1] - top[0])[::-1]))
    angle_bottom = math.degrees(math.atan2(*(bottom[1] - bottom[0])[::-1]))
    angle = abs(angle_top - angle_bottom)
    return float(180 - angle if angle > 90 else angle)


@dataclass
class SpatialRecord:
    identifier: str
    image_path: str
    points_norm: np.ndarray
    width: int
    height: int
    targets: np.ndarray
    upper: int = -1
    lower: int = -1
    patient_id: int = -1


def aasce_splits() -> dict[str, list[SpatialRecord]]:
    all_rows = json.loads((ROOT / "AASCE" / "aasce_gt.json").read_text())
    test_rows = json.loads((ROOT / "AASCE" / "aasce_test_split.json").read_text())
    test_ids = {row["image_name"] for row in test_rows}
    order = np.random.RandomState(42).permutation(len(all_rows))
    n_train, n_val = int(.70 * len(all_rows)), int(.15 * len(all_rows))
    train_ids = {all_rows[index]["image_name"] for index in order[:n_train]}
    val_ids = {all_rows[index]["image_name"] for index in order[n_train:n_train + n_val]}
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)

    def convert(rows) -> list[SpatialRecord]:
        records = []
        for row in rows:
            with Image.open(row["image_path"]) as image:
                width, height = image.size
            scale = np.asarray([width, height], np.float32)
            points = np.asarray(row["keypoints"], np.float32) / scale
            records.append(SpatialRecord(
                row["image_name"], row["image_path"], points, width, height,
                np.asarray([row["cobb_gt"]], np.float32),
                int(row["upper_end_vertebra"]), int(row["lower_end_vertebra"]),
            ))
        return records

    remaining = [row for row in all_rows if row["image_name"] not in test_ids]
    return {
        "train": convert([row for row in remaining if row["image_name"] in train_ids]),
        "val": convert([row for row in remaining if row["image_name"] in val_ids]),
        "test": convert(test_rows),
    }


def hvangle_splits(split_seed: int = 42) -> dict[str, list[SpatialRecord]]:
    data_dir = ROOT / "HVAngleEst" / "HVAngleEst"
    split_dir = ROOT / "HVAngleEst" / "splits"
    frame = pd.read_csv(data_dir / "datasets.csv")
    frame = frame[~frame.filename.duplicated(keep=False)]
    saved_manifest = split_manifest("hvangle")
    saved_splits = (json.loads(saved_manifest.read_text())["split_ids"]
                    if saved_manifest.exists() else None)
    released_names = (set().union(*(set(saved_splits[name]) for name in saved_splits))
                      if saved_splits is not None else set().union(*(
                          set(pd.read_csv(split_dir / f"{name}.csv")["filename"])
                          for name in ("train", "val", "test"))))
    frame = frame[frame.filename.isin(released_names)]
    patients = np.asarray(sorted(frame.patient_id.unique()))
    patients = patients[np.random.RandomState(split_seed).permutation(len(patients))]
    n_train, n_val = int(.70 * len(patients)), int(.15 * len(patients))
    patient_sets = {
        "train": set(patients[:n_train]),
        "val": set(patients[n_train:n_train + n_val]),
        "test": set(patients[n_train + n_val:]),
    }
    result = {}
    for split in ("train", "val", "test"):
        if saved_splits is not None and split_seed == 42:
            rows = frame.set_index("filename", drop=False).loc[saved_splits[split]]
        else:
            rows = frame[frame.patient_id.isin(patient_sets[split])]
        records = []
        for row in rows.itertuples():
            points = np.concatenate([
                parse_segment(row.great_toe), parse_segment(row.first_metatarsal),
                parse_segment(row.second_metatarsal),
            ])
            image_path = data_dir / "images" / row.filename
            if image_path.is_file():
                records.append(SpatialRecord(
                    row.filename, str(image_path), points,
                    int(row.image_width), int(row.image_height),
                    np.asarray([row.HVA, row.IMA], np.float32),
                    patient_id=int(row.patient_id),
                ))
        result[split] = records
    observed = {name: {record.patient_id for record in records}
                for name, records in result.items()}
    assert not (observed["train"] & observed["val"] or
                observed["train"] & observed["test"] or
                observed["val"] & observed["test"])
    return result


def flip_landmarks(points: torch.Tensor, dataset_name: str) -> torch.Tensor:
    flipped = points.clone()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    if dataset_name == "aasce":
        flipped = flipped.reshape(17, 4, 2)[:, [1, 0, 3, 2], :].reshape(68, 2)
    return flipped


class LandmarkDataset(Dataset):
    def __init__(self, records, dataset_name: str, image_size: int, heatmap_size: int,
                 sigma: float, augment: bool):
        self.records = records
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment
        grid = torch.arange(heatmap_size, dtype=torch.float32) + .5
        self.grid_y, self.grid_x = torch.meshgrid(grid, grid, indexing="ij")
        self.normalize = T.Normalize([.485, .456, .406], [.229, .224, .225])

    def __len__(self):
        return len(self.records)

    def make_heatmaps(self, points: torch.Tensor) -> torch.Tensor:
        centers = points * self.heatmap_size
        dx = self.grid_x[None] - centers[:, 0, None, None]
        dy = self.grid_y[None] - centers[:, 1, None, None]
        return torch.exp(-(dx.square() + dy.square()) / (2 * self.sigma ** 2))

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR)
        points = torch.from_numpy(record.points_norm.copy())
        if self.augment and random.random() < .5:
            image = TF.hflip(image)
            points = flip_landmarks(points, self.dataset_name)
        if self.augment:
            image = T.ColorJitter(.15, .15, .10, .03)(image)
        image = self.normalize(T.ToTensor()(image))
        metadata = torch.tensor(
            [record.width, record.height, record.upper, record.lower], dtype=torch.float32)
        return (image, self.make_heatmaps(points), points,
                torch.from_numpy(record.targets.copy()), metadata, record.identifier)


class ResNetFPNLandmarkDetector(nn.Module):
    """ImageNet ResNet50 with a small top-down spatial prediction head."""
    def __init__(self, number_of_landmarks: int, pretrained: bool = True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "resnet50", pretrained=pretrained, features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        channels = self.backbone.feature_info.channels()
        self.lateral = nn.ModuleList([nn.Conv2d(channel, 128, 1) for channel in channels])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=False),
                          nn.BatchNorm2d(128), nn.ReLU(inplace=True))
            for _ in channels[:-1]
        ])
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, number_of_landmarks, 1),
        )

    def forward(self, images):
        features = self.backbone(images)
        pyramid = self.lateral[-1](features[-1])
        for level in range(len(features) - 2, -1, -1):
            pyramid = F.interpolate(pyramid, size=features[level].shape[-2:], mode="bilinear",
                                    align_corners=False) + self.lateral[level](features[level])
            pyramid = self.smooth[level](pyramid)
        return self.head(pyramid)


def decode_heatmaps(logits: torch.Tensor) -> torch.Tensor:
    """Quarter-pixel argmax decoder returning normalized x/y coordinates."""
    batch, landmarks, height, width = logits.shape
    flat_index = logits.reshape(batch, landmarks, -1).argmax(-1)
    y = (flat_index // width).float()
    x = (flat_index % width).float()
    padded = F.pad(logits, (1, 1, 1, 1), mode="replicate")
    batch_index = torch.arange(batch, device=logits.device)[:, None]
    landmark_index = torch.arange(landmarks, device=logits.device)[None, :]
    yp, xp = y.long() + 1, x.long() + 1
    dx = padded[batch_index, landmark_index, yp, xp + 1] - padded[batch_index, landmark_index, yp, xp - 1]
    dy = padded[batch_index, landmark_index, yp + 1, xp] - padded[batch_index, landmark_index, yp - 1, xp]
    x = x + .25 * torch.sign(dx)
    y = y + .25 * torch.sign(dy)
    return torch.stack([(x + .5) / width, (y + .5) / height], dim=-1).clamp(0, 1)


def soft_argmax_coordinates(logits: torch.Tensor, temperature: float = 10.0) -> torch.Tensor:
    """Differentiable normalized coordinates used only by the training loss."""
    batch, landmarks, height, width = logits.shape
    probability = torch.softmax(logits.reshape(batch, landmarks, -1) * temperature, dim=-1)
    grid_y, grid_x = torch.meshgrid(
        (torch.arange(height, device=logits.device, dtype=logits.dtype) + .5) / height,
        (torch.arange(width, device=logits.device, dtype=logits.dtype) + .5) / width,
        indexing="ij",
    )
    x = (probability * grid_x.reshape(1, 1, -1)).sum(-1)
    y = (probability * grid_y.reshape(1, 1, -1)).sum(-1)
    return torch.stack([x, y], dim=-1)


def landmark_loss(logits: torch.Tensor, heatmaps: torch.Tensor,
                  points: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Foreground-weighted heatmap loss plus sub-pixel coordinate supervision."""
    squared_error = (torch.sigmoid(logits) - heatmaps).square()
    heatmap_loss = (squared_error * (1.0 + 20.0 * heatmaps)).mean()
    coordinates = soft_argmax_coordinates(logits)
    coordinate_loss = F.smooth_l1_loss(coordinates, points, beta=.02)
    total = heatmap_loss + .5 * coordinate_loss
    return total, {"heatmap": heatmap_loss.detach(), "coordinate": coordinate_loss.detach()}


def spatial_metrics(truth: np.ndarray, prediction: np.ndarray,
                    widths: np.ndarray, heights: np.ndarray) -> dict:
    scale = np.stack([widths, heights], axis=-1)[:, None, :]
    distance_px = np.linalg.norm((prediction - truth) * scale, axis=-1)
    diagonal = np.sqrt(widths ** 2 + heights ** 2)[:, None]
    normalized_distance = distance_px / diagonal
    return {
        "mean_radial_error_px": float(distance_px.mean()),
        "normalized_mean_error": float(normalized_distance.mean()),
        "pck_at_0.025": float(100 * np.mean(normalized_distance <= .025)),
        "pck_at_0.05": float(100 * np.mean(normalized_distance <= .05)),
    }


def evaluate(model, loader, device, dataset_name: str) -> tuple[dict, float]:
    model.eval()
    predicted_points, true_points, targets, metadata, identifiers = [], [], [], [], []
    validation_loss = 0.0
    with torch.no_grad():
        for images, heatmaps, points, batch_targets, batch_metadata, batch_ids in loader:
            logits = model(images.to(device))
            validation_loss += F.mse_loss(torch.sigmoid(logits), heatmaps.to(device)).item()
            predicted_points.append(decode_heatmaps(logits).cpu().numpy())
            true_points.append(points.numpy())
            targets.append(batch_targets.numpy())
            metadata.append(batch_metadata.numpy())
            identifiers.extend(batch_ids)
    prediction = np.concatenate(predicted_points)
    truth = np.concatenate(true_points)
    target = np.concatenate(targets)
    meta = np.concatenate(metadata)
    widths, heights = meta[:, 0], meta[:, 1]
    result = {
        "identifiers": identifiers,
        "spatial": spatial_metrics(truth, prediction, widths, heights),
    }

    if dataset_name == "hvangle":
        hva, ima = [], []
        for points_norm, width, height in zip(prediction, widths, heights):
            points_px = points_norm * np.asarray([width, height])
            hva.append(acute_angle(points_px[0:2], points_px[2:4]))
            ima.append(acute_angle(points_px[2:4], points_px[4:6]))
        result["measurements"] = {
            "hva": measurement_metrics(target[:, 0], hva, 5.0),
            "ima": measurement_metrics(target[:, 1], ima, 3.0),
        }
    else:
        automatic, known_end, selected_upper, selected_lower = [], [], [], []
        for points_norm, width, height, item_meta in zip(prediction, widths, heights, meta):
            points_px = points_norm * np.asarray([width, height])
            angle, upper, lower = cobb_from_landmarks(points_px)
            automatic.append(angle)
            selected_upper.append(upper)
            selected_lower.append(lower)
            known_end.append(cobb_with_known_end_vertebrae(
                points_px, int(item_meta[2]), int(item_meta[3])))
        true_upper, true_lower = meta[:, 2].astype(int), meta[:, 3].astype(int)
        result["end_vertebra"] = {
            "upper_accuracy_percent": float(100 * np.mean(np.asarray(selected_upper) == true_upper)),
            "lower_accuracy_percent": float(100 * np.mean(np.asarray(selected_lower) == true_lower)),
            "pair_accuracy_percent": float(100 * np.mean(
                (np.asarray(selected_upper) == true_upper) &
                (np.asarray(selected_lower) == true_lower))),
        }
        result["measurements"] = {
            "cobb_automatic": measurement_metrics(target[:, 0], automatic, 5.0),
            "cobb_known_end_vertebra": measurement_metrics(target[:, 0], known_end, 5.0),
        }
    result["landmark_ground_truth"] = truth.tolist()
    result["landmark_predictions"] = prediction.tolist()
    return result, validation_loss / max(len(loader), 1)


def train_one_seed(dataset_name: str, splits, args, seed: int, device) -> dict:
    seed_everything(seed)
    number_of_landmarks = 68 if dataset_name == "aasce" else 6
    train_dataset = LandmarkDataset(splits["train"], dataset_name, args.image_size,
                                    args.heatmap_size, args.sigma, augment=True)
    evaluation_datasets = {
        split: LandmarkDataset(records, dataset_name, args.image_size,
                               args.heatmap_size, args.sigma, augment=False)
        for split, records in splits.items()
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(train_dataset, args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            generator=generator),
        **{
            split: DataLoader(dataset, args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
            for split, dataset in evaluation_datasets.items() if split != "train"
        },
    }
    model = ResNetFPNLandmarkDetector(number_of_landmarks, pretrained=not args.no_pretrained).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    checkpoint = args.output / f"{dataset_name}_seed{seed}.pt"
    best_loss, best_epoch, stale = float("inf"), 0, 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, heatmaps, points, _, _, _ in loaders["train"]:
            images, heatmaps, points = images.to(device), heatmaps.to(device), points.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                prediction = model(images)
                loss, _ = landmark_loss(prediction, heatmaps, points)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        scheduler.step()
        validation_result, validation_loss = evaluate(model, loaders["val"], device, dataset_name)
        selection_score = validation_result["spatial"]["normalized_mean_error"]
        improved = selection_score < best_loss - 1e-7
        if improved:
            best_loss, best_epoch, stale = selection_score, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        print(json.dumps({
            "dataset": dataset_name, "seed": seed, "epoch": epoch,
            "train_loss": running_loss / max(len(loaders["train"]), 1),
            "val_heatmap_loss": validation_loss, "val_nme": selection_score,
            "best_epoch": best_epoch,
        }), flush=True)
        if stale >= args.patience:
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    validation, _ = evaluate(model, loaders["val"], device, dataset_name)
    test, _ = evaluate(model, loaders["test"], device, dataset_name)
    return {
        "dataset": dataset_name,
        "seed": seed,
        "architecture": "ImageNet ResNet50-FPN landmark detector",
        "parameters": parameter_count,
        "trainable_parameters": parameter_count,
        "image_size": args.image_size,
        "heatmap_size": args.heatmap_size,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.time() - started,
        "split_sizes": {split: len(records) for split, records in splits.items()},
        "validation": validation,
        "test": test,
    }


def geometry_sanity_checks() -> None:
    hv = hvangle_splits()["test"]
    hva_error, ima_error = [], []
    for record in hv:
        points = record.points_norm * np.asarray([record.width, record.height])
        hva_error.append(abs(acute_angle(points[:2], points[2:4]) - record.targets[0]))
        ima_error.append(abs(acute_angle(points[2:4], points[4:6]) - record.targets[1]))
    aasce = aasce_splits()["test"]
    cobb_error = []
    for record in aasce:
        points = record.points_norm * np.asarray([record.width, record.height])
        angle, _, _ = cobb_from_landmarks(points)
        cobb_error.append(abs(angle - record.targets[0]))
    assert max(hva_error) < 1e-3 and max(ima_error) < 1e-3 and max(cobb_error) < 1e-3
    print(json.dumps({
        "geometry_sanity_check": "passed",
        "max_hva_error_deg": max(hva_error),
        "max_ima_error_deg": max(ima_error),
        "max_cobb_error_deg": max(cobb_error),
    }), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["aasce", "hvangle"],
                        default=["aasce", "hvangle"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--heatmap-size", type=int, default=56)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sanity-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    geometry_sanity_checks()
    if args.sanity_only:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The full benchmark requires CUDA; use --sanity-only for CPU validation")
    all_results = []
    result_path = args.output / "results.json"
    if result_path.is_file():
        all_results = json.loads(result_path.read_text())
    completed = {(row["dataset"], int(row["seed"])) for row in all_results}
    for dataset_name in args.datasets:
        splits = aasce_splits() if dataset_name == "aasce" else hvangle_splits()
        for seed in args.seeds:
            if (dataset_name, seed) in completed:
                print(f"Skipping completed {dataset_name} seed {seed}", flush=True)
                continue
            result = train_one_seed(dataset_name, splits, args, seed, device)
            all_results.append(result)
            result_path.write_text(json.dumps(all_results, indent=2))
            print(f"Saved {dataset_name} seed {seed} to {result_path}", flush=True)


if __name__ == "__main__":
    main()
