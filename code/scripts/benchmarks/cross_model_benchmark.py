#!/usr/bin/env python3
"""Controlled multi-model regression benchmark for AASCE and HVAngleEst."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from workspace_paths import ROOT, split_manifest
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT = ROOT / "outputs" / "benchmarks" / "controlled_cross_model"
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


def icc_2_1(y_true, y_pred):
    """Two-way random-effects, absolute-agreement, single-measure ICC(2,1)."""
    x = np.column_stack([np.asarray(y_true, float), np.asarray(y_pred, float)])
    n, k = x.shape
    if n < 2:
        return float("nan")
    mean_rows, mean_cols, grand = x.mean(1), x.mean(0), x.mean()
    msr = k * np.square(mean_rows - grand).sum() / (n - 1)
    msc = n * np.square(mean_cols - grand).sum() / (k - 1)
    residual = np.square(x - mean_rows[:, None] - mean_cols + grand).sum()
    mse = residual / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denominator) if denominator else float("nan")


def measurement_metrics(y_true, y_pred, tolerances):
    truth, pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    error, diff = np.abs(pred - truth), pred - truth
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    result = {
        "n": int(len(truth)), "mae": float(error.mean()),
        "mae_sd": float(error.std(ddof=1)), "icc_2_1": icc_2_1(truth, pred),
        "bias": bias, "loa_lower": bias - 1.96 * sd, "loa_upper": bias + 1.96 * sd,
        "ground_truth": truth.tolist(), "predictions": pred.tolist()
    }
    for tolerance in tolerances:
        result[f"within_{tolerance:g}deg"] = float(100 * np.mean(error <= tolerance))
    return result


def target_tolerances(config, target):
    return config.get("tolerances_by_target", {}).get(target, config.get("tolerances_deg", [5.0]))


class MeasurementDataset(Dataset):
    def __init__(self, records, transform):
        self.records, self.transform = records, transform
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        row = self.records[index]
        image = self.transform(Image.open(row["image_path"]).convert("RGB"))
        return image, torch.tensor(row["targets"], dtype=torch.float32), row["id"]


def aasce_splits():
    filename_table = pd.read_csv(
        ROOT / "AASCE" / "data" / "train_txt" / "filenames.csv", header=None)
    official_angles = pd.read_csv(
        ROOT / "AASCE" / "data" / "train_txt" / "angles.csv", header=None)
    if filename_table.shape != (481, 1) or official_angles.shape != (481, 3):
        raise ValueError(
            "Expected the public AASCE release to contain 481 filenames and a 481x3 angle table")
    if filename_table[0].isna().any() or filename_table[0].duplicated().any():
        raise ValueError("AASCE filenames must be present and unique")
    filenames = filename_table[0].astype(str)
    angle_values = official_angles.to_numpy(dtype=np.float64)
    if not np.isfinite(angle_values).all():
        raise ValueError("AASCE angle table contains non-finite values")
    if not np.allclose(angle_values[:, 0], angle_values.max(axis=1), rtol=0.0, atol=1e-8):
        raise ValueError("AASCE angle column 0 is not the largest released curve angle for every case")
    image_dir = ROOT / "AASCE" / "data" / "train"
    missing_images = [name for name in filenames if not (image_dir / name).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"AASCE release is missing {len(missing_images)} referenced image(s): {missing_images[:3]}")
    # The AASCE release stores three landmark-derived curve angles per image.
    # Column 0 is the primary (largest) angle for every released case.  Use that
    # organizer-supplied target instead of the legacy local end-plate heuristic.
    primary_cobb = dict(zip(filenames, official_angles[0].astype(float)))
    saved_manifest = split_manifest("aasce")
    if saved_manifest.exists():
        # The review package includes this record so the exact submitted order,
        # as well as split membership, is replayed without private helper files.
        split_ids = json.loads(saved_manifest.read_text())["split_ids"]
    else:
        # Deterministic reconstruction from the two organizer tables. Training
        # and validation retain release order; test retains permutation order.
        names = filenames.tolist()
        order = np.random.RandomState(42).permutation(len(names))
        n_train, n_val = int(.70 * len(names)), int(.15 * len(names))
        train_set = {names[index] for index in order[:n_train]}
        val_set = {names[index] for index in order[n_train:n_train + n_val]}
        split_ids = {
            "train": [name for name in names if name in train_set],
            "val": [name for name in names if name in val_set],
            "test": [names[index] for index in order[n_train + n_val:]],
        }
    observed = [name for part in ("train", "val", "test") for name in split_ids[part]]
    if len(observed) != len(set(observed)) or set(observed) != set(filenames):
        raise ValueError("AASCE split identifiers do not partition the organizer filenames")
    expected_sizes = {"train": 336, "val": 72, "test": 73}
    if {part: len(split_ids[part]) for part in expected_sizes} != expected_sizes:
        raise ValueError(f"AASCE split sizes do not match the declared partition: {expected_sizes}")

    def convert(names):
        return [{"id": name, "image_path": str(image_dir / name),
                 "targets": [primary_cobb[name]]} for name in names]
    return {part: convert(split_ids[part]) for part in ("train", "val", "test")}, ["cobb"]


def hvangle_splits(unilateral_only=True, split_mode="official_image", split_seed=42):
    data_dir, split_dir = ROOT / "HVAngleEst" / "HVAngleEst", ROOT / "HVAngleEst" / "splits"
    frame = pd.read_csv(data_dir / "datasets.csv")
    if unilateral_only:
        frame = frame[~frame.filename.duplicated(keep=False)]
    saved_manifest = split_manifest("hvangle")
    saved_splits = (json.loads(saved_manifest.read_text())["split_ids"]
                    if saved_manifest.exists() else None)
    if split_mode == "patient":
        # Keep the same image pool as the released split files, changing only
        # the assignment unit from image to patient.
        released_names = (set().union(*(set(saved_splits[name]) for name in saved_splits))
                          if saved_splits is not None else set().union(*(
                              set(pd.read_csv(split_dir / f"{name}.csv")["filename"])
                              for name in ("train", "val", "test"))))
        frame = frame[frame.filename.isin(released_names)]
        # Group the retained radiographs by the dataset's explicit patient_id so
        # repeat acquisitions cannot cross partitions.  The patient permutation
        # and 70/15/15 assignment are fixed independently of model training.
        patients = np.asarray(sorted(frame.patient_id.unique()))
        patients = patients[np.random.RandomState(split_seed).permutation(len(patients))]
        n_train, n_val = int(.70 * len(patients)), int(.15 * len(patients))
        patient_sets = {
            "train": set(patients[:n_train]),
            "val": set(patients[n_train:n_train + n_val]),
            "test": set(patients[n_train + n_val:]),
        }
    elif split_mode != "official_image":
        raise ValueError(f"unsupported HVAngleEst split mode: {split_mode}")
    result = {}
    for split in ("train", "val", "test"):
        if split_mode == "patient" and saved_splits is not None and split_seed == 42:
            row_by_name = frame.set_index("filename", drop=False)
            rows = row_by_name.loc[saved_splits[split]]
        elif split_mode == "patient":
            rows = frame[frame.patient_id.isin(patient_sets[split])]
        else:
            names = set(pd.read_csv(split_dir / f"{split}.csv")["filename"])
            rows = frame[frame.filename.isin(names)]
        result[split] = [{"id": r.filename, "image_path": str(data_dir / "images" / r.filename),
                          "patient_id": int(r.patient_id),
                          "targets": [float(r.HVA), float(r.IMA)]} for r in rows.itertuples()
                         if (data_dir / "images" / r.filename).exists()]
    if split_mode == "patient":
        observed = {name: {row["patient_id"] for row in rows} for name, rows in result.items()}
        assert not (observed["train"] & observed["val"] or
                    observed["train"] & observed["test"] or
                    observed["val"] & observed["test"])
    return result, ["hva", "ima"]


class LetterboxResize:
    """Fit an image inside a square canvas without changing its aspect ratio."""
    def __init__(self, image_size: int, fill: int = 0):
        self.image_size = int(image_size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        contained = ImageOps.contain(
            image, (self.image_size, self.image_size), Image.Resampling.BILINEAR)
        canvas = Image.new(image.mode, (self.image_size, self.image_size), color=self.fill)
        offset = ((self.image_size - contained.width) // 2,
                  (self.image_size - contained.height) // 2)
        canvas.paste(contained, offset)
        return canvas


def transforms(image_size, resize_mode="distort"):
    normalize = T.Normalize([.485, .456, .406], [.229, .224, .225])
    if resize_mode == "distort":
        resize = T.Resize(
            (image_size, image_size),
            interpolation=T.InterpolationMode.BILINEAR,
        )
    elif resize_mode == "letterbox":
        resize = LetterboxResize(image_size, fill=0)
    else:
        raise ValueError(f"unsupported resize mode: {resize_mode}")
    train = T.Compose([resize, T.RandomHorizontalFlip(.3),
                       T.RandomRotation(10), T.ColorJitter(.2, .2), T.ToTensor(), normalize])
    test = T.Compose([resize, T.ToTensor(), normalize])
    return train, test


def build_timm_model(spec, outputs, pretrained=True):
    import timm
    model = timm.create_model(spec["architecture"], pretrained=pretrained, num_classes=outputs)
    checkpoint = spec.get("checkpoint")
    if spec["family"] == "radimagenet":
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError("RadImageNet requires models.<id>.checkpoint in config.json")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        # The released PyTorch checkpoint stores torchvision ResNet children in
        # a Sequential named ``backbone``. Map those keys to timm's equivalent
        # module names and deliberately omit any task head.
        child_names = {
            "0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2",
            "6": "layer3", "7": "layer4",
        }
        mapped = {}
        for key, value in state.items():
            parts = key.split(".")
            if len(parts) >= 3 and parts[0] == "backbone" and parts[1] in child_names:
                mapped[".".join([child_names[parts[1]], *parts[2:]])] = value
        if not mapped:
            raise RuntimeError("RadImageNet checkpoint contains no recognized ResNet backbone keys")
        incompatible = model.load_state_dict(mapped, strict=False)
        loaded = len(mapped) - len(incompatible.unexpected_keys)
        if loaded < 250:
            raise RuntimeError(f"Only {loaded} RadImageNet tensors loaded; checkpoint mapping is invalid")
    return model


def parameter_counts(model):
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def evaluate(model, loader, device, target_names, config):
    model.eval(); predictions, truths, ids = [], [], []
    with torch.no_grad():
        for images, targets, batch_ids in loader:
            output = model(images.to(device)).cpu()
            if output.ndim == 1: output = output[:, None]
            predictions.append(output.numpy()); truths.append(targets.numpy()); ids.extend(batch_ids)
    pred, truth = np.concatenate(predictions), np.concatenate(truths)
    return {name: measurement_metrics(truth[:, i], pred[:, i], target_tolerances(config, name))
            for i, name in enumerate(target_names)}, ids


def train_trial(model, loaders, device, target_names, hp, config, checkpoint):
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["max_epochs"])
    criterion, best, best_state, best_epoch, stale = (
        nn.HuberLoss(delta=5.0, reduction="mean"),
        float("inf"), None, None, 0)
    amp_enabled = bool(config.get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    for epoch in range(config["max_epochs"]):
        model.train()
        for images, targets, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = criterion(model(images.to(device)), targets.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        scheduler.step()
        metrics, _ = evaluate(model, loaders["val"], device, target_names, config)
        score = float(np.mean([metrics[name]["mae"] for name in target_names]))
        if (epoch + 1) % 10 == 0:
            print(f"      epoch={epoch + 1} current_val_mae={score:.3f} best={best:.3f}",
                  flush=True)
        if score < best:
            best, best_epoch, stale = score, epoch + 1, 0
            # Keep the validation winner in host memory and write it once at
            # the end of the trial.  Large ViT/ConvNeXt checkpoints otherwise
            # block the training loop on every small validation improvement.
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config["patience"]: break
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    torch.save(best_state, checkpoint)
    return {"validation_mae": best, "best_epoch": best_epoch,
            "epochs_run": epoch + 1, "epochs": epoch + 1}


def medical_availability(spec):
    try:
        if spec["family"] == "biomedclip":
            import open_clip  # noqa: F401
            return True, "available"
        if spec["family"] == "medimageinsight":
            sys.path.insert(0, str(ROOT / "MedImageInsights"))
            import medimageinsightmodel  # noqa: F401
            weight = (ROOT / "MedImageInsights" / "2024.09.27" / "vision_model" /
                      "medimageinsigt-v1.0.0.pt")
            return weight.is_file(), f"weight={weight}"
        if spec["family"] == "radimagenet":
            path = spec.get("checkpoint")
            return bool(path and Path(path).is_file()), "set checkpoint in config"
        if spec["family"] in {"medclip_resnet", "medclip_vit", "dax_resnet50", "medsam_encoder"}:
            path = ROOT / spec["checkpoint"]
            return path.is_file(), f"checkpoint={path}"
        if spec["family"] == "bonevlp":
            path = ROOT / spec["checkpoint"]
            return path.is_dir(), (
                f"checkpoint={path}; official download host orthovlp.aenglecloud.com unavailable"
            )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "available"


def extract_medical_features(family, records, batch_size, device):
    """Return fixed pretrained embeddings without fitting on evaluation labels."""
    features, labels, ids = [], [], []
    if family == "biomedclip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        model.eval().to(device)
        # The probe consumes only the visual encoder.  Counting the unused
        # text tower would overstate both the deployed feature extractor and
        # the model-size comparison in the paper.
        total_parameters = sum(p.numel() for p in model.visual.parameters())
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            images = torch.stack([preprocess(LetterboxResize(224)(
                Image.open(r["image_path"]).convert("RGB"))) for r in batch]).to(device)
            with torch.no_grad():
                encoded = model.encode_image(images)
                encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            features.append(encoded.cpu().numpy()); labels.extend(r["targets"] for r in batch); ids.extend(r["id"] for r in batch)
    elif family == "medimageinsight":
        sys.path.insert(0, str(ROOT / "MedImageInsights"))
        from medimageinsightmodel import MedImageInsight
        model_dir = ROOT / "MedImageInsights" / "2024.09.27"
        encoder = MedImageInsight(str(model_dir), "medimageinsigt-v1.0.0.pt", "language_model.pth")
        encoder.load_model()
        encoder.model.eval()
        total_parameters = (
            sum(p.numel() for p in encoder.model.image_encoder.parameters()) +
            encoder.model.image_projection.numel()
        )
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            encoded_images = []
            for row in batch:
                buffer = io.BytesIO()
                LetterboxResize(224)(Image.open(row["image_path"]).convert("RGB")).save(
                    buffer, format="JPEG", quality=95)
                encoded_images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
            encoded = encoder.encode(images=encoded_images)["image_embeddings"]
            features.append(np.asarray(encoded)); labels.extend(r["targets"] for r in batch); ids.extend(r["id"] for r in batch)
    elif family in {"medclip_resnet", "medclip_vit"}:
        sys.path.insert(0, str(ROOT / "external_models" / "MedCLIP"))
        from medclip import MedCLIPVisionModel, MedCLIPVisionModelViT
        checkpoint_dir = ROOT / "checkpoints" / "MedCLIP" / ("resnet" if family.endswith("resnet") else "vit")
        vision_cls = MedCLIPVisionModel if family.endswith("resnet") else MedCLIPVisionModelViT
        model = vision_cls(medclip_checkpoint=str(checkpoint_dir)).eval().to(device)
        total_parameters = sum(p.numel() for p in model.parameters())
        preprocess = T.Compose([LetterboxResize(224), T.ToTensor(),
                                T.Normalize([.5862786], [.2795009])])
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            images = torch.stack([preprocess(Image.open(r["image_path"]).convert("L")) for r in batch]).to(device)
            with torch.no_grad(): encoded = model(images)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            features.append(encoded.cpu().numpy()); labels.extend(r["targets"] for r in batch); ids.extend(r["id"] for r in batch)
    elif family == "dax_resnet50":
        import torchvision.models as tv_models
        checkpoint = torch.load(ROOT / "checkpoints" / "DAX" / "dax-resnet50-a.pth",
                                map_location="cpu", weights_only=False)
        teacher = checkpoint["teacher"]
        state = {k.replace("module.backbone.", ""): v for k, v in teacher.items()
                 if k.startswith("module.backbone.")}
        model = tv_models.resnet50(weights=None); model.fc = nn.Identity()
        model.load_state_dict(state, strict=True); model.eval().to(device)
        total_parameters = sum(p.numel() for p in model.parameters())
        preprocess = T.Compose([LetterboxResize(224), T.ToTensor(),
                                T.Normalize([.5, .5, .5], [.5, .5, .5])])
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            images = torch.stack([preprocess(Image.open(r["image_path"]).convert("RGB")) for r in batch]).to(device)
            with torch.no_grad(): encoded = model(images)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            features.append(encoded.cpu().numpy()); labels.extend(r["targets"] for r in batch); ids.extend(r["id"] for r in batch)
    elif family == "medsam_encoder":
        sys.path.insert(0, str(ROOT / "external_models" / "segment-anything"))
        from segment_anything import sam_model_registry
        checkpoint = ROOT / "cvpr" / "checkpoints" / "medsam_vit_b.pth"
        model = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).eval().to(device)
        total_parameters = sum(p.numel() for p in model.image_encoder.parameters())
        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
        preprocess = T.Compose([LetterboxResize(1024), T.ToTensor()])
        medsam_batch = min(batch_size, 2)
        for start in range(0, len(records), medsam_batch):
            batch = records[start:start + medsam_batch]
            images = torch.stack([preprocess(Image.open(r["image_path"]).convert("RGB")) for r in batch]).to(device) * 255.0
            with torch.no_grad(): encoded = model.image_encoder((images - pixel_mean) / pixel_std).mean(dim=(-2, -1))
            encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            features.append(encoded.cpu().numpy()); labels.extend(r["targets"] for r in batch); ids.extend(r["id"] for r in batch)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    elif family == "bonevlp":
        raise RuntimeError("Bone X-ray VLP adapter requires the authors' unavailable checkpoint directory")
    else:
        raise ValueError(f"unsupported medical feature family: {family}")
    return np.concatenate(features), np.asarray(labels, np.float32), ids, total_parameters


class Probe(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        hidden = min(512, max(64, inputs // 2))
        self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(.2), nn.Linear(hidden, outputs))
    def forward(self, x):
        return self.network(x)


def run_medical_probe(model_id, spec, splits, targets, config, output_dir, dry_run):
    available, detail = medical_availability(spec)
    if not available:
        return {"status": "unavailable", "detail": detail, "adaptation": spec["adaptation"]}
    if dry_run:
        return {"status": "adapter_ready", "detail": detail, "adaptation": spec["adaptation"]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load each large frozen encoder once, then slice the combined feature
    # matrix back into the declared partitions.  Reloading BiomedCLIP,
    # MedImageInsight, or MedSAM once per split triples initialization cost.
    split_names = list(splits)
    combined_records = [row for name in split_names for row in splits[name]]
    print(
        "  extracting frozen features: " +
        ", ".join(f"{name}={len(splits[name])}" for name in split_names),
        flush=True,
    )
    all_features, all_labels, all_ids, encoder_parameters = extract_medical_features(
        spec["family"], combined_records, config["batch_size"], device)
    encoded, offset = {}, 0
    for name in split_names:
        stop = offset + len(splits[name])
        encoded[name] = (
            all_features[offset:stop], all_labels[offset:stop], all_ids[offset:stop],
            encoder_parameters,
        )
        offset = stop
    input_dim, encoder_params = encoded["train"][0].shape[1], encoded["train"][3]
    runs = []
    for seed in config["seeds"]:
        print(f"  probe seed={seed}", flush=True)
        trials = []
        for trial_index, hp in enumerate(config["search_space"]):
            print(f"    trial={trial_index} hp={hp}", flush=True)
            seed_everything(seed)
            probe = Probe(input_dim, len(targets)).to(device)
            optimizer = torch.optim.AdamW(probe.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
            criterion, best, best_state, best_epoch, stale = (
                nn.HuberLoss(delta=5.0, reduction="mean"),
                float("inf"), None, None, 0)
            xtr = torch.tensor(encoded["train"][0], dtype=torch.float32, device=device)
            ytr = torch.tensor(encoded["train"][1], dtype=torch.float32, device=device)
            xva = torch.tensor(encoded["val"][0], dtype=torch.float32, device=device)
            yva = torch.tensor(encoded["val"][1], dtype=torch.float32, device=device)
            for epoch in range(config["max_epochs"]):
                probe.train(); order = torch.randperm(len(xtr), device=device)
                for start in range(0, len(order), config["batch_size"]):
                    idx = order[start:start + config["batch_size"]]
                    optimizer.zero_grad(set_to_none=True); loss = criterion(probe(xtr[idx]), ytr[idx]); loss.backward(); optimizer.step()
                probe.eval()
                with torch.no_grad():
                    # Equal-weight mean of target-specific validation MAEs.
                    # With complete joint targets this is also the mean over
                    # all case-by-output absolute errors.
                    score = torch.abs(probe(xva) - yva).mean(dim=0).mean().item()
                if score < best:
                    best, best_epoch, stale = score, epoch + 1, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
                else:
                    stale += 1
                    if stale >= config["patience"]: break
            trials.append({"trial": trial_index, "hyperparameters": hp,
                           "validation_mae": best, "best_epoch": best_epoch,
                           "epochs_run": epoch + 1, "epochs": epoch + 1,
                           "state": best_state})
            print(f"      val_mae={best:.3f} epochs={epoch + 1}", flush=True)
        best = min(trials, key=lambda x: x["validation_mae"])
        probe = Probe(input_dim, len(targets)).to(device); probe.load_state_dict(best.pop("state")); probe.eval()
        xte = torch.tensor(encoded["test"][0], dtype=torch.float32, device=device)
        with torch.no_grad(): pred = probe(xte).cpu().numpy()
        truth = encoded["test"][1]
        metrics = {name: measurement_metrics(truth[:, i], pred[:, i], target_tolerances(config, name))
                   for i, name in enumerate(targets)}
        for trial in trials: trial.pop("state", None)
        runs.append({"seed": seed, "best_trial": best, "trials": trials,
                     "metrics": metrics, "test_ids": encoded["test"][2]})
    probe_params = sum(p.numel() for p in Probe(input_dim, len(targets)).parameters())
    return {"status": "complete", "parameters": encoder_params + probe_params,
            "trainable_parameters": probe_params, "encoder_parameters": encoder_params, "runs": runs}


def add_seed_summary(result, target_names, config):
    if result.get("status") != "complete":
        return result
    summary = {}
    for target in target_names:
        summary[target] = {}
        fields = ["mae", "icc_2_1", "bias", "loa_lower", "loa_upper"] + [
            f"within_{t:g}deg" for t in target_tolerances(config, target)]
        for field in fields:
            values = [run["metrics"][target][field] for run in result["runs"]]
            summary[target][field] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
    result["seed_summary"] = summary
    return result


def run_supervised(model_id, spec, splits, targets, config, output_dir, dry_run):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_t, test_t = transforms(config["image_size"], config.get("resize_mode", "distort"))
    loaders = {name: DataLoader(MeasurementDataset(rows, train_t if name == "train" else test_t),
                                batch_size=config["batch_size"], shuffle=name == "train",
                                num_workers=config["num_workers"], pin_memory=torch.cuda.is_available())
               for name, rows in splits.items()}
    probe = build_timm_model(spec, len(targets), pretrained=False)
    counts = parameter_counts(probe)
    if dry_run:
        images, labels, _ = next(iter(loaders["train"])); output = probe(images[:2])
        return {"status": "dry_run_ok", **counts, "input_shape": list(images[:2].shape),
                "output_shape": list(output.shape)}
    runs = []
    for seed in config["seeds"]:
        print(f"  seed={seed}", flush=True)
        seed_everything(seed); trials = []
        for trial_index, hp in enumerate(config["search_space"]):
            print(f"    trial={trial_index} hp={hp}", flush=True)
            model = build_timm_model(spec, len(targets), pretrained=True).to(device)
            ckpt = output_dir / f"{model_id}_seed{seed}_trial{trial_index}.pt"
            outcome = train_trial(model, loaders, device, targets, hp, config, ckpt)
            print(f"      val_mae={outcome['validation_mae']:.3f} epochs={outcome['epochs']}", flush=True)
            trials.append({"trial": trial_index, "hyperparameters": hp, **outcome, "checkpoint": str(ckpt)})
        best = min(trials, key=lambda x: x["validation_mae"])
        model = build_timm_model(spec, len(targets), pretrained=False).to(device)
        model.load_state_dict(torch.load(
            best["checkpoint"], map_location=device, weights_only=True))
        metrics, ids = evaluate(model, loaders["test"], device, targets, config)
        runs.append({"seed": seed, "best_trial": best, "metrics": metrics, "test_ids": ids})
    return {"status": "complete", **counts, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", choices=["aasce", "hvangle"], required=True)
    parser.add_argument("--models", nargs="*", help="model IDs; default: all configured")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resize-mode", choices=["distort", "letterbox"],
                        help="override supervised image resize mode from the config")
    parser.add_argument("--hvangle-split-mode", choices=["official_image", "patient"],
                        help="override the HVAngleEst split; patient mode groups by patient_id")
    parser.add_argument("--dry-run", action="store_true", help="validate data/model shapes without downloads or training")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.resize_mode is not None:
        config["resize_mode"] = args.resize_mode
    if args.dataset == "aasce":
        splits, targets = aasce_splits()
    else:
        split_mode = args.hvangle_split_mode or config.get("hvangle_split_mode", "official_image")
        config["hvangle_split_mode"] = split_mode
        splits, targets = hvangle_splits(
            config["hvangle_unilateral_only"], split_mode, config.get("split_seed", 42))
    chosen = args.models or list(config["models"])
    unknown = sorted(set(chosen) - set(config["models"]))
    if unknown: parser.error(f"unknown models: {', '.join(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": config["protocol_version"], "dataset": args.dataset,
                "targets": targets, "split_sizes": {k: len(v) for k, v in splits.items()},
                "split_ids": {k: [r["id"] for r in v] for k, v in splits.items()},
                "split_patient_ids": {k: sorted({r["patient_id"] for r in v if "patient_id" in r})
                                      for k, v in splits.items()},
                "config": config, "models": {}, "created_unix": time.time()}
    result_path = args.output_dir / f"{args.dataset}_{'dry_run' if args.dry_run else 'results'}.json"
    for model_id in chosen:
        spec = config["models"][model_id]
        print(f"[{model_id}] {spec['family']} / {spec['adaptation']}")
        if spec["family"] in {"timm", "radimagenet"}:
            if spec["family"] == "radimagenet":
                available, detail = medical_availability(spec)
                if not available:
                    manifest["models"][model_id] = {"status": "unavailable", "detail": detail,
                                                     "adaptation": spec["adaptation"]}
                    continue
            try:
                result = run_supervised(model_id, spec, splits, targets, config, args.output_dir, args.dry_run)
                manifest["models"][model_id] = add_seed_summary(result, targets, config)
            except Exception as exc:
                manifest["models"][model_id] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                result_path.write_text(json.dumps(manifest, indent=2))
                if not args.dry_run: raise
        else:
            try:
                result = run_medical_probe(
                    model_id, spec, splits, targets, config, args.output_dir, args.dry_run)
                manifest["models"][model_id] = add_seed_summary(result, targets, config)
            except Exception as exc:
                manifest["models"][model_id] = {
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "adaptation": spec["adaptation"],
                }
                print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
        # Persist after every model so a later adapter failure cannot discard
        # earlier completed predictions.
        result_path.write_text(json.dumps(manifest, indent=2))
    result_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
