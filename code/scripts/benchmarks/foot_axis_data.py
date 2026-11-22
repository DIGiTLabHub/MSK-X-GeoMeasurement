"""Executed foot-axis data and angle helpers; no landmark-heatmap training."""
from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from workspace_paths import ROOT, split_manifest, validate_paper_records

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
    validate_paper_records("hvangle", result)
    return result
