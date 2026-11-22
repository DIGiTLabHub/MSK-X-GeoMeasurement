"""Read immutable execution records and public-release split/target metadata."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
RECORDS = CODE_ROOT / "records" / "controlled_cross_model_v2"
SUPERVISED = {"ResNet50": "resnet50", "EfficientNet-B0": "efficientnet_b0",
              "ConvNeXt-T": "convnext_tiny", "ViT-B/16": "vit_b16"}
PAPER_MODELS = ("EfficientNet-B0", "ConvNeXt-T", "ViT-B/16")
FROZEN = {"BiomedCLIP": "biomedclip_probe", "MedImageInsight": "medimageinsight_probe",
          "MedCLIP-ResNet50": "medclip_resnet_probe", "MedCLIP-Swin": "medclip_vit_probe",
          "DAX-ResNet50-vA": "dax_resnet50_probe", "MedSAM ViT-B": "medsam_encoder_probe"}
PAPER_ENCODERS = ("BiomedCLIP", "MedImageInsight", "MedCLIP-Swin", "MedSAM ViT-B")


def load_json(path):
    return json.loads(Path(path).read_text())


def target_info():
    result = {}
    for dataset, targets in [("aasce", ("cobb",)), ("hvangle", ("hva", "ima"))]:
        with (CODE_ROOT / "splits" / f"{dataset}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for target in targets:
            test = [row for row in rows if row["split"] == "test"]
            result[target] = {"train": np.asarray([float(r[target]) for r in rows if r["split"] == "train"]),
                              "test": np.asarray([float(r[target]) for r in test]),
                              "ids": [r["id"] for r in test], "tolerance": 3.0 if target == "ima" else 5.0,
                              "clusters": np.asarray([r["id"] if target == "cobb" else int(r["patient_id"]) for r in test])}
    return result


def clusters_for(target, ids, info):
    mapping = dict(zip(info[target]["ids"], info[target]["clusters"]))
    return np.asarray([mapping[value] for value in ids])


def arrays(record, target):
    runs = record["runs"]
    if [run["seed"] for run in runs] != [17, 42, 73]:
        raise ValueError("Execution record must contain seeds 17, 42, 73 in order")
    ids = runs[0]["test_ids"]
    truth = np.asarray(runs[0]["metrics"][target]["ground_truth"], float)
    for run in runs:
        if run["test_ids"] != ids or not np.array_equal(run["metrics"][target]["ground_truth"], truth):
            raise ValueError("Run identifiers or reference values differ across seeds")
    predictions = np.asarray([run["metrics"][target]["predictions"] for run in runs], float)
    return ids, truth, predictions


def model_record(dataset, model, frozen=False):
    path = RECORDS / ("medical" if frozen else "") / dataset / f"{dataset}_results.json"
    return load_json(path)["models"][(FROZEN if frozen else SUPERVISED)[model]]
