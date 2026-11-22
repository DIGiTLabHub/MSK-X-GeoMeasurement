"""Portable storage paths; input records are separate from generated outputs."""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("MSK_WORKSPACE", str(CODE_ROOT))).expanduser().resolve()


def split_manifest(dataset):
    """Return the immutable split manifest distributed with this release."""
    return CODE_ROOT / "records" / "controlled_cross_model_v2" / dataset / f"{dataset}_results.json"


def validate_paper_records(dataset, splits):
    """Reject altered case order, targets, patient groups or missing images."""
    import csv
    import numpy as np
    with (CODE_ROOT / "splits" / f"{dataset}.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    targets = ("cobb",) if dataset == "aasce" else ("hva", "ima")
    for part in ("train", "val", "test"):
        expected = [row for row in rows if row["split"] == part]
        actual = splits[part]
        ids = [row["id"] if isinstance(row, dict) else row.identifier for row in actual]
        if ids != [row["id"] for row in expected]:
            raise ValueError(f"{dataset}: {part} differs from published cases/order or has missing images")
        for row, reference in zip(actual, expected):
            values = row["targets"] if isinstance(row, dict) else row.targets
            atol = 1e-10 if isinstance(row, dict) else 1e-5
            if not np.allclose(values, [float(reference[t]) for t in targets], rtol=0, atol=atol):
                raise ValueError(f"{dataset}: changed target for {reference['id']}")
            image_path = row["image_path"] if isinstance(row, dict) else row.image_path
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"{dataset}: missing image {reference['id']}")
            if dataset == "hvangle":
                patient = row["patient_id"] if isinstance(row, dict) else row.patient_id
                if int(patient) != int(reference["patient_id"]):
                    raise ValueError(f"HVAngleEst: changed patient grouping for {reference['id']}")
