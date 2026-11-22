#!/usr/bin/env python3
"""Validate local public datasets against the paper's immutable partitions."""
import argparse
import json
import os
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("MSK_WORKSPACE", str(CODE_ROOT))))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    os.environ["MSK_WORKSPACE"] = str(args.workspace.expanduser().resolve())
    sys.path.insert(0, str(CODE_ROOT / "scripts/benchmarks"))
    from cross_model_benchmark import aasce_splits, hvangle_splits
    from workspace_paths import ROOT, split_manifest
    result = {}
    for dataset in ("aasce", "hvangle"):
        splits, _ = aasce_splits() if dataset == "aasce" else hvangle_splits(True, "patient", 42)
        expected = json.loads(split_manifest(dataset).read_text())["split_ids"]
        for part, records in splits.items():
            if [r["id"] for r in records] != expected[part]:
                raise ValueError(f"{dataset}: split order or membership differs for {part}")
        result[dataset] = {"images": {part: len(rows) for part, rows in splits.items()}}
        if dataset == "hvangle":
            result[dataset]["patients"] = {part: len({r["patient_id"] for r in rows}) for part, rows in splits.items()}
            missing = [r["id"] for rows in splits.values() for r in rows
                       if not (ROOT / "HVAngleEst/HVAngleEst/annotations/masks" / Path(r["id"]).with_suffix(".png")).is_file()]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} released axis masks")
            result[dataset]["axis_masks"] = "present"
    result["passed"] = True
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
