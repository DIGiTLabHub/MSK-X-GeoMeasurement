#!/usr/bin/env python3
"""Verify the release or recompute current-paper statistics without images/GPU."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def verify_archive():
    import csv
    problems, count = [], 0
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = ROOT / name
        count += 1
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(name)
    splits = {}
    for dataset, expected in [("aasce", (336, 72, 73)), ("hvangle", (829, 172, 176))]:
        with (ROOT / "splits" / f"{dataset}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        parts = {name: [r for r in rows if r["split"] == name] for name in ("train", "val", "test")}
        sizes = tuple(len(parts[name]) for name in parts)
        if sizes != expected or len({row["id"] for row in rows}) != len(rows):
            problems.append(f"{dataset}: split sizes or identifiers")
        splits[dataset] = {name: len(value) for name, value in parts.items()}
        if dataset == "hvangle":
            patients = {name: {r["patient_id"] for r in group} for name, group in parts.items()}
            if any(patients[a] & patients[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
                problems.append("HVAngleEst: patient overlap")
            if tuple(map(len, patients.values())) != (617, 132, 133):
                problems.append("HVAngleEst: patient counts")
    return {"passed": not problems, "verified_files": count, "split_sizes": splits, "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify", help="Check reference checksums and split integrity; standard library only")
    check.add_argument("--report", type=Path)
    stats = sub.add_parser("statistics", help="Recompute and verify paper-scoped results from saved predictions")
    stats.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "paper_statistics")
    stats.add_argument("--replicates", type=int, default=10000)
    stats.add_argument("--figures", action="store_true", help="Also regenerate the prompt-distribution figure")
    args = parser.parse_args()
    integrity = verify_archive()
    if not integrity["passed"]:
        print(json.dumps(integrity, indent=2)); return 1
    if args.command == "verify":
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(integrity, indent=2) + "\n")
        print(json.dumps(integrity, indent=2)); return 0
    if args.replicates < 2:
        parser.error("--replicates must be at least 2")
    # Keep the small CPU replay independent of a machine's BLAS thread defaults.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    import pandas as pd
    from msk_geomeasurement.replay import recompute, verify_results
    print("Recomputing saved-prediction statistics; no training or provider calls.", flush=True)
    result, distributions = recompute(args.replicates)
    output = args.output_dir.expanduser().resolve()
    protected = (ROOT / "records", ROOT / "results" / "expected", ROOT / "splits")
    if any(output == p or p in output.parents for p in protected):
        parser.error("output directory must be separate from immutable reference inputs")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in result.items():
        (output / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    report = verify_results(result) if args.replicates == 10000 else {"passed": None, "reason": "Exploratory resample count; paper uses 10000"}
    report.update({"bootstrap_replicates": args.replicates, "comparison_families": {"architecture": 9, "proportional_bias": 9, "frozen_pair": 3, "prompt_to_median": 6}})
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.figures:
        from msk_geomeasurement.figures import prompt_distribution
        prompt_distribution(distributions, result["constant_baselines"], output)
    print(json.dumps(report, indent=2))
    return 1 if report["passed"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
