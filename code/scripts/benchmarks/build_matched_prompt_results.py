#!/usr/bin/env python3
"""Validate matched-prompt JSONL records and build paper result manifests.

The API runner appends audit records and may retain failed attempts.  This
script accepts only the latest successful record for each expected
provider-target-case tuple, verifies the frozen test order and references,
then writes the compact metric manifests consumed by revision_statistics.py.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from cross_model_benchmark import ROOT, aasce_splits, hvangle_splits, measurement_metrics


INPUTS = {
    "GPT-4o": ROOT / "outputs/results/matched_prompt_openai.jsonl",
    "Claude Sonnet 5": ROOT / "outputs/results/matched_prompt_claude.jsonl",
}
ALIASES = {"GPT-4o": "gpt-4o", "Claude Sonnet 5": "claude-sonnet-5"}
PROVIDERS = {"GPT-4o": "openai", "Claude Sonnet 5": "claude"}
OUT = ROOT / "outputs/benchmarks/controlled_cross_model_v2/prompt_matched"
BOOTSTRAPS = 10_000
RNG_SEED = 20260809
PARSER_RANGES = {"cobb": (1.0, 90.0), "hva": (5.0, 90.0), "ima": (5.0, 30.0)}


def parse_prediction(text: str, target: str) -> float | None:
    low, high = PARSER_RANGES[target]
    values = [float(token) for token in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    valid = [value for value in values if low <= value <= high]
    return valid[0] if len(valid) == 1 else None


def audit_records(path: Path, provider: str, alias: str) -> list[dict]:
    """Re-parse every raw response and summarize retained attempts by target."""
    attempts, successes, failures = Counter(), Counter(), Counter()
    successful_cases = {target: set() for target in PARSER_RANGES}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if record.get("provider") != provider or record.get("model_alias") != alias:
                raise ValueError(f"unexpected provider/model at {path}:{line_number}")
            target = record["target"]
            attempts[target] += 1
            if record.get("status") != "ok":
                failures[target] += 1
                continue
            parsed = parse_prediction(str(record.get("raw_response", "")), target)
            if parsed is None or not np.isclose(parsed, float(record["prediction"]), rtol=0, atol=1e-12):
                raise ValueError(f"saved prediction fails strict parser replay at {path}:{line_number}")
            successes[target] += 1
            successful_cases[target].add(record["image_id"])
    return [{
        "provider": provider,
        "model_alias": alias,
        "target": target.upper(),
        "parser_interval": f"{PARSER_RANGES[target][0]:g} <= x <= {PARSER_RANGES[target][1]:g}",
        "n_attempts": attempts[target],
        "n_successful_attempts": successes[target],
        "n_unique_successful_cases": len(successful_cases[target]),
        "n_failed_attempts": failures[target],
        "all_successful_predictions_match_parser_replay": True,
    } for target in ("cobb", "hva", "ima")]


def load_successes(path: Path, provider: str, alias: str) -> dict[tuple[str, str], dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    successes: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if record.get("provider") != provider or record.get("model_alias") != alias:
                raise ValueError(f"unexpected provider/model at {path}:{line_number}")
            if record.get("prompt_variant") != "matched_single_target":
                raise ValueError(f"unexpected prompt variant at {path}:{line_number}")
            if record.get("status") == "ok":
                key = (record["target"], record["image_id"])
                successes[key] = record
    return successes


def bootstrap_mae(truth: np.ndarray, prediction: np.ndarray, rng) -> list[float]:
    values = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        sample = rng.integers(0, len(truth), len(truth))
        values[index] = np.mean(np.abs(prediction[sample] - truth[sample]))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def metric_block(truth: np.ndarray, prediction: np.ndarray, tolerance: float, rng) -> dict:
    metric = measurement_metrics(truth, prediction, [tolerance])
    metric["mae_ci"] = bootstrap_mae(truth, prediction, rng)
    metric["prediction_sd"] = float(np.std(prediction, ddof=1))
    metric["reference_sd"] = float(np.std(truth, ddof=1))
    return metric


def ordered_predictions(records, target: str, identifiers: list[str], truth: np.ndarray) -> np.ndarray:
    expected = {(target, identifier) for identifier in identifiers}
    observed = {key for key in records if key[0] == target}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"{target}: missing {len(missing)} and extra {len(extra)} successful records"
        )
    selected = [records[(target, identifier)] for identifier in identifiers]
    saved_truth = np.asarray([record["reference"] for record in selected], float)
    if not np.allclose(saved_truth, truth, rtol=0.0, atol=1e-9):
        raise ValueError(f"{target}: saved references differ from the frozen split")
    return np.asarray([record["prediction"] for record in selected], float)


def main() -> None:
    aasce, _ = aasce_splits()
    aasce_test = aasce["test"]
    cobb_ids = [row["id"] for row in aasce_test]
    cobb_truth = np.asarray([row["targets"][0] for row in aasce_test], float)

    hvangle, target_names = hvangle_splits(True, "patient", 42)
    foot_test = hvangle["test"]
    foot_ids = [row["id"] for row in foot_test]
    foot_truth = np.asarray([row["targets"] for row in foot_test], float)

    protocol = "matched single-target prompts"
    model_sources = {
        "GPT-4o": {
            "source": "outputs/results/matched_prompt_openai.jsonl",
            "interaction": "August 9, 2026 matched single-target rerun",
        },
        "Claude Sonnet 5": {
            "source": "outputs/results/matched_prompt_claude.jsonl",
            "interaction": "matched single-target rerun with thinking disabled",
        },
    }
    cobb_output = {
        "protocol": protocol,
        "model_sources": model_sources,
        "target_definition": "AASCE angles.csv column 0 (released primary/largest curve angle)",
        "test_ids": cobb_ids,
        "model_aliases": ALIASES,
        "models": {},
    }
    foot_output = {
        "protocol": protocol,
        "model_sources": model_sources,
        "split": "patient-disjoint unilateral pool, seed 42",
        "test_ids": foot_ids,
        "model_aliases": ALIASES,
        "models": {},
    }
    rng = np.random.default_rng(RNG_SEED)
    parser_audit = []
    for display_name, path in INPUTS.items():
        parser_audit.extend(audit_records(
            path, PROVIDERS[display_name], ALIASES[display_name]))
        records = load_successes(path, PROVIDERS[display_name], ALIASES[display_name])
        if len(records) != 425:
            raise ValueError(f"{display_name}: expected 425 successful unique records, found {len(records)}")

        cobb_prediction = ordered_predictions(records, "cobb", cobb_ids, cobb_truth)
        cobb_output["models"][display_name] = metric_block(
            cobb_truth, cobb_prediction, 5.0, rng
        )

        foot_output["models"][display_name] = {}
        for index, target in enumerate(target_names):
            prediction = ordered_predictions(
                records, target, foot_ids, foot_truth[:, index]
            )
            tolerance = 5.0 if target == "hva" else 3.0
            foot_output["models"][display_name][target] = metric_block(
                foot_truth[:, index], prediction, tolerance, rng
            )

    OUT.mkdir(parents=True, exist_ok=True)
    cobb_path = OUT / "aasce_matched_prompt_results.json"
    foot_path = OUT / "hvangle_matched_prompt_patient_results.json"
    cobb_path.write_text(json.dumps(cobb_output, indent=2) + "\n", encoding="utf-8")
    foot_path.write_text(json.dumps(foot_output, indent=2) + "\n", encoding="utf-8")
    audit_path = OUT / "prompt_parser_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parser_audit[0]))
        writer.writeheader()
        writer.writerows(parser_audit)
    print(cobb_path)
    print(foot_path)
    print(audit_path)


if __name__ == "__main__":
    main()
