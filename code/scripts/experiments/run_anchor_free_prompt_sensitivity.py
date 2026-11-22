#!/usr/bin/env python3
"""Run matched single-target prompts on the paper's exact test cases.

The case order and reference values are loaded from the frozen benchmark
splits used by the paper.  This prevents a sensitivity run from silently
changing the evaluation sample.  Results are appended as JSON Lines so an
interrupted paid API run can be resumed without repeating completed calls.

Examples
--------
Verify all case IDs and prompts without making API calls:

    python scripts/experiments/run_anchor_free_prompt_sensitivity.py \
        --provider openai --dry-run

Run or resume one provider over every target:

    OPENAI_API_KEY=... python scripts/experiments/run_anchor_free_prompt_sensitivity.py \
        --provider openai

    ANTHROPIC_API_KEY=... python scripts/experiments/run_anchor_free_prompt_sensitivity.py \
        --provider claude
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "scripts" / "benchmarks"))
from cross_model_benchmark import ROOT, aasce_splits, hvangle_splits  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs/results/matched_prompt_evaluation.jsonl"

PROMPTS = {
    "cobb": (
        "This is a spinal AP radiograph used for orthopedic research. "
        "Estimate the Cobb angle for scoliosis. Your response must be ONLY a "
        "single number between 1 and 90. Do not include any text, units, or "
        "explanation. Example: 32"
    ),
    "hva": (
        "This is a foot X-ray used for orthopedic research. Estimate the "
        "Hallux Valgus Angle (HVA): the angle between the long axis of the "
        "first metatarsal and the long axis of the proximal phalanx of the "
        "great toe. Your response must be ONLY a single number between 5 and "
        "90. Do not include any text, units, or explanation. Example: 23"
    ),
    "ima": (
        "This is a foot X-ray used for orthopedic research. Estimate the "
        "Intermetatarsal Angle (IMA): the angle between the long axes of the "
        "first and second metatarsals. Your response must be ONLY a single "
        "number between 5 and 30. Do not include any text, units, or explanation. "
        "Example: 12"
    ),
}

RANGES = {"cobb": (1.0, 90.0), "hva": (5.0, 90.0), "ima": (5.0, 30.0)}
MODEL_DEFAULTS = {
    "openai": "gpt-4o",
    "claude": "claude-sonnet-5",
}


@dataclass(frozen=True)
class Case:
    target: str
    image_id: str
    image_path: Path
    reference: float


def load_cases(targets: Iterable[str]) -> list[Case]:
    requested = set(targets)
    cases: list[Case] = []
    if "cobb" in requested:
        split, _ = aasce_splits()
        test_rows = split["test"]
        if len(test_rows) != 73:
            raise ValueError("AASCE split is not the frozen 73-case test set")
        for row in test_rows:
            image_id, reference = row["id"], float(row["targets"][0])
            cases.append(
                Case("cobb", image_id, ROOT / "AASCE/data/train" / image_id, reference)
            )

    foot_requested = requested.intersection({"hva", "ima"})
    if foot_requested:
        split, target_names = hvangle_splits(True, "patient", 42)
        test_rows = split["test"]
        if len(test_rows) != 176:
            raise ValueError("HVAngleEst split is not the frozen 176-case test set")
        for target_index, target in enumerate(target_names):
            if target in foot_requested:
                for row in test_rows:
                    image_id = row["id"]
                    reference = float(row["targets"][target_index])
                    cases.append(
                        Case(
                            target,
                            image_id,
                            ROOT / "HVAngleEst/HVAngleEst/images" / image_id,
                            reference,
                        )
                    )

    missing = [str(case.image_path) for case in cases if not case.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images; first: {missing[0]}")
    return cases


def encode_jpeg(path: Path) -> str:
    with Image.open(path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_prediction(text: str, target: str) -> float | None:
    low, high = RANGES[target]
    values = [float(token) for token in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    valid = [value for value in values if low <= value <= high]
    return valid[0] if len(valid) == 1 else None


def post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def query_openai(api_key: str, model: str, image_b64: str, prompt: str) -> str:
    data = post_json(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 20,
        },
        {"Authorization": f"Bearer {api_key}"},
    )
    return data["choices"][0]["message"]["content"].strip()


def query_claude(api_key: str, model: str, image_b64: str, prompt: str) -> str:
    data = post_json(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": 20,
            "thinking": {"type": "disabled"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        },
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    return " ".join(
        block["text"] for block in data["content"] if block.get("type") == "text"
    ).strip()


def load_completed(path: Path, provider: str, model: str) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
            if (
                record.get("provider") == provider
                and record.get("model_alias") == model
                and record.get("prompt_variant") == "matched_single_target"
                and record.get("status") == "ok"
            ):
                completed.add((record["target"], record["image_id"]))
    return completed


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def query_with_retry(
    provider: str,
    api_key: str,
    model: str,
    image_b64: str,
    prompt: str,
    attempts: int,
) -> str:
    for attempt in range(1, attempts + 1):
        try:
            if provider == "openai":
                return query_openai(api_key, model, image_b64, prompt)
            return query_claude(api_key, model, image_b64, prompt)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == attempts:
                raise
            wait = min(60, 5 * (2 ** (attempt - 1)))
            detail = type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                detail = f"HTTP {exc.code}"
            print(f"API attempt {attempt} failed ({detail}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "claude"), required=True)
    parser.add_argument("--model", help="Provider model alias; defaults match the paper")
    parser.add_argument(
        "--target", choices=("all", "cobb", "hva", "ima"), default="all"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        help="Development-only cap per target; omit for the paper analysis",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = args.model or MODEL_DEFAULTS[args.provider]
    targets = ("cobb", "hva", "ima") if args.target == "all" else (args.target,)
    cases = load_cases(targets)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        cases = [
            case
            for target in targets
            for case in [item for item in cases if item.target == target][: args.limit]
        ]

    counts = {target: sum(case.target == target for case in cases) for target in targets}
    print(f"provider={args.provider} model={model} cases={counts}")
    for target in targets:
        print(f"{target}: {PROMPTS[target]}")
    if args.dry_run:
        print("Dry run complete: manifests, references, and image paths validated.")
        return 0

    key_names = (
        ("OPENAI_API_KEY",)
        if args.provider == "openai"
        else ("ANTHROPIC_API_KEY",)
    )
    api_key = next(
        (os.environ.get(name, "").strip() for name in key_names if os.environ.get(name, "").strip()),
        "",
    )
    if not api_key:
        raise RuntimeError(f"none of {', '.join(key_names)} is configured")

    completed = load_completed(args.output, args.provider, model)
    pending = [case for case in cases if (case.target, case.image_id) not in completed]
    print(f"completed={len(cases) - len(pending)} pending={len(pending)} output={args.output}")
    for index, case in enumerate(pending, 1):
        image_b64 = encode_jpeg(case.image_path)
        started = time.monotonic()
        try:
            raw = query_with_retry(
                args.provider,
                api_key,
                model,
                image_b64,
                PROMPTS[case.target],
                args.attempts,
            )
            prediction = parse_prediction(raw, case.target)
            status = "ok" if prediction is not None else "parse_error"
            error = None
        except Exception as exc:  # preserve the failed case for audit and resume
            raw = ""
            prediction = None
            status = "api_error"
            error = (
                f"HTTPError status={exc.code}"
                if isinstance(exc, urllib.error.HTTPError)
                else type(exc).__name__
            )

        record = {
            "provider": args.provider,
            "model_alias": model,
            "prompt_variant": "matched_single_target",
            "target": case.target,
            "image_id": case.image_id,
            "reference": case.reference,
            "prediction": prediction,
            "raw_response": raw,
            "status": status,
            "error": error,
            "temperature": 0.0 if args.provider != "claude" else None,
            "thinking": "disabled" if args.provider == "claude" else None,
            "prompt": PROMPTS[case.target],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        append_record(args.output, record)
        print(
            f"[{index}/{len(pending)}] {case.target} {case.image_id}: "
            f"prediction={prediction} status={status}"
        )
        if status == "api_error":
            print("Stopping after an API error; rerun the same command to resume.", file=sys.stderr)
            return 1
        time.sleep(max(0.0, args.delay))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
