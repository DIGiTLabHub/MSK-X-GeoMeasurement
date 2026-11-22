#!/usr/bin/env python3
"""Create publication-ready CSV and Markdown tables from completed benchmark JSON."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    rows = []
    for model_id, model in data["models"].items():
        base = {"dataset": data["dataset"], "model": model_id, "status": model["status"],
                "parameters_m": model.get("parameters", 0) / 1e6,
                "trainable_parameters_m": model.get("trainable_parameters", 0) / 1e6}
        if model["status"] != "complete":
            rows.append(base); continue
        for target, metrics in model["seed_summary"].items():
            row = {**base, "target": target}
            for metric, values in metrics.items():
                row[f"{metric}_mean"] = values["mean"]
                row[f"{metric}_sd"] = values["sd"]
            rows.append(row)
    frame = pd.DataFrame(rows)
    prefix = args.output_prefix or args.result.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(str(prefix) + ".csv", index=False)
    markdown = frame.to_markdown(index=False, floatfmt=".3f")
    Path(str(prefix) + ".md").write_text("# Controlled benchmark summary\n\n" + markdown + "\n")
    print(f"Saved {prefix}.csv and {prefix}.md")


if __name__ == "__main__":
    main()
