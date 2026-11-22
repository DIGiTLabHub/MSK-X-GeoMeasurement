#!/usr/bin/env python3
"""Plot the retained patient-split HVA/IMA spatial-to-measurement experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


from workspace_paths import ROOT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from foot_axis_data import hvangle_splits  # noqa: E402


REFERENCE_COLOR = "#0072BD"
PREDICTION_COLOR = "#D95319"
AXIS_NAMES = ("GT", "M1", "M2")


def axial_error(vector_a, vector_b):
    cosine = abs(float(np.dot(vector_a, vector_b))) / max(
        float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b)), 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))


def draw_axis(axis, center, direction, length, color, linestyle, marker, zorder):
    direction = np.asarray(direction, float)
    direction /= max(np.linalg.norm(direction), 1e-8)
    center = np.asarray(center, float)
    endpoints = np.stack([center - direction * length / 2, center + direction * length / 2])
    line, = axis.plot(
        endpoints[:, 0], endpoints[:, 1], color=color, linewidth=2.8,
        linestyle=linestyle, marker=marker, markersize=5.6,
        markeredgecolor="white", markeredgewidth=.8, zorder=zorder,
    )
    line.set_path_effects([
        path_effects.Stroke(linewidth=4.2, foreground="black", alpha=.65),
        path_effects.Normal(),
    ])
    return endpoints


def overlay_legend(axis):
    handles = [
        Line2D([0], [0], color=REFERENCE_COLOR, linewidth=2.8, linestyle="-",
               marker="s", markersize=5.5, markeredgecolor="white",
               markeredgewidth=.8, label="Reference"),
        Line2D([0], [0], color=PREDICTION_COLOR, linewidth=2.8,
               linestyle=(0, (6, 2)), marker="o", markersize=5.5,
               markeredgecolor="white", markeredgewidth=.8, label="Prediction"),
    ]
    axis.legend(handles=handles, loc="lower center", fontsize=8, framealpha=.94,
                facecolor="white", edgecolor="0.35", ncol=2,
                handlelength=2.8, columnspacing=1.1, borderpad=.35)


def style_error_axis(axis):
    axis.grid(True, color="0.86", linewidth=.65)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("0.2")
        spine.set_linewidth(.8)
    axis.tick_params(direction="out", width=.8, length=3.5)


def add_fit(axis, x, y):
    coefficients = np.polyfit(x, y, 1)
    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    axis.plot(x_fit, np.polyval(coefficients, x_fit), color=PREDICTION_COLOR,
              linewidth=1.8, zorder=2)
    correlation = float(np.corrcoef(x, y)[0, 1])
    axis.text(.97, .93, f"Pearson $r={correlation:.2f}$", transform=axis.transAxes,
              ha="right", va="top",
              bbox={"boxstyle": "square,pad=.2", "fc": "white", "ec": "0.65"})


def case_data(run, records):
    record_map = {record.identifier: record for record in records}
    hva = run["test"]["measurements"]["hva"]
    ima = run["test"]["measurements"]["ima"]
    hva_error = np.abs(np.asarray(hva["predictions"]) - np.asarray(hva["ground_truth"]))
    ima_error = np.abs(np.asarray(ima["predictions"]) - np.asarray(ima["ground_truth"]))
    rows, axis_errors = [], []
    for index, (identifier, output) in enumerate(zip(
            run["test"]["identifiers"], run["test"]["predicted_axis_directions"])):
        record = record_map[identifier]
        true_points = record.points_norm * np.asarray([record.width, record.height])
        true_directions = np.stack([
            true_points[2 * axis_index + 1] - true_points[2 * axis_index]
            for axis_index in range(3)
        ])
        predicted_directions = np.asarray(output["directions"])
        orientation_errors = [
            axial_error(predicted_directions[axis_index], true_directions[axis_index])
            for axis_index in range(3)
        ]
        axis_errors.append(orientation_errors)
        rows.append({
            "record": record,
            "true_points": true_points,
            "output": output,
            "mean_measurement_error": float((hva_error[index] + ima_error[index]) / 2),
        })
    return rows, np.asarray(axis_errors), hva_error, ima_error


def draw_case(axis, row, title):
    record = row["record"]
    image = np.asarray(Image.open(record.image_path).convert("L"))
    height, width = image.shape
    side = max(height, width)
    offset_x = (side - width) // 2
    offset_y = (side - height) // 2
    canvas = np.zeros((side, side), dtype=image.dtype)
    canvas[offset_y:offset_y + height, offset_x:offset_x + width] = image
    display_offset = np.asarray([offset_x, offset_y], dtype=float)

    # Preserve the acquisition aspect ratio and use display-only padding so
    # both case panels occupy equal square plotting areas.  Overlay coordinates
    # are translated by the same offset; no image or geometry is rescaled.
    true_points = row["true_points"] + display_offset
    output = row["output"]
    axis.imshow(canvas, cmap="gray", vmin=0, vmax=255)
    predicted_centers = np.asarray(output["centers_px"]) + display_offset
    predicted_directions = np.asarray(output["directions"])
    for axis_index, name in enumerate(AXIS_NAMES):
        true_line = true_points[2 * axis_index:2 * axis_index + 2]
        length = float(np.linalg.norm(true_line[1] - true_line[0]))
        draw_axis(axis, predicted_centers[axis_index], predicted_directions[axis_index],
                  length, PREDICTION_COLOR, (0, (6, 2)), "o", 5)
        endpoints = draw_axis(
            axis, true_line.mean(0), true_line[1] - true_line[0], length,
            REFERENCE_COLOR, "-", "s", 4,
        )
        label = axis.text(endpoints[1, 0] + 5, endpoints[1, 1], name,
                          color="white", fontsize=8, weight="bold", zorder=7)
        label.set_path_effects([
            path_effects.Stroke(linewidth=2.2, foreground="black"),
            path_effects.Normal(),
        ])
    axis.set_title(title)
    overlay_legend(axis)
    axis.axis("off")


def plot(args):
    results = json.loads(args.results.read_text())
    run = next(item for item in results if int(item["seed"]) == args.seed)
    rows, axis_errors, hva_error, ima_error = case_data(
        run, hvangle_splits(args.split_seed)["test"])
    hva_axis_error = axis_errors[:, :2].mean(axis=1)
    ima_axis_error = axis_errors[:, 1:].mean(axis=1)
    measurement_error = np.asarray([row["mean_measurement_error"] for row in rows])
    order = np.argsort(measurement_error)
    representative = rows[int(order[len(order) // 2])]
    high_error = rows[int(order[min(len(order) - 1, round(.9 * (len(order) - 1)))])]

    figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.2))
    draw_case(axes[0, 0], representative, "(a) Representative three-axis output")
    draw_case(axes[0, 1], high_error, "(b) High-error three-axis output")

    axes[1, 0].scatter(hva_axis_error, hva_error, s=24, facecolor="white",
                       edgecolor=REFERENCE_COLOR, linewidth=.9, zorder=3)
    add_fit(axes[1, 0], hva_axis_error, hva_error)
    axes[1, 0].set_xlabel("Mean GT/M1 orientation error ($^\\circ$)")
    axes[1, 0].set_ylabel("Absolute HVA error ($^\\circ$)")
    axes[1, 0].set_title("(c) HVA error propagation")
    style_error_axis(axes[1, 0])

    axes[1, 1].scatter(ima_axis_error, ima_error, s=24, facecolor="white",
                       edgecolor=REFERENCE_COLOR, linewidth=.9, marker="s", zorder=3)
    add_fit(axes[1, 1], ima_axis_error, ima_error)
    axes[1, 1].set_xlabel("Mean M1/M2 orientation error ($^\\circ$)")
    axes[1, 1].set_ylabel("Absolute IMA error ($^\\circ$)")
    axes[1, 1].set_title("(d) IMA error propagation")
    style_error_axis(axes[1, 1])

    # Both quantities are angular errors. Use one common degree scale so the
    # panel geometry and fitted slopes can be compared directly.
    for axis in axes[1, :]:
        axis.set_xlim(-1, 50)
        axis.set_ylim(-1, 50)
        axis.set_xticks(np.arange(0, 51, 10))
        axis.set_yticks(np.arange(0, 51, 10))
        axis.set_aspect("equal", adjustable="box")

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    if args.png is not None:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.png, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path,
        default=ROOT / "outputs/benchmarks/controlled_cross_model_v2/"
        "hvangle_axis_geometry/results.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/benchmarks/controlled_cross_model_v2/"
        "hvangle_axis_geometry/fig_protocol_geometry.pdf",
    )
    parser.add_argument("--png", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    plot(parse_args())
