"""Data-free regeneration of the matched-prompt distribution figure."""
from __future__ import annotations
import numpy as np


def prompt_distribution(distributions, constants, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    labels = ["Reference", "GPT-4o", "Claude Sonnet 5"]
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), sharey=True)
    for ax, target in zip(axes, ("cobb", "hva", "ima")):
        values = [distributions[target][label] for label in labels]
        parts = ax.violinplot(values, positions=np.arange(3), vert=False,
                              showmeans=False, showmedians=True, widths=.75)
        for body, color in zip(parts["bodies"], ("#4D4D4D", "#0072B2", "#D55E00")):
            body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.42)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[key].set_color("0.25"); parts[key].set_linewidth(.8)
        for row in constants:
            if row["target"] != target.upper():
                continue
            median = "median" in row["predictor"].lower()
            ax.axvline(row["value"], color="#A6761D" if median else "#666666",
                       linestyle="--" if median else ":", linewidth=1.2,
                       label=row["predictor"] if ax is axes[0] else None)
        ax.set_title("Cobb" if target == "cobb" else target.upper())
        ax.set_xlabel("Prediction / reference ($^\\circ$)")
        ax.set_yticks(np.arange(3))
        if ax is axes[0]:
            ax.set_yticklabels(labels)
        ax.grid(axis="x", color="0.88", linewidth=.6); ax.set_axisbelow(True)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, .08, 1, 1))
    figure.savefig(output / "fig_prompt_distributions.pdf", bbox_inches="tight")
    figure.savefig(output / "fig_prompt_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
