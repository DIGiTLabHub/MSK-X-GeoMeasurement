"""Measurement statistics used by the executed benchmark."""
from __future__ import annotations
import numpy as np
from scipy import stats


def icc_2_1(truth, prediction):
    x = np.column_stack([np.asarray(truth, float), np.asarray(prediction, float)])
    n, k = x.shape
    if n < 2:
        return float("nan")
    rows, columns, grand = x.mean(1), x.mean(0), x.mean()
    msr = k * np.square(rows - grand).sum() / (n - 1)
    msc = n * np.square(columns - grand).sum() / (k - 1)
    residual = np.square(x - rows[:, None] - columns + grand).sum()
    mse = residual / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denominator) if denominator else float("nan")


def measurement_metrics(truth, prediction, tolerance):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    difference = prediction - truth
    bias, sd = float(difference.mean()), float(difference.std(ddof=1))
    return {"mae": float(np.abs(difference).mean()), "icc": icc_2_1(truth, prediction),
            "bias": bias, "loa_lower": bias - 1.96 * sd, "loa_upper": bias + 1.96 * sd,
            "tolerance": float(100 * np.mean(np.abs(difference) <= tolerance))}


def unit_means(values, clusters):
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    return np.asarray([values[clusters == unit].mean() for unit in np.unique(clusters)])


def paired_test(difference, clusters):
    values = unit_means(difference, clusters)
    result = stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox", method="auto")
    return float(result.statistic), float(result.pvalue)


def holm_adjust(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    adjusted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def proportional_bias(truth, prediction, clusters):
    truth, prediction, clusters = map(np.asarray, (truth, prediction, clusters))
    magnitude, difference = (truth + prediction) / 2, prediction - truth
    if len(np.unique(clusters)) == len(clusters):
        result = stats.linregress(magnitude, difference)
        return float(result.slope), float(result.pvalue)
    design = np.column_stack([np.ones(len(magnitude)), magnitude])
    bread = np.linalg.inv(design.T @ design)
    coefficient = bread @ design.T @ difference
    residual = difference - design @ coefficient
    meat = np.zeros((2, 2), float)
    for unit in np.unique(clusters):
        selected = clusters == unit
        score = design[selected].T @ residual[selected]
        meat += np.outer(score, score)
    n, k, groups = len(magnitude), 2, len(np.unique(clusters))
    covariance = groups / (groups - 1) * (n - 1) / (n - k) * bread @ meat @ bread
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    statistic = float(coefficient[1] / standard_error) if standard_error else np.inf
    return float(coefficient[1]), float(2 * stats.t.sf(abs(statistic), groups - 1))
