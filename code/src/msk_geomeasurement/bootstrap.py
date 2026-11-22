"""Vectorized replay of the original patient-cluster bootstrap.

Draw order and RNG consumption match the executed scalar implementation.
Image-weighted summaries retain all images in each sampled patient block;
paired contrasts first average differences within each patient.
"""
from __future__ import annotations
import numpy as np
from .metrics import unit_means


def interval(values):
    values = np.asarray(values, float)
    return np.quantile(values[np.isfinite(values)], [.025, .975]).tolist()


def draw_counts(clusters, rng, replicates):
    unique, inverse = np.unique(clusters, return_inverse=True)
    sampled = rng.integers(0, len(unique), (replicates, len(unique)))
    counts = np.zeros((replicates, len(unique)), float)
    np.add.at(counts, (np.arange(replicates)[:, None], sampled), 1)
    return counts, inverse, len(unique)


def mean_interval(values, clusters, rng, replicates):
    counts, inverse, groups = draw_counts(clusters, rng, replicates)
    size = np.bincount(inverse, minlength=groups)
    total = np.bincount(inverse, weights=values, minlength=groups)
    return interval((counts @ total) / (counts @ size))


def paired_interval(difference, clusters, rng, replicates):
    means = unit_means(difference, clusters)
    sampled = rng.integers(0, len(means), (replicates, len(means)))
    return interval(means[sampled].mean(1))


def agreement_intervals(truth, prediction, clusters, rng, replicates):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    counts, inverse, groups = draw_counts(clusters, rng, replicates)
    def summed(values):
        return counts @ np.bincount(inverse, weights=values, minlength=groups)
    n = summed(np.ones(len(truth)))
    difference, magnitude = prediction - truth, (prediction + truth) / 2
    sd, sm = summed(difference), summed(magnitude)
    bias = sd / n
    spread = np.sqrt(np.maximum((summed(difference ** 2) - sd ** 2 / n) / (n - 1), 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (summed(magnitude * difference) - sm * sd / n) / (summed(magnitude ** 2) - sm ** 2 / n)
        sy, sp = summed(truth), summed(prediction)
        sy2, sp2, syp = summed(truth ** 2), summed(prediction ** 2), summed(truth * prediction)
        grand = (sy + sp) / (2 * n)
        ssr = .5 * (sy2 + sp2 + 2 * syp) - 2 * n * grand ** 2
        msc = (sy - sp) ** 2 / (2 * n)
        sse = sy2 + sp2 - 2 * n * grand ** 2 - ssr - msc
        msr, mse = ssr / (n - 1), sse / (n - 1)
        icc = (msr - mse) / (msr + mse + 2 / n * (msc - mse))
    return {"bias_ci": interval(bias), "loa_lower_ci": interval(bias - 1.96 * spread),
            "loa_upper_ci": interval(bias + 1.96 * spread), "icc_ci": interval(icc),
            "proportional_bias_slope_ci": interval(slope)}
