"""Independent statistical checks for unequal patient-group sizes and replay."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from msk_geomeasurement.bootstrap import agreement_intervals, paired_interval, mean_interval
from msk_geomeasurement.metrics import icc_2_1, unit_means, holm_adjust


class MeasurementTests(unittest.TestCase):
    def test_perfect_agreement_and_constant(self):
        y = np.array([2., 4., 7., 8., 11.])
        self.assertAlmostEqual(icc_2_1(y, y), 1.)
        self.assertAlmostEqual(icc_2_1(y, np.full_like(y, y.mean())), 0.)

    def test_holm_unsorted_pvalues(self):
        np.testing.assert_allclose(holm_adjust([.02, .001, .04]), [.04, .003, .04])

    def test_patient_weighting_differs_from_image_weighting(self):
        differences = np.array([6., 6., 0.])
        groups = np.array([10, 10, 20])
        self.assertEqual(differences.mean(), 4.)
        self.assertEqual(unit_means(differences, groups).mean(), 3.)

    def test_vectorized_agreement_matches_literal_resampling(self):
        y = np.array([7., 9., 12., 16., 18., 22., 30.])
        p = np.array([5., 10., 10., 18., 19., 20., 26.])
        clusters = np.array([10, 10, 20, 30, 30, 30, 40])
        count, seed = 73, 123
        fast = agreement_intervals(y, p, clusters, np.random.default_rng(seed), count)
        rng, result = np.random.default_rng(seed), {k: [] for k in fast}
        members = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
        for _ in range(count):
            selected = rng.integers(0, len(members), len(members))
            ids = np.concatenate([members[i] for i in selected])
            yy, pp = y[ids], p[ids]
            difference, magnitude = pp - yy, (pp + yy) / 2
            bias, sd = difference.mean(), difference.std(ddof=1)
            result['bias_ci'].append(bias)
            result['loa_lower_ci'].append(bias - 1.96 * sd)
            result['loa_upper_ci'].append(bias + 1.96 * sd)
            result['icc_ci'].append(icc_2_1(yy, pp))
            denominator = np.sum((magnitude - magnitude.mean()) ** 2)
            result['proportional_bias_slope_ci'].append(np.sum((magnitude - magnitude.mean()) * (difference - difference.mean())) / denominator if denominator > 0 else np.nan)
        for key, values in result.items():
            np.testing.assert_allclose(fast[key], np.quantile(np.asarray(values)[np.isfinite(values)], [.025, .975]), rtol=1e-10, atol=1e-10)

    def test_resampling_units_for_paired_and_descriptive_intervals(self):
        v, groups = np.array([6., 6., 0.]), np.array([10, 10, 20])
        count, seed = 500, 7
        rng = np.random.default_rng(seed)
        draw = rng.integers(0, 2, (count, 2))
        patient_means = np.array([6., 0.])
        paired = np.quantile(patient_means[draw].mean(1), [.025, .975])
        np.testing.assert_allclose(paired_interval(v, groups, np.random.default_rng(seed), count), paired)
        total, size = np.array([12., 0.]), np.array([2., 1.])
        image_weighted = np.quantile(total[draw].sum(1) / size[draw].sum(1), [.025, .975])
        np.testing.assert_allclose(mean_interval(v, groups, np.random.default_rng(seed), count), image_weighted)


if __name__ == "__main__":
    unittest.main()
