#!/usr/bin/env python3
"""Statistical analyses requested during the TMLR major revision.

The script is deliberately read-only with respect to experiment outputs.  It
reconstructs constant baselines from training targets, audits prediction
dispersion, and performs paired inference on absolute errors averaged across
training seeds for each image. AASCE uses images as independent units;
HVAngleEst uses patient-cluster resampling and patient-mean paired differences.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

from cross_model_benchmark import aasce_splits, hvangle_splits, measurement_metrics


from workspace_paths import ROOT
OUT = ROOT / "outputs" / "benchmarks" / "revision_statistics_v2"
BOOTSTRAPS = 10_000
RNG_SEED = 20260807
PROMPT_RESPONSE_RANGES = {
    "cobb": (1.0, 90.0),
    "hva": (5.0, 90.0),
    "ima": (5.0, 30.0),
}


MEDICAL_FILES = {
    "BiomedCLIP": {
        "aasce": ROOT / "outputs/benchmarks/controlled_cross_model_v2/medical/aasce/aasce_results.json",
        "hvangle": ROOT / "outputs/benchmarks/controlled_cross_model_v2/medical/hvangle/hvangle_results.json",
        "key": "biomedclip_probe",
    },
    "MedImageInsight": {
        "aasce": ROOT / "outputs/benchmarks/controlled_cross_model_v2/medical/aasce/aasce_results.json",
        "hvangle": ROOT / "outputs/benchmarks/controlled_cross_model_v2/medical/hvangle/hvangle_results.json",
        "key": "medimageinsight_probe",
    },
    "MedCLIP-ResNet50": {
        "aasce": ROOT / "outputs/benchmarks/controlled_cross_model_v2/medical/aasce/aasce_results.json",
        "hvangle": ROOT / "outputs/benchmarks/controlled_cross_model_v2/medical/hvangle/hvangle_results.json",
        "key": "medclip_resnet_probe",
    },
    "MedCLIP-Swin": {
        "aasce": ROOT / "outputs/benchmarks/controlled_cross_model_v2/medical/aasce/aasce_results.json",
        "hvangle": ROOT / "outputs/benchmarks/controlled_cross_model_v2/medical/hvangle/hvangle_results.json",
        "key": "medclip_vit_probe",
    },
    "DAX-ResNet50-vA": {
        "aasce": ROOT / "outputs/benchmarks/controlled_cross_model_v2/medical/aasce/aasce_results.json",
        "hvangle": ROOT / "outputs/benchmarks/controlled_cross_model_v2/medical/hvangle/hvangle_results.json",
        "key": "dax_resnet50_probe",
    },
    "MedSAM ViT-B": {
        "aasce": ROOT / "outputs/benchmarks/controlled_cross_model_v2/medical/aasce/aasce_results.json",
        "hvangle": ROOT / "outputs/benchmarks/controlled_cross_model_v2/medical/hvangle/hvangle_results.json",
        "key": "medsam_encoder_probe",
    },
}

RECOVERED_MEDICAL_FILES = {
    "aasce": ROOT / "outputs/benchmarks/frozen_probe_epoch_recovery/aasce/aasce_results.json",
    "hvangle": ROOT / "outputs/benchmarks/frozen_probe_epoch_recovery/hvangle/hvangle_results.json",
}


SUPERVISED_FILES = {
    "aasce": [ROOT / "outputs/benchmarks/controlled_cross_model_v2/aasce/aasce_results.json"],
    "hvangle": [ROOT / "outputs/benchmarks/controlled_cross_model_v2/hvangle/hvangle_results.json"],
}

ABLATION_FILES = {
    "aasce_radimagenet": ROOT / "outputs/benchmarks/controlled_cross_model_v2/aasce_rad/aasce_results.json",
    "hvangle_radimagenet": ROOT / "outputs/benchmarks/controlled_cross_model_v2/hvangle_rad/hvangle_results.json",
    "aasce_distort": ROOT / "outputs/benchmarks/controlled_cross_model_v2/aasce_distort/aasce_results.json",
}


DISPLAY_MODELS = {
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-T",
    "vit_b16": "ViT-B/16",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def target_arrays():
    aasce, _ = aasce_splits()
    hvangle, _ = hvangle_splits(True, "patient", 42)
    return {
        "cobb": {
            "train": np.asarray([row["targets"][0] for row in aasce["train"]]),
            "test": np.asarray([row["targets"][0] for row in aasce["test"]]),
            "test_ids": [row["id"] for row in aasce["test"]],
            "test_clusters": [row["id"] for row in aasce["test"]],
            "tolerance": 5.0,
        },
        "hva": {
            "train": np.asarray([row["targets"][0] for row in hvangle["train"]]),
            "test": np.asarray([row["targets"][0] for row in hvangle["test"]]),
            "test_ids": [row["id"] for row in hvangle["test"]],
            "test_clusters": [row["patient_id"] for row in hvangle["test"]],
            "tolerance": 5.0,
        },
        "ima": {
            "train": np.asarray([row["targets"][1] for row in hvangle["train"]]),
            "test": np.asarray([row["targets"][1] for row in hvangle["test"]]),
            "test_ids": [row["id"] for row in hvangle["test"]],
            "test_clusters": [row["patient_id"] for row in hvangle["test"]],
            "tolerance": 3.0,
        },
    }


def aggregate_runs(result: dict, target: str):
    runs = result["runs"]
    ids = runs[0]["test_ids"]
    truth = np.asarray(runs[0]["metrics"][target]["ground_truth"], float)
    predictions = []
    for run in runs:
        if run["test_ids"] != ids:
            raise ValueError("test identifier order differs across seeds")
        current_truth = np.asarray(run["metrics"][target]["ground_truth"], float)
        if not np.allclose(current_truth, truth):
            raise ValueError("ground truth differs across seeds")
        predictions.append(run["metrics"][target]["predictions"])
    return ids, truth, np.mean(np.asarray(predictions, float), axis=0), np.asarray(predictions, float)


def percentile_ci(values, alpha=.05):
    return [float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))]


def sampling_clusters(target: str, identifiers, targets) -> np.ndarray:
    """Return the independent sampling-unit identifier for each saved image."""
    mapping = dict(zip(targets[target]["test_ids"], targets[target]["test_clusters"]))
    missing = [identifier for identifier in identifiers if identifier not in mapping]
    if missing:
        raise KeyError(f"missing sampling clusters for {target}: {missing[:3]}")
    return np.asarray([mapping[identifier] for identifier in identifiers])


def cluster_members(clusters):
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    return unique, [np.flatnonzero(clusters == value) for value in unique]


def bootstrap_indices(members, rng):
    """Resample independent units and retain every image within each sampled unit."""
    sampled = rng.integers(0, len(members), len(members))
    return np.concatenate([members[index] for index in sampled])


def patient_mean_difference(error, reference_error, clusters):
    difference = np.asarray(error, float) - np.asarray(reference_error, float)
    unique, members = cluster_members(clusters)
    return np.asarray([np.mean(difference[index]) for index in members]), len(unique)


def independent_unit_means(values, clusters):
    """Average image-level values within each independent sampling unit."""
    values = np.asarray(values, float)
    unique, members = cluster_members(clusters)
    return np.asarray([np.mean(values[index]) for index in members]), len(unique)


def paired_signed_rank(error, reference_error, clusters):
    differences, independent_n = patient_mean_difference(error, reference_error, clusters)
    test = stats.wilcoxon(differences, alternative="two-sided",
                          zero_method="wilcox", method="auto")
    return test, independent_n


def cluster_robust_slope_test(magnitude, difference, clusters):
    """OLS slope with a cluster-robust sandwich standard error."""
    magnitude, difference, clusters = map(np.asarray, (magnitude, difference, clusters))
    if len(np.unique(clusters)) == len(clusters):
        regression = stats.linregress(magnitude, difference)
        return float(regression.slope), float(regression.pvalue)
    design = np.column_stack([np.ones(len(magnitude)), magnitude])
    bread = np.linalg.inv(design.T @ design)
    coefficient = bread @ design.T @ difference
    residual = difference - design @ coefficient
    meat = np.zeros((2, 2), float)
    unique, members = cluster_members(clusters)
    for index in members:
        score = design[index].T @ residual[index]
        meat += np.outer(score, score)
    n, k, g = len(magnitude), design.shape[1], len(unique)
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    statistic = float(coefficient[1] / standard_error) if standard_error else np.inf
    p_value = float(2 * stats.t.sf(abs(statistic), df=g - 1))
    return float(coefficient[1]), p_value


def bootstrap_metrics(truth, prediction, tolerance, rng, clusters):
    n = len(truth)
    _, members = cluster_members(clusters)
    mae = np.empty(BOOTSTRAPS)
    bias = np.empty(BOOTSTRAPS)
    lower = np.empty(BOOTSTRAPS)
    upper = np.empty(BOOTSTRAPS)
    icc = np.empty(BOOTSTRAPS)
    proportional_slope = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        sample = bootstrap_indices(members, rng)
        y, p = truth[sample], prediction[sample]
        difference = p - y
        magnitude = (p + y) / 2
        mae[index] = np.mean(np.abs(difference))
        bias[index] = np.mean(difference)
        sd = np.std(difference, ddof=1)
        lower[index], upper[index] = bias[index] - 1.96 * sd, bias[index] + 1.96 * sd
        icc[index] = measurement_metrics(y, p, [tolerance])["icc_2_1"]
        centered_magnitude = magnitude - magnitude.mean()
        denominator = np.sum(centered_magnitude ** 2)
        proportional_slope[index] = (
            np.sum(centered_magnitude * (difference - difference.mean())) / denominator
            if denominator > 0 else np.nan
        )
    return {
        "mae_ci": percentile_ci(mae),
        "bias_ci": percentile_ci(bias),
        "loa_lower_ci": percentile_ci(lower),
        "loa_upper_ci": percentile_ci(upper),
        "icc_ci": percentile_ci(icc[np.isfinite(icc)]),
        "proportional_bias_slope_ci": percentile_ci(
            proportional_slope[np.isfinite(proportional_slope)]),
    }


def bootstrap_delta_mae(truth, prediction, reference_prediction, rng, clusters):
    error = np.abs(np.asarray(prediction, float) - np.asarray(truth, float))
    reference_error = np.abs(
        np.asarray(reference_prediction, float) - np.asarray(truth, float))
    return bootstrap_delta_case_error(error, reference_error, rng, clusters)


def case_seed_mean_absolute_error(truth, seed_predictions):
    """Average absolute error across seeds for each image.

    Subsequent inference uses either the image or the patient's mean paired
    difference as its unit. This does not turn the saved runs into an ensemble.
    """
    seed_predictions = np.asarray(seed_predictions, float)
    return np.mean(np.abs(seed_predictions - np.asarray(truth, float)[None, :]), axis=0)


def bootstrap_mean(values, rng, clusters):
    values = np.asarray(values, float)
    _, members = cluster_members(clusters)
    replicates = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        sample = bootstrap_indices(members, rng)
        replicates[index] = np.mean(values[sample])
    return percentile_ci(replicates)


def bootstrap_delta_case_error(error, reference_error, rng, clusters):
    unit_differences, _ = patient_mean_difference(error, reference_error, clusters)
    replicates = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        sample = rng.integers(0, len(unit_differences), len(unit_differences))
        replicates[index] = np.mean(unit_differences[sample])
    return percentile_ci(replicates)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate((len(p_values) - np.arange(len(p_values))) * p_values[order])
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def constant_baselines(targets):
    rows = []
    for target, item in targets.items():
        for estimator in ("mean", "median"):
            value = float(np.mean(item["train"]) if estimator == "mean" else np.median(item["train"]))
            prediction = np.full_like(item["test"], value, dtype=float)
            metric = measurement_metrics(item["test"], prediction, [item["tolerance"]])
            rows.append({
                "target": target.upper(), "predictor": f"Training {estimator}", "value": value,
                "reference_sd": float(np.std(item["test"], ddof=1)), "prediction_sd": 0.0,
                "mae": metric["mae"], "tolerance": metric[f"within_{item['tolerance']:g}deg"],
                "bias": metric["bias"], "loa_lower": metric["loa_lower"],
                "loa_upper": metric["loa_upper"], "icc": metric["icc_2_1"],
            })
    return pd.DataFrame(rows)


def medical_dispersion(targets):
    rows, distributions = [], {target: {} for target in targets}
    rng = np.random.default_rng(RNG_SEED + 1)
    for label, spec in MEDICAL_FILES.items():
        for dataset, target_names in (("aasce", ["cobb"]), ("hvangle", ["hva", "ima"])):
            manifest = load_json(spec[dataset])
            model = manifest["models"][spec["key"]]
            for target in target_names:
                identifiers, truth, prediction, seed_predictions = aggregate_runs(model, target)
                clusters = sampling_clusters(target, identifiers, targets)
                prediction_sd = float(np.std(prediction, ddof=1))
                reference_sd = float(np.std(truth, ddof=1))
                constant_value = float(np.median(targets[target]["train"]))
                constant_prediction = np.full_like(truth, constant_value)
                model_error = case_seed_mean_absolute_error(truth, seed_predictions)
                constant_error = np.abs(constant_prediction - truth)
                unit_difference, _ = patient_mean_difference(
                    model_error, constant_error, clusters)
                test, independent_n = paired_signed_rank(model_error, constant_error, clusters)
                delta_ci = bootstrap_delta_case_error(
                    model_error, constant_error, rng, clusters)
                equivalence_margin = .1 * targets[target]["tolerance"]
                rows.append({
                    "target": target.upper(), "encoder": label,
                    "n_cases": int(len(truth)), "n_independent_units": independent_n,
                    "reference_sd": reference_sd, "prediction_sd": prediction_sd,
                    "sd_ratio": prediction_sd / reference_sd,
                    "near_constant_sd_ratio_lt_0_1": bool(prediction_sd / reference_sd < .1),
                    "seed_prediction_sd_mean": float(np.mean(np.std(seed_predictions, axis=1, ddof=1))),
                    "mae_seed_mean": float(np.mean(model_error)),
                    "training_median_constant": constant_value,
                    "constant_mae": float(np.mean(constant_error)),
                    "image_weighted_delta_mae_vs_constant": float(
                        np.mean(model_error) - np.mean(constant_error)),
                    "delta_mae_vs_constant": float(np.mean(unit_difference)),
                    "delta_mae_ci": delta_ci,
                    "equivalence_margin_deg": equivalence_margin,
                    "mae_equivalent_to_median_constant": bool(
                        delta_ci[0] > -equivalence_margin and delta_ci[1] < equivalence_margin),
                    "constant_wilcoxon_p_raw": float(test.pvalue),
                })
                distributions[target][label] = prediction
                distributions[target]["Reference"] = truth
    frame = pd.DataFrame(rows)
    frame["constant_wilcoxon_p_holm_18"] = holm_adjust(
        frame["constant_wilcoxon_p_raw"].to_numpy())
    return frame, distributions


def medical_probe_pair_inference(targets):
    """Paired comparison requested for the two leading probes.

    Absolute error is calculated for each seed and then averaged across seeds
    for each image. HVAngleEst differences are then averaged within patient
    before the signed-rank test. The declared family contains one encoder pair
    across the three targets.
    """
    rng = np.random.default_rng(RNG_SEED + 4)
    rows = []
    for dataset, target_names in (("aasce", ["cobb"]), ("hvangle", ["hva", "ima"])):
        path = MEDICAL_FILES["BiomedCLIP"][dataset]
        manifest = load_json(path)["models"]
        biomed = manifest[MEDICAL_FILES["BiomedCLIP"]["key"]]
        insight = manifest[MEDICAL_FILES["MedImageInsight"]["key"]]
        for target in target_names:
            ids_b, truth_b, _, seed_predictions_b = aggregate_runs(biomed, target)
            ids_i, truth_i, _, seed_predictions_i = aggregate_runs(insight, target)
            if ids_b != ids_i or not np.allclose(truth_b, truth_i):
                raise ValueError(f"medical-probe pairing mismatch for {target}")
            error_b = case_seed_mean_absolute_error(truth_b, seed_predictions_b)
            error_i = case_seed_mean_absolute_error(truth_b, seed_predictions_i)
            clusters = sampling_clusters(target, ids_b, targets)
            unit_error_b, _ = independent_unit_means(error_b, clusters)
            unit_error_i, _ = independent_unit_means(error_i, clusters)
            test, independent_n = paired_signed_rank(error_i, error_b, clusters)
            rows.append({
                "target": target.upper(),
                "comparison": "MedImageInsight - BiomedCLIP",
                "n_cases": int(len(truth_b)),
                "n_independent_units": independent_n,
                "biomedclip_image_weighted_mae_seed_mean": float(np.mean(error_b)),
                "medimageinsight_image_weighted_mae_seed_mean": float(np.mean(error_i)),
                "biomedclip_mae_seed_mean": float(np.mean(unit_error_b)),
                "medimageinsight_mae_seed_mean": float(np.mean(unit_error_i)),
                "delta_mae_seed_mean": float(np.mean(unit_error_i - unit_error_b)),
                "delta_mae_ci": bootstrap_delta_case_error(
                    error_i, error_b, rng, clusters),
                "wilcoxon_statistic": float(test.statistic),
                "p_raw": float(test.pvalue),
            })
    adjusted = holm_adjust([row["p_raw"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_holm_3"] = float(value)
    return pd.DataFrame(rows)


def structured_seed_summary(selector: str = "dice"):
    """Expose run-level EVL-2 and EVL-3 results for either foot selector."""
    directories = {
        "dice": "hvangle_axis_geometry",
        "measurement": "hvangle_axis_geometry_measurement_selected",
    }
    path = (ROOT / "outputs/benchmarks/controlled_cross_model_v2" /
            directories[selector] / "results.json")
    rows = []
    for run in load_json(path):
        spatial = run["test"]
        row = {
            "selector": selector,
            "seed": int(run["seed"]),
            "best_epoch": int(run["best_epoch"]),
            "dice": float(spatial["segmentation"]["mean_dice"]),
            "cldice": float(spatial["segmentation"]["mean_cldice"]),
            "axis_mae": float(spatial["axis"]["mean_angular_mae_deg"]),
        }
        for target in ("hva", "ima"):
            metric = spatial["measurements"][target]
            row.update({
                f"{target}_mae": float(metric["mae"]),
                f"{target}_icc": float(metric["icc_2_1"]),
                f"{target}_tolerance": float(metric["within_tolerance_percent"]),
                f"{target}_bias": float(metric["bias"]),
                f"{target}_loa_lower": float(metric["loa_lower"]),
                f"{target}_loa_upper": float(metric["loa_upper"]),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed")


def structured_selector_inference(targets):
    """Paired patient-level MAE sensitivity for the two validation selectors."""
    root = ROOT / "outputs/benchmarks/controlled_cross_model_v2"
    dice_runs = load_json(root / "hvangle_axis_geometry/results.json")
    measurement_runs = load_json(
        root / "hvangle_axis_geometry_measurement_selected/results.json")
    if [run["seed"] for run in dice_runs] != [run["seed"] for run in measurement_runs]:
        raise ValueError("structured selectors do not contain the same seed order")
    rng = np.random.default_rng(RNG_SEED + 5)
    rows = []
    for target in ("hva", "ima"):
        identifiers = dice_runs[0]["test"]["identifiers"]
        truth = np.asarray(
            dice_runs[0]["test"]["measurements"][target]["ground_truth"], float)
        dice_predictions, measurement_predictions = [], []
        for dice_run, measurement_run in zip(dice_runs, measurement_runs):
            if (dice_run["test"]["identifiers"] != identifiers or
                    measurement_run["test"]["identifiers"] != identifiers):
                raise ValueError("structured selector identifier order differs across runs")
            for run, destination in ((dice_run, dice_predictions),
                                     (measurement_run, measurement_predictions)):
                metric = run["test"]["measurements"][target]
                if not np.allclose(metric["ground_truth"], truth, rtol=0, atol=1e-9):
                    raise ValueError("structured selector reference values differ across runs")
                destination.append(metric["predictions"])
        dice_error = case_seed_mean_absolute_error(truth, dice_predictions)
        measurement_error = case_seed_mean_absolute_error(truth, measurement_predictions)
        clusters = sampling_clusters(target, identifiers, targets)
        dice_units, independent_n = independent_unit_means(dice_error, clusters)
        measurement_units, _ = independent_unit_means(measurement_error, clusters)
        rows.append({
            "target": target.upper(),
            "comparison": "measurement selector - Dice selector",
            "n_cases": len(truth),
            "n_independent_units": independent_n,
            "dice_selector_patient_weighted_mae_seed_mean": float(np.mean(dice_units)),
            "measurement_selector_patient_weighted_mae_seed_mean": float(
                np.mean(measurement_units)),
            "delta_mae_seed_mean": float(np.mean(measurement_units - dice_units)),
            "delta_mae_ci": bootstrap_delta_case_error(
                measurement_error, dice_error, rng, clusters),
            "bootstrap_replicates": BOOTSTRAPS,
            "inference_status": "post hoc exploratory; unadjusted interval",
        })
    return pd.DataFrame(rows)


def reference_axis_target_consistency(targets):
    """Reconstruct released foot angles directly from released axis endpoints."""
    data_dir = ROOT / "HVAngleEst" / "HVAngleEst"
    split_dir = ROOT / "HVAngleEst" / "splits"
    frame = pd.read_csv(data_dir / "datasets.csv")
    frame = frame[~frame.filename.duplicated(keep=False)]
    saved_manifest = (ROOT / "outputs" / "benchmarks" / "controlled_cross_model_v2" /
                      "hvangle" / "hvangle_results.json")
    if saved_manifest.exists():
        saved_splits = json.loads(saved_manifest.read_text())["split_ids"]
        released_names = set().union(*(set(saved_splits[name]) for name in saved_splits))
    else:
        released_names = set().union(*(
            set(pd.read_csv(split_dir / f"{name}.csv")["filename"])
            for name in ("train", "val", "test")))
    frame = frame[frame.filename.isin(released_names)].copy()
    if len(frame) != 1177:
        raise ValueError(f"unexpected retained HVAngleEst count: {len(frame)}")

    def parse_segment(value, width, height):
        points = np.asarray(
            [float(item) for item in str(value).split(",")], float).reshape(2, 2)
        return points * np.asarray([width, height], float)

    def angle(segment_a, segment_b):
        vector_a = segment_a[1] - segment_a[0]
        vector_b = segment_b[1] - segment_b[0]
        cosine = abs(float(np.dot(vector_a, vector_b))) / (
            np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))

    records = []
    for row in frame.itertuples():
        great_toe = parse_segment(row.great_toe, row.image_width, row.image_height)
        first = parse_segment(row.first_metatarsal, row.image_width, row.image_height)
        second = parse_segment(row.second_metatarsal, row.image_width, row.image_height)
        records.append({
            "id": row.filename,
            "hva_reconstructed": angle(great_toe, first),
            "hva_released": float(row.HVA),
            "ima_reconstructed": angle(first, second),
            "ima_released": float(row.IMA),
        })
    reconstructed = pd.DataFrame(records)
    rows = []
    test_ids = set(targets["hva"]["test_ids"])
    for scope, current in (
            ("retained_release", reconstructed),
            ("test", reconstructed[reconstructed.id.isin(test_ids)])):
        for target in ("hva", "ima"):
            difference = np.abs(
                current[f"{target}_reconstructed"] - current[f"{target}_released"])
            rows.append({
                "scope": scope,
                "target": target.upper(),
                "n_images": int(len(current)),
                "mae_deg": float(difference.mean()),
                "max_absolute_difference_deg": float(difference.max()),
            })
    return pd.DataFrame(rows)


def frozen_probe_training_records():
    """Report selected probe settings and recover validation-winner epochs."""
    rows = []
    for dataset, task in (("aasce", "Cobb"), ("hvangle", "HVA/IMA")):
        recovered_path = RECOVERED_MEDICAL_FILES[dataset]
        if not recovered_path.is_file():
            raise FileNotFoundError(f"missing frozen-probe epoch recovery: {recovered_path}")
        recovered = load_json(recovered_path)["models"]
        for label, spec in MEDICAL_FILES.items():
            original = load_json(spec[dataset])["models"][spec["key"]]
            replay = recovered[spec["key"]]
            lr_wd, best_epochs, stop_epochs = [], [], []
            for original_run, replay_run in zip(original["runs"], replay["runs"]):
                targets = ("cobb",) if dataset == "aasce" else ("hva", "ima")
                for target in targets:
                    original_prediction = np.asarray(
                        original_run["metrics"][target]["predictions"], float)
                    replay_prediction = np.asarray(
                        replay_run["metrics"][target]["predictions"], float)
                    if not np.array_equal(original_prediction, replay_prediction):
                        raise ValueError(
                            f"epoch-recovery replay changed predictions: {dataset} {label} {target}")
                best = replay_run["best_trial"]
                hp = best["hyperparameters"]
                lr_wd.append(f"{hp['lr']:g}/{hp['weight_decay']:g}")
                best_epochs.append(str(best["best_epoch"]))
                stop_epochs.append(str(best["epochs_run"]))
            rows.append({
                "dataset": dataset,
                "task": task,
                "encoder": label,
                "head_batch_size": 16,
                "feature_extraction_batch_size": 2 if label == "MedSAM ViT-B" else 16,
                "selected_lr_wd_seed_17_42_73": ";".join(lr_wd),
                "best_epoch_seed_17_42_73": "/".join(best_epochs),
                "stop_epoch_seed_17_42_73": "/".join(stop_epochs),
                "replay_predictions_match_original": True,
            })
    return pd.DataFrame(rows)


def prompt_dispersion(target_info_by_name):
    rows = []
    distributions = {target: {} for target in target_info_by_name}
    display_name = {
        "gpt-4o": "GPT-4o",
        "claude-sonnet-5": "Claude Sonnet 5",
    }
    rng = np.random.default_rng(RNG_SEED + 2)
    aasce = load_json(
        ROOT / "outputs/benchmarks/controlled_cross_model_v2/prompt_matched/aasce_matched_prompt_results.json")
    hvangle = load_json(
        ROOT / "outputs/benchmarks/controlled_cross_model_v2/prompt_matched/hvangle_matched_prompt_patient_results.json")
    for model, metric in aasce["models"].items():
        truth = np.asarray(metric["ground_truth"], float)
        prediction = np.asarray(metric["predictions"], float)
        prompt_low, prompt_high = PROMPT_RESPONSE_RANGES["cobb"]
        prompt_supported = (truth >= prompt_low) & (truth <= prompt_high)
        clusters = sampling_clusters("cobb", aasce["test_ids"], target_info_by_name)
        constant = np.full_like(truth, np.median(target_info_by_name["cobb"]["train"]))
        delta_ci = bootstrap_delta_mae(truth, prediction, constant, rng, clusters)
        paired_test, independent_n = paired_signed_rank(
            np.abs(prediction - truth), np.abs(constant - truth), clusters)
        unit_difference, _ = patient_mean_difference(
            np.abs(prediction - truth), np.abs(constant - truth), clusters)
        margin = .1 * target_info_by_name["cobb"]["tolerance"]
        rows.append({"model": model, "target": "COBB", "n": metric["n"],
                     "n_independent_units": independent_n,
                     "reference_sd": metric["reference_sd"],
                     "prediction_sd": metric["prediction_sd"],
                     "unique_predictions": int(np.unique(prediction).size),
                     "prompt_response_min": prompt_low,
                     "prompt_response_max": prompt_high,
                     "n_reference_outside_prompt_range": int(np.sum(~prompt_supported)),
                     "mae_within_prompt_reference_range": float(np.mean(
                         np.abs(prediction[prompt_supported] - truth[prompt_supported]))),
                     "sd_ratio": metric["prediction_sd"] / metric["reference_sd"],
                     "near_constant_sd_ratio_lt_0_1": bool(
                         metric["prediction_sd"] / metric["reference_sd"] < .1),
                     "mae": metric["mae"],
                     "median_constant_mae": float(np.mean(np.abs(constant - truth))),
                     "image_weighted_delta_mae_vs_median_constant": float(
                         np.mean(np.abs(prediction - truth)) - np.mean(np.abs(constant - truth))),
                     "delta_mae_vs_median_constant": float(np.mean(unit_difference)),
                     "delta_mae_ci": delta_ci,
                     "wilcoxon_p_raw": float(paired_test.pvalue),
                     "equivalence_margin_deg": margin,
                     "mae_equivalent_to_median_constant": bool(
                         delta_ci[0] > -margin and delta_ci[1] < margin)})
        distributions["cobb"][display_name.get(model, model)] = prediction
        distributions["cobb"]["Reference"] = truth
    for model, model_targets in hvangle["models"].items():
        for target, metric in model_targets.items():
            truth = np.asarray(metric["ground_truth"], float)
            prediction = np.asarray(metric["predictions"], float)
            prompt_low, prompt_high = PROMPT_RESPONSE_RANGES[target]
            prompt_supported = (truth >= prompt_low) & (truth <= prompt_high)
            target_info = target_info_by_name[target]
            clusters = sampling_clusters(target, hvangle["test_ids"], target_info_by_name)
            constant = np.full_like(truth, np.median(target_info["train"]))
            delta_ci = bootstrap_delta_mae(truth, prediction, constant, rng, clusters)
            paired_test, independent_n = paired_signed_rank(
                np.abs(prediction - truth), np.abs(constant - truth), clusters)
            unit_difference, _ = patient_mean_difference(
                np.abs(prediction - truth), np.abs(constant - truth), clusters)
            margin = .1 * target_info["tolerance"]
            rows.append({"model": model, "target": target.upper(), "n": metric["n"],
                         "n_independent_units": independent_n,
                         "reference_sd": metric["reference_sd"],
                         "prediction_sd": metric["prediction_sd"],
                         "unique_predictions": int(np.unique(prediction).size),
                         "prompt_response_min": prompt_low,
                         "prompt_response_max": prompt_high,
                         "n_reference_outside_prompt_range": int(np.sum(~prompt_supported)),
                         "mae_within_prompt_reference_range": float(np.mean(
                             np.abs(prediction[prompt_supported] - truth[prompt_supported]))),
                         "sd_ratio": metric["prediction_sd"] / metric["reference_sd"],
                         "near_constant_sd_ratio_lt_0_1": bool(
                             metric["prediction_sd"] / metric["reference_sd"] < .1),
                         "mae": metric["mae"],
                         "median_constant_mae": float(np.mean(np.abs(constant - truth))),
                         "image_weighted_delta_mae_vs_median_constant": float(
                             np.mean(np.abs(prediction - truth)) - np.mean(np.abs(constant - truth))),
                         "delta_mae_vs_median_constant": float(np.mean(unit_difference)),
                         "delta_mae_ci": delta_ci,
                         "wilcoxon_p_raw": float(paired_test.pvalue),
                         "equivalence_margin_deg": margin,
                         "mae_equivalent_to_median_constant": bool(
                             delta_ci[0] > -margin and delta_ci[1] < margin)})
            distributions[target][display_name.get(model, model)] = prediction
            distributions[target]["Reference"] = truth
    adjusted = holm_adjust([row["wilcoxon_p_raw"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["wilcoxon_p_holm_6"] = float(value)
    return pd.DataFrame(rows), distributions


def supervised_inference(targets):
    manifests = {dataset: [load_json(path) for path in paths] for dataset, paths in SUPERVISED_FILES.items()}
    model_results = {"cobb": {}, "hva": {}, "ima": {}}
    for dataset, target_names in (("aasce", ["cobb"]), ("hvangle", ["hva", "ima"])):
        merged = {}
        for manifest in manifests[dataset]:
            merged.update(manifest["models"])
        for model_key, label in DISPLAY_MODELS.items():
            if model_key not in merged:
                raise KeyError(f"missing {model_key} in {dataset}")
            for target in target_names:
                identifiers, truth, prediction, seed_predictions = aggregate_runs(
                    merged[model_key], target)
                model_results[target][label] = (
                    identifiers, truth, prediction, seed_predictions)

    rng = np.random.default_rng(RNG_SEED)
    rows, comparisons = [], []
    raw_p = []
    for target, models in model_results.items():
        tolerance = targets[target]["tolerance"]
        for label, (identifiers, truth, prediction, seed_predictions) in models.items():
            clusters = sampling_clusters(target, identifiers, targets)
            metric = measurement_metrics(truth, prediction, [tolerance])
            ci = bootstrap_metrics(truth, prediction, tolerance, rng, clusters)
            case_error = case_seed_mean_absolute_error(truth, seed_predictions)
            ci["mae_ci"] = bootstrap_mean(case_error, rng, clusters)
            mean_value = (truth + prediction) / 2
            difference = prediction - truth
            slope, slope_p = cluster_robust_slope_test(mean_value, difference, clusters)
            rows.append({"target": target.upper(), "model": label, **ci,
                         "n_cases": int(len(truth)),
                         "n_independent_units": int(len(np.unique(clusters))),
                         "mae": float(np.mean(case_error)), "bias": metric["bias"],
                         "loa_lower": metric["loa_lower"], "loa_upper": metric["loa_upper"],
                         "proportional_bias_slope": slope,
                         "proportional_bias_p": slope_p})
        # All six architecture pairs are declared before inspecting test
        # performance.  This includes the ConvNeXt--EfficientNet comparison
        # highlighted by the reviewer and avoids selecting a test-set winner
        # as the sole reference condition.
        for (label_a, (ids_a, truth_a, _, seed_predictions_a)), (label_b, (ids_b, truth_b, _, seed_predictions_b)) in \
                combinations(models.items(), 2):
            if ids_a != ids_b or not np.allclose(truth_a, truth_b):
                raise ValueError(f"truth mismatch for {target}: {label_a}, {label_b}")
            clusters = sampling_clusters(target, ids_a, targets)
            error_a = case_seed_mean_absolute_error(truth_a, seed_predictions_a)
            error_b = case_seed_mean_absolute_error(truth_a, seed_predictions_b)
            unit_difference, _ = patient_mean_difference(error_b, error_a, clusters)
            test, independent_n = paired_signed_rank(error_b, error_a, clusters)
            comparisons.append({
                "target": target.upper(), "comparison": f"{label_b} - {label_a}",
                "n_cases": int(len(truth_a)), "n_independent_units": independent_n,
                "image_weighted_delta_mae": float(np.mean(error_b) - np.mean(error_a)),
                "delta_mae": float(np.mean(unit_difference)),
                "delta_mae_ci": bootstrap_delta_case_error(
                    error_b, error_a, rng, clusters),
                "wilcoxon_statistic": float(test.statistic), "p_raw": float(test.pvalue),
            })
            raw_p.append(float(test.pvalue))
    adjusted = holm_adjust(raw_p)
    for row, value in zip(comparisons, adjusted):
        row["p_holm_18"] = float(value)
    inference = pd.DataFrame(rows)
    inference["proportional_bias_p_holm_12"] = holm_adjust(
        inference["proportional_bias_p"].to_numpy())
    return inference, pd.DataFrame(comparisons)


def cobb_max_reference_sensitivity(inference):
    """Repeat Cobb proportional-bias tests after excluding its largest target.

    The main analysis retains every case.  This post hoc diagnostic checks
    whether the single 156.39-degree reference drives the Cobb slopes or the
    ordering of image-weighted MAE across supervised architectures.
    """
    manifests = [load_json(path) for path in SUPERVISED_FILES["aasce"]]
    merged = {}
    for manifest in manifests:
        merged.update(manifest["models"])

    records = []
    revised_p = inference["proportional_bias_p"].to_numpy(copy=True)
    cobb_rows = {
        row.model: index
        for index, row in inference.iterrows()
        if row.target == "COBB"
    }
    for model_key, label in DISPLAY_MODELS.items():
        identifiers, truth, prediction, seed_predictions = aggregate_runs(
            merged[model_key], "cobb")
        maximum_index = int(np.argmax(truth))
        keep = np.arange(len(truth)) != maximum_index
        clusters_full = np.asarray(identifiers)
        clusters_without = clusters_full[keep]
        mean_full = (truth + prediction) / 2
        difference_full = prediction - truth
        mean_without = mean_full[keep]
        difference_without = difference_full[keep]
        slope_full, p_full = cluster_robust_slope_test(
            mean_full, difference_full, clusters_full)
        slope_without, p_without = cluster_robust_slope_test(
            mean_without, difference_without, clusters_without)
        case_error = case_seed_mean_absolute_error(truth, seed_predictions)
        inference_index = cobb_rows[label]
        revised_p[inference_index] = p_without
        records.append({
            "model": label,
            "excluded_case_id": identifiers[maximum_index],
            "maximum_reference_deg": float(truth[maximum_index]),
            "n_full": int(len(truth)),
            "n_without_maximum": int(np.sum(keep)),
            "mae_full": float(np.mean(case_error)),
            "mae_without_maximum": float(np.mean(case_error[keep])),
            "slope_full": float(slope_full),
            "p_full_raw": float(p_full),
            "slope_without_maximum": float(slope_without),
            "p_without_maximum_raw": float(p_without),
            "slope_direction_preserved": bool(
                np.sign(slope_full) == np.sign(slope_without)),
        })

    revised_holm = holm_adjust(revised_p)
    for record in records:
        index = cobb_rows[record["model"]]
        full_holm = float(
            inference.loc[index, "proportional_bias_p_holm_12"])
        without_holm = float(revised_holm[index])
        record["p_full_holm_12"] = full_holm
        record["p_without_maximum_holm_12"] = without_holm
        record["holm_significance_at_0_05_preserved"] = bool(
            (full_holm < .05) == (without_holm < .05))

    sensitivity = pd.DataFrame(records)
    sensitivity["mae_rank_full"] = sensitivity["mae_full"].rank(
        method="min").astype(int)
    sensitivity["mae_rank_without_maximum"] = sensitivity[
        "mae_without_maximum"].rank(method="min").astype(int)
    return sensitivity


def ablation_inference(targets):
    main_aasce = load_json(SUPERVISED_FILES["aasce"][0])["models"]["resnet50"]
    main_hvangle = load_json(SUPERVISED_FILES["hvangle"][0])["models"]["resnet50"]
    rad_aasce = load_json(ABLATION_FILES["aasce_radimagenet"])["models"]["radimagenet_resnet50"]
    rad_hvangle = load_json(ABLATION_FILES["hvangle_radimagenet"])["models"]["radimagenet_resnet50"]
    distort_aasce = load_json(ABLATION_FILES["aasce_distort"])["models"]["resnet50"]
    rng = np.random.default_rng(RNG_SEED + 3)
    rows = []
    for target, reference, condition, name in (
        ("cobb", main_aasce, rad_aasce, "RadImageNet - ImageNet"),
        ("hva", main_hvangle, rad_hvangle, "RadImageNet - ImageNet"),
        ("ima", main_hvangle, rad_hvangle, "RadImageNet - ImageNet"),
        ("cobb", main_aasce, distort_aasce, "Distort - letterbox"),
    ):
        ids_ref, truth_ref, _, seed_predictions_ref = aggregate_runs(reference, target)
        ids_condition, truth_condition, _, seed_predictions_condition = aggregate_runs(condition, target)
        if ids_ref != ids_condition or not np.allclose(truth_ref, truth_condition):
            raise ValueError(f"ablation pairing mismatch: {name} {target}")
        clusters = sampling_clusters(target, ids_ref, targets)
        error_ref = case_seed_mean_absolute_error(truth_ref, seed_predictions_ref)
        error_condition = case_seed_mean_absolute_error(truth_ref, seed_predictions_condition)
        unit_error_ref, _ = independent_unit_means(error_ref, clusters)
        unit_error_condition, _ = independent_unit_means(error_condition, clusters)
        test, independent_n = paired_signed_rank(error_condition, error_ref, clusters)
        rows.append({
            "target": target.upper(), "comparison": name,
            "n_cases": int(len(truth_ref)), "n_independent_units": independent_n,
            "reference_image_weighted_mae": float(np.mean(error_ref)),
            "condition_image_weighted_mae": float(np.mean(error_condition)),
            "reference_mae": float(np.mean(unit_error_ref)),
            "condition_mae": float(np.mean(unit_error_condition)),
            "delta_mae": float(np.mean(unit_error_condition - unit_error_ref)),
            "delta_mae_ci": bootstrap_delta_case_error(
                error_condition, error_ref, rng, clusters),
            "wilcoxon_p_raw": float(test.pvalue),
        })
    pretraining_indices = [index for index, row in enumerate(rows)
                           if row["comparison"] == "RadImageNet - ImageNet"]
    adjusted = holm_adjust([rows[index]["wilcoxon_p_raw"] for index in pretraining_indices])
    for index, value in zip(pretraining_indices, adjusted):
        rows[index]["wilcoxon_p_holm_3"] = float(value)
    return pd.DataFrame(rows)


def distribution_figure(targets, distributions, constant_frame):
    labels = ["Reference", "BiomedCLIP", "MedImageInsight", "MedCLIP-ResNet50",
              "MedCLIP-Swin", "DAX-ResNet50-vA", "MedSAM ViT-B"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), sharey=True)
    colors = ["#4D4D4D", "#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#CC79A7"]
    for axis, target in zip(axes, ("cobb", "hva", "ima")):
        values = [distributions[target][label] for label in labels]
        positions = np.arange(len(labels))
        parts = axis.violinplot(values, positions=positions, vert=False,
                                showmeans=False, showmedians=True, widths=.78)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.42)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[key].set_color("0.25"); parts[key].set_linewidth(.8)
        constants = constant_frame[constant_frame.target == target.upper()]
        for _, row in constants.iterrows():
            axis.axvline(row.value, color="#A6761D" if "mean" in row.predictor.lower() else "#666666",
                         linestyle="--" if "mean" in row.predictor.lower() else ":", linewidth=1.2,
                         label=row.predictor if axis is axes[0] else None)
        axis.set_title("Cobb" if target.lower() == "cobb" else target.upper())
        axis.set_xlabel("Prediction / reference ($^\\circ$)")
        axis.grid(axis="x", color="0.88", linewidth=.6)
        axis.set_axisbelow(True)
        axis.set_yticks(positions)
        if axis is axes[0]:
            axis.set_yticklabels(labels)
        else:
            axis.tick_params(axis="y", labelleft=False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("Frozen-probe prediction dispersion and constant baselines", y=.99)
    figure.tight_layout()
    figure.savefig(OUT / "fig_prediction_distributions.pdf", bbox_inches="tight")
    figure.savefig(OUT / "fig_prediction_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def prompt_distribution_figure(distributions, constant_frame):
    labels = ["Reference", "GPT-4o", "Claude Sonnet 5"]
    colors = ["#4D4D4D", "#0072B2", "#D55E00"]
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), sharey=True)
    for axis, target in zip(axes, ("cobb", "hva", "ima")):
        values = [distributions[target][label] for label in labels]
        positions = np.arange(len(labels))
        parts = axis.violinplot(values, positions=positions, vert=False,
                                showmeans=False, showmedians=True, widths=.75)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.42)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[key].set_color("0.25"); parts[key].set_linewidth(.8)
        constants = constant_frame[constant_frame.target == target.upper()]
        for _, row in constants.iterrows():
            is_median = "median" in row.predictor.lower()
            axis.axvline(
                row.value,
                color="#A6761D" if is_median else "#666666",
                linestyle="--" if is_median else ":",
                linewidth=1.2,
                label=row.predictor if axis is axes[0] else None,
            )
        axis.set_title("Cobb" if target.lower() == "cobb" else target.upper())
        axis.set_xlabel("Prediction / reference ($^\\circ$)")
        axis.grid(axis="x", color="0.88", linewidth=.6)
        axis.set_axisbelow(True)
        axis.set_yticks(positions)
        if axis is axes[0]:
            axis.set_yticklabels(labels)
        else:
            axis.tick_params(axis="y", labelleft=False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("Prompt-only prediction dispersion and constant baselines", y=.99)
    figure.tight_layout()
    figure.savefig(OUT / "fig_prompt_distributions.pdf", bbox_inches="tight")
    figure.savefig(OUT / "fig_prompt_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = target_arrays()
    constants = constant_baselines(targets)
    medical, distributions = medical_dispersion(targets)
    medical_pairs = medical_probe_pair_inference(targets)
    structured_seeds = pd.concat([
        structured_seed_summary("dice"),
        structured_seed_summary("measurement"),
    ], ignore_index=True)
    structured_selector_pairs = structured_selector_inference(targets)
    target_consistency = reference_axis_target_consistency(targets)
    frozen_training = frozen_probe_training_records()
    prompt, prompt_distributions = prompt_dispersion(targets)
    inference, comparisons = supervised_inference(targets)
    cobb_sensitivity = cobb_max_reference_sensitivity(inference)
    ablations = ablation_inference(targets)
    frozen_equivalence = medical[[
        "target", "encoder", "delta_mae_vs_constant", "delta_mae_ci",
        "equivalence_margin_deg", "mae_equivalent_to_median_constant",
    ]].copy()
    frozen_equivalence["ci_lower"] = frozen_equivalence["delta_mae_ci"].map(
        lambda value: value[0])
    frozen_equivalence["ci_upper"] = frozen_equivalence["delta_mae_ci"].map(
        lambda value: value[1])
    frozen_equivalence = frozen_equivalence.drop(columns="delta_mae_ci").rename(columns={
        "delta_mae_vs_constant": "delta_mae_vs_training_median",
        "equivalence_margin_deg": "equivalence_margin",
        "mae_equivalent_to_median_constant": "equivalence_established",
    })[[
        "target", "encoder", "delta_mae_vs_training_median", "ci_lower", "ci_upper",
        "equivalence_margin", "equivalence_established",
    ]]
    constants.to_csv(OUT / "constant_baselines.csv", index=False)
    medical.to_csv(OUT / "medical_probe_dispersion.csv", index=False)
    frozen_equivalence.to_csv(OUT / "frozen_baseline_equivalence.csv", index=False)
    medical[[
        "target", "encoder", "delta_mae_vs_constant", "delta_mae_ci",
        "equivalence_margin_deg", "mae_equivalent_to_median_constant",
    ]].rename(columns={
        "delta_mae_vs_constant": "delta_mae_vs_training_median",
        "equivalence_margin_deg": "equivalence_margin",
        "mae_equivalent_to_median_constant": "equivalence_established",
    }).to_json(OUT / "frozen_baseline_equivalence.json", orient="records", indent=2)
    medical_pairs.to_json(OUT / "medical_probe_paired_tests.json", orient="records", indent=2)
    structured_seeds.to_csv(OUT / "structured_geometry_per_seed.csv", index=False)
    structured_selector_pairs.to_csv(
        OUT / "structured_selector_paired_sensitivity.csv", index=False)
    target_consistency.to_csv(OUT / "reference_axis_target_consistency.csv", index=False)
    frozen_training.to_csv(OUT / "frozen_probe_training_records.csv", index=False)
    prompt.to_csv(OUT / "prompt_dispersion.csv", index=False)
    prompt[[
        "model", "target", "n", "prompt_response_min", "prompt_response_max",
        "n_reference_outside_prompt_range", "mae",
        "mae_within_prompt_reference_range",
    ]].to_csv(OUT / "prompt_response_range_sensitivity.csv", index=False)
    inference.to_json(OUT / "supervised_bootstrap_ci.json", orient="records", indent=2)
    cobb_sensitivity.to_csv(
        OUT / "cobb_max_reference_sensitivity.csv", index=False)
    comparisons.to_json(OUT / "supervised_paired_tests.json", orient="records", indent=2)
    ablations.to_json(OUT / "ablation_paired_tests.json", orient="records", indent=2)
    distribution_figure(targets, distributions, constants)
    prompt_distribution_figure(prompt_distributions, constants)
    summary = {
        "bootstrap_replicates": BOOTSTRAPS, "rng_seed": RNG_SEED,
        "constant_baselines": constants.to_dict(orient="records"),
        "medical_probe_dispersion": medical.to_dict(orient="records"),
        "medical_probe_paired_tests": medical_pairs.to_dict(orient="records"),
        "structured_geometry_per_seed": structured_seeds.to_dict(orient="records"),
        "structured_selector_paired_sensitivity": structured_selector_pairs.to_dict(
            orient="records"),
        "reference_axis_target_consistency": target_consistency.to_dict(orient="records"),
        "frozen_probe_training_records": frozen_training.to_dict(orient="records"),
        "prompt_dispersion": prompt.to_dict(orient="records"),
        "supervised_inference": inference.to_dict(orient="records"),
        "cobb_max_reference_sensitivity": cobb_sensitivity.to_dict(orient="records"),
        "paired_tests": comparisons.to_dict(orient="records"),
        "ablation_tests": ablations.to_dict(orient="records"),
    }
    (OUT / "revision_statistics.json").write_text(json.dumps(summary, indent=2))
    print(OUT / "revision_statistics.json")


if __name__ == "__main__":
    main()
