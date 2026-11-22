"""Recompute paper-scoped results from saved predictions, without API calls."""
from __future__ import annotations
from itertools import combinations
import json
import numpy as np
import pandas as pd
from .bootstrap import agreement_intervals, mean_interval, paired_interval
from .metrics import measurement_metrics, holm_adjust, proportional_bias, paired_test, unit_means
from .records import (CODE_ROOT, RECORDS, SUPERVISED, PAPER_MODELS, FROZEN,
                      PAPER_ENCODERS, load_json, target_info, clusters_for, arrays, model_record)


def supervised(info, replicates):
    rng = np.random.default_rng(20260807)
    summaries, comparisons = [], []
    # Preserve draws consumed by historical auxiliary records, then apply Holm
    # only to the paper's nine retained comparisons. No auxiliary model is trained.
    for target in ("cobb", "hva", "ima"):
        dataset = "aasce" if target == "cobb" else "hvangle"
        current = {}
        for label in SUPERVISED:
            ids, truth, predictions = arrays(model_record(dataset, label), target)
            clusters = clusters_for(target, ids, info)
            prediction = predictions.mean(0)
            errors = np.abs(predictions - truth).mean(0)
            ci = agreement_intervals(truth, prediction, clusters, rng, replicates)
            ci["mae_ci"] = mean_interval(errors, clusters, rng, replicates)
            current[label] = (ids, truth, predictions, clusters)
            if label not in PAPER_MODELS:
                continue
            metrics = measurement_metrics(truth, prediction, info[target]["tolerance"])
            slope, p = proportional_bias(truth, prediction, clusters)
            summaries.append({"target": target.upper(), "model": label, **ci,
                              "n_cases": len(truth), "n_independent_units": len(np.unique(clusters)),
                              "mae": float(errors.mean()), "bias": metrics["bias"],
                              "loa_lower": metrics["loa_lower"], "loa_upper": metrics["loa_upper"],
                              "proportional_bias_slope": slope, "proportional_bias_p": p})
        for (a, first), (b, second) in combinations(current.items(), 2):
            ids, truth, pa, clusters = first
            ib, yb, pb, _ = second
            if ids != ib or not np.array_equal(truth, yb):
                raise ValueError("Supervised comparison is not paired")
            differences = np.abs(pb - truth).mean(0) - np.abs(pa - truth).mean(0)
            ci = paired_interval(differences, clusters, rng, replicates)
            if a not in PAPER_MODELS or b not in PAPER_MODELS:
                continue
            statistic, p = paired_test(differences, clusters)
            comparisons.append({"target": target.upper(), "comparison": f"{b} - {a}",
                                "n_cases": len(truth), "n_independent_units": len(np.unique(clusters)),
                                "image_weighted_delta_mae": float(differences.mean()),
                                "delta_mae": float(unit_means(differences, clusters).mean()),
                                "delta_mae_ci": ci, "wilcoxon_statistic": statistic, "p_raw": p})
    for row, value in zip(comparisons, holm_adjust([r["p_raw"] for r in comparisons])):
        row["p_holm_9"] = float(value)
    for row, value in zip(summaries, holm_adjust([r["proportional_bias_p"] for r in summaries])):
        row["proportional_bias_p_holm_9"] = float(value)
    return summaries, comparisons


def frozen(info, replicates):
    rng = np.random.default_rng(20260808)
    rows = []
    for label in FROZEN:
        for target in ("cobb", "hva", "ima"):
            dataset = "aasce" if target == "cobb" else "hvangle"
            ids, truth, predictions = arrays(model_record(dataset, label, True), target)
            clusters = clusters_for(target, ids, info)
            prediction = predictions.mean(0)
            constant = float(np.median(info[target]["train"]))
            errors = np.abs(predictions - truth).mean(0)
            difference = errors - np.abs(constant - truth)
            ci = paired_interval(difference, clusters, rng, replicates)
            if label not in PAPER_ENCODERS:
                continue
            margin = .1 * info[target]["tolerance"]
            rows.append({"target": target.upper(), "encoder": label,
                         "reference_sd": float(truth.std(ddof=1)),
                         "prediction_sd": float(prediction.std(ddof=1)),
                         "sd_ratio": float(prediction.std(ddof=1) / truth.std(ddof=1)),
                         "mae_seed_mean": float(errors.mean()),
                         "delta_mae_vs_training_median": float(unit_means(difference, clusters).mean()),
                         "ci_lower": ci[0], "ci_upper": ci[1], "margin": margin,
                         "baseline_similarity_criterion_met": bool(ci[0] > -margin and ci[1] < margin)})
    return rows


def frozen_pair(info, replicates):
    rng, rows = np.random.default_rng(20260811), []
    for target in ("cobb", "hva", "ima"):
        dataset = "aasce" if target == "cobb" else "hvangle"
        ids, truth, pi = arrays(model_record(dataset, "MedImageInsight", True), target)
        ib, yb, pb = arrays(model_record(dataset, "BiomedCLIP", True), target)
        if ids != ib or not np.array_equal(truth, yb):
            raise ValueError("Frozen probes are not paired")
        clusters = clusters_for(target, ids, info)
        ei, eb = np.abs(pi - truth).mean(0), np.abs(pb - truth).mean(0)
        statistic, p = paired_test(ei - eb, clusters)
        rows.append({"target": target.upper(), "comparison": "MedImageInsight - BiomedCLIP",
                     "n_cases": len(truth), "n_independent_units": len(np.unique(clusters)),
                     "biomedclip_image_weighted_mae_seed_mean": float(eb.mean()),
                     "medimageinsight_image_weighted_mae_seed_mean": float(ei.mean()),
                     "biomedclip_mae_seed_mean": float(unit_means(eb, clusters).mean()),
                     "medimageinsight_mae_seed_mean": float(unit_means(ei, clusters).mean()),
                     "delta_mae_seed_mean": float(unit_means(ei - eb, clusters).mean()),
                     "delta_mae_ci": paired_interval(ei - eb, clusters, rng, replicates),
                     "wilcoxon_statistic": statistic, "p_raw": p})
    for row, value in zip(rows, holm_adjust([r["p_raw"] for r in rows])):
        row["p_holm_3"] = float(value)
    return rows


def constants(info):
    rows = []
    for target, item in info.items():
        for estimator in ("mean", "median"):
            value = float(getattr(np, estimator)(item["train"]))
            metrics = measurement_metrics(item["test"], np.full_like(item["test"], value), item["tolerance"])
            rows.append({"target": target.upper(), "predictor": f"Training {estimator}", "value": value,
                         "reference_sd": float(item["test"].std(ddof=1)), "prediction_sd": 0.0, **metrics})
    return rows


def prompts(info, replicates):
    rng = np.random.default_rng(20260809)
    cobb = load_json(RECORDS / "prompt_matched/aasce_matched_prompt_results.json")
    foot = load_json(RECORDS / "prompt_matched/hvangle_matched_prompt_patient_results.json")
    rows, distributions = [], {target: {} for target in info}
    sequence = [("cobb", label) for label in cobb["models"]]
    sequence += [(target, label) for label in foot["models"] for target in foot["models"][label]]
    for target, label in sequence:
        source = cobb if target == "cobb" else foot
        metric = source["models"][label] if target == "cobb" else source["models"][label][target]
        truth, prediction = np.asarray(metric["ground_truth"], float), np.asarray(metric["predictions"], float)
        clusters = clusters_for(target, source["test_ids"], info)
        constant = float(np.median(info[target]["train"]))
        difference = np.abs(prediction - truth) - np.abs(constant - truth)
        low, high = {"cobb": (1, 90), "hva": (5, 90), "ima": (5, 30)}[target]
        supported = (truth >= low) & (truth <= high)
        statistic, p = paired_test(difference, clusters)
        rows.append({"model": label, "target": target.upper(), "n": len(truth),
                     "n_independent_units": len(np.unique(clusters)),
                     "mae": float(np.abs(prediction - truth).mean()),
                     "icc": measurement_metrics(truth, prediction, info[target]["tolerance"])["icc"],
                     "delta_mae_vs_median_constant": float(unit_means(difference, clusters).mean()),
                     "delta_mae_ci": paired_interval(difference, clusters, rng, replicates),
                     "wilcoxon_p_raw": p, "n_reference_outside_prompt_range": int((~supported).sum()),
                     "mae_within_prompt_reference_range": float(np.abs(prediction[supported] - truth[supported]).mean())})
        distributions[target][label] = prediction
        distributions[target]["Reference"] = truth
    for row, value in zip(rows, holm_adjust([r["wilcoxon_p_raw"] for r in rows])):
        row["wilcoxon_p_holm_6"] = float(value)
    return rows, distributions


def structured():
    rows = []
    for selector, directory in [("dice", "hvangle_axis_geometry"), ("measurement", "hvangle_axis_geometry_measurement_selected")]:
        for run in load_json(RECORDS / directory / "results.json"):
            spatial = run["test"]
            row = {"selector": selector, "seed": run["seed"], "best_epoch": run["best_epoch"],
                   "dice": spatial["segmentation"]["mean_dice"], "cldice": spatial["segmentation"]["mean_cldice"],
                   "axis_mae": spatial["axis"]["mean_angular_mae_deg"]}
            for target in ("hva", "ima"):
                metric = spatial["measurements"][target]
                for key, original in [("mae", "mae"), ("icc", "icc_2_1"), ("tolerance", "within_tolerance_percent"),
                                      ("bias", "bias"), ("loa_lower", "loa_lower"), ("loa_upper", "loa_upper")]:
                    row[f"{target}_{key}"] = metric[original]
            rows.append(row)
    return rows


def verify_table(expected, actual, keys, columns):
    checks = []
    lookup = {tuple(str(row[k]) for k in keys): row for row in actual}
    for row in expected:
        key = tuple(str(row[k]) for k in keys)
        if key not in lookup:
            raise ValueError(f"Missing computed row: {key}")
        other = lookup[key]
        for column in columns:
            value = row[column]
            if isinstance(value, str) and value.startswith("["):
                value = json.loads(value)
            checks.append({"row": list(key), "field": column,
                           "pass": bool(np.allclose(value, other[column], rtol=1e-7, atol=2e-7))})
    return checks


def verify_results(results):
    expected = CODE_ROOT / "results/expected"
    checks = []
    for name, keys in [("supervised_bootstrap_ci", ("target", "model")),
                       ("supervised_paired_tests", ("target", "comparison")),
                       ("medical_probe_paired_tests", ("target", "comparison"))]:
        reference = load_json(expected / f"{name}.json")
        columns = [k for k in reference[0] if k not in keys and k not in ("n_cases", "n_independent_units")]
        checks += verify_table(reference, results[name], keys, columns)
    for name, keys, columns in [
        ("constant_baselines", ("target", "predictor"), ("value", "reference_sd", "mae", "tolerance", "bias", "loa_lower", "loa_upper", "icc")),
        ("structured_geometry_per_seed", ("selector", "seed"), ("best_epoch", "dice", "cldice", "axis_mae", "hva_mae", "hva_icc", "ima_mae", "ima_icc")),
        ("prompt_paired_tests", ("target", "model"), ("mae", "delta_mae_vs_median_constant", "delta_mae_ci", "wilcoxon_p_raw", "wilcoxon_p_holm_6", "n_reference_outside_prompt_range", "mae_within_prompt_reference_range")),
    ]:
        reference = pd.read_csv(expected / f"{name}.csv").to_dict("records")
        checks += verify_table(reference, results[name], keys, columns)
    reference = pd.read_csv(expected / "frozen_baseline_equivalence.csv").to_dict("records")
    # Keep the historical CSV field name only at the reference boundary.
    for row in reference:
        row["margin"] = row["equivalence_margin"]
    checks += verify_table(reference, results["frozen_baseline_similarity"], ("target", "encoder"),
                           ("delta_mae_vs_training_median", "ci_lower", "ci_upper", "margin"))
    reference = pd.read_csv(expected / "frozen_probe_dispersion.csv").to_dict("records")
    checks += verify_table(reference, results["frozen_baseline_similarity"], ("target", "encoder"),
                           ("reference_sd", "prediction_sd", "sd_ratio", "mae_seed_mean"))
    return {"checks": len(checks), "failed": [x for x in checks if not x["pass"]],
            "passed": all(x["pass"] for x in checks), "absolute_tolerance": 2e-7}


def recompute(replicates=10000):
    info = target_info()
    summaries, pairs = supervised(info, replicates)
    prompt_rows, distributions = prompts(info, replicates)
    results = {"supervised_bootstrap_ci": summaries, "supervised_paired_tests": pairs,
               "medical_probe_paired_tests": frozen_pair(info, replicates),
               "constant_baselines": constants(info), "frozen_baseline_similarity": frozen(info, replicates),
               "structured_geometry_per_seed": structured(), "prompt_paired_tests": prompt_rows}
    return results, distributions
