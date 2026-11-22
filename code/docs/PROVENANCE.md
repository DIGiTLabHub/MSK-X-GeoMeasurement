# Source and result provenance

## Executed source baseline

The training, bounded-prompt, reconstruction and historical analysis scripts were recovered from the authors' existing `execution_code.zip` package. Original per-file hashes and the package hash are in `source_manifest.json`. The Apache-2.0 license already distributed with that package is retained in `code/LICENSE`.

The publication preparation makes these changes:

1. Storage paths use `workspace_paths.py`; `MSK_WORKSPACE` selects data, weights and newly generated outputs.
2. Training data loaders read immutable published split manifests from `records/` rather than from an output directory that may contain a different run.
3. The default model configuration contains only the seven executable model conditions used in the current paper (three supervised and four frozen). Unavailable Bone X-ray VLP is documented, not substituted.
4. `src/msk_geomeasurement/` and `reproduce.py` provide a CPU-only statistical replay and explicit current-paper comparison families.
5. A vectorized bootstrap preserves the executed random draws. Synthetic tests compare it with literal block resampling; the paper replay verifies 432 reference fields at an absolute tolerance of 0.0000002 degrees or statistic units.

Network definitions, training losses, retained model-specific transforms, fitting rules, HVA/IMA constructions and exact prompt strings remain unchanged. Unused RadImageNet/DAX/MedCLIP-RN50/Bone-VLP execution branches, the surrogate Cobb heatmap trainer and the old analysis/summary entry points are excluded from the public entry points. Foot helper functions are retained in `foot_axis_data.py`. The prompt-protocol table points to the renamed public runner; its prompt strings, request settings and numerical entries are unchanged. Data loaders reject changed cohort identifiers, targets, patient groups or missing images. `docs/source_manifest.json` identifies the source of each executed script. The released helper for statistical replay uses the same mathematical definitions and is checked against the archived outputs.

## Current paper versus historical records

The executed analysis originally processed a larger model inventory. Its random-number stream therefore includes draws for historical auxiliary models. `records/` retains those saved numerical records so confidence-interval endpoints can be replayed; this does not require retraining or making new comparisons for those models.

The current-paper replay keeps three supervised architectures and four frozen encoders. Holm correction is recomputed **after** selecting the declared testing families:

| Family | Comparisons |
| --- | ---: |
| Supervised architecture pairs across three targets | 9 |
| Proportional-bias tests across three models and three targets | 9 |
| MedImageInsight versus BiomedCLIP across three targets | 3 |
| Two prompt conditions versus the training median across three targets | 6 |

The separate families do not constitute a global study-wide error guarantee. Some historical CSV field names contain `equivalence` or a broader family size; those original schemas are retained only as reference inputs and must not be used as current-paper inferential results. The public replay uses `baseline_similarity_criterion_met` and the current family names. The baseline-similarity rule is a post-hoc descriptive margin, not a formal equivalence conclusion.

`results/expected/` contains the manuscript-scoped reference tables from the authors' Frontiers materials. `records/` contains numerical execution records rather than model checkpoints. `splits/` records are derived from the original public dataset tables using the archived case order and patient grouping. No filenames, group assignments, predictions or values were invented.

## What the verification establishes

The offline replay checks numerical reproduction from saved predictions, split metadata and reference values. It is not a rerun of GPU training or of mutable provider models. It does not establish identical behavior for different checkpoints, environments, prompts or datasets.

The full image dataset audit verifies the original AASCE target identity and the declared patient-disjoint HVAngleEst pool. Image-based target reconstruction and figures still require the source datasets. Raw provider responses, images and pretrained weights are not included in this public code package.
