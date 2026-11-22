# Reproduction guide

Commands below are run from the repository root unless noted. Dataset and model paths are described in DATA.md and MODELS.md.

## 1. Offline statistical replay

```bash
python code/reproduce.py verify
python code/reproduce.py statistics --figures
```

The statistical replay checks paper-scoped reference values and regenerates JSON/CSV tables and the prompt-distribution figure. It preserves the original random-number consumption of the historical analysis before retaining the paper's three supervised architectures and four frozen encoders. The primary resampling count is 10,000; `--replicates` is available for development, but other counts are not checked against paper confidence-interval endpoints.

- Absolute errors are calculated by seed and averaged within image before paired inference.
- HVAngleEst paired contrasts average differences within patient; their point estimates are patient weighted.
- Descriptive confidence intervals resample patients with replacement and keep all images in each sampled block.
- AASCE uses images as independent units.
- Agreement intervals use predictions averaged across seeds. This is distinct from averaging seed-wise absolute errors for MAE.
- Holm correction is applied separately within each stated family: nine architecture contrasts, nine proportional-bias tests, three selected encoder contrasts, and six prompt-to-baseline contrasts.
- Baseline-similarity intervals are descriptive and do not establish formal equivalence or noninferiority.

## 2. Data audit and direct regression

```bash
export MSK_WORKSPACE=/absolute/path/to/WORKSPACE
python code/scripts/check_data.py --workspace "$MSK_WORKSPACE"

python code/scripts/benchmarks/cross_model_benchmark.py \
  --config code/configs/paper.json --dataset aasce --dry-run \
  --models efficientnet_b0 convnext_tiny vit_b16 \
  --output-dir "$MSK_WORKSPACE/outputs/audit/aasce"
python code/scripts/benchmarks/cross_model_benchmark.py \
  --config code/configs/paper.json --dataset hvangle --dry-run \
  --models efficientnet_b0 convnext_tiny vit_b16 --hvangle-split-mode patient \
  --output-dir "$MSK_WORKSPACE/outputs/audit/hvangle"
```

For full training, remove `--dry-run` and use separate output directories. The paper budget is seeds 17/42/73, three learning-rate/weight-decay candidates, at most 60 epochs and patience 10. The configuration retains Huber loss, augmentation and validation selection settings. GPU retraining can differ across software/hardware; the saved-record replay is the check of the historical numerical tables.

## 3. Frozen readouts

After installing the upstream source packages and weights:

```bash
python code/scripts/benchmarks/cross_model_benchmark.py \
  --config code/configs/paper.json --dataset aasce \
  --models biomedclip_probe medimageinsight_probe medclip_vit_probe \
  --output-dir "$MSK_WORKSPACE/outputs/new_runs/frozen/aasce"
python code/scripts/benchmarks/cross_model_benchmark.py \
  --config code/configs/paper.json --dataset hvangle --hvangle-split-mode patient \
  --models biomedclip_probe medimageinsight_probe medclip_vit_probe \
  --output-dir "$MSK_WORKSPACE/outputs/new_runs/frozen/hvangle"
```

Run `--models medsam_encoder_probe` separately to retain the model-specific 1024-pixel condition. The frozen head is the executed nonlinear MLP. Encoder parameters are not updated.

## 4. Structured foot pathway

```bash
python code/scripts/benchmarks/hvangle_axis_geometry_benchmark.py \
  --seeds 17 42 73 --epochs 60 --patience 10 --lr 0.001 \
  --image-size 256 --batch-size 16 --checkpoint-metric dice \
  --output "$MSK_WORKSPACE/outputs/new_runs/axes_dice"
python code/scripts/benchmarks/hvangle_axis_geometry_benchmark.py \
  --seeds 17 42 73 --epochs 60 --patience 10 --lr 0.001 \
  --image-size 256 --batch-size 16 --checkpoint-metric measurement \
  --output "$MSK_WORKSPACE/outputs/new_runs/axes_measurement"
```

The second condition is the recorded post-hoc selector sensitivity. It changes checkpoint selection and early stopping, without adding an angle loss. Soft probability-weighted PCA and original-image coordinate reconstruction are retained. Reference axes are endpoint defined; no reference fallback is used during inference.

## 5. Recorded prompt protocol

First inspect the dataset/prompt dry run:

```bash
python code/scripts/experiments/run_matched_prompts.py \
  --provider openai --dry-run
python code/scripts/experiments/run_matched_prompts.py \
  --provider claude --dry-run
```

Real calls require `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the environment and incur provider charges. They are never called by the offline replay. Use distinct output files:

```bash
python code/scripts/experiments/run_matched_prompts.py \
  --provider openai --output "$MSK_WORKSPACE/outputs/results/matched_prompt_openai.jsonl"
python code/scripts/experiments/run_matched_prompts.py \
  --provider claude --output "$MSK_WORKSPACE/outputs/results/matched_prompt_claude.jsonl"
python code/scripts/benchmarks/build_matched_prompt_results.py
```

The executed prompts include numeric examples and explicit response bounds. Their exact text and original parser are retained. A prompt/model/payload change requires a distinct run/variant identifier; do not resume a changed interaction into historical output files. Mutable model aliases and provider transformations can change new results.

## 6. Image-based figure

The original axis-overlay/error figure requires the downloaded radiographs. With the archived structured results:

```bash
python code/scripts/benchmarks/plot_hvangle_protocol_geometry.py \
  --results code/records/controlled_cross_model_v2/hvangle_axis_geometry/results.json \
  --seed 42 --output "$MSK_WORKSPACE/outputs/fig_protocol_geometry.pdf"
```

Visual styling can differ from final manuscript layout edits. Axis directions, errors, and case selection come from the saved run. The newer prompt-distribution renderer preserves the numerical data and distinguishes training median and mean.

The public entry point is `reproduce.py statistics`. Historical training and analysis entry points outside the manuscript scope are excluded.
