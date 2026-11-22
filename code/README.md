# Measurement readiness: code and reproducibility

Implementation and saved-prediction replay for **Measurement Readiness in Musculoskeletal Radiography: A Geometry-Centric Framework and Evaluation**.

## Quick start: verify the reported results

From the repository root, with Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python code/reproduce.py verify
python code/reproduce.py statistics --figures
```

`verify` checks reference-file hashes, all partition sizes, and patient separation using the Python standard library. `statistics` uses saved predictions and public-release split/target records. It makes **no API calls, trains no models, and requires no radiographs, checkpoints, CUDA or PyTorch**. Generated files go to `code/outputs/paper_statistics/`; reference records remain unchanged.

The default replay uses 10,000 bootstrap draws and verifies 432 numerical fields against the manuscript reference tables. It reports separate Holm families of nine architecture contrasts, nine proportional-bias tests, three selected frozen-encoder contrasts, and six prompt-to-median contrasts. The post-hoc baseline-similarity margin is descriptive, not a formal equivalence test.

## Experiments represented here

| Inference examined | Conditions | Main execution code |
| --- | --- | --- |
| A numerical output establishes case-level agreement | Training-median baseline; matched GPT-4o / Claude Sonnet 5 prompts | `scripts/experiments/run_matched_prompts.py` |
| Lower MAE establishes stronger agreement | Frozen readouts; end-to-end EfficientNet-B0, ConvNeXt-T and ViT-B/16 | `scripts/benchmarks/cross_model_benchmark.py` |
| Spatial accuracy establishes final-angle agreement | Four-class foot-axis U-Net; fixed HVA/IMA construction; Dice and angle-error checkpoint selectors | `scripts/benchmarks/hvangle_axis_geometry_benchmark.py` |

The empirical split contains 336/72/73 AASCE images and 829/172/176 HVAngleEst images from 617/132/133 disjoint patients. AASCE has no verifiable patient identifiers. GT in the structured code means the great-toe proximal phalanx, not ground truth.

## Layout

```text
code/
├── reproduce.py                 # public verification/statistics entry point
├── requirements*.txt            # CPU replay; training; frozen-model extras
├── configs/paper.json           # models and training budget used in the paper
├── scripts/benchmarks/          # executed training, reconstruction and analysis code
├── scripts/experiments/         # matched provider-call runner
├── src/msk_geomeasurement/      # data-free statistical replay
├── records/                    # immutable saved predictions; no image/weight files
├── splits/                     # case order, references and released patient grouping
├── results/expected/            # manuscript-scoped numerical reference tables/ledgers
├── docs/                       # data, model and reproduction instructions
└── tests/                      # synthetic agreement and patient-bootstrap checks
```

For training and image-based figures, follow [data preparation](docs/DATA.md), [model setup](docs/MODELS.md), and the [full reproduction guide](docs/REPRODUCIBILITY.md). [Provenance](docs/PROVENANCE.md) distinguishes historical execution records from the current paper's comparison families. Use `reproduce.py statistics` for the manuscript comparison families.

Run the statistical unit checks with:

```bash
python -m unittest discover -s code/tests -v
```

No dataset images, pretrained weights, API credentials or raw prompt payloads are distributed. Model and dataset sources retain their own licenses and access conditions. Source code retains the Apache-2.0 license already included in the executed-code package; see [LICENSE](LICENSE).
