# Data preparation

Statistical replay does not require image downloads. Training and image overlays require the original public releases, obtained under their own terms:

- **AASCE:** the [reviewed MICCAI 2019 mirror](https://www.kaggle.com/datasets/wahyurahmaniar/aasce-miccai-2019-x-ray-dataset). Use the 481-image release with organizer-supplied filename, angle and landmark tables. The first angle column is the primary target; a different local endplate heuristic is not a replacement.
- **HVAngleEst:** obtain the image release, annotations and official manifests from the sources linked by the [dataset publication](https://doi.org/10.1038/s41597-025-05261-9). Retain its `patient_id` metadata and four-pixel axis masks.

Arrange an external workspace as follows:

```text
WORKSPACE/
├── AASCE/data/
│   ├── train/                         # original radiographs
│   └── train_txt/
│       ├── filenames.csv
│       ├── angles.csv
│       └── landmarks.csv
├── HVAngleEst/
│   ├── HVAngleEst/
│   │   ├── datasets.csv
│   │   ├── images/
│   │   └── annotations/masks/
│   └── splits/{train,val,test}.csv
├── checkpoints/                       # optional model weights; see MODELS.md
├── external_models/                   # optional official implementations
└── outputs/                           # newly generated runs, never reference inputs
```

Set `MSK_WORKSPACE` to this directory. Without it, training expects this layout inside `code/`. No symlink or machine-specific path is needed in source code.

```bash
export MSK_WORKSPACE=/absolute/path/to/WORKSPACE
python code/scripts/check_data.py --workspace "$MSK_WORKSPACE"
```

The check validates the released target tables, exact archived split order, patient counts and availability of axis masks. It neither trains models nor changes the input releases.

The release's `splits/` CSVs are derived case/target records, not radiographs. They retain the original benchmark order and the dataset's de-identified patient grouping. AASCE supplies no patient grouping. Their source data and identifiers remain governed by the original dataset terms; the code license does not relicense the datasets.

The HVAngleEst retained pool has 1,177 unilateral cases. It uses the official-manifest pool and removes duplicated filenames representing bilateral images. The immutable split manifests are used by every published training entry point. Missing images cause a failed data audit rather than a replacement cohort.
