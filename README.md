# MSK-X-GeoMeasurement

Research repository for **Measurement Readiness in Musculoskeletal Radiography: A Geometry-Centric Framework and Evaluation**.

## Authors

Hongcheng Jiang<sup>1</sup>, ZhiQiang Chen<sup>1</sup>, Caleb W. Grote<sup>2</sup>, Sherwin S. Chan<sup>2</sup>, Cheng Han<sup>1</sup>, Rong Wang<sup>3</sup>, and Dianxiang Xu<sup>1</sup>

1. School of Science and Engineering, University of Missouri-Kansas City, Kansas City, MO, USA
2. Children's Mercy Research Institute, Children's Mercy Hospital, Kansas City, MO, USA
3. School of Dentistry, University of Missouri-Kansas City, Kansas City, MO, USA

**Correspondence:** ZhiQiang Chen ([chenzhiq@umkc.edu](mailto:chenzhiq@umkc.edu)); Caleb W. Grote ([cwgrote@cmh.edu](mailto:cwgrote@cmh.edu)).

## Abstract

**Introduction:** Artificial-intelligence (AI) systems increasingly return protocol-defined measurements in musculoskeletal (MSK) radiography, including Cobb, hallux valgus (HVA), and intermetatarsal (IMA) angles. Dataset annotations, model outputs, and evaluations often describe different objects. Developers and clinicians lack shared terminology to connect them, despite the collaboration called for by machine-learning operations (MLOps) and sound machine-learning practice. We asked what evidence establishes readiness for a protocol-defined radiographic measurement.

**Methods:** The proposed Geometric Construction Pathway traces measurement construction from a radiograph. Three dimensions map evidence onto them: Annotation Support Level (ASL) for datasets, Capability Description Level (CDL) for outputs, and Evaluation Variable Level (EVL) for evaluations. We applied the framework in a purposive review of 15 public or research-accessible datasets and 19 model-task outputs from 13 studies, and experiments on public Cobb (73 test images) and HVA/IMA targets (176 test images from 133 patients). Case studies compared constant baselines and prompt-only vision-language models, frozen-encoder readouts and end-to-end regressors, and a structured foot pathway. Metrics included mean absolute error (MAE), the intraclass correlation coefficient (ICC), and Bland–Altman agreement.

**Results:** Four of 15 datasets supplied structured final targets; five of 19 outputs produced protocol-defined measurements. All five were evaluated against measurement references, none for clinical use. A constant training-median predictor attained MAEs of 16.31°, 8.44°, and 3.02° for Cobb, HVA, and IMA. Prompt-only GPT-4o and Claude Sonnet 5 had higher HVA and IMA error than the constant (Holm-adjusted *p* < 0.001), with ICCs of 0.024–0.118. Two structured foot runs with nearly identical centerline Dice (0.820 and 0.819) had IMA ICCs of 0.762 and 0.409.

**Discussion:** Output form, favorable MAE, and spatial accuracy each fail to establish case-level measurement agreement. The pathway specifies information dependencies independently of architecture, providing developers and clinicians with a common specification for data preparation, training, and evaluation across task-specific and foundation models. Datasets should release final values with defining anatomical references, provenance, and calibration information. Studies should evaluate the final quantity directly and report absolute error and agreement for the intended use.

**Keywords:** Musculoskeletal radiography, Measurement readiness, Anatomical geometry, Literature review, Artificial intelligence, Machine-learning operations.

## Repository structure

```text
MSK-X-GeoMeasurement/
├── README.md
├── code/          # Implementations and experiment resources
└── paper/         # Manuscript and supporting materials
```

The repository is being reorganized around this paper. The [code](code/) and [paper](paper/) directories currently contain organizational READMEs; implementation files, experiment resources, and manuscript materials have not yet been added. The abstract above reports the manuscript findings supplied by the authors.

## Availability and citation

Publication details, a paper link, and a formal citation will be added when available. Installation and reproduction instructions will accompany the code release.

## License

A license has not yet been specified.
