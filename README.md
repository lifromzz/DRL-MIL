# DRL-MIL: Reformulating Single-Stream MIL as Cross-Branch Adversarial Dual-Stream Learning for Improved WSI Classification

## Repository layout

```text
CRL-MIL/
  process/             # Patient-level and slide-level five-fold split generation
  feature_extraction/  # ResNet and Prov-GigaPath instance feature extraction
  data/                # Existing dataset split CSVs
  Models/              # MIL backbones and attention modules
  Lightning/           # Feature-bag loading, training, and evaluation
  Config/              # Paths, model dimensions, and training configuration
  train/               # Training entry points
```

## Dataset preparation and five-fold splits

The repository now includes both split generators. See
[split preparation and verification](CRL-MIL/process/README.md) for input schemas,
commands, label rules, and the official CPTAC case mapping.

| Script | Unit | Five-fold protocol |
| --- | --- | --- |
| [create_patient_split.py](CRL-MIL/process/create_patient_split.py) | Patient/case | Stratified outer folds on case labels; all slides of a patient stay together within each fold |
| [create_kflod.py](CRL-MIL/process/create_kflod.py) | Slide | Ordinary outer `KFold`; the inner train/validation split is stratified by slide label |

Both default to seed 42 and five test folds. Validation uses 20% of the remaining
80%, giving approximately **64% train / 16% validation / 20% test** at the
respective splitting unit. Slide-level splits can share patients across
partitions and should not be described as patient-independent evaluation.

The patient-level code uses BRACS `Patient Id` and CPTAC case identifiers.
The four exceptional CPTAC filenames were checked against the
[official TCIA subject/image table](https://pathdb.cancerimagingarchive.net/listofimages/528?page=0);
their case mapping is included in
[cptac_patient_map.csv](CRL-MIL/process/cptac_patient_map.csv).
The default preserves historical identifiers and fold assignments. Supplying
the explicit map with strict mode produces official IDs but may change the
sorted order and fold assignments; keep that as a separate split version.

Run from the repository root:

```bash
cd CRL-MIL
python -m pip install -r process/requirements.txt
python process/create_kflod.py --input-csv /path/to/cptac_lung.csv --output-dir data/cptac_lung_slide
python process/create_patient_split.py --dataset bracs --bracs-xlsx /path/to/BRACS.xlsx --output-root data
python process/create_patient_split.py --dataset cptac --cptac-csv /path/to/cptac_lung.csv --output-root data
```

Each split folder contains `fold0_train.csv`, `fold0_val.csv`, `fold0_test.csv`,
through fold 4. These scripts use locally supplied metadata; this update does
not replace the existing split CSVs in `data/`.

## Instance feature extraction

The standalone [feature_extraction folder](CRL-MIL/feature_extraction/README.md)
contains the two original extraction pipelines, packaged with configurable
paths, shared patch loading, dependency lists, and weight references:

| Extractor | Instance representation | Output |
| --- | --- | --- |
| [save_represention.py](CRL-MIL/feature_extraction/save_represention.py) | ResNet layer 3 / layer 4 / concatenation | 1024 / 2048 / 3072 dimensions |
| [save_represention_Prov-GigaPath.py](CRL-MIL/feature_extraction/save_represention_Prov-GigaPath.py) | Prov-GigaPath tile encoder | 1536 dimensions |

Input is one folder of image patches per slide. Each output bag contains one
feature vector per patch. The GigaPath slide encoder is not used. ResNet retains
its original dictionary serialization by default; use **`--output-format tensor`**
with the current CRL-MIL loader, which expects an `[N,D]` tensor.

```bash
# From CRL-MIL/; install a matching PyTorch/torchvision pair for your platform.
python -m pip install -r feature_extraction/requirements.txt
python feature_extraction/save_represention.py --data_dir /path/to/patches --repre_dir /path/to/features_resnet --checkpoint /path/to/resnet50-19c8e357.pth --stage stage4 --output-format tensor
python feature_extraction/save_represention_Prov-GigaPath.py --data_dir /path/to/patches --repre_dir /path/to/features_gigapath --checkpoint-dir /path/to/gigapath
```

Large pretrained weights are not stored in this repository:

- **ResNet:** the supplied local ResNet-50 checkpoint matches the historical
  [official ImageNet checkpoint](https://download.pytorch.org/models/resnet50-19c8e357.pth).
  Cite He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
  [Paper](https://arxiv.org/abs/1512.03385).
- **Prov-GigaPath:** obtain `config.json` and `pytorch_model.bin` from the
  [official model repository](https://huggingface.co/prov-gigapath/prov-gigapath),
  following its access requirements. Cite Xu et al., *A whole-slide foundation
  model for digital pathology from real-world data*, Nature 630, 181-188 (2024).
  [Paper](https://doi.org/10.1038/s41586-024-07441-w).

See the [extractor documentation](CRL-MIL/feature_extraction/README.md) for exact
preprocessing, checkpoint checksum, output formats, and release validation;
[BibTeX references](CRL-MIL/feature_extraction/references.bib) are included.
The supplied ResNet checkpoint is ImageNet pretrained; historical `BT` directory
names in the training configuration do not identify its pretraining method.

## Training and evaluation

From `CRL-MIL/`, configure dataset, split, and feature paths in `Config/base.py`,
`Config/args_base.py`, and `Config/train_args.py`, then run:

```bash
python -m train.manage
```

The configuration contains machine-specific defaults that must be updated for
your environment. Set `input_dim` and the relevant hidden/input dimensions to
match the selected feature extractor, and point train/validation/test feature
paths to its output folder (for ResNet, include the `stage3`, `stage4`, or
`stage34` subfolder). For patient-level experiments, point the split paths to the
generated patient split folder and add dataset keys/class counts if using new
dataset names. The existing `BRACS` configuration uses a fixed fold-0 split;
`BRACS_` and other configured datasets use fold-indexed paths. Select the intended
protocol explicitly rather than assuming all entry-point runs use patient CV.

## Models and results

`CRL-MIL/Models/` includes attention-based MIL (`abmil.py`), transformer backbones
(`TransMIL.py`, `ILRA.py`), the graph-based `WiKG.py`, and model-loading utilities.
Evaluation reports accuracy, AUC, F1 score, and MCC on datasets including BRACS
and CPTAC-Lung.

## License and contact

This project is intended for academic research purposes only. Please contact
the authors for commercial licensing. Refer to the original model providers
for pretrained model terms. For questions or issues, open a GitHub issue or
contact the authors.
