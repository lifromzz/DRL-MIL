# DRL-MIL: Reformulating Single-Stream MIL as Cross-Branch Adversarial Dual-Stream Learning for Improved WSI Classification

## Repository layout

```text
DRL-MIL/
  process/             # Patient-level and slide-level five-fold split generation
    metadata/          # Ordered BRACS and CPTAC-Lung slide/label manifests
  feature_extraction/  # ResNet and Prov-GigaPath instance feature extraction
  tests/               # Split regression tests and real-weight feature smoke tests
  data/                # Existing dataset split CSVs
  Models/              # MIL backbones and attention modules
  Lightning/           # Feature-bag loading, training, and evaluation
  Config/              # Paths, model dimensions, and training configuration
  train/               # Training entry points
```

## Start from a fresh clone

The preprocessing workflow is tested with **Python 3.12 on Windows** in an isolated CPU
environment and Python 3.9 with CUDA 12.4. The commands below cover preprocessing;
they do not install the separate full MIL training environment.

```bash
git clone https://github.com/lifromzz/DRL-MIL.git
cd DRL-MIL/DRL-MIL
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. If activation is unavailable,
use `.venv/bin/python` or `.\.venv\Scripts\python.exe` instead of `python` below.
Keep the working directory at **`DRL-MIL/DRL-MIL`** for all remaining commands.
The first `DRL-MIL` is the cloned repository; the second is its source directory.
Paths containing spaces must be quoted.

```bash
# Windows/Linux CPU baseline; works without an NVIDIA GPU.
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-preprocessing.txt -r process/requirements.txt -r feature_extraction/requirements.txt
python -m pip check
```

For a compatible NVIDIA system, use
`--index-url https://download.pytorch.org/whl/cu124` in the first installation
command instead. See the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).
On macOS, use the matching platform-specific PyTorch installation command from
that page instead of the Windows/Linux wheel-index commands above.
[constraints-preprocessing.txt](DRL-MIL/constraints-preprocessing.txt) pins the
main libraries used for this audit. CPU is a functional fallback; a suitable GPU
is recommended for full-cohort GigaPath extraction.
If a package download times out, retry the same pip command with
`--timeout 120 --retries 5`.

## Dataset preparation and five-fold splits

The repository now includes both split generators. See
[split preparation and verification](DRL-MIL/process/README.md) for input schemas,
commands, label rules, and the official CPTAC case mapping.

| Script | Unit | Five-fold protocol |
| --- | --- | --- |
| [create_patient_split.py](DRL-MIL/process/create_patient_split.py) | Patient/case | Stratified outer folds on case labels; all slides of a patient stay together within each fold |
| [create_kflod.py](DRL-MIL/process/create_kflod.py) | Slide | Ordinary outer `KFold`; the inner train/validation split is stratified by slide label |

Both default to seed 42 and five test folds. Validation uses 20% of the remaining
80%, giving approximately **64% train / 16% validation / 20% test** at the
respective splitting unit. Slide-level splits can share patients across
partitions and should not be described as patient-independent evaluation.

The patient-level code uses BRACS `Patient Id` and CPTAC case identifiers.
The four exceptional CPTAC filenames were checked against the
[official TCIA subject/image table](https://pathdb.cancerimagingarchive.net/listofimages/528?page=0);
their case mapping is included in
[cptac_patient_map.csv](DRL-MIL/process/cptac_patient_map.csv).
The default preserves historical identifiers and fold assignments. Supplying
the explicit map with strict mode produces official IDs but may change the
sorted order and fold assignments; keep that as a separate split version.

The two ordered `path,STAS` manifests are included under
[process/metadata/](DRL-MIL/process/metadata/README.md), so the slide workflows and
CPTAC patient workflow can run immediately after installation. BRACS patient
splits additionally require the official `BRACS.xlsx` from the
[BRACS download portal](https://www.bracs.icar.cnr.it/download/) (dataset access
required). Save it as `inputs/BRACS.xlsx`, or substitute its actual path.

From `DRL-MIL/DRL-MIL`:

```bash
python process/create_kflod.py --input-csv process/metadata/BRACS.csv --output-dir outputs/splits/BRACS_slide
python process/create_kflod.py --input-csv process/metadata/cptac_lung.csv --output-dir outputs/splits/cptac_lung_slide
python process/create_patient_split.py --dataset cptac --cptac-csv process/metadata/cptac_lung.csv --output-root outputs/splits
python process/create_patient_split.py --dataset bracs --bracs-xlsx inputs/BRACS.xlsx --output-root outputs/splits
```

Each split folder contains `fold0_train.csv`, `fold0_val.csv`, `fold0_test.csv`,
through fold 4: **15 fold CSVs per dataset/protocol**. Patient folders also contain
one summary CSV. Examples write under `outputs/` and leave the historical
`data/` splits intact. Keep manifest row order unchanged to reproduce slide folds.

## Instance feature extraction

The standalone [feature_extraction folder](DRL-MIL/feature_extraction/README.md)
contains the two original extraction pipelines, packaged with configurable
paths, shared patch loading, dependency lists, and weight references:

| Extractor | Instance representation | Output |
| --- | --- | --- |
| [save_represention.py](DRL-MIL/feature_extraction/save_represention.py) | ResNet layer 3 / layer 4 / concatenation | 1024 / 2048 / 3072 dimensions |
| [save_represention_Prov-GigaPath.py](DRL-MIL/feature_extraction/save_represention_Prov-GigaPath.py) | Prov-GigaPath tile encoder | 1536 dimensions |

Input is one folder of image patches per slide. Each output bag contains one
feature vector per patch. The GigaPath slide encoder is not used. ResNet retains
its original dictionary serialization by default; use **`--output-format tensor`**
with the current DRL-MIL loader, which expects an `[N,D]` tensor.

```bash
# First obtain weights as described in feature_extraction/README.md.
# Replace /path/to/patches with your slide-folder tree of image patches.
python feature_extraction/save_represention.py --data_dir /path/to/patches --repre_dir outputs/features_resnet --checkpoint feature_extraction/Checkpoint/resnet50-19c8e357.pth --stage stage4 --output-format tensor --bs 1 --num_workers 0 --device cpu
python feature_extraction/save_represention_Prov-GigaPath.py --data_dir /path/to/patches --repre_dir outputs/features_gigapath --checkpoint-dir feature_extraction/gigapath --bs 1 --num_workers 0 --device cpu
```

These are conservative first-run settings; use `--device cuda` and increase
batch size/workers as memory permits. Clone alone does not provide WSI patches
or pretrained weights. Raw WSI tiling is outside these extraction scripts.

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

See the [extractor documentation](DRL-MIL/feature_extraction/README.md) for exact
preprocessing, checkpoint checksum, output formats, and release validation;
[BibTeX references](DRL-MIL/feature_extraction/references.bib) are included.
The supplied ResNet checkpoint is ImageNet pretrained; historical `BT` directory
names in the training configuration do not identify its pretraining method.

## Check your installation

```bash
# No images, model weights, or BRACS account needed for the split tests.
python -m unittest discover -s tests -p test_splits.py -v

# Actual encoders on generated RGB test patches; download weights first.
python tests/smoke_features.py --resnet-checkpoint feature_extraction/Checkpoint/resnet50-19c8e357.pth --device cpu
python tests/smoke_features.py --gigapath-dir feature_extraction/gigapath --device cpu
```

The split tests check all five folds, coverage, label preservation, patient
isolation, official CPTAC mappings, and equality to the published slide/CPTAC
patient folds. The BRACS test generates an Excel fixture with the documented
schema; real BRACS metadata is still required for the actual cohort. Feature
tests use real supplied weights, verify 1024/2048/3072-dimensional ResNet and
1536-dimensional GigaPath outputs, and check patch order and ResNet resume/output
format compatibility. Tests use temporary directories and remove their own
generated outputs. See the [reproduction audit](DRL-MIL/tests/README.md) for the
checked environment and limits.

## Training and evaluation

From `DRL-MIL/`, configure dataset, split, and feature paths in `Config/base.py`,
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

`DRL-MIL/Models/` includes attention-based MIL (`abmil.py`), transformer backbones
(`TransMIL.py`, `ILRA.py`), the graph-based `WiKG.py`, and model-loading utilities.
Evaluation reports accuracy, AUC, F1 score, and MCC on datasets including BRACS
and CPTAC-Lung.

## License and contact

This project is intended for academic research purposes only. Please contact
the authors for commercial licensing. Refer to the original model providers
for pretrained model terms. For questions or issues, open a GitHub issue or
contact the authors.
