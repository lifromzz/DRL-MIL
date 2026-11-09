# Preprocessing reproduction checks

Run from `DRL-MIL/DRL-MIL` using the environment setup in the
[root README](../../README.md#start-from-a-fresh-clone).

## Splits without restricted data or model weights

```bash
python -m unittest discover -s tests -p test_splits.py -v
```

The four tests invoke the public command-line scripts in subprocesses:

| Check | Input and assertion |
| --- | --- |
| BRACS/CPTAC slide folds | Bundled ordered manifests; all 30 output CSVs equal the published five-fold files |
| CPTAC patient folds | Historical and strict official-ID modes; label/coverage checks, case isolation, and historical CSV equality |
| BRACS patient CLI | Generated Excel workbook with the documented sheet/columns and seven label classes; no patient overlap |
| Invalid inputs | Duplicate slide IDs and unknown patient IDs in strict mode are rejected |

The test fixture is generated in a temporary directory. It is not patient data
and is not used as evidence for the actual BRACS cohort's patient identity.
To regenerate the actual BRACS cohort, obtain the official workbook and run the
documented `--bracs-xlsx` command.

## Real encoder checks

After obtaining the weights described in
[feature_extraction/README.md](../feature_extraction/README.md):

```bash
python tests/smoke_features.py --resnet-checkpoint feature_extraction/Checkpoint/resnet50-19c8e357.pth --device cpu
python tests/smoke_features.py --gigapath-dir feature_extraction/gigapath --device cpu
```

Both commands create two RGB patches in a temporary `slide.with.dots` folder,
include a non-image file to check filtering, and invoke the actual extraction
scripts. No mocked encoder is used. ResNet checks all three stages, CPU tensor
serialization, legacy/tensor numerical agreement, filename ordering, and resume.
GigaPath checks that the first slide is processed and produces finite 1536-D
instance features. All temporary test files are removed after the check.

Use `--device cuda` to check a GPU and `--num-workers 2` to check multiprocessing.
The default worker count is zero for a portable first run. Only the specified
encoder is loaded if you supply just one checkpoint argument.

## Fresh-clone audit on 2026-11-08

The starting point was a new clone of published commit
`ec9d2cfa80af721f164e544334148332365e3975`. The documentation/input additions in
this revision were then exercised from that checkout, without importing the
author's separate preprocessing source directory.

- A new **Windows / Python 3.12.7 venv** was created without system site packages.
  PyTorch `2.6.0+cpu`, torchvision `0.21.0+cpu`, and the constrained dependencies
  were installed through pip. `pip check` reported no broken requirements.
- All four `test_splits.py` tests passed in this environment.
- The four split command lines in the root README were executed directly, using
  the two included manifests and the actual official BRACS workbook. All **60
  generated fold CSVs** matched the published files in `data/` when read as
  tables, including row order and labels. The BRACS input had 547 slides and 189
  patients; CPTAC-Lung had 2,136 slides and 441 case groups.
- The ResNet checkpoint was freshly downloaded from the documented official
  URL and passed its full SHA-256 check. CPU smoke tests passed all three
  feature dimensions, legacy/tensor equality, row order, and resume checks.
- The original project's full local Prov-GigaPath checkpoint/configuration was
  used by the fresh CPU environment. Its smoke test passed with finite
  `[2,1536]` CPU features, using the same CLI and preprocessing as documented.

The actual BRACS workbook, model checkpoints, test patch images, venv, and
generated feature bags are not committed. The gated GigaPath download was not
repeated in this audit; its local checkpoint was supplied as an external input.
Linux/macOS installations and full-dataset inference were not executed in this
audit. The earlier CUDA tests used Python 3.9.25, PyTorch 2.6.0+cu124, and
torchvision 0.21.0+cu124; encoder code is unchanged by this documentation audit.

## Scope

These checks cover preprocessing and output contracts. They do not run
whole-cohort feature extraction, WSI tiling, or full MIL training, and do not
reproduce the paper's classification metrics. Access to the BRACS workbook and
GigaPath model files is controlled by their providers; a successful local
smoke test cannot grant download permission to another user.
