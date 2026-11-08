# Instance feature extraction

This folder packages the two extraction pipelines used in the local project:
`save_represention.py` (ResNet) and `save_represention_Prov-GigaPath.py`
(Prov-GigaPath **tile encoder**). Both take pre-extracted image patches and
produce one feature bag per slide. WSI tiling, tissue masking, and patch
selection are upstream steps and are not performed here. The GigaPath slide
encoder is not used by these scripts.

## Installation and input layout

Follow the [fresh-clone environment setup](../../README.md#start-from-a-fresh-clone)
first. From `DRL-MIL/CRL-MIL`, the tested CPU baseline is:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-preprocessing.txt -r feature_extraction/requirements.txt
python -m pip check
```

Use the `cu124` wheel index instead of `cpu` for a compatible NVIDIA system.
Keep the selected Python environment active (or use its Python executable path).
The quick start uses `--bs 1 --num_workers 0 --device cpu`; these settings also
avoid multiprocessing setup issues on the first Windows run. CPU GigaPath is
slow and needs substantial host memory for the roughly 4.5 GB checkpoint and
model; a machine with at least 16 GB system RAM and enough free memory is a
practical starting point. For throughput, switch to `--device cuda` and increase
batch size/workers gradually according to available GPU and host memory.

Create the following patch layout before running the real data commands:

```text
patches/
  slide_001/
    patch_001.png
    patch_002.png
  slide_002/
    patch_001.jpg
```

Each immediate subdirectory is one slide. Its name must match the split CSV
`path` value; output names are `<slide_id>.pt`. Supported extensions are PNG,
JPG/JPEG, TIFF/TIF, and BMP. Images are read as RGB. Patch filenames are sorted
lexicographically, and `<slide_id>.patches.json` records the row-to-patch mapping.
The original filesystem enumeration order was not guaranteed; use the sidecar
when aligning features to coordinates or visualizations.

## ResNet

The original backbone is preserved in `resnet_backbone.py`: pooled layer 3
features have dimension 1024; pooled layer 4 features have dimension 2048;
their concatenation has dimension 3072. The default remains `stage3`.
All patches are resized directly to 224 x 224, then normalized with the original
defaults `mean=[0.7476,0.5583,0.6705]`, `std=[0.1884,0.2474,0.1961]`.
These are the source script's CPTAC statistics, not universal defaults for all
cohorts; `--mean` and `--std` accept three floats each.

First download the exact official ImageNet checkpoint (no login required).
The command checks its full SHA-256:

```bash
python -c "from pathlib import Path; from torch.hub import download_url_to_file; Path('feature_extraction/Checkpoint').mkdir(parents=True, exist_ok=True); download_url_to_file('https://download.pytorch.org/models/resnet50-19c8e357.pth', 'feature_extraction/Checkpoint/resnet50-19c8e357.pth', hash_prefix='19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097')"
```

Then extract features:

```bash
# Current CRL-MIL BagDataset_GPU.py expects a plain [N,D] tensor.
python feature_extraction/save_represention.py --data_dir /path/to/patches --repre_dir outputs/features_resnet --checkpoint feature_extraction/Checkpoint/resnet50-19c8e357.pth --arch resnet50 --stage stage4 --output-format tensor --bs 1 --num_workers 0 --device cpu

# Historical serialization (default): {"l3": tensor}, {"l4": tensor}, or {"l34": tensor}.
python feature_extraction/save_represention.py --data_dir /path/to/patches --repre_dir outputs/features_legacy --checkpoint feature_extraction/Checkpoint/resnet50-19c8e357.pth --stage stage3 --output-format legacy --bs 1 --num_workers 0 --device cpu
```

Outputs go to `<repre_dir>/<stage>/<slide_id>.pt`. Choose `--output-format tensor`
for the repository's current training loader, and set the MIL input dimension
to 1024/2048/3072 to match the selected stage. `--ckt` is retained as an
alternative to `--checkpoint` only when the directory contains exactly one
`.pth` filename matching `--arch`. ResNet-101/152 architectures are also retained;
use compatible state-dictionary checkpoints.

### ResNet weights and reference

The local ResNet-50 checkpoint was verified as the historical official
ImageNet-pretrained file:

- [Download resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth)
- [Official torchvision checkpoint definition](https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py)
- SHA-256: `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097`
- He, K., Zhang, X., Ren, S., and Sun, J. **Deep Residual Learning for Image
  Recognition.** CVPR, pp. 770-778, 2016.
  [Paper](https://arxiv.org/abs/1512.03385), [DOI](https://doi.org/10.1109/CVPR.2016.90).

Use this exact checkpoint to reproduce this extractor. Newer torchvision
`DEFAULT` weights need not match it. PyTorch 2.6's weights-only loader cannot
read this old tar format; the script verifies the exact SHA-256 before allowing
the legacy load. Other checkpoints use weights-only loading and all backbone
parameters must match (only unused classifier parameters are ignored).
The training configuration's historical `BT`/`ssl_BT` names do not establish a
Barlow Twins checkpoint provenance: the supplied ResNet checkpoint is ImageNet
pretrained. A different encoder requires its actual checkpoint and citation.

## Prov-GigaPath tile encoder

Obtain **`config.json` and `pytorch_model.bin`** from the original
[prov-gigapath/prov-gigapath model repository](https://huggingface.co/prov-gigapath/prov-gigapath).
Complete the model provider's access process and follow its applicable terms.
The checkpoint is about 4.54 GB (4.23 GiB) and is intentionally not committed.
The two files must come from the same checkpoint version. No `slide_encoder.pth`
or GigaPath-Flash checkpoint is required.

1. Sign in to the linked model page and complete its access requirements.
2. Authenticate locally with a token that can read that model (do not put it in
   source code). The login command below prompts interactively; skip it if
   already authenticated.
3. Download the two files, then run extraction:

```bash
python -c "from huggingface_hub import login; login()"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='prov-gigapath/prov-gigapath', allow_patterns=['config.json','pytorch_model.bin'], local_dir='feature_extraction/gigapath')"

python feature_extraction/save_represention_Prov-GigaPath.py --data_dir /path/to/patches --repre_dir outputs/features_gigapath --checkpoint-dir feature_extraction/gigapath --bs 1 --num_workers 0 --device cpu
```

The extraction command loads local files and makes no model-download request.
The official configuration specifies a 1536-dimensional tile encoder.
Preprocessing matches the original script and official tile example: bicubic
resize of the shorter side to 256, center crop to 224, and ImageNet normalization
`mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`.
Output is a plain CPU tensor of shape `[number_of_patches,1536]` at
`<repre_dir>/<slide_id>.pt`; set the MIL input dimension to 1536.

Reference: Xu, H., Usuyama, N., Bagga, J., et al. **A whole-slide foundation model
for digital pathology from real-world data.** Nature 630, 181-188 (2024).
[Paper and DOI](https://doi.org/10.1038/s41586-024-07441-w),
[official implementation](https://github.com/prov-gigapath/prov-gigapath),
[model and weights](https://huggingface.co/prov-gigapath/prov-gigapath).
Machine-readable references are provided in [references.bib](references.bib).

## Verify before extracting a cohort

These commands generate temporary RGB patches and call the actual extraction
scripts. They require the downloaded weights but no WSI data:

```bash
python tests/smoke_features.py --resnet-checkpoint feature_extraction/Checkpoint/resnet50-19c8e357.pth --device cpu
python tests/smoke_features.py --gigapath-dir feature_extraction/gigapath --device cpu
```

Success prints `PASS ResNet` / `PASS Prov-GigaPath`. ResNet's three output
dimensions and both serialization formats are checked, along with patch row
order and skipping completed slides. GigaPath must process the first slide and
produce finite `[2,1536]` features. To test worker spawning on your platform,
add `--num-workers 2`; to test your GPU, use `--device cuda`.

For your real data, inspect a bag with:

```bash
python -c "import torch; x=torch.load('outputs/features_resnet/stage4/slide_001.pt', map_location='cpu', weights_only=True); print(x.shape)"
```

Replace `slide_001` with an actual slide ID. Tensor-mode stage 4 should print
`[N,2048]`; GigaPath should print `[N,1536]`. `N` is the number of supported
image files directly inside that slide folder. Empty folders are skipped;
patches nested another level down are not read. A 401/403 during GigaPath
download usually means the model access or local login step is incomplete.

## Release behavior and reproducibility

- Original encoder architecture, image transforms, feature dimensions, and
  default output layouts are preserved. ResNet's tensor output is an explicit
  compatibility option for the current MIL loader.
- Absolute machine-specific paths, embedded credentials, and the fixed
  GigaPath `[152:]` resume offset were removed. Every slide folder is considered.
- Existing `.pt` files are skipped. Use a separate empty output directory when
  changing the encoder, stage, transforms, or serialization format.
- Features are accumulated on CPU after each batch and saved on CPU. Use a
  smaller `--bs` if GPU memory is limited; `--device cpu` is also supported.
- No checkpoint, patch image, extracted feature, private metadata, or access
  token is included. No whole-cohort feature extraction is run by installation.

Validation used Python 3.9.25, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124,
timm 1.0.30, and Pillow 11.3.0. ResNet outputs were compared exactly against the
original backbone with the same checkpoint. Synthetic RGB patch smoke tests
cover stage dimensions, legacy/tensor consistency, CPU output, sorted row
manifests, and skipping completed slides. The actual local Prov-GigaPath
checkpoint was also run on two synthetic RGB patches, producing finite
`[2,1536]` CPU features from the first slide folder. These checks verify the release;
they do not reproduce the paper's full training experiments.
Both real encoders were subsequently tested in a fresh Python 3.12.7 CPU venv;
see the [fresh-clone audit](../tests/README.md#fresh-clone-audit-on-2026-11-08).
