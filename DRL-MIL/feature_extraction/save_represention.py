"""Extract ResNet instance features using the original backbone and preprocessing."""

import argparse
import hashlib
from pathlib import Path

import torch
from torchvision import transforms

from common import add_common_arguments, extract_slides
from resnet_backbone import Bottleneck_Baseline, ResNet_Baseline

LAYERS = {"resnet50": [3, 4, 6, 3], "resnet101": [3, 4, 23, 3], "resnet152": [3, 8, 36, 3]}
STAGE_KEYS = {"stage3": "l3", "stage4": "l4", "stage34": "l34"}
OFFICIAL_LEGACY_RESNET50_SHA256 = "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097"


def read_checkpoint(checkpoint):
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    except RuntimeError as error:
        if "legacy .tar format" not in str(error):
            raise
        # PyTorch 2.6 cannot read legacy tar checkpoints in weights-only mode.
        # Only allow the exact official ResNet-50 artifact used by this project.
        digest = hashlib.sha256()
        with open(checkpoint, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != OFFICIAL_LEGACY_RESNET50_SHA256:
            raise ValueError("Unrecognized legacy checkpoint. Use a modern state_dict checkpoint.") from error
        return torch.load(checkpoint, map_location="cpu", weights_only=False)


def load_encoder(arch, checkpoint):
    model = ResNet_Baseline(Bottleneck_Baseline, LAYERS[arch])
    state_dict = read_checkpoint(checkpoint)
    # Official ImageNet checkpoints include a classifier that this encoder omits.
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = set(incompatible.unexpected_keys) - {"fc.weight", "fc.bias"}
    if incompatible.missing_keys or unexpected:
        raise ValueError(f"Checkpoint does not match {arch}: missing={incompatible.missing_keys}, "
                         f"unexpected={sorted(unexpected)}")
    return model


def get_transform(mean, std):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, batch_size=128)
    parser.add_argument("--arch", choices=list(LAYERS), default="resnet50")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path, help="Exact checkpoint file")
    checkpoint_group.add_argument("--ckt", type=Path, help="Legacy checkpoint directory; must contain exactly one matching .pth")
    parser.add_argument("--stage", choices=list(STAGE_KEYS), default="stage3")
    parser.add_argument("--mean", nargs=3, type=float, default=[0.7476, 0.5583, 0.6705])
    parser.add_argument("--std", nargs=3, type=float, default=[0.1884, 0.2474, 0.1961])
    parser.add_argument("--output-format", choices=["legacy", "tensor"], default="legacy",
                        help="legacy: original dictionary; tensor: current DRL-MIL loader format")
    args = parser.parse_args()
    if args.ckt:
        candidates = sorted(args.ckt.glob(f"*{args.arch}*.pth"))
        if len(candidates) != 1:
            parser.error("--ckt must contain exactly one matching checkpoint; use --checkpoint otherwise")
        args.checkpoint = candidates[0]
    model = load_encoder(args.arch, args.checkpoint)

    def select_features(outputs):
        l3, l4 = outputs
        if args.stage == "stage3":
            return l3
        if args.stage == "stage4":
            return l4
        return torch.cat([l3, l4], dim=1)

    extract_slides(model, get_transform(args.mean, args.std), args,
                   Path(args.repre_dir) / args.stage, select_features,
                   dict_key=STAGE_KEYS[args.stage] if args.output_format == "legacy" else None)


if __name__ == "__main__":
    main()
