"""Extract 1536-dimensional instance features with the Prov-GigaPath tile encoder."""

import argparse
import json
from pathlib import Path

import timm
from torchvision import transforms

from common import add_common_arguments, extract_slides, identity


def load_encoder(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    return timm.create_model(
        model_name=config["architecture"],
        pretrained=False,
        checkpoint_path=str(checkpoint_dir / "pytorch_model.bin"),
        **config["model_args"],
    )


def get_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, batch_size=64)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Directory containing official config.json and pytorch_model.bin")
    args = parser.parse_args()
    model = load_encoder(args.checkpoint_dir)
    extract_slides(model, get_transform(), args, args.repre_dir, identity)


if __name__ == "__main__":
    main()
