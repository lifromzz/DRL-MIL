"""Shared patch loading and per-slide feature serialization."""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class PatchDataset(Dataset):
    def __init__(self, slide_dir, transform):
        self.paths = sorted(
            path for path in Path(slide_dir).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def add_common_arguments(parser, batch_size):
    parser.add_argument("--data_dir", required=True, help="Root containing one patch folder per slide")
    parser.add_argument("--repre_dir", required=True, help="Output feature directory")
    parser.add_argument("--bs", type=int, default=batch_size)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def extract_slides(encoder, transform, args, output_dir, select_features, dict_key=None):
    if args.bs < 1 or args.num_workers < 0:
        raise ValueError("bs must be positive and num_workers must be nonnegative")
    slides = sorted(path for path in Path(args.data_dir).iterdir() if path.is_dir())
    if not slides:
        raise ValueError("No slide subdirectories found in data_dir")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    encoder.to(device).eval()
    with torch.inference_mode():
        for slide in tqdm(slides, desc="Slides"):
            destination = output_dir / (slide.name + ".pt")
            if destination.exists():
                continue
            dataset = PatchDataset(slide, transform)
            if not len(dataset):
                tqdm.write(f"Skipping empty slide folder: {slide.name}")
                continue
            loader = DataLoader(dataset, batch_size=args.bs, shuffle=False,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda")
            batches = []
            for images in loader:
                features = select_features(encoder(images.to(device, non_blocking=True)))
                if features.ndim != 2 or features.shape[0] != images.shape[0]:
                    raise RuntimeError("Encoder must return one feature vector per patch")
                batches.append(features.cpu())
            features = torch.cat(batches, dim=0)
            payload = {dict_key: features} if dict_key else features
            # The sidecar records row order without changing the original .pt format.
            manifest = output_dir / (slide.name + ".patches.json")
            manifest.write_text(json.dumps([p.name for p in dataset.paths], indent=2), encoding="utf-8")
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(destination)


def identity(features):
    return features
