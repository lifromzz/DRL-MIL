"""Run the real feature-extraction CLIs on synthetic patches with local weights."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resnet-checkpoint", type=Path)
    parser.add_argument("--gigapath-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if not (args.resnet_checkpoint or args.gigapath_dir):
        parser.error("provide --resnet-checkpoint and/or --gigapath-dir")
    # Normalize before the child process changes its working directory.
    if args.resnet_checkpoint:
        args.resnet_checkpoint = args.resnet_checkpoint.resolve()
    if args.gigapath_dir:
        args.gigapath_dir = args.gigapath_dir.resolve()

    with tempfile.TemporaryDirectory(prefix="crl_features_") as directory:
        tmp = Path(directory)
        slide = tmp / "patches/slide.with.dots"
        slide.mkdir(parents=True)
        Image.new("RGB", (256, 256), (100, 150, 200)).save(slide / "b.png")
        Image.new("RGB", (256, 256), (50, 60, 70)).save(slide / "a.png")
        (slide / "ignore.txt").write_text("not an image", encoding="utf-8")
        common = ["--data_dir", str(slide.parent), "--bs", "1", "--num_workers",
                  str(args.num_workers), "--device", args.device]

        def run(script, *options):
            subprocess.run([sys.executable, str(ROOT / "feature_extraction" / script),
                            *common, *map(str, options)], cwd=ROOT, check=True)

        def check(path, dimension, key=None):
            features = torch.load(path, weights_only=True)
            if key:
                features = features[key]
            if features.device.type != "cpu" or features.shape != (2, dimension) or not torch.isfinite(features).all():
                raise AssertionError(f"Invalid feature bag at {path}")
            names = json.loads(path.with_suffix(".patches.json").read_text(encoding="utf-8"))
            if names != ["a.png", "b.png"]:
                raise AssertionError("Feature rows do not follow the recorded patch order")
            return features

        if args.resnet_checkpoint:
            script = "save_represention.py"
            base_options = ["--checkpoint", args.resnet_checkpoint, "--repre_dir", tmp / "resnet"]
            for stage, dimension in [("stage3", 1024), ("stage4", 2048), ("stage34", 3072)]:
                options = [*base_options, "--stage", stage, "--output-format", "tensor"]
                run(script, *options)
                path = tmp / "resnet" / stage / "slide.with.dots.pt"
                check(path, dimension)
                if stage == "stage3":
                    stamp = path.stat().st_mtime_ns
                    run(script, *options)
                    if path.stat().st_mtime_ns != stamp:
                        raise AssertionError("Resume overwrote a completed slide")
            run(script, "--checkpoint", args.resnet_checkpoint, "--repre_dir", tmp / "legacy")
            original_format = check(tmp / "legacy/stage3/slide.with.dots.pt", 1024, key="l3")
            tensor_format = check(tmp / "resnet/stage3/slide.with.dots.pt", 1024)
            torch.testing.assert_close(original_format, tensor_format, rtol=0, atol=0)
            print("PASS ResNet: all stages, tensor/legacy agreement, row order, and resume.", flush=True)
        if args.gigapath_dir:
            run("save_represention_Prov-GigaPath.py", "--checkpoint-dir", args.gigapath_dir,
                "--repre_dir", tmp / "gigapath")
            check(tmp / "gigapath/slide.with.dots.pt", 1536)
            print("PASS Prov-GigaPath: first slide processed; finite [2,1536] features.", flush=True)


if __name__ == "__main__":
    main()
