"""Compose interpretability and feature-attribution figures into a 4x3 plate."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = REPO_ROOT / "LaTex-MIPGraph" / "Fig"
DEFAULT_INTERPRETABILITY = FIG_DIR / "interpretability_results.png"
DEFAULT_FEATURE = FIG_DIR / "feature_importance_heatmap.png"
DEFAULT_OUTPUT = FIG_DIR / "interpretability_feature_importance_4x3.png"


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = int(round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def compose(interp_path: Path, feature_path: Path, output_path: Path, height_scale: float) -> None:
    interp = Image.open(interp_path).convert("RGB")
    feature = Image.open(feature_path).convert("RGB")
    feature = resize_to_width(feature, interp.width)

    canvas = Image.new("RGB", (interp.width, interp.height + feature.height), "white")
    canvas.paste(interp, (0, 0))
    canvas.paste(feature, (0, interp.height))
    if height_scale <= 0:
        raise ValueError("--height-scale must be positive")
    if height_scale != 1.0:
        target_height = int(round(canvas.height * height_scale))
        canvas = canvas.resize((canvas.width, target_height), Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(600, 600))
    canvas.save(output_path.with_suffix(".pdf"), resolution=600.0)
    canvas.save(output_path.with_suffix(".tiff"), dpi=(600, 600))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose a 4x3 interpretability figure plate.")
    parser.add_argument("--interpretability", type=Path, default=DEFAULT_INTERPRETABILITY)
    parser.add_argument("--feature", type=Path, default=DEFAULT_FEATURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compose(args.interpretability, args.feature, args.output, args.height_scale)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
