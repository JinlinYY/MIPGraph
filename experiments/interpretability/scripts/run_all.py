

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="Viscosity")
    parser.add_argument("--n-edge-samples", type=int, default=150)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--correlation-split", default="test")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-edge", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    py = sys.executable

    if not args.skip_extract:
        run([py, str(here / "extract_latent.py")])

    run([py, str(here / "latent_correlation.py"), "--split", args.correlation_split])
    run([py, str(here / "visualize_umap.py")])

    if not args.skip_edge:
        run(
            [
                py,
                str(here / "edge_attribution.py"),
                "--n-samples",
                str(args.n_edge_samples),
                "--ig-steps",
                str(args.ig_steps),
                "--target",
                args.target,
            ]
        )

    run([py, str(here / "plot_main_figure.py"), "--target", args.target])


if __name__ == "__main__":
    main()
