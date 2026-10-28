"""Redraw the released two-panel derived-metric figure with four highlighted ILs."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.computational_application_case.scripts import (
    build_refactored_application_case as builder,
    package_chapter_artifacts as chapter,
)
from experiments.computational_application_case.scripts.render_screening_figure import publish_figure


def main():
    builder.DATA_DIR = builder.APPLICATION_SOURCE_DIR
    builder.FIGURE_DIR = chapter.CHAPTER_FIGURE_DIR
    builder.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = builder.make_derived_metrics_figure()
    publish_figure(paths, "figureS_application_derived_metrics", "figureS",
                   provenance="Re-exported released metrics; added UPR-0581 in purple; no inference or metric recomputation")
    print("Highlights: UPR-0343, UPR-0024, UPR-0558, UPR-0581")


if __name__ == "__main__":
    main()
