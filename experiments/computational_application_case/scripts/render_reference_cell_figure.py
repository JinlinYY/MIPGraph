"""Re-export the post hoc reference-cell figure using published metrics only.

The original three highlighted candidates retain their colours. UPR-0581 is
added in purple; no inference, constraint selection, or metric recomputation
is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.computational_application_case.scripts import (  # noqa: E402
    build_refactored_application_case as builder,
    package_chapter_artifacts as chapter,
)
from experiments.computational_application_case.scripts.render_screening_figure import (  # noqa: E402
    publish_figure,
)


def main() -> None:
    builder.DATA_DIR = builder.APPLICATION_SOURCE_DIR
    builder.FIGURE_DIR = chapter.CHAPTER_FIGURE_DIR
    final = pd.read_csv(builder.DATA_DIR / "final_prioritized_candidates.csv", low_memory=False)
    display = pd.read_csv(builder.DATA_DIR / "panel_f_qualification_display.csv")
    paths = builder.make_figure6(
        final,
        display,
        output_stem="reference_cell_scenario_audited",
        write_source_data=False,
    )
    publish_figure(
        paths,
        "reference_cell_scenario_audited",
        "figure6",
        provenance="Post hoc mapping from released metrics; four highlighted candidates; no inference",
    )
    print("Highlights: UPR-0343 (blue), UPR-0024 (orange), UPR-0558 (green), UPR-0581 (purple)")


if __name__ == "__main__":
    main()
