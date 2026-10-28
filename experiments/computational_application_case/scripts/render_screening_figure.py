"""Re-export the screening figure from released CSVs without model inference.

Run from the project root:
    python experiments/computational_application_case/scripts/render_screening_figure.py
"""

from __future__ import annotations

import shutil
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
from experiments.manuscript_figure_source_data import rebuild_manifest  # noqa: E402
from experiments.result_analysis.scripts import package_manuscript_results as package  # noqa: E402


FIGURE_ID = "auditable_virtual_screening_validation"


def main() -> None:
    source_dir = builder.APPLICATION_SOURCE_DIR
    builder.DATA_DIR = source_dir
    builder.FIGURE_DIR = chapter.CHAPTER_FIGURE_DIR
    builder.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(source_dir / name, low_memory=False)

    final = read("final_prioritized_candidates.csv")
    protocol = read("cross_protocol_decision_matrix.csv")
    funnel = read("panel_a_candidate_space_funnel.csv").set_index("stage")
    paths = builder.make_figure5(
        {"new_valid_unseen_pool": int(funnel.loc["InChI_identity_new_pool", "count"])},
        read("reference_bootstrap_candidate_selection.csv"),
        final,
        read("final_candidate_constraint_margins.csv"),
        protocol,
        read("downstream_qualification_priorities.csv"),
        output_stem=FIGURE_ID,
    )
    publish_figure(
        paths,
        FIGURE_ID,
        "figure5",
        source_names=("panel_f_qualification_display.csv",),
        provenance="Re-exported from audited CSVs; panel f display corrected; no inference",
    )
    print("Panel f: UPR-0581 / UPR-0024 / UPR-0343 / UPR-0558")


def publish_figure(
    paths: list[Path],
    figure_id: str,
    paper_prefix: str,
    *,
    source_names: tuple[str, ...] = (),
    provenance: str,
) -> None:
    """Publish one re-export and refresh only its figure inventory records."""

    result_dir = package.MODULE_ROOT / "figures" / figure_id
    for path in paths:
        shutil.copy2(path, result_dir / path.name)
        shutil.copy2(path, builder.PAPER_FIG_DIR / f"{paper_prefix}_{path.name}")
    package.write_tiff(
        result_dir / f"{figure_id}.png", result_dir / f"{figure_id}.tiff"
    )

    # Only rows belonging to the changed figure are refreshed.
    figure_manifest = pd.read_csv(package.MODULE_ROOT / "manifest.csv", keep_default_na=False)
    for index, row in figure_manifest.iterrows():
        if row["figure_id"] != figure_id:
            continue
        path = package.MODULE_ROOT / row["relative_path"]
        figure_manifest.loc[index, "bytes"] = path.stat().st_size
        figure_manifest.loc[index, "sha256"] = package.sha256(path)
        origin_suffix = ".png" if path.suffix == ".tiff" else path.suffix
        figure_manifest.loc[index, "origin"] = (
            chapter.CHAPTER_FIGURE_DIR / f"{figure_id}{origin_suffix}"
        ).relative_to(PROJECT_ROOT).as_posix()
        figure_manifest.loc[index, "provenance"] = (
            "600-dpi LZW conversion from re-exported manuscript PNG"
            if path.suffix == ".tiff"
            else provenance
        )
        if path.suffix in {".png", ".tiff"}:
            with package.Image.open(path) as image:
                figure_manifest.loc[index, ["width_px", "height_px"]] = image.size
    figure_manifest.to_csv(package.MODULE_ROOT / "manifest.csv", index=False, encoding="utf-8-sig")

    # Keep the chapter release inventory and optional display CSVs in sync.
    chapter_manifest = pd.read_csv(chapter.CHAPTER_RESULTS_DIR / "manifest.csv", keep_default_na=False)
    changed_names = {path.name for path in paths} | set(source_names)
    records = chapter_manifest[~chapter_manifest["file"].isin(changed_names)].to_dict("records")
    for path in paths:
        records.append({
            "bundle": "application_chapter_results",
            "file": path.name,
            "destination_relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
            "source_relative_path": f"LaTex-MIPGraph/Fig/{paper_prefix}_{path.name}",
            "bytes": path.stat().st_size,
            "rows": "",
            "columns": "",
            "sha256": package.sha256(path),
        })
    for name in source_names:
        records.append(chapter.copy_record(
            builder.APPLICATION_SOURCE_DIR / name,
            chapter.CHAPTER_CSV_DIR / name,
            "application_chapter_results",
        ))
    chapter.write_manifest(chapter.CHAPTER_RESULTS_DIR / "manifest.csv", records)
    rebuild_manifest.main()
    package.package_source_data(package.MODULE_ROOT)
    chapter.write_code_inventory()
    print(f"Updated {result_dir}")


if __name__ == "__main__":
    main()
