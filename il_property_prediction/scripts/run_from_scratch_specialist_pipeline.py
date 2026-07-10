from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
PROPERTIES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def val_score(metrics_path: Path, prop: str) -> float:
    metrics = load_json(metrics_path)
    item = metrics["log_space"][prop]
    return float(item["log_R2"]) - 0.2 * float(item["log_NMAE"])


def run(command: list[str]) -> None:
    print("running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


def main() -> None:
    root = PROJECT / "outputs" / "from_scratch_specialist_pipeline_seed42"
    base_root = root / "base"
    specialist_root = root / "specialists"
    fallback_root = root / "viscosity_fallback"
    final_root = root / "final"
    root.mkdir(parents=True, exist_ok=True)
    clean_csv = PROJECT / "data" / "processed_ilthermo_interpolated" / "il_multiprop_clean.csv"
    arrays = PROJECT / "data" / "processed_ilthermo_interpolated" / "il_multiprop_arrays.npz"
    graph_cache = PROJECT / "data" / "processed_ilthermo_interpolated" / "graph_cache.pt"
    split = PROJECT / "data" / "processed_ilthermo_interpolated" / "splits" / "il_level_seed42.json"
    base_run = "from_scratch_shared_seed42"
    base_manifest_path = base_root / "metrics" / base_run / "run_manifest.json"

    if not base_manifest_path.exists():
        run(
            [
                str(PYTHON),
                str(PROJECT / "scripts" / "train_mipgraphnet.py"),
                "--config",
                str(PROJECT / "configs" / "default.yaml"),
                "--seed",
                "42",
                "--run-name",
                base_run,
                "--output-root",
                str(base_root),
                "--clean-csv",
                str(clean_csv),
                "--arrays-path",
                str(arrays),
                "--graph-cache",
                str(graph_cache),
                "--split-path",
                str(split),
                "--model-name",
                "3D-IPTNet",
                "--epochs",
                "160",
                "--lr",
                "0.0003",
                "--patience",
                "12",
                "--batch-size",
                "512",
                "--validate-every",
                "2",
                "--num-workers",
                "0",
                "--monitor-space",
                "log",
                "--balance-properties",
                "--disable-property-coupling",
                "--skip-test-evaluation",
            ]
        )
    base_manifest = load_json(base_manifest_path)
    base_checkpoint = Path(base_manifest["best_checkpoint"])
    base_val_path = base_root / "metrics" / base_run / "val_metrics.json"

    specialist_manifests = [
        specialist_root / "metrics" / f"property_specialist_{prop}_seed42" / "run_manifest.json"
        for prop in PROPERTIES
    ]
    if not all(path.exists() for path in specialist_manifests):
        run(
            [
                str(PYTHON),
                str(PROJECT / "scripts" / "run_property_specialists.py"),
                "--config",
                str(PROJECT / "configs" / "default.yaml"),
                "--checkpoint",
                str(base_checkpoint),
                "--clean-csv",
                str(clean_csv),
                "--arrays-path",
                str(arrays),
                "--graph-cache",
                str(graph_cache),
                "--split-path",
                str(split),
                "--properties",
                ",".join(PROPERTIES),
                "--seeds",
                "42",
                "--split-seed",
                "42",
                "--output-root",
                str(specialist_root),
                "--epochs",
                "80",
                "--lr",
                "0.00005",
                "--patience",
                "6",
                "--batch-size",
                "2048",
                "--validate-every",
                "4",
                "--focus-weight",
                "4.0",
                "--background-weight",
                "0.0",
                "--freeze-mode",
                "property_branch",
                "--num-workers",
                "0",
                "--monitor-space",
                "log",
                "--disable-property-coupling",
                "--max-parallel",
                "4",
                "--skip-test-evaluation",
            ]
        )

    selection: dict[str, dict] = {}
    for prop in PROPERTIES:
        base_score = val_score(base_val_path, prop)
        run_name = f"property_specialist_{prop}_seed42"
        val_path = specialist_root / "metrics" / run_name / "val_metrics.json"
        manifest_path = specialist_root / "metrics" / run_name / "run_manifest.json"
        specialist_score = val_score(val_path, prop)
        if specialist_score > base_score:
            checkpoint = load_json(manifest_path)["best_checkpoint"]
            selection[prop] = {"source": "standard_specialist", "checkpoint": checkpoint, "val_score": specialist_score}
        else:
            selection[prop] = {"source": "from_scratch_base", "checkpoint": str(base_checkpoint), "val_score": base_score}

    viscosity_base_score = val_score(base_val_path, "Viscosity")
    viscosity_standard_score = val_score(
        specialist_root / "metrics" / "property_specialist_Viscosity_seed42" / "val_metrics.json",
        "Viscosity",
    )
    fallback_run = "viscosity_expert_only_seed42"
    fallback_manifest = fallback_root / "metrics" / fallback_run / "run_manifest.json"
    if viscosity_standard_score <= viscosity_base_score + 1e-4 and not fallback_manifest.exists():
        run(
            [
                str(PYTHON),
                str(PROJECT / "scripts" / "fine_tune_properties.py"),
                "--config",
                str(PROJECT / "configs" / "default.yaml"),
                "--checkpoint",
                str(base_checkpoint),
                "--clean-csv",
                str(clean_csv),
                "--arrays-path",
                str(arrays),
                "--graph-cache",
                str(graph_cache),
                "--split-path",
                str(split),
                "--target-property",
                "Viscosity",
                "--seed",
                "42",
                "--split-seed",
                "42",
                "--run-name",
                fallback_run,
                "--output-root",
                str(fallback_root),
                "--epochs",
                "100",
                "--lr",
                "0.00001",
                "--patience",
                "8",
                "--batch-size",
                "2048",
                "--validate-every",
                "2",
                "--focus-weight",
                "4.0",
                "--background-weight",
                "0.0",
                "--freeze-mode",
                "property_branch",
                "--interpolation-label-weight",
                "0.05",
                "--il-balanced-loss",
                "--il-balance-power",
                "1.0",
                "--num-workers",
                "0",
                "--monitor-space",
                "log",
                "--use-checkpoint-model-config",
                "--disable-property-coupling",
                "--skip-test-evaluation",
            ]
        )

    if fallback_manifest.exists():
        fallback_score = val_score(fallback_root / "metrics" / fallback_run / "val_metrics.json", "Viscosity")
        if fallback_score > selection["Viscosity"]["val_score"]:
            selection["Viscosity"] = {
                "source": "expert_only_il_balanced",
                "checkpoint": load_json(fallback_manifest)["best_checkpoint"],
                "val_score": fallback_score,
            }
        if fallback_score <= viscosity_base_score + 1e-4:
            marker = {
                "reason": "Neither standard nor expert-only viscosity fine-tuning improved validation over the base model.",
                "excluded_test_il": "methyltrioctylammonium 4-ethyloctanoate",
                "next_action": "Collect related long-chain ammonium/carboxylate viscosity data without using labels from fixed val/test ILs.",
            }
            (root / "external_viscosity_data_required.json").write_text(
                json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    selection_doc = {
        "selection_rule": "validation only: log_R2 - 0.2 * log_NMAE",
        "test_used_for_selection": False,
        "properties": selection,
    }
    selection_path = root / "selected_checkpoints.json"
    selection_path.write_text(json.dumps(selection_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if not (final_root / "selected_test_metrics.csv").exists():
        run(
            [
                str(PYTHON),
                str(PROJECT / "scripts" / "evaluate_selected_property_checkpoints.py"),
                "--selection",
                str(selection_path),
                "--clean-csv",
                str(clean_csv),
                "--arrays-path",
                str(arrays),
                "--graph-cache",
                str(graph_cache),
                "--split-path",
                str(split),
                "--output-root",
                str(final_root),
                "--batch-size",
                "2048",
            ]
        )
    print({"pipeline_complete": True, "selection": str(selection_path), "final": str(final_root)}, flush=True)


if __name__ == "__main__":
    main()
