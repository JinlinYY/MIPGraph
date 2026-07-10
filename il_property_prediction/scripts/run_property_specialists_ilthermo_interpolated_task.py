from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    output_root = project / "outputs" / "property_specialists_ilthermo_interpolated_seed42"
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        r"E:\anaconda\envs\ggnn39\python.exe",
        str(project / "scripts" / "run_property_specialists.py"),
        "--config",
        str(project / "configs" / "default.yaml"),
        "--checkpoint",
        str(
            project
            / "outputs"
            / "property_branch_sequence_decoupled_log_seed42"
            / "checkpoints"
            / "decoupled_log_branch_step06_ThermalConductivity_seed42"
            / "best_model.pt"
        ),
        "--clean-csv",
        str(project / "data" / "processed_ilthermo_interpolated" / "il_multiprop_clean.csv"),
        "--arrays-path",
        str(project / "data" / "processed_ilthermo_interpolated" / "il_multiprop_arrays.npz"),
        "--graph-cache",
        str(project / "data" / "processed_ilthermo_interpolated" / "graph_cache.pt"),
        "--split-path",
        str(project / "data" / "processed_ilthermo_interpolated" / "splits" / "il_level_seed42.json"),
        "--properties",
        "Density,ElectricalConductivity,HeatCapacity,SurfaceTension,ThermalConductivity,Viscosity",
        "--seeds",
        "42",
        "--split-seed",
        "42",
        "--output-root",
        str(output_root),
        "--epochs",
        "80",
        "--lr",
        "0.0001",
        "--patience",
        "5",
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
        "2",
        "--skip-existing",
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with (output_root / "run3_stdout.log").open("w", encoding="utf-8") as stdout, (
        output_root / "run3_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            command,
            cwd=project,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
        stdout.write(f"\nrunner_exit_code={result.returncode}\n")
        stdout.flush()


if __name__ == "__main__":
    main()
