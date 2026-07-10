from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    output_root = project / "outputs" / "groupkfold_ilthermo_interpolated_seed42"
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        r"E:\anaconda\envs\ggnn39\python.exe",
        str(project / "scripts" / "run_groupkfold_cv.py"),
        "--config",
        str(project / "configs" / "default.yaml"),
        "--base-split",
        str(project / "data" / "processed_ilthermo_interpolated" / "splits" / "il_level_seed42.json"),
        "--clean-csv",
        str(project / "data" / "processed_ilthermo_interpolated" / "il_multiprop_clean.csv"),
        "--arrays-path",
        str(project / "data" / "processed_ilthermo_interpolated" / "il_multiprop_arrays.npz"),
        "--graph-cache",
        str(project / "data" / "processed_ilthermo_interpolated" / "graph_cache.pt"),
        "--output-root",
        str(output_root),
        "--pool",
        "train",
        "--folds",
        "5",
        "--seed",
        "42",
        "--epochs",
        "80",
        "--patience",
        "20",
        "--batch-size",
        "512",
        "--validate-every",
        "4",
        "--num-workers",
        "0",
        "--disable-property-coupling",
        "--skip-existing",
    ]
    creation_flags = (
        subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.CREATE_NO_WINDOW
    )
    with (output_root / "run_stdout.log").open("w", encoding="utf-8") as stdout, (
        output_root / "run_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=project,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
            close_fds=True,
        )
    print(process.pid)


if __name__ == "__main__":
    if sys.platform != "win32":
        raise RuntimeError("This launcher uses Windows detached-process flags")
    main()
