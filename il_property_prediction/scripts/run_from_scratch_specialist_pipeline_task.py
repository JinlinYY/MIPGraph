from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    root = project / "outputs" / "from_scratch_specialist_pipeline_seed42"
    root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    conda_prefix = Path(r"E:\anaconda\envs\ggnn39")
    conda_paths = [
        conda_prefix,
        conda_prefix / "Library" / "mingw-w64" / "bin",
        conda_prefix / "Library" / "usr" / "bin",
        conda_prefix / "Library" / "bin",
        conda_prefix / "Scripts",
        conda_prefix / "bin",
    ]
    inherited_path = next((value for key, value in environment.items() if key.lower() == "path"), "")
    for key in [key for key in environment if key.lower() == "path"]:
        environment.pop(key)
    environment["Path"] = os.pathsep.join([str(path) for path in conda_paths] + [inherited_path])
    environment["CONDA_PREFIX"] = str(conda_prefix)
    environment["CONDA_DLL_SEARCH_MODIFICATION_ENABLE"] = "1"
    environment.pop("PYTHONHOME", None)
    command = [
        r"E:\anaconda\envs\ggnn39\python.exe",
        str(project / "scripts" / "run_from_scratch_specialist_pipeline.py"),
    ]
    with (root / "pipeline_stdout.log").open("w", encoding="utf-8") as stdout, (
        root / "pipeline_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        preflight = subprocess.run(
            [str(conda_prefix / "python.exe"), "-c", "import ctypes, numpy, torch; print('preflight_ok')"],
            cwd=project,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
        if preflight.returncode != 0:
            stdout.write(f"preflight_exit_code={preflight.returncode}\n")
            stdout.flush()
            return
        result = subprocess.run(
            command,
            cwd=project,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=False,
        )
        stdout.write(f"\npipeline_exit_code={result.returncode}\n")
        stdout.flush()


if __name__ == "__main__":
    main()
