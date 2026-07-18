"""Portable path resolution with strict project and output boundaries."""

from __future__ import annotations

from pathlib import Path


PROJECT_MARKERS = ("README.md", "il_property_prediction", "data", "experiments")


def locate_project_root(start: str | Path | None = None) -> Path:
    """Find the MIPGraph root by walking upward from a caller-provided path."""

    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.extend([Path.cwd().resolve(), Path(__file__).resolve()])
    checked: set[Path] = set()
    for candidate in candidates:
        current = candidate if candidate.is_dir() else candidate.parent
        for directory in (current, *current.parents):
            if directory in checked:
                continue
            checked.add(directory)
            if all((directory / marker).exists() for marker in PROJECT_MARKERS):
                return directory
    raise FileNotFoundError(
        "Could not locate the MIPGraph project root from the current path"
    )


def resolve_project_path(root: Path, value: str | Path) -> Path:
    """Resolve a project-relative path and reject the deprecated archive tree."""

    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    relative = resolved.relative_to(root.resolve())
    if relative.parts and relative.parts[0].lower() == "web":
        raise ValueError("Deprecated archive paths are forbidden in this application case")
    return resolved


def ensure_output_within_case(root: Path, output: str | Path) -> Path:
    """Resolve and validate that generated output stays in this experiment tree."""

    resolved = resolve_project_path(root, output)
    case_root = (root / "experiments" / "computational_application_case").resolve()
    try:
        resolved.relative_to(case_root)
    except ValueError as exc:
        raise ValueError(f"Output directory must remain below {case_root}: {resolved}") from exc
    return resolved

