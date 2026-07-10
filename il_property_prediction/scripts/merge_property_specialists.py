from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

PROPERTY_NAMES = ["Density", "ElectricalConductivity", "HeatCapacity", "SurfaceTension", "ThermalConductivity", "Viscosity"]


def _load_property_checkpoints(selection_json: str | None, property_checkpoints: str | None) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    if selection_json:
        with Path(selection_json).open("r", encoding="utf-8") as f:
            selection = json.load(f)
        properties = selection.get("properties", selection)
        for prop, item in properties.items():
            checkpoint = item.get("checkpoint") if isinstance(item, dict) else item
            checkpoints[prop] = Path(checkpoint)
    if property_checkpoints:
        for item in property_checkpoints.split(";"):
            if not item.strip():
                continue
            prop, checkpoint = item.split("=", 1)
            checkpoints[prop.strip()] = Path(checkpoint.strip())
    bad = [prop for prop in checkpoints if prop not in PROPERTY_NAMES]
    if bad:
        raise ValueError(f"Unknown properties in checkpoint selection: {bad}")
    missing = [prop for prop in PROPERTY_NAMES if prop not in checkpoints]
    if missing:
        raise ValueError(f"Missing specialist checkpoints for properties: {missing}")
    return checkpoints


def _target_prefixes(prop: str) -> tuple[str, ...]:
    return (
        f"property_adapters.adapters.{prop}.",
        f"decoder.heads.{prop}.",
        f"physics_moe.router_heads.{prop}.",
    )


def _clone_tensor(value):
    return value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-property specialist branches into one PhysicsMoE checkpoint.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--selection-json", default=None)
    parser.add_argument(
        "--property-checkpoints",
        default=None,
        help="Semicolon-separated mapping, e.g. Density=a.pt;Viscosity=b.pt. Overrides/extends --selection-json.",
    )
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--clean-csv", default=None)
    parser.add_argument("--arrays-path", default=None)
    parser.add_argument("--graph-cache", default=None)
    parser.add_argument("--split-path", default=None)
    args = parser.parse_args()

    checkpoints = _load_property_checkpoints(args.selection_json, args.property_checkpoints)
    base_payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    merged_state = {key: _clone_tensor(value) for key, value in base_payload["model_state_dict"].items()}

    first_payload = torch.load(next(iter(checkpoints.values())), map_location="cpu", weights_only=False)
    first_state = first_payload["model_state_dict"]
    for key, value in first_state.items():
        if key.startswith("property_adapters.") and key not in merged_state:
            merged_state[key] = _clone_tensor(value)

    copied: dict[str, dict[str, int | str]] = {}
    for prop in PROPERTY_NAMES:
        payload = torch.load(checkpoints[prop], map_location="cpu", weights_only=False)
        state = payload["model_state_dict"]
        prefixes = _target_prefixes(prop)
        copied[prop] = {"checkpoint": str(checkpoints[prop]), "parameters": 0}
        for key, value in state.items():
            if key.startswith(prefixes):
                merged_state[key] = _clone_tensor(value)
                copied[prop]["parameters"] = int(copied[prop]["parameters"]) + 1
        if copied[prop]["parameters"] == 0:
            raise RuntimeError(f"No target branch parameters were copied for {prop} from {checkpoints[prop]}")

    merged_payload = copy.deepcopy(base_payload)
    merged_payload["model_state_dict"] = merged_state
    output = Path(args.output_checkpoint)
    merged_config = copy.deepcopy(base_payload.get("config", {}))
    first_config = first_payload.get("config", {})
    if "model" in first_config:
        merged_config["model"] = copy.deepcopy(first_config["model"])
    if "chem" in first_config:
        merged_config["chem"] = copy.deepcopy(first_config["chem"])
    merged_config.setdefault("model", {})["use_property_adapters"] = True
    if args.clean_csv:
        merged_config["data"]["clean_csv"] = str(Path(args.clean_csv).resolve())
    if args.arrays_path:
        merged_config["data"]["arrays_path"] = str(Path(args.arrays_path).resolve())
    if args.graph_cache:
        merged_config["data"]["graph_cache_path"] = str(Path(args.graph_cache).resolve())
    if args.split_path:
        merged_config["data"]["split_path"] = str(Path(args.split_path).resolve())
    run_name = args.run_name or output.stem
    output_root = Path(args.output_root).resolve() if args.output_root else output.parent.resolve()
    merged_config.setdefault("outputs", {})
    merged_config["outputs"].update(
        {
            "run_name": run_name,
            "output_dir": str(output_root / run_name),
            "checkpoint_dir": str(output.parent.resolve()),
            "log_dir": str(output_root / "logs" / run_name),
            "metric_dir": str(output_root / "metrics" / run_name),
            "prediction_dir": str(output_root / "predictions" / run_name),
            "figure_dir": str(output_root / "figures" / run_name),
        }
    )
    merged_payload["config"] = merged_config
    merged_payload["merged_property_specialists"] = copied

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_payload, output)
    print({"merged_checkpoint": str(output), "copied": copied})


if __name__ == "__main__":
    main()
