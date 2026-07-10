from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataset import ILPropertyDataset, PROPERTY_NAMES
from src.data.split import load_split
from src.models.factory import build_model
from src.training.evaluate import evaluate_model
from src.utils.io import save_json


LEGACY_UNUSED_PREFIXES = (
    "interaction.",
    "interaction_fusion.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation-selected property checkpoints once on fixed test.")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--clean-csv", required=True)
    parser.add_argument("--arrays-path", required=True)
    parser.add_argument("--graph-cache", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    external_marker = Path(args.selection).parent / "external_viscosity_data_required.json"
    if external_marker.exists():
        raise RuntimeError(
            f"Refusing final test evaluation while external viscosity data is required: {external_marker}"
        )
    arrays = dict(np.load(args.arrays_path, allow_pickle=True))
    clean_df = pd.read_csv(args.clean_csv)
    test_indices = load_split(args.split_path)["test"]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache: dict[str, tuple[dict, pd.DataFrame]] = {}
    rows = []

    for prop in PROPERTY_NAMES:
        selected = selection["properties"][prop]
        checkpoint = str(Path(selected["checkpoint"]).resolve())
        if checkpoint not in cache:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = build_model(state["config"])
            missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
            invalid_unexpected = [
                key for key in unexpected if not key.startswith(LEGACY_UNUSED_PREFIXES)
            ]
            if missing or invalid_unexpected:
                raise RuntimeError(
                    f"Checkpoint is incompatible with the current model: "
                    f"missing={missing}, unexpected={invalid_unexpected}"
                )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            condition = state["condition_scaler"].transform(arrays["temperature"], arrays["pressure"])
            y_scaled = state["target_scaler"].transform(arrays["y"], arrays["mask"])
            weights = np.ones_like(arrays["y"], dtype=np.float32)
            dataset = ILPropertyDataset(
                args.clean_csv,
                args.arrays_path,
                args.graph_cache,
                test_indices,
                condition,
                y_scaled,
                weights,
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            metrics, predictions, _ = evaluate_model(
                model,
                loader,
                device,
                state["target_scaler"],
                clean_df,
                "test",
                PROPERTY_NAMES,
            )
            cache[checkpoint] = (metrics, predictions)
        metrics, predictions = cache[checkpoint]
        raw = metrics[prop]
        log = metrics["log_space"][prop]
        rows.append(
            {
                "property": prop,
                "source": selected["source"],
                "checkpoint": checkpoint,
                "val_score": selected["val_score"],
                "log_MAE": log["log_MAE"],
                "log_RMSE": log["log_RMSE"],
                "log_R2": log["log_R2"],
                "log_NMAE": log["log_NMAE"],
                "raw_MAE": raw["MAE"],
                "raw_RMSE": raw["RMSE"],
                "raw_R2": raw["R2"],
                "raw_NMAE": raw["NMAE"],
            }
        )
        predictions[predictions["property"] == prop].to_csv(
            output_root / f"test_predictions_{prop}.csv", index=False
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_root / "selected_test_metrics.csv", index=False)
    average = {
        "property": "Average",
        "log_MAE": float(result["log_MAE"].mean()),
        "log_RMSE": float(result["log_RMSE"].mean()),
        "log_R2": float(result["log_R2"].mean()),
        "log_NMAE": float(result["log_NMAE"].mean()),
    }
    save_json({"rows": rows, "average": average}, output_root / "selected_test_metrics.json")
    print(result.to_string(index=False))
    print(average)


if __name__ == "__main__":
    main()
