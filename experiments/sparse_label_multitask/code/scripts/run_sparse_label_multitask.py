from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.split import create_split, load_split
from src.utils.io import load_config, resolve_path


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_csv(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _without_runtime_keys(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    out.pop("_config_path", None)
    out.pop("_base_dir", None)
    return out


def _set_absolute_data_paths(cfg: dict, base: str | Path) -> None:
    for key in ["raw_path", "processed_dir", "clean_csv", "arrays_path", "graph_cache_path", "failed_smiles_path"]:
        if key in cfg["data"]:
            cfg["data"][key] = str(resolve_path(cfg["data"][key], base))


def _set_output_dirs(cfg: dict, base: str | Path, experiment_root: Path, run_name: str) -> None:
    root = resolve_path(experiment_root, base) / run_name
    cfg["outputs"]["run_name"] = run_name
    cfg["outputs"]["output_dir"] = str(root)
    cfg["outputs"]["checkpoint_dir"] = str(root / "checkpoints")
    cfg["outputs"]["log_dir"] = str(root / "logs")
    cfg["outputs"]["metric_dir"] = str(root / "metrics")
    cfg["outputs"]["prediction_dir"] = str(root / "predictions")
    cfg["outputs"]["figure_dir"] = str(root / "figures")
    cfg["outputs"]["ablation_dir"] = str(root / "ablation")


def _write_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_without_runtime_keys(cfg), f, sort_keys=False, allow_unicode=True)


def _sample_target_labels(base_mask: np.ndarray, train_indices: list[int], target_idx: int, ratio: float, seed: int) -> np.ndarray:
    train_idx = np.asarray(train_indices, dtype=np.int64)
    candidates = train_idx[base_mask[train_idx, target_idx] > 0]
    keep = np.zeros(base_mask.shape[0], dtype=bool)
    if len(candidates) == 0 or ratio <= 0:
        return keep
    n_keep = int(round(len(candidates) * ratio))
    n_keep = min(len(candidates), max(1, n_keep))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=n_keep, replace=False)
    keep[chosen] = True
    return keep


def _make_train_mask(
    base_mask: np.ndarray,
    train_indices: list[int],
    mode: str,
    target_idx: int | None,
    ratio: float,
    seed: int,
) -> np.ndarray:
    train_idx = np.asarray(train_indices, dtype=np.int64)
    visible = np.zeros_like(base_mask, dtype=np.float32)
    if mode == "joint_all":
        visible[train_idx] = base_mask[train_idx]
        return visible
    if target_idx is None:
        raise ValueError(f"target_idx is required for mode {mode}")
    keep_target = _sample_target_labels(base_mask, train_indices, target_idx, ratio, seed)
    if mode == "single_full":
        visible[train_idx, target_idx] = base_mask[train_idx, target_idx]
    elif mode == "target_only_low_label":
        visible[keep_target, target_idx] = base_mask[keep_target, target_idx]
    elif mode == "multitask_transfer":
        visible[train_idx] = base_mask[train_idx]
        hidden_target = np.zeros(base_mask.shape[0], dtype=bool)
        hidden_target[train_idx] = base_mask[train_idx, target_idx] > 0
        hidden_target &= ~keep_target
        visible[hidden_target, target_idx] = 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return visible


def _label_counts(mask: np.ndarray, indices: list[int], property_names: list[str]) -> dict[str, int]:
    idx = np.asarray(indices, dtype=np.int64)
    return {name: int(mask[idx, j].sum()) for j, name in enumerate(property_names)}


def _metrics_for_property(metrics: dict, property_name: str) -> dict:
    raw = metrics.get(property_name, {})
    log = metrics.get("log_space", {}).get(property_name, {})
    return {
        "MAE": raw.get("MAE"),
        "RMSE": raw.get("RMSE"),
        "R2": raw.get("R2"),
        "MAPE": raw.get("MAPE"),
        "NMAE_mean": raw.get("NMAE_mean"),
        "NRMSE_mean": raw.get("NRMSE_mean"),
        "label_count": raw.get("label_count"),
        "log_MAE": log.get("log_MAE"),
        "log_RMSE": log.get("log_RMSE"),
        "log_R2": log.get("log_R2"),
    }


def _summarize_delta(summary: pd.DataFrame, out_path: Path) -> None:
    rows = []
    key_cols = ["seed", "target_property", "target_train_ratio"]
    left = summary[summary["mode"] == "target_only_low_label"]
    right = summary[summary["mode"] == "multitask_transfer"]
    if left.empty or right.empty:
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        return
    merged = left.merge(right, on=key_cols, suffixes=("_target_only", "_transfer"))
    for _, row in merged.iterrows():
        rows.append(
            {
                "seed": row["seed"],
                "target_property": row["target_property"],
                "target_train_ratio": row["target_train_ratio"],
                "target_only_train_labels": row["target_train_labels_target_only"],
                "transfer_target_train_labels": row["target_train_labels_transfer"],
                "transfer_aux_train_labels": row["aux_train_labels_transfer"],
                "delta_R2": _delta(row.get("R2_transfer"), row.get("R2_target_only")),
                "delta_log_R2": _delta(row.get("log_R2_transfer"), row.get("log_R2_target_only")),
                "delta_MAE": _delta(row.get("MAE_transfer"), row.get("MAE_target_only")),
                "relative_MAE_reduction": _relative_reduction(row.get("MAE_target_only"), row.get("MAE_transfer")),
                "delta_log_MAE": _delta(row.get("log_MAE_transfer"), row.get("log_MAE_target_only")),
                "relative_log_MAE_reduction": _relative_reduction(row.get("log_MAE_target_only"), row.get("log_MAE_transfer")),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def _delta(a, b):
    if pd.isna(a) or pd.isna(b):
        return None
    return float(a) - float(b)


def _relative_reduction(before, after):
    if pd.isna(before) or pd.isna(after) or abs(float(before)) < 1e-12:
        return None
    return (float(before) - float(after)) / abs(float(before))


def _make_experiments(property_names: list[str], target_properties: list[str], ratios: list[float], modes: list[str]) -> list[dict]:
    experiments = []
    if "joint" in modes:
        experiments.append({"mode": "joint_all", "target_property": "ALL", "target_train_ratio": 1.0})
    if "single" in modes:
        for prop in property_names:
            experiments.append({"mode": "single_full", "target_property": prop, "target_train_ratio": 1.0})
    if "transfer" in modes:
        for prop in target_properties:
            for ratio in ratios:
                experiments.append({"mode": "target_only_low_label", "target_property": prop, "target_train_ratio": ratio})
                experiments.append({"mode": "multitask_transfer", "target_property": prop, "target_train_ratio": ratio})
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sparse-label multi-task experiments: single-property, joint-property, and low-label property-transfer."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--modes", default="single,joint,transfer", help="Comma-separated subset of: single,joint,transfer.")
    parser.add_argument("--target-properties", default=None, help="Transfer targets. Default: all configured properties.")
    parser.add_argument("--transfer-ratios", default="0.1,0.25,0.5")
    parser.add_argument("--run-prefix", default="sparse_label")
    parser.add_argument("--experiment-root", default="outputs/sparse_label_multitask")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--balance-properties", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--uniform-property-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monitor-space", choices=["raw", "log"], default="log")
    args = parser.parse_args()

    cfg0 = load_config(args.config)
    base = cfg0["_base_dir"]
    property_names = list(cfg0["properties"]["names"])
    target_properties = _parse_csv(args.target_properties) if args.target_properties else property_names
    bad = [p for p in target_properties if p not in property_names]
    if bad:
        raise ValueError(f"Unknown target properties: {bad}. Valid properties: {property_names}")
    modes = _parse_csv(args.modes)
    seeds = [int(item) for item in _parse_csv(args.seeds)]
    ratios = _parse_float_csv(args.transfer_ratios)
    experiment_root = Path(args.experiment_root)
    root_abs = resolve_path(experiment_root, base)
    config_dir = root_abs / "configs"
    mask_dir = root_abs / "masks"
    summary_rows = []

    arrays_path = resolve_path(cfg0["data"]["arrays_path"], base)
    clean_csv = resolve_path(cfg0["data"]["clean_csv"], base)
    graph_cache = resolve_path(cfg0["data"]["graph_cache_path"], base)
    processed_dir = resolve_path(cfg0["data"]["processed_dir"], base)
    arrays = dict(np.load(arrays_path, allow_pickle=True))

    experiments = _make_experiments(property_names, target_properties, ratios, modes)
    for seed in seeds:
        split_name = cfg0["data"].get("split_type", "il_level")
        split_path = processed_dir / "splits" / f"{split_name}_seed{seed}.json"
        if not split_path.exists():
            create_split(
                split_name,
                clean_csv,
                processed_dir,
                cfg0["data"]["train_ratio"],
                cfg0["data"]["val_ratio"],
                cfg0["data"]["test_ratio"],
                seed,
            )
        split = load_split(split_path)
        for exp in experiments:
            target = exp["target_property"]
            ratio = float(exp["target_train_ratio"])
            mode = exp["mode"]
            target_idx = None if target == "ALL" else property_names.index(target)
            mask_seed = seed + (0 if target_idx is None else (target_idx + 1) * 1009) + int(round(ratio * 10000))
            train_mask = _make_train_mask(arrays["mask"], split["train"], mode, target_idx, ratio, mask_seed)
            train_counts = _label_counts(train_mask, split["train"], property_names)
            original_train_counts = _label_counts(arrays["mask"], split["train"], property_names)
            ratio_tag = str(ratio).replace(".", "p")
            run_name = f"{args.run_prefix}_{mode}_{target}_r{ratio_tag}_seed{seed}"

            cfg = copy.deepcopy(cfg0)
            _set_absolute_data_paths(cfg, base)
            _set_output_dirs(cfg, base, experiment_root, run_name)
            cfg["data"]["seed"] = seed
            cfg["training"]["seed"] = seed
            cfg["loss"]["balance_properties"] = bool(args.balance_properties)
            if args.uniform_property_weights:
                cfg["loss"]["property_loss_weights"] = {name: 1.0 for name in property_names}
            if args.epochs is not None:
                cfg["training"]["epochs"] = args.epochs
            if args.batch_size is not None:
                cfg["training"]["batch_size"] = args.batch_size
            if args.num_workers is not None:
                cfg["training"]["num_workers"] = args.num_workers
                if args.num_workers == 0:
                    cfg["training"]["persistent_workers"] = False
            if args.validate_every is not None:
                cfg["training"]["validate_every"] = args.validate_every
            if args.patience is not None:
                cfg["training"]["patience"] = args.patience
            if mode == "joint_all":
                cfg["training"]["monitor_metric"] = "macro_log_R2" if args.monitor_space == "log" else "macro_R2"
            else:
                cfg["training"]["monitor_metric"] = "focus_macro_log_R2" if args.monitor_space == "log" else "focus_macro_R2"
                cfg["training"]["monitor_properties"] = [target]
            cfg["training"]["monitor_mode"] = "max"

            cfg_path = config_dir / f"{run_name}.yaml"
            mask_path = mask_dir / f"{run_name}_train_visible_mask.npz"
            _write_config(cfg, cfg_path)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                mask_path,
                train_visible_mask=train_mask.astype(np.float32),
                original_mask=arrays["mask"].astype(np.float32),
                property_names=np.array(property_names),
                train_indices=np.asarray(split["train"], dtype=np.int64),
                val_indices=np.asarray(split["val"], dtype=np.int64),
                test_indices=np.asarray(split["test"], dtype=np.int64),
            )

            metrics_path = Path(cfg["outputs"]["metric_dir"]) / "test_metrics.json"
            metrics = None
            status = "config_only"
            if args.run_training:
                if args.skip_existing and metrics_path.exists():
                    status = "done_existing"
                else:
                    from src.training.trainer import Trainer
                    from src.utils.seed import set_seed

                    print(f"running {run_name}")
                    set_seed(seed)
                    trainer = Trainer(cfg, arrays, clean_csv, graph_cache, split, train_label_mask=train_mask)
                    trainer.train()
                    status = "done"
                if metrics_path.exists():
                    with metrics_path.open("r", encoding="utf-8") as f:
                        metrics = json.load(f)
            elif metrics_path.exists():
                status = "done_existing"
                with metrics_path.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)

            props_to_record = property_names if mode == "joint_all" else [target]
            for prop in props_to_record:
                metric_values = _metrics_for_property(metrics or {}, prop)
                summary_rows.append(
                    {
                        "run_name": run_name,
                        "status": status,
                        "seed": seed,
                        "mode": mode,
                        "target_property": prop if mode == "joint_all" else target,
                        "target_train_ratio": ratio,
                        "target_train_labels": train_counts[prop],
                        "target_original_train_labels": original_train_counts[prop],
                        "aux_train_labels": int(sum(v for k, v in train_counts.items() if k != prop)),
                        "total_visible_train_labels": int(sum(train_counts.values())),
                        "config_path": str(cfg_path),
                        "mask_path": str(mask_path),
                        **metric_values,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary_path = root_abs / f"{args.run_prefix}_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    delta_path = root_abs / f"{args.run_prefix}_transfer_delta.csv"
    _summarize_delta(summary, delta_path)

    manifest_path = root_abs / f"{args.run_prefix}_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "status", "config_path", "mask_path"])
        writer.writeheader()
        seen = set()
        for row in summary_rows:
            key = row["run_name"]
            if key in seen:
                continue
            seen.add(key)
            writer.writerow({k: row[k] for k in ["run_name", "status", "config_path", "mask_path"]})

    print(f"saved summary: {summary_path}")
    print(f"saved transfer delta: {delta_path}")
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
