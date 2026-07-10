from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataset import PROPERTY_NAMES
from src.data.split import load_split
from src.training.trainer import Trainer
from src.utils.io import load_config, resolve_path, save_json
from src.utils.seed import set_seed


def _set_output_dirs(cfg: dict, root: Path, run_name: str) -> None:
    cfg["outputs"]["run_name"] = run_name
    cfg["outputs"]["output_dir"] = str(root)
    cfg["outputs"]["checkpoint_dir"] = str(root / "checkpoints" / run_name)
    cfg["outputs"]["log_dir"] = str(root / "logs" / run_name)
    cfg["outputs"]["metric_dir"] = str(root / "metrics" / run_name)
    cfg["outputs"]["prediction_dir"] = str(root / "predictions" / run_name)
    cfg["outputs"]["figure_dir"] = str(root / "figures" / run_name)


def _test_hash(indices: list[int]) -> str:
    payload = ",".join(str(int(i)) for i in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _fold_rows(root: Path, fold_count: int) -> pd.DataFrame:
    rows = []
    for fold in range(fold_count):
        run_name = f"groupkfold_fold{fold}"
        path = root / "metrics" / run_name / "val_metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        for prop in PROPERTY_NAMES:
            item = metrics.get("log_space", {}).get(prop, {})
            rows.append(
                {
                    "fold": fold,
                    "property": prop,
                    "log_MAE": item.get("log_MAE"),
                    "log_RMSE": item.get("log_RMSE"),
                    "log_R2": item.get("log_R2"),
                    "log_NMAE": item.get("log_NMAE"),
                    "label_count": item.get("label_count"),
                }
            )
    return pd.DataFrame(rows)


def _save_summaries(root: Path, fold_count: int) -> None:
    fold_df = _fold_rows(root, fold_count)
    if fold_df.empty:
        return
    fold_df.to_csv(root / "fold_metrics.csv", index=False)
    summary = (
        fold_df.groupby("property")[["log_MAE", "log_RMSE", "log_R2", "log_NMAE"]]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(root / "cv_summary.csv", index=False)

    prediction_frames = []
    for fold in range(fold_count):
        path = root / "predictions" / f"groupkfold_fold{fold}" / "val_predictions.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["fold"] = fold
            prediction_frames.append(frame)
    if len(prediction_frames) != fold_count:
        return
    oof = pd.concat(prediction_frames, ignore_index=True)
    oof.to_csv(root / "oof_predictions.csv", index=False)
    rows = []
    for prop in PROPERTY_NAMES:
        frame = oof[oof["property"] == prop]
        y_true = frame["y_true"].to_numpy(dtype=float)
        y_pred = frame["y_pred"].to_numpy(dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
        y_true_log = np.log(y_true[valid])
        y_pred_log = np.log(y_pred[valid])
        err = y_pred_log - y_true_log
        rows.append(
            {
                "property": prop,
                "label_count": int(valid.sum()),
                "log_MAE": float(np.mean(np.abs(err))),
                "log_RMSE": float(np.sqrt(np.mean(err**2))),
                "log_R2": float(r2_score(y_true_log, y_pred_log)),
                "log_NMAE_std": float(np.mean(np.abs(err)) / max(float(np.std(y_true_log)), 1e-8)),
            }
        )
    average = {"property": "Average", "label_count": int(sum(row["label_count"] for row in rows))}
    for key in ["log_MAE", "log_RMSE", "log_R2", "log_NMAE_std"]:
        average[key] = float(np.mean([row[key] for row in rows]))
    rows.append(average)
    pd.DataFrame(rows).to_csv(root / "oof_metrics_log.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-free IL-level GroupKFold CV while keeping test locked.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--base-split", default="data/processed/splits/il_level_seed42.json")
    parser.add_argument("--clean-csv", default=None)
    parser.add_argument("--arrays-path", default=None)
    parser.add_argument("--graph-cache", default=None)
    parser.add_argument("--output-root", default="outputs/groupkfold_seed42")
    parser.add_argument("--pool", choices=["train", "trainval"], default="train")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validate-every", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-property-coupling", action="store_true")
    parser.add_argument("--fold-indices", default="", help="Optional comma-separated subset of folds to train.")
    parser.add_argument("--splits-only", action="store_true", help="Create and audit folds without training.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    cfg_base = load_config(args.config)
    base = cfg_base["_base_dir"]
    output_root = resolve_path(args.output_root, base)
    output_root.mkdir(parents=True, exist_ok=True)
    clean_csv = resolve_path(args.clean_csv or cfg_base["data"]["clean_csv"], base)
    arrays_path = resolve_path(args.arrays_path or cfg_base["data"]["arrays_path"], base)
    graph_cache = resolve_path(args.graph_cache or cfg_base["data"]["graph_cache_path"], base)
    base_split_path = resolve_path(args.base_split, base)
    base_split = load_split(base_split_path)
    pool_indices = list(base_split["train"])
    if args.pool == "trainval":
        pool_indices.extend(base_split["val"])
    pool_indices = list(dict.fromkeys(pool_indices))
    test_indices = list(base_split["test"])
    if set(pool_indices) & set(test_indices):
        raise ValueError("Development pool overlaps the locked test set")

    clean_df = pd.read_csv(clean_csv)
    pool_array = np.asarray(pool_indices, dtype=np.int64)
    groups = clean_df.loc[pool_array, "IL_SMILES"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=args.folds)
    folds = []
    validation_union: set[int] = set()
    test_groups = set(clean_df.loc[test_indices, "IL_SMILES"].astype(str))
    for fold, (train_pos, val_pos) in enumerate(splitter.split(pool_array, groups=groups)):
        train_indices = pool_array[train_pos].astype(int).tolist()
        val_indices = pool_array[val_pos].astype(int).tolist()
        train_groups = set(clean_df.loc[train_indices, "IL_SMILES"].astype(str))
        val_groups = set(clean_df.loc[val_indices, "IL_SMILES"].astype(str))
        if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
            raise ValueError(f"IL leakage detected in fold {fold}")
        validation_union.update(val_indices)
        split = {"train": train_indices, "val": val_indices, "test": test_indices}
        fold_path = output_root / "splits" / f"fold{fold}.json"
        save_json(split, fold_path)
        folds.append(
            {
                "fold": fold,
                "split_path": str(fold_path),
                "train_rows": len(train_indices),
                "val_rows": len(val_indices),
                "train_ils": len(train_groups),
                "val_ils": len(val_groups),
            }
        )
    if validation_union != set(pool_indices):
        raise ValueError("Validation folds do not cover the development pool exactly once")
    manifest = {
        "base_split": str(base_split_path),
        "pool": args.pool,
        "fold_count": args.folds,
        "locked_test_rows": len(test_indices),
        "locked_test_sha256": _test_hash(test_indices),
        "test_evaluated_during_cv": False,
        "folds": folds,
    }
    save_json(manifest, output_root / "folds_manifest.json")
    print(f"saved: {output_root / 'folds_manifest.json'}")
    if args.splits_only:
        return

    selected_folds = set(range(args.folds))
    if args.fold_indices:
        selected_folds = {int(value.strip()) for value in args.fold_indices.split(",") if value.strip()}
        invalid_folds = selected_folds - set(range(args.folds))
        if invalid_folds:
            raise ValueError(f"Invalid --fold-indices values: {sorted(invalid_folds)}")

    arrays = dict(np.load(arrays_path, allow_pickle=True))
    for fold_info in folds:
        fold = int(fold_info["fold"])
        if fold not in selected_folds:
            continue
        run_name = f"groupkfold_fold{fold}"
        metric_path = output_root / "metrics" / run_name / "val_metrics.json"
        checkpoint_path = output_root / "checkpoints" / run_name / "best_model.pt"
        prediction_path = output_root / "predictions" / run_name / "val_predictions.csv"
        completion_path = output_root / "metrics" / run_name / "fold_complete.json"
        completed = completion_path.exists() or (
            metric_path.exists() and checkpoint_path.exists() and prediction_path.exists()
        )
        if completed and not completion_path.exists():
            save_json({"completed": True, "migrated": True}, completion_path)
        if args.skip_existing and completed:
            print(f"skip existing: {run_name}")
            _save_summaries(output_root, args.folds)
            continue
        cfg = copy.deepcopy(cfg_base)
        fold_seed = args.seed + fold
        cfg["data"]["arrays_path"] = str(arrays_path)
        cfg["data"]["seed"] = args.seed
        cfg["training"].update(
            {
                "seed": fold_seed,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "validate_every": args.validate_every,
                "num_workers": args.num_workers,
                "persistent_workers": False if args.num_workers == 0 else cfg["training"].get("persistent_workers", True),
                "monitor_metric": "macro_log_val_score",
                "monitor_mode": "max",
                "evaluate_test": False,
                "evaluate_initial": False,
                "save_last_checkpoint": False,
                "versioned_best_checkpoints": False,
                "atomic_checkpoint_save": False,
            }
        )
        if args.lr is not None:
            cfg["training"]["lr"] = args.lr
        if args.disable_property_coupling:
            cfg["model"]["use_property_coupling"] = False
        cfg["augmentation"] = {"enabled": False, "properties": []}
        _set_output_dirs(cfg, output_root, run_name)
        set_seed(fold_seed)
        split = load_split(fold_info["split_path"])
        trainer = Trainer(cfg, arrays, clean_csv, graph_cache, split)
        print(
            {
                "run_name": run_name,
                "fold_seed": fold_seed,
                "train_rows": len(split["train"]),
                "val_rows": len(split["val"]),
                "locked_test_rows": len(split["test"]),
                "evaluate_test": False,
                "use_property_coupling": cfg["model"].get("use_property_coupling", True),
            }
        )
        trainer.train()
        save_json(
            {
                "completed": True,
                "fold": fold,
                "fold_seed": fold_seed,
                "best_checkpoint": str(trainer.best_checkpoint_path),
            },
            completion_path,
        )
        _save_summaries(output_root, args.folds)

    print(f"saved: {output_root / 'cv_summary.csv'}")
    if (output_root / "oof_metrics_log.csv").exists():
        print(f"saved: {output_root / 'oof_metrics_log.csv'}")


if __name__ == "__main__":
    main()
