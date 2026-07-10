from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataset import PROPERTY_NAMES
from src.data.split import load_split
from src.training.trainer import Trainer
from src.utils.io import load_config, resolve_path, save_json
from src.utils.seed import set_seed


def _save_json_atomic(obj: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


VARIANTS = {
    "unimol2_concat": {
        "label": "Uni-Mol2 + concat",
        "model": {
            "fusion_mode": "concat",
            "use_atom_cross_attention": False,
            "use_transformer_fusion": False,
        },
    },
    "unimol2_cross_concat": {
        "label": "Uni-Mol2 + cross-ion attention + concat",
        "model": {
            "fusion_mode": "concat",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": False,
        },
    },
    "unimol2_bilinear": {
        "label": "Uni-Mol2 + bilinear fusion",
        "model": {
            "fusion_mode": "bilinear",
            "use_atom_cross_attention": False,
            "use_transformer_fusion": False,
        },
    },
    "unimol2_cross_bilinear": {
        "label": "Uni-Mol2 + cross-ion attention + bilinear fusion",
        "model": {
            "fusion_mode": "bilinear",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": False,
        },
    },
    "unimol2_cross_transformer_global_desc": {
        "label": "Uni-Mol2 + cross-ion attention + Transformer fusion + global descriptors",
        "model": {
            "fusion_mode": "transformer",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": True,
            "use_global_descriptors": True,
            "use_functional_group_descriptors": False,
        },
    },
    "unimol2_cross_transformer_global_fg_desc": {
        "label": "Uni-Mol2 + cross-ion attention + Transformer fusion + global and functional-group descriptors",
        "model": {
            "fusion_mode": "transformer",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": True,
            "use_global_descriptors": True,
            "use_functional_group_descriptors": True,
        },
    },
    "unimol2_cross_transformer_unfreeze1": {
        "label": "Uni-Mol2 + cross-ion attention + Transformer fusion + unfreeze last 1 Uni-Mol2 layer",
        "model": {
            "fusion_mode": "transformer",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": True,
            "freeze_unimol2_backbone": True,
            "unimol2_unfreeze_last_n_layers": 1,
        },
    },
    "unimol2_cross_transformer_unfreeze2": {
        "label": "Uni-Mol2 + cross-ion attention + Transformer fusion + unfreeze last 2 Uni-Mol2 layers",
        "model": {
            "fusion_mode": "transformer",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": True,
            "freeze_unimol2_backbone": True,
            "unimol2_unfreeze_last_n_layers": 2,
        },
    },
    "unimol2_cross_transformer_topk1": {
        "label": "Uni-Mol2 + cross-ion attention + Transformer fusion + MoE top_k=1",
        "model": {
            "fusion_mode": "transformer",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": True,
            "physics_moe_top_k": 1,
        },
    },
    "unimol2_cross_transformer_topk3": {
        "label": "Uni-Mol2 + cross-ion attention + Transformer fusion + MoE top_k=3",
        "model": {
            "fusion_mode": "transformer",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": True,
            "physics_moe_top_k": 3,
        },
    },
    "unimol2_cross_transformer_no_moe_prior": {
        "label": "Uni-Mol2 + cross-ion attention + Transformer fusion + MoE prior loss off",
        "model": {
            "fusion_mode": "transformer",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": True,
        },
        "loss": {
            "use_moe_prior_loss": False,
            "moe_prior_weight": 0.0,
        },
    },
    "unimol2_cross_transformer_no_moe_load_balance": {
        "label": "Uni-Mol2 + cross-ion attention + Transformer fusion + MoE load-balance loss off",
        "model": {
            "fusion_mode": "transformer",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": True,
        },
        "loss": {
            "use_moe_load_balance_loss": False,
            "moe_load_balance_weight": 0.0,
        },
    },
    "unimol2_cross_transformer_no_moe_regularizers": {
        "label": "Uni-Mol2 + cross-ion attention + Transformer fusion + MoE prior and load-balance losses off",
        "model": {
            "fusion_mode": "transformer",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": True,
        },
        "loss": {
            "use_moe_prior_loss": False,
            "moe_prior_weight": 0.0,
            "use_moe_load_balance_loss": False,
            "moe_load_balance_weight": 0.0,
        },
    },
    "unimol2_cross_transformer_no_condition_film": {
        "label": "Uni-Mol2 + cross-ion attention + Transformer fusion + Condition FiLM off",
        "model": {
            "fusion_mode": "transformer",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": True,
            "use_condition_film": False,
        },
    },
    "unimol2_cross_transformer": {
        "label": "Uni-Mol2 + cross-ion attention + Transformer fusion",
        "model": {
            "fusion_mode": "transformer",
            "use_atom_cross_attention": True,
            "use_transformer_fusion": True,
        },
    },
    "unimol2_transformer": {
        "label": "Uni-Mol2 + Transformer fusion",
        "model": {
            "fusion_mode": "transformer",
            "use_atom_cross_attention": False,
            "use_transformer_fusion": True,
        },
    },
}


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


def _parse_variants(text: str) -> list[str]:
    if text.strip().lower() == "all":
        return list(VARIANTS)
    names = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown ablation variants: {unknown}. Available: {sorted(VARIANTS)}")
    return names


def _apply_variant(cfg: dict, variant_name: str, use_descriptors: bool) -> None:
    variant = VARIANTS[variant_name]
    model_cfg = cfg["model"]
    model_cfg["ion_encoder"] = "unimol2"
    if not use_descriptors:
        model_cfg["use_global_descriptors"] = False
        model_cfg["use_functional_group_descriptors"] = False
    model_cfg.update(variant.get("model", {}))
    cfg["loss"].update(variant.get("loss", {}))


def _fold_rows(root: Path, fold_count: int, variant_name: str) -> pd.DataFrame:
    rows = []
    for fold in range(fold_count):
        run_name = f"fold{fold}"
        path = root / "metrics" / run_name / "val_metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        for prop in PROPERTY_NAMES:
            item = metrics.get("log_space", {}).get(prop, {})
            rows.append(
                {
                    "variant": variant_name,
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


def _save_variant_summaries(root: Path, fold_count: int, variant_name: str) -> None:
    fold_df = _fold_rows(root, fold_count, variant_name)
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
        path = root / "predictions" / f"fold{fold}" / "val_predictions.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["variant"] = variant_name
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
                "variant": variant_name,
                "property": prop,
                "label_count": int(valid.sum()),
                "log_MAE": float(np.mean(np.abs(err))),
                "log_RMSE": float(np.sqrt(np.mean(err**2))),
                "log_R2": float(r2_score(y_true_log, y_pred_log)),
                "log_NMAE_std": float(np.mean(np.abs(err)) / max(float(np.std(y_true_log)), 1e-8)),
            }
        )
    average = {
        "variant": variant_name,
        "property": "Average",
        "label_count": int(sum(row["label_count"] for row in rows)),
    }
    for key in ["log_MAE", "log_RMSE", "log_R2", "log_NMAE_std"]:
        average[key] = float(np.mean([row[key] for row in rows]))
    rows.append(average)
    pd.DataFrame(rows).to_csv(root / "oof_metrics_log.csv", index=False)


def _save_combined_summaries(output_root: Path, variant_names: list[str]) -> None:
    fold_frames = []
    oof_metric_frames = []
    for variant_name in variant_names:
        variant_root = output_root / variant_name
        fold_path = variant_root / "fold_metrics.csv"
        if fold_path.exists():
            fold_frames.append(pd.read_csv(fold_path))
        oof_metric_path = variant_root / "oof_metrics_log.csv"
        if oof_metric_path.exists():
            oof_metric_frames.append(pd.read_csv(oof_metric_path))
    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(output_root / "ablation_fold_metrics.csv", index=False)
    if oof_metric_frames:
        metrics = pd.concat(oof_metric_frames, ignore_index=True)
        metrics.to_csv(output_root / "ablation_oof_metrics_log.csv", index=False)
        average = metrics[metrics["property"] == "Average"].copy()
        if not average.empty:
            average.sort_values("log_MAE").to_csv(output_root / "ablation_ranking_log_mae.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random-point KFold ablations with a locked test split.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--base-split", default="data/processed/splits/row_level_seed42.json")
    parser.add_argument("--clean-csv", default=None)
    parser.add_argument("--arrays-path", default=None)
    parser.add_argument("--graph-cache", default=None)
    parser.add_argument("--output-root", default="outputs/ablation_random_point_kfold_seed42")
    parser.add_argument("--pool", choices=["train", "trainval"], default="trainval")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--variants", default="all", help="Comma-separated variant names or 'all'.")
    parser.add_argument("--fold-indices", default="", help="Optional comma-separated subset of folds to train.")
    parser.add_argument("--splits-only", action="store_true", help="Create shared folds without training.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--use-descriptors", action="store_true", help="Keep global and functional-group descriptors.")
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    variant_names = _parse_variants(args.variants)
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

    pool_array = np.asarray(pool_indices, dtype=np.int64)
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = []
    validation_union: set[int] = set()
    for fold, (train_pos, val_pos) in enumerate(splitter.split(pool_array)):
        train_indices = pool_array[train_pos].astype(int).tolist()
        val_indices = pool_array[val_pos].astype(int).tolist()
        validation_union.update(val_indices)
        split = {"train": train_indices, "val": val_indices, "test": test_indices}
        fold_path = output_root / "splits" / f"fold{fold}.json"
        _save_json_atomic(split, fold_path)
        folds.append(
            {
                "fold": fold,
                "split_path": str(fold_path),
                "train_rows": len(train_indices),
                "val_rows": len(val_indices),
            }
        )
    if validation_union != set(pool_indices):
        raise ValueError("Validation folds do not cover the development pool exactly once")

    manifest = {
        "split_strategy": "random_point_kfold",
        "base_split": str(base_split_path),
        "pool": args.pool,
        "fold_count": args.folds,
        "locked_test_rows": len(test_indices),
        "locked_test_sha256": _test_hash(test_indices),
        "test_evaluated_during_cv": False,
        "default_descriptors_enabled": bool(args.use_descriptors),
        "variants": {name: VARIANTS[name] for name in variant_names},
        "folds": folds,
    }
    _save_json_atomic(manifest, output_root / "folds_manifest.json")
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
    for variant_name in variant_names:
        variant_root = output_root / variant_name
        variant_root.mkdir(parents=True, exist_ok=True)
        save_json(VARIANTS[variant_name], variant_root / "variant_manifest.json")
        for fold_info in folds:
            fold = int(fold_info["fold"])
            if fold not in selected_folds:
                continue
            run_name = f"fold{fold}"
            metric_path = variant_root / "metrics" / run_name / "val_metrics.json"
            checkpoint_path = variant_root / "checkpoints" / run_name / "best_model.pt"
            prediction_path = variant_root / "predictions" / run_name / "val_predictions.csv"
            completion_path = variant_root / "metrics" / run_name / "fold_complete.json"
            completed = completion_path.exists() or (
                metric_path.exists() and checkpoint_path.exists() and prediction_path.exists()
            )
            if completed and not completion_path.exists():
                save_json({"completed": True, "migrated": True}, completion_path)
            if args.skip_existing and completed:
                print(f"skip existing: {variant_name}/{run_name}")
                _save_variant_summaries(variant_root, args.folds, variant_name)
                _save_combined_summaries(output_root, variant_names)
                continue

            cfg = copy.deepcopy(cfg_base)
            _apply_variant(cfg, variant_name, args.use_descriptors)
            fold_seed = args.seed + fold
            cfg["data"]["arrays_path"] = str(arrays_path)
            cfg["data"]["clean_csv"] = str(clean_csv)
            cfg["data"]["graph_cache_path"] = str(graph_cache)
            cfg["data"]["split_path"] = str(fold_info["split_path"])
            cfg["data"]["seed"] = args.seed
            training_updates = {
                "seed": fold_seed,
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
            for key, value in (
                ("epochs", args.epochs),
                ("lr", args.lr),
                ("patience", args.patience),
                ("batch_size", args.batch_size),
                ("validate_every", args.validate_every),
            ):
                if value is not None:
                    training_updates[key] = value
            cfg["training"].update(training_updates)
            cfg["augmentation"] = {"enabled": False, "properties": []}
            _set_output_dirs(cfg, variant_root, run_name)
            set_seed(fold_seed)
            split = load_split(fold_info["split_path"])
            trainer = Trainer(cfg, arrays, clean_csv, graph_cache, split)
            print(
                {
                    "variant": variant_name,
                    "run_name": run_name,
                    "fold_seed": fold_seed,
                    "train_rows": len(split["train"]),
                    "val_rows": len(split["val"]),
                    "locked_test_rows": len(split["test"]),
                    "fusion_mode": cfg["model"].get("fusion_mode"),
                    "use_atom_cross_attention": cfg["model"].get("use_atom_cross_attention"),
                    "use_global_descriptors": cfg["model"].get("use_global_descriptors"),
                    "use_functional_group_descriptors": cfg["model"].get("use_functional_group_descriptors"),
                    "unimol2_unfreeze_last_n_layers": cfg["model"].get("unimol2_unfreeze_last_n_layers"),
                    "physics_moe_top_k": cfg["model"].get("physics_moe_top_k"),
                    "use_moe_prior_loss": cfg["loss"].get("use_moe_prior_loss"),
                    "use_moe_load_balance_loss": cfg["loss"].get("use_moe_load_balance_loss"),
                    "use_condition_film": cfg["model"].get("use_condition_film"),
                }
            )
            trainer.train()
            save_json(
                {
                    "completed": True,
                    "variant": variant_name,
                    "fold": fold,
                    "fold_seed": fold_seed,
                    "best_checkpoint": str(trainer.best_checkpoint_path),
                },
                completion_path,
            )
            _save_variant_summaries(variant_root, args.folds, variant_name)
            _save_combined_summaries(output_root, variant_names)

    _save_combined_summaries(output_root, variant_names)
    print(f"saved: {output_root / 'ablation_fold_metrics.csv'}")
    if (output_root / "ablation_ranking_log_mae.csv").exists():
        print(f"saved: {output_root / 'ablation_ranking_log_mae.csv'}")


if __name__ == "__main__":
    main()
