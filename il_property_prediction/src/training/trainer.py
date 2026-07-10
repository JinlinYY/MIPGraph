from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..data.dataset import ILPropertyDataset
from ..data.augmentation import build_temperature_interpolation_samples
from ..data.scaler import fit_scalers
from ..models.factory import build_model
from ..utils.io import ensure_dir, resolve_path, save_json, save_log_metrics_csv, save_metrics_csv
from .evaluate import evaluate_model
from .losses import PROPERTY_INDEX, build_loss, needs_temperature_grad


class Trainer:
    def __init__(self, config: dict, arrays: dict, clean_csv: str | Path, graph_cache_path: str | Path, split: dict, scalers=None) -> None:
        self.config = config
        self.arrays = arrays
        self.clean_csv = Path(clean_csv)
        self.clean_df = pd.read_csv(clean_csv)
        self.split = split
        loss_cfg = config["loss"]
        if scalers is None:
            self.condition_scaler, self.target_scaler, self.y_scaled, self.condition, self.error_weights = fit_scalers(
                arrays,
                split["train"],
                loss_cfg.get("error_weight_clip_min", 0.1),
                loss_cfg.get("error_weight_clip_max", 10.0),
                loss_cfg.get("target_scaler_mask", "mask"),
            )
        else:
            self.condition_scaler, self.target_scaler, self.y_scaled, self.condition, self.error_weights = scalers
        self.error_weights = np.asarray(self.error_weights, dtype=np.float32).copy()
        self.il_balance_report = self._apply_il_balance_weights()
        self.augmented_train_samples = []
        self.augmentation_report = None
        augmentation_cfg = config.get("augmentation", {})
        if bool(augmentation_cfg.get("enabled", False)):
            self.augmented_train_samples, self.augmentation_report = build_temperature_interpolation_samples(
                self.clean_df,
                arrays,
                split["train"],
                self.condition_scaler,
                self.target_scaler,
                augmentation_cfg.get("properties", []),
                int(augmentation_cfg.get("points_per_interval", 1)),
                float(augmentation_cfg.get("max_temperature_gap", 40.0)),
                int(augmentation_cfg.get("pressure_round_decimals", 1)),
                float(augmentation_cfg.get("sample_weight", 0.5)),
                int(augmentation_cfg.get("max_samples_per_property", 0)),
                int(augmentation_cfg.get("seed", config["training"].get("seed", 42))),
            )
        self.graph_cache_path = graph_cache_path
        self.device = self._device()
        self.model = build_model(config).to(self.device)
        self.loss_fn = build_loss(config)
        self.needs_temperature_grad = needs_temperature_grad(config)
        self.teacher_model = None
        self.distill_weight = float(config["training"].get("distill_weight", 0.0) or 0.0)
        self.distill_indices = [
            PROPERTY_INDEX[name]
            for name in config["training"].get("protect_properties", [])
            if name in PROPERTY_INDEX
        ]
        distill_checkpoint = config["training"].get("distill_checkpoint")
        if distill_checkpoint and self.distill_weight > 0.0 and self.distill_indices:
            base = config.get("_base_dir")
            state = torch.load(resolve_path(distill_checkpoint, base), map_location="cpu", weights_only=False)
            self.teacher_model = build_model(state["config"]).to(self.device)
            self.teacher_model.load_state_dict(state["model_state_dict"])
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def _apply_il_balance_weights(self) -> dict:
        properties = list(self.config.get("loss", {}).get("il_balance_properties", []))
        if not properties:
            return {"enabled": False, "properties": []}
        property_names = self.config["properties"]["names"]
        bad = [name for name in properties if name not in property_names]
        if bad:
            raise ValueError(f"Unknown IL-balanced properties: {bad}")
        power = float(self.config.get("loss", {}).get("il_balance_power", 1.0))
        train_idx = np.asarray(self.split["train"], dtype=np.int64)
        train_smiles = self.clean_df.iloc[train_idx]["IL_SMILES"].astype(str).to_numpy()
        report = {"enabled": True, "power": power, "properties": {}}
        for prop in properties:
            prop_idx = property_names.index(prop)
            active = np.asarray(self.arrays["mask"])[train_idx, prop_idx] > 0
            if not active.any():
                report["properties"][prop] = {"labels": 0, "ils": 0}
                continue
            active_smiles = train_smiles[active]
            counts = pd.Series(active_smiles).value_counts()
            factors = np.asarray([float(counts[smi]) ** (-power) for smi in active_smiles], dtype=np.float32)
            factors /= max(float(factors.mean()), 1e-12)
            self.error_weights[train_idx[active], prop_idx] *= factors
            report["properties"][prop] = {
                "labels": int(active.sum()),
                "ils": int(len(counts)),
                "factor_min": float(factors.min()),
                "factor_max": float(factors.max()),
                "factor_mean": float(factors.mean()),
            }
        return report

    def _metric_value(self, metrics: dict, metric_name: str):
        value = metrics.get(metric_name)
        if value is not None:
            return value
        target = self.config["training"].get("monitor_target_property")
        if metric_name == "target_val_score" and target:
            item = metrics.get(target, {})
            r2 = item.get("R2")
            nmae = item.get("NMAE")
            if r2 is None or nmae is None:
                return None
            return float(r2) - 0.2 * float(nmae)
        if metric_name == "target_R2" and target:
            value = metrics.get(target, {}).get("R2")
            return float(value) if value is not None else None
        if metric_name == "target_MAE" and target:
            value = metrics.get(target, {}).get("MAE")
            return float(value) if value is not None else None
        if metric_name == "target_NMAE" and target:
            value = metrics.get(target, {}).get("NMAE")
            return float(value) if value is not None else None
        if metric_name == "target_log_val_score" and target:
            item = metrics.get("log_space", {}).get(target, {})
            r2 = item.get("log_R2")
            nmae = item.get("log_NMAE")
            if r2 is None or nmae is None:
                return None
            return float(r2) - 0.2 * float(nmae)
        if metric_name == "target_log_R2" and target:
            value = metrics.get("log_space", {}).get(target, {}).get("log_R2")
            return float(value) if value is not None else None
        if metric_name == "target_log_MAE" and target:
            value = metrics.get("log_space", {}).get(target, {}).get("log_MAE")
            return float(value) if value is not None else None
        if metric_name == "target_log_NMAE" and target:
            value = metrics.get("log_space", {}).get(target, {}).get("log_NMAE")
            return float(value) if value is not None else None
        if metric_name == "macro_log_val_score":
            r2 = metrics.get("macro_log_R2")
            nmae = metrics.get("macro_log_NMAE")
            if r2 is None or nmae is None:
                return None
            return float(r2) - 0.2 * float(nmae)
        focus = self.config["training"].get("monitor_properties", [])
        if not focus:
            return None
        if metric_name == "focus_macro_R2":
            vals = [metrics.get(prop, {}).get("R2") for prop in focus]
        elif metric_name == "focus_macro_log_R2":
            log_metrics = metrics.get("log_space", {})
            vals = [log_metrics.get(prop, {}).get("log_R2") for prop in focus]
        elif metric_name == "focus_macro_MAE":
            vals = [metrics.get(prop, {}).get("MAE") for prop in focus]
        elif metric_name == "focus_macro_log_MAE":
            log_metrics = metrics.get("log_space", {})
            vals = [log_metrics.get(prop, {}).get("log_MAE") for prop in focus]
        else:
            return None
        vals = [float(v) for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _device(self):
        dev = self.config["training"].get("device", "auto")
        if dev == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(dev)

    def loader(self, split_name: str, shuffle: bool = False):
        indices = list(self.split[split_name])
        split_limit = int(self.config["training"].get(f"limit_{split_name}_samples", 0) or 0)
        if split_limit <= 0 and split_name != "train":
            split_limit = int(self.config["training"].get("limit_eval_samples", 0) or 0)
        if split_limit > 0:
            indices = indices[:split_limit]
        augmented_samples = self.augmented_train_samples if split_name == "train" else None
        ds = ILPropertyDataset(
            self.clean_csv,
            self.config["data"]["arrays_path"],
            self.graph_cache_path,
            indices,
            self.condition,
            self.y_scaled,
            self.error_weights,
            augmented_samples=augmented_samples,
        )
        num_workers = int(self.config["training"].get("num_workers", 0))
        kwargs = {
            "batch_size": self.config["training"]["batch_size"],
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": bool(self.config["training"].get("pin_memory", False)) and self.device.type == "cuda",
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(self.config["training"].get("persistent_workers", True))
            kwargs["prefetch_factor"] = int(self.config["training"].get("prefetch_factor", 2))
        return DataLoader(ds, **kwargs)

    def train(self):
        out = self.config["outputs"]
        ckpt_dir = ensure_dir(out["checkpoint_dir"])
        log_dir = ensure_dir(out["log_dir"])
        metric_dir = ensure_dir(out["metric_dir"])
        pred_dir = ensure_dir(out["prediction_dir"])
        versioned_best = bool(self.config["training"].get("versioned_best_checkpoints", False))

        def best_path(epoch: int) -> Path:
            if versioned_best:
                return ckpt_dir / f"best_model_pid{os.getpid()}_epoch{epoch:03d}.pt"
            return ckpt_dir / "best_model.pt"

        def prune_stale_best_checkpoints(keep_path: Path) -> None:
            if not versioned_best:
                return
            try:
                keep_resolved = keep_path.resolve()
            except OSError:
                keep_resolved = keep_path
            for stale_path in ckpt_dir.glob("best_model*.pt"):
                try:
                    stale_resolved = stale_path.resolve()
                except OSError:
                    stale_resolved = stale_path
                if stale_resolved == keep_resolved:
                    continue
                last_error = None
                for attempt in range(1, 4):
                    try:
                        stale_path.unlink(missing_ok=True)
                        last_error = None
                        break
                    except OSError as exc:
                        last_error = exc
                        time.sleep(0.5 * attempt)
                if last_error is not None:
                    print(
                        {
                            "warning": "failed_to_delete_stale_best_checkpoint",
                            "path": str(stale_path),
                            "error": str(last_error),
                        }
                    )

        best_checkpoint_path = None
        if self.augmentation_report is not None:
            save_json(self.augmentation_report, metric_dir / "augmentation_report.json")
            print({"augmentation": self.augmentation_report})
        train_loader = self.loader("train", shuffle=True)
        val_loader = self.loader("val")
        evaluate_test = bool(self.config["training"].get("evaluate_test", True))
        test_loader = self.loader("test") if evaluate_test else None
        train_cfg = self.config["training"]
        base_lr = float(train_cfg["lr"])
        backbone_lr = train_cfg.get("backbone_lr")
        backbone_parameters = []
        other_parameters = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("ion_encoder.backbone."):
                backbone_parameters.append(parameter)
            else:
                other_parameters.append(parameter)
        parameter_groups = [{"params": other_parameters, "lr": base_lr}]
        if backbone_parameters:
            parameter_groups.append(
                {
                    "params": backbone_parameters,
                    "lr": float(backbone_lr) if backbone_lr is not None else base_lr,
                }
            )
        opt = torch.optim.AdamW(
            parameter_groups,
            lr=base_lr,
            weight_decay=float(train_cfg["weight_decay"]),
        )
        use_amp = bool(self.config["training"].get("use_amp", True)) and self.device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        validate_every = max(1, int(self.config["training"].get("validate_every", 1)))
        monitor_metric = self.config["training"].get("monitor_metric", "macro_R2")
        monitor_mode = self.config["training"].get("monitor_mode", "max")
        maximize_monitor = monitor_mode == "max"
        print(
            {
                "device": str(self.device),
                "batch_size": self.config["training"]["batch_size"],
                "num_workers": self.config["training"].get("num_workers", 0),
                "pin_memory": self.config["training"].get("pin_memory", False),
                "use_amp": use_amp,
                "validate_every": validate_every,
                "monitor_metric": monitor_metric,
                "monitor_mode": monitor_mode,
                "monitor_target_property": self.config["training"].get("monitor_target_property"),
                "protect_properties": self.config["training"].get("protect_properties", []),
                "distill_weight": self.distill_weight,
                "property_loss_weights": self.config["loss"].get("property_loss_weights", {}),
                "augmented_train_samples": len(self.augmented_train_samples),
                "evaluate_test": evaluate_test,
                "il_balance": self.il_balance_report,
            }
        )
        resume_checkpoint = train_cfg.get("resume_checkpoint")
        if resume_checkpoint:
            resume_path = resolve_path(resume_checkpoint, self.config.get("_base_dir"))
            state = torch.load(resume_path, map_location=self.device, weights_only=False)
            missing, unexpected = self.model.load_state_dict(state["model_state_dict"], strict=False)
            print(
                {
                    "resume_checkpoint": str(resume_path),
                    "resume_epoch": state.get("epoch"),
                    "missing_keys": list(missing),
                    "unexpected_keys": list(unexpected),
                }
            )
        best = -float("inf") if maximize_monitor else float("inf")
        bad_epochs = 0
        last_val_metrics = {}
        if bool(self.config["training"].get("evaluate_initial", True)):
            initial_val_metrics, _, _ = evaluate_model(
                self.model,
                val_loader,
                self.device,
                self.target_scaler,
                self.clean_df,
                "val",
                self.config["properties"]["names"],
            )
            initial_score = self._metric_value(initial_val_metrics, monitor_metric)
            if initial_score is None:
                raise KeyError(f"Monitor metric {monitor_metric!r} was not found in initial validation metrics")
            best = initial_score
            last_val_metrics = initial_val_metrics
            best_checkpoint_path = best_path(0)
            self.save_checkpoint(best_checkpoint_path, 0, initial_val_metrics)
            prune_stale_best_checkpoints(best_checkpoint_path)
            save_json(initial_val_metrics, metric_dir / "val_metrics.json")
            save_log_metrics_csv(initial_val_metrics, metric_dir / "val_metrics_log.csv", self.config["properties"]["names"])
            save_metrics_csv(initial_val_metrics, metric_dir / "val_metrics_raw.csv", self.config["properties"]["names"])
            print({"epoch": 0, "monitor_metric": monitor_metric, "monitor_score": initial_score, "checkpoint": "initial"})
        log_path = log_dir / "train_log.csv"
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "processed_batches",
                    "skipped_nonfinite_batches",
                    "monitor_metric",
                    "monitor_score",
                    "val_macro_MAE",
                    "val_macro_R2",
                    "val_weighted_R2",
                    "val_macro_log_R2",
                    "val_weighted_log_R2",
                    "val_target_R2",
                    "val_target_NMAE",
                    "val_target_score",
                    "val_target_log_R2",
                    "val_target_log_NMAE",
                    "val_target_log_score",
                ],
            )
            writer.writeheader()
            for epoch in range(1, self.config["training"]["epochs"] + 1):
                self.model.train()
                losses = []
                processed_batches = 0
                skipped_nonfinite_batches = 0
                for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} train"), start=1):
                    batch = batch.to(self.device, non_blocking=True)
                    if self.needs_temperature_grad:
                        batch.raw_condition = batch.raw_condition.clone().detach().requires_grad_(True)
                    opt.zero_grad(set_to_none=True)
                    try:
                        autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
                    except (AttributeError, TypeError):
                        autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)
                    with autocast_ctx:
                        pred, aux = self.model(batch)
                        loss = self.loss_fn(pred, batch.y, batch.mask, batch.error_weight, aux)
                        if self.teacher_model is not None:
                            with torch.no_grad():
                                teacher_pred, _ = self.teacher_model(batch)
                            protected_pred = pred[:, self.distill_indices]
                            protected_teacher = teacher_pred[:, self.distill_indices].detach()
                            distill_loss = torch.mean((protected_pred - protected_teacher) ** 2)
                            loss = loss + self.distill_weight * distill_loss
                    if not torch.isfinite(loss):
                        skipped_nonfinite_batches += 1
                        sample_preview = None
                        if hasattr(batch, "sample_id"):
                            sample_preview = batch.sample_id[:8].detach().cpu().tolist()
                        print(
                            {
                                "warning": "skipped_nonfinite_loss_batch",
                                "epoch": epoch,
                                "batch": batch_idx,
                                "loss": float(loss.detach().cpu()),
                                "pred_finite": bool(torch.isfinite(pred).all().detach().cpu().item()),
                                "sample_id_preview": sample_preview,
                            }
                        )
                        opt.zero_grad(set_to_none=True)
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["gradient_clip_norm"])
                    scaler.step(opt)
                    scaler.update()
                    losses.append(float(loss.detach().cpu()))
                    processed_batches += 1
                if skipped_nonfinite_batches:
                    print(
                        {
                            "warning": "skipped_nonfinite_batches",
                            "epoch": epoch,
                            "skipped": skipped_nonfinite_batches,
                            "processed": processed_batches,
                        }
                    )
                should_validate = epoch == 1 or epoch % validate_every == 0
                if should_validate:
                    val_metrics, _, _ = evaluate_model(self.model, val_loader, self.device, self.target_scaler, self.clean_df, "val", self.config["properties"]["names"])
                    last_val_metrics = val_metrics
                    monitor_score = self._metric_value(val_metrics, monitor_metric)
                    if monitor_score is None:
                        raise KeyError(f"Monitor metric {monitor_metric!r} was not found in validation metrics")
                else:
                    val_metrics = last_val_metrics
                    monitor_score = best
                row = {
                    "epoch": epoch,
                    "train_loss": sum(losses) / len(losses) if losses else None,
                    "processed_batches": processed_batches,
                    "skipped_nonfinite_batches": skipped_nonfinite_batches,
                    "monitor_metric": monitor_metric,
                    "monitor_score": monitor_score,
                    "val_macro_MAE": val_metrics.get("macro_MAE"),
                    "val_macro_R2": val_metrics.get("macro_R2"),
                    "val_weighted_R2": val_metrics.get("weighted_R2"),
                    "val_macro_log_R2": val_metrics.get("macro_log_R2"),
                    "val_weighted_log_R2": val_metrics.get("weighted_log_R2"),
                }
                target = self.config["training"].get("monitor_target_property")
                if target:
                    target_metrics = val_metrics.get(target, {})
                    row["val_target_R2"] = target_metrics.get("R2")
                    row["val_target_NMAE"] = target_metrics.get("NMAE")
                    row["val_target_score"] = self._metric_value(val_metrics, "target_val_score")
                    target_log_metrics = val_metrics.get("log_space", {}).get(target, {})
                    row["val_target_log_R2"] = target_log_metrics.get("log_R2")
                    row["val_target_log_NMAE"] = target_log_metrics.get("log_NMAE")
                    row["val_target_log_score"] = self._metric_value(val_metrics, "target_log_val_score")
                else:
                    row["val_target_R2"] = None
                    row["val_target_NMAE"] = None
                    row["val_target_score"] = None
                    row["val_target_log_R2"] = None
                    row["val_target_log_NMAE"] = None
                    row["val_target_log_score"] = None
                writer.writerow(row)
                f.flush()
                print(row)
                if bool(self.config["training"].get("save_last_checkpoint", True)):
                    self.save_checkpoint(ckpt_dir / "last_model.pt", epoch, val_metrics)
                if should_validate:
                    improved = monitor_score > best if maximize_monitor else monitor_score < best
                    if improved:
                        best = monitor_score
                        bad_epochs = 0
                        best_checkpoint_path = best_path(epoch)
                        self.save_checkpoint(best_checkpoint_path, epoch, val_metrics)
                        prune_stale_best_checkpoints(best_checkpoint_path)
                        save_json(val_metrics, metric_dir / "val_metrics.json")
                        save_log_metrics_csv(val_metrics, metric_dir / "val_metrics_log.csv", self.config["properties"]["names"])
                        save_metrics_csv(val_metrics, metric_dir / "val_metrics_raw.csv", self.config["properties"]["names"])
                    else:
                        bad_epochs += 1
                if bad_epochs >= self.config["training"].get("patience", 40):
                    break
        if best_checkpoint_path is None:
            raise RuntimeError("Training finished without producing a best checkpoint")
        self.best_checkpoint_path = best_checkpoint_path
        state = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        if not evaluate_test:
            val_metrics, pred_df, pred_wide_df = evaluate_model(
                self.model,
                val_loader,
                self.device,
                self.target_scaler,
                self.clean_df,
                "val",
                self.config["properties"]["names"],
            )
            save_json(val_metrics, metric_dir / "val_metrics.json")
            save_log_metrics_csv(val_metrics, metric_dir / "val_metrics_log.csv", self.config["properties"]["names"])
            save_metrics_csv(val_metrics, metric_dir / "val_metrics_raw.csv", self.config["properties"]["names"])
            pred_df.to_csv(pred_dir / "val_predictions.csv", index=False)
            pred_wide_df.to_csv(pred_dir / "val_predictions_wide.csv", index=False)
            return val_metrics
        test_metrics, pred_df, pred_wide_df = evaluate_model(self.model, test_loader, self.device, self.target_scaler, self.clean_df, "test", self.config["properties"]["names"])
        save_json(test_metrics, metric_dir / "test_metrics.json")
        save_log_metrics_csv(test_metrics, metric_dir / "test_metrics_log.csv", self.config["properties"]["names"])
        save_metrics_csv(test_metrics, metric_dir / "test_metrics_raw.csv", self.config["properties"]["names"])
        pred_df.to_csv(pred_dir / "test_predictions.csv", index=False)
        pred_wide_df.to_csv(pred_dir / "test_predictions_wide.csv", index=False)
        return test_metrics

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict):
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "condition_scaler": self.condition_scaler,
            "target_scaler": self.target_scaler,
            "property_names": self.config["properties"]["names"],
            "target_means": self.target_scaler.means,
            "target_stds": self.target_scaler.stds,
            "metrics": metrics,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        if not bool(self.config["training"].get("atomic_checkpoint_save", True)):
            torch.save(payload, path)
            return
        last_error = None
        for attempt in range(1, 6):
            temp_path = path.with_name(f".{path.name}.{os.getpid()}.{attempt}.tmp")
            try:
                torch.save(payload, temp_path)
                os.replace(temp_path, path)
                return
            except (OSError, RuntimeError) as exc:
                last_error = exc
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt < 5:
                    time.sleep(0.5 * attempt)
        raise RuntimeError(f"Failed to save checkpoint after 5 attempts: {path}") from last_error
