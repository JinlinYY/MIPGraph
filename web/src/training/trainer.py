from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..data.dataset import ILPropertyDataset
from ..data.scaler import fit_scalers
from ..models.iptnet import IPTNet
from ..utils.io import ensure_dir, save_json, save_log_metrics_csv, save_metrics_csv
from .evaluate import evaluate_model
from .losses import build_loss


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
            )
        else:
            self.condition_scaler, self.target_scaler, self.y_scaled, self.condition, self.error_weights = scalers
        self.graph_cache_path = graph_cache_path
        self.device = self._device()
        self.model = IPTNet(config).to(self.device)
        self.loss_fn = build_loss(config)

    def _metric_value(self, metrics: dict, metric_name: str):
        value = metrics.get(metric_name)
        if value is not None:
            return value
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
        ds = ILPropertyDataset(self.clean_csv, self.config["data"]["arrays_path"], self.graph_cache_path, self.split[split_name], self.condition, self.y_scaled, self.error_weights)
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
        train_loader = self.loader("train", shuffle=True)
        val_loader = self.loader("val")
        test_loader = self.loader("test")
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config["training"]["lr"], weight_decay=self.config["training"]["weight_decay"])
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
                "property_loss_weights": self.config["loss"].get("property_loss_weights", {}),
            }
        )
        best = -float("inf") if maximize_monitor else float("inf")
        bad_epochs = 0
        last_val_metrics = {}
        log_path = log_dir / "train_log.csv"
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "monitor_metric",
                    "monitor_score",
                    "val_macro_MAE",
                    "val_macro_R2",
                    "val_weighted_R2",
                    "val_macro_log_R2",
                    "val_weighted_log_R2",
                ],
            )
            writer.writeheader()
            for epoch in range(1, self.config["training"]["epochs"] + 1):
                self.model.train()
                losses = []
                for batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
                    batch = batch.to(self.device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    try:
                        autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
                    except (AttributeError, TypeError):
                        autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)
                    with autocast_ctx:
                        pred, aux = self.model(batch)
                        loss = self.loss_fn(pred, batch.y, batch.mask, batch.error_weight, aux)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["gradient_clip_norm"])
                    scaler.step(opt)
                    scaler.update()
                    losses.append(float(loss.detach().cpu()))
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
                    "train_loss": sum(losses) / max(len(losses), 1),
                    "monitor_metric": monitor_metric,
                    "monitor_score": monitor_score,
                    "val_macro_MAE": val_metrics.get("macro_MAE"),
                    "val_macro_R2": val_metrics.get("macro_R2"),
                    "val_weighted_R2": val_metrics.get("weighted_R2"),
                    "val_macro_log_R2": val_metrics.get("macro_log_R2"),
                    "val_weighted_log_R2": val_metrics.get("weighted_log_R2"),
                }
                writer.writerow(row)
                f.flush()
                print(row)
                self.save_checkpoint(ckpt_dir / "last_model.pt", epoch, val_metrics)
                if should_validate:
                    improved = monitor_score > best if maximize_monitor else monitor_score < best
                    if improved:
                        best = monitor_score
                        bad_epochs = 0
                        self.save_checkpoint(ckpt_dir / "best_model.pt", epoch, val_metrics)
                        save_json(val_metrics, metric_dir / "val_metrics.json")
                        save_metrics_csv(val_metrics, metric_dir / "val_metrics.csv", self.config["properties"]["names"])
                        save_log_metrics_csv(val_metrics, metric_dir / "val_metrics_log.csv", self.config["properties"]["names"])
                    else:
                        bad_epochs += 1
                if bad_epochs >= self.config["training"].get("patience", 40):
                    break
        state = torch.load(ckpt_dir / "best_model.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        test_metrics, pred_df, pred_wide_df = evaluate_model(self.model, test_loader, self.device, self.target_scaler, self.clean_df, "test", self.config["properties"]["names"])
        save_json(test_metrics, metric_dir / "test_metrics.json")
        save_metrics_csv(test_metrics, metric_dir / "test_metrics.csv", self.config["properties"]["names"])
        save_log_metrics_csv(test_metrics, metric_dir / "test_metrics_log.csv", self.config["properties"]["names"])
        pred_df.to_csv(pred_dir / "test_predictions.csv", index=False)
        pred_wide_df.to_csv(pred_dir / "test_predictions_wide.csv", index=False)
        return test_metrics

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "condition_scaler": self.condition_scaler,
                "target_scaler": self.target_scaler,
                "property_names": self.config["properties"]["names"],
                "target_means": self.target_scaler.means,
                "target_stds": self.target_scaler.stds,
                "metrics": metrics,
            },
            path,
        )
