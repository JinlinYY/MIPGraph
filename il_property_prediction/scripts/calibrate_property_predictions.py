from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _load_xy(path: str | Path, prop: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    pred_col = f"{prop}_pred"
    true_col = f"{prop}_true"
    if pred_col in df.columns and true_col in df.columns:
        g = df[[true_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
        return g[true_col].to_numpy(dtype=float), g[pred_col].to_numpy(dtype=float), df
    required = {"property", "y_true", "y_pred"}
    if required.issubset(df.columns):
        g = df[df["property"] == prop][["y_true", "y_pred"]].replace([np.inf, -np.inf], np.nan).dropna()
        return g["y_true"].to_numpy(dtype=float), g["y_pred"].to_numpy(dtype=float), df
    raise ValueError(f"{path} is neither wide prediction CSV nor long prediction CSV for {prop}")


def _fit_predict(method: str, y_val: np.ndarray, pred_val: np.ndarray, pred_test: np.ndarray):
    if method == "identity":
        return pred_test, None
    if method == "raw_affine":
        model = LinearRegression()
        model.fit(pred_val.reshape(-1, 1), y_val)
        return model.predict(pred_test.reshape(-1, 1)), model
    if method == "log_affine":
        valid = (y_val > 0) & (pred_val > 0)
        if valid.sum() < 2 or np.any(pred_test <= 0):
            raise ValueError("log_affine requires positive validation and test predictions")
        model = LinearRegression()
        model.fit(np.log(pred_val[valid]).reshape(-1, 1), np.log(y_val[valid]))
        return np.exp(model.predict(np.log(pred_test).reshape(-1, 1))), model
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(pred_val, y_val)
        return model.predict(pred_test), model
    if method == "log_isotonic":
        valid = (y_val > 0) & (pred_val > 0)
        if valid.sum() < 2 or np.any(pred_test <= 0):
            raise ValueError("log_isotonic requires positive validation and test predictions")
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(np.log(pred_val[valid]), np.log(y_val[valid]))
        return np.exp(model.predict(np.log(pred_test))), model
    raise ValueError(f"Unknown calibration method: {method}")


def _apply_to_dataframe(df: pd.DataFrame, prop: str, calibrated: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    pred_col = f"{prop}_pred"
    true_col = f"{prop}_true"
    err_col = f"{prop}_absolute_error"
    if pred_col in out.columns and true_col in out.columns:
        valid = out[[true_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna().index
        if len(valid) != len(calibrated):
            raise ValueError("Calibrated prediction count does not match valid wide rows")
        out.loc[valid, pred_col] = calibrated
        out.loc[valid, err_col] = np.abs(out.loc[valid, pred_col].to_numpy(dtype=float) - out.loc[valid, true_col].to_numpy(dtype=float))
        return out
    if {"property", "y_true", "y_pred", "absolute_error"}.issubset(out.columns):
        valid = out[(out["property"] == prop) & out["y_true"].notna() & out["y_pred"].notna()].index
        if len(valid) != len(calibrated):
            raise ValueError("Calibrated prediction count does not match valid long rows")
        out.loc[valid, "y_pred"] = calibrated
        out.loc[valid, "absolute_error"] = np.abs(out.loc[valid, "y_pred"].to_numpy(dtype=float) - out.loc[valid, "y_true"].to_numpy(dtype=float))
        return out
    raise ValueError("Unsupported prediction CSV format")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-fitted post-hoc calibration for one property prediction.")
    parser.add_argument("--property", required=True)
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--method", choices=["auto", "identity", "raw_affine", "log_affine", "isotonic", "log_isotonic"], default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    y_val, pred_val, _ = _load_xy(args.val_predictions, args.property)
    y_test, pred_test, test_df = _load_xy(args.test_predictions, args.property)
    methods = ["identity", "raw_affine", "log_affine", "isotonic", "log_isotonic"] if args.method == "auto" else [args.method]

    candidates = []
    for method in methods:
        try:
            pred_val_cal, _ = _fit_predict(method, y_val, pred_val, pred_val)
            pred_test_cal, model = _fit_predict(method, y_val, pred_val, pred_test)
        except ValueError as exc:
            print(f"skip {method}: {exc}")
            continue
        val_metrics = _metrics(y_val, pred_val_cal)
        test_metrics = _metrics(y_test, pred_test_cal)
        candidates.append((method, val_metrics, test_metrics, pred_test_cal, model))
    if not candidates:
        raise RuntimeError("No calibration method was valid")

    best = max(candidates, key=lambda item: item[1]["R2"])
    method, val_metrics, test_metrics, pred_test_cal, _ = best
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calibrated_df = _apply_to_dataframe(test_df, args.property, pred_test_cal)
    calibrated_df.to_csv(out_dir / f"test_predictions_{args.property}_calibrated.csv", index=False)

    with (out_dir / f"{args.property}_calibration_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "method", "MAE", "RMSE", "R2"])
        writer.writeheader()
        writer.writerow({"split": "val", "method": method, "MAE": _fmt(val_metrics["MAE"]), "RMSE": _fmt(val_metrics["RMSE"]), "R2": _fmt(val_metrics["R2"])})
        writer.writerow({"split": "test", "method": method, "MAE": _fmt(test_metrics["MAE"]), "RMSE": _fmt(test_metrics["RMSE"]), "R2": _fmt(test_metrics["R2"])})
    print({"selected_method": method, "val": val_metrics, "test": test_metrics})


if __name__ == "__main__":
    main()
