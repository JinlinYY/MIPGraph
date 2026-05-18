from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    default_output = repo_root / "exp2_performance" / "outputs"
    parser = argparse.ArgumentParser(description="Build CSV and LaTeX summary table for metrics.")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MAE"] = out["MAE"].map(lambda x: f"{x:.4f}")
    out["RMSE"] = out["RMSE"].map(lambda x: f"{x:.4f}")
    out["R2"] = out["R2"].map(lambda x: f"{x:.4f}")
    out["MAPE"] = out["MAPE"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
    out["N"] = out["N"].astype(int)
    return out


def make_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="lrrrrr",
        caption=caption,
        label=label,
    )
    latex = latex.replace("\\toprule", "\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline")
    return latex


def main() -> None:
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    metrics_path = tables_dir / "metrics_by_split_property.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} does not exist. Please run evaluate.py first."
        )
    metrics = pd.read_csv(metrics_path)
    split_df = metrics[metrics["split"] == args.split].copy()
    if split_df.empty:
        raise ValueError(f"No metrics found for split={args.split!r}")

    metric_space = "raw"
    if "metric_space" in split_df.columns and not split_df["metric_space"].dropna().empty:
        metric_space = str(split_df["metric_space"].dropna().iloc[0])

    split_df = split_df[["property", "MAE", "RMSE", "R2", "MAPE", "N"]].sort_values("property").reset_index(drop=True)
    formatted = format_metric_table(split_df)
    formatted_path = tables_dir / "metrics_summary_formatted.csv"
    try:
        formatted.to_csv(formatted_path, index=False)
    except PermissionError:
        print(f"Skipped locked auxiliary table: {formatted_path}")

    caption_prefix = "Log-space performance" if metric_space == "log" else "Performance"
    latex = make_latex_table(
        formatted,
        caption=f"{caption_prefix} summary on the {args.split} split.",
        label=f"tab:performance_{args.split}",
    )
    tex_path = tables_dir / "results_table.tex"
    tex_path.write_text(latex, encoding="utf-8")

    project_level_tex = args.output_dir.parent / "results_table.tex"
    project_level_tex.write_text(latex, encoding="utf-8")
    print(f"Saved LaTeX table: {tex_path}")
    print(f"Saved LaTeX table copy: {project_level_tex}")


if __name__ == "__main__":
    main()
