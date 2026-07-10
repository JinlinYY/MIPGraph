from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "il_property_prediction"

MODEL_ORDER = [
    "rf",
    "xgboost",
    "lgbm",
    "chemberta",
    "mpnn_concat",
    "gcn",
    "gat",
    "graphsage",
    "gin",
    "mipgraph",
]

DISPLAY_NAMES = {
    "rf": "RF",
    "xgboost": "XGBoost",
    "lgbm": "LGBM",
    "chemberta": "ChemBERTa",
    "mpnn_concat": "MPNN-Concat",
    "gcn": "GCN",
    "gat": "GAT",
    "graphsage": "GraphSAGE",
    "gin": "GIN",
    "mipgraph": "MIPGraph",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline comparison metrics.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DIR / "outputs" / "baseline_comparison_random_point_seed42",
    )
    parser.add_argument(
        "--mipgraph-metrics",
        type=Path,
        default=PROJECT_DIR
        / "outputs"
        / "fg_transformer_random_point_seed42_noamp"
        / "metrics"
        / "unimol2_fg_transformer_random_point_seed42_noamp_resume56"
        / "test_metrics_log.csv",
    )
    return parser.parse_args()


def load_metric_file(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(0, "model", model)
    return df


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def write_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    models = [m for m in MODEL_ORDER if m in set(summary["model"])]
    props = [p for p in summary["property"].drop_duplicates().tolist() if p != "Average"]
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\caption{Predictive performance comparison of baseline models and MIPGraph under the random-point split. Metrics are computed in logarithmic target space.}")
    lines.append("  \\label{tab:comparison_updated}")
    lines.append("  \\resizebox{\\textwidth}{!}{%")
    header = "Property & Metric & " + " & ".join(DISPLAY_NAMES.get(m, m) for m in models) + " \\\\"
    lines.append("  \\begin{tabular}{ll " + "c" * len(models) + "}")
    lines.append("    \\hline")
    lines.append("    " + header)
    lines.append("    \\hline")
    for prop in props:
        for metric in ["log_RMSE", "log_R2"]:
            row = [latex_escape(prop), "RMSE" if metric == "log_RMSE" else "$\\mathrm{R}^2$"]
            for model in models:
                sub = summary[(summary["model"] == model) & (summary["property"] == prop)]
                if sub.empty or pd.isna(sub.iloc[0][metric]):
                    row.append("--")
                else:
                    row.append(f"{float(sub.iloc[0][metric]):.4f}")
            lines.append("    " + " & ".join(row) + " \\\\")
        lines.append("    \\midrule")
    lines.append("  \\end{tabular}%")
    lines.append("  }")
    lines.append("\\end{table*}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    frames = []
    for metric_file in sorted((output_root / "metrics").glob("*/test_metrics_log.csv")):
        model = metric_file.parent.name
        frames.append(load_metric_file(metric_file, model))
    if args.mipgraph_metrics.exists():
        frames.append(load_metric_file(args.mipgraph_metrics, "mipgraph"))
    if not frames:
        raise FileNotFoundError(f"No test_metrics_log.csv files found under {output_root / 'metrics'}")
    long_df = pd.concat(frames, ignore_index=True)
    long_df["model_display"] = long_df["model"].map(DISPLAY_NAMES).fillna(long_df["model"])
    output_root.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(output_root / "baseline_metrics_long.csv", index=False)
    summary = long_df.pivot_table(index=["property"], columns="model", values=["log_RMSE", "log_R2", "log_NMAE"], aggfunc="first")
    summary.to_csv(output_root / "baseline_metrics_summary.csv")
    write_latex_table(long_df, output_root / "baseline_comparison_table.tex")
    print({"long": str(output_root / "baseline_metrics_long.csv"), "tex": str(output_root / "baseline_comparison_table.tex")})


if __name__ == "__main__":
    main()
