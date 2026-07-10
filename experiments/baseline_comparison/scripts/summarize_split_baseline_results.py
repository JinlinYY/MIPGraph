from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "il_property_prediction"

CASE_LABELS = {
    "random_point": "Random-point",
    "random_il_level": "Random IL-level",
    "property_balanced_il_level": "Property-balanced IL-level",
    "ion_family": "Ion-family",
}

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

PROPERTY_ORDER = [
    "Density",
    "Viscosity",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Average",
]

PROPERTY_DISPLAY = {
    "Density": "Density",
    "Viscosity": "Viscosity",
    "ElectricalConductivity": "Elec. Cond.",
    "HeatCapacity": "Heat Capacity",
    "SurfaceTension": "Surface Ten.",
    "ThermalConductivity": "Thermal Cond.",
    "Average": "Average",
}

LATEX_DISPLAY_NAMES = {
    "rf": r"RF~\cite{breiman_random_2001}",
    "xgboost": r"XGBoost~\cite{chen_xgboost_2016}",
    "lgbm": r"LGBM~\cite{ke_lightgbm_2017}",
    "chemberta": r"ChemBERTa~\cite{chithrananda_chemberta_2020}",
    "mpnn_concat": r"MPNN-Concat~\cite{gilmer_neural_2017}",
    "gcn": r"GCN~\cite{kipf_semi-supervised_2017}",
    "gat": r"GAT~\cite{velickovic_graph_2018}",
    "graphsage": r"GraphSAGE~\cite{hamilton_inductive_2018}",
    "gin": r"GIN~\cite{xu_how_2019}",
    "mipgraph": "MIPGraph",
}

PROPERTY_TABLE_MODEL_NAMES = {
    "rf": "RF",
    "xgboost": "XGB",
    "lgbm": "LGBM",
    "chemberta": "ChemBERTa",
    "mpnn_concat": "MPNN",
    "gcn": "GCN",
    "gat": "GAT",
    "graphsage": "SAGE",
    "gin": "GIN",
    "mipgraph": "MIPGraph",
}

METRIC_ORDER = [
    ("R2", "log_R2", r"$R^2$"),
    ("MAE", "log_MAE", "MAE"),
    ("RMSE", "log_RMSE", "RMSE"),
    ("NMAE", "log_NMAE", "NMAE"),
]

MIPGRAPH_METRICS = {
    "random_point": PROJECT_DIR
    / "outputs"
    / "fg_transformer_random_point_seed42_noamp"
    / "metrics"
    / "unimol2_fg_transformer_random_point_seed42_noamp_resume56"
    / "test_metrics_log.csv",
    "random_il_level": PROJECT_DIR
    / "outputs"
    / "split_strategy_comparison_seed42"
    / "metrics"
    / "il_level_random_seed42"
    / "test_metrics_log.csv",
    "property_balanced_il_level": PROJECT_DIR
    / "outputs"
    / "split_strategy_comparison_seed42"
    / "metrics"
    / "il_level_property_balanced_seed42"
    / "test_metrics_log.csv",
    "ion_family": PROJECT_DIR
    / "outputs"
    / "split_strategy_comparison_seed42"
    / "metrics"
    / "il_level_family_pair_seed42"
    / "test_metrics_log.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline metrics across split strategies.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DIR / "outputs" / "baseline_comparison_by_split_seed42",
    )
    parser.add_argument(
        "--random-point-root",
        type=Path,
        default=PROJECT_DIR / "outputs" / "baseline_comparison_random_point_seed42",
    )
    return parser.parse_args()


def average_row(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df.loc[df["property"].astype(str).str.lower() == "average"]
    if row.empty:
        return None
    item = row.iloc[0]
    return {
        "macro_log_MAE": float(item["log_MAE"]),
        "macro_log_RMSE": float(item["log_RMSE"]),
        "macro_log_R2": float(item["log_R2"]),
        "macro_log_NMAE": float(item["log_NMAE"]),
    }


def load_metric_rows(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    expected = {"property", "log_MAE", "log_RMSE", "log_R2", "log_NMAE"}
    if not expected.issubset(df.columns):
        return None
    return df[["property", "log_R2", "log_MAE", "log_RMSE", "log_NMAE"]].copy()


def metric_path(output_root: Path, random_point_root: Path, case: str, model: str) -> Path:
    if case == "random_point":
        return random_point_root / "metrics" / model / "test_metrics_log.csv"
    return output_root / case / "metrics" / model / "test_metrics_log.csv"


def collect(output_root: Path, random_point_root: Path) -> pd.DataFrame:
    rows = []
    for case in CASE_LABELS:
        for model in MODEL_ORDER:
            if model == "mipgraph":
                avg = average_row(MIPGRAPH_METRICS[case])
            else:
                avg = average_row(metric_path(output_root, random_point_root, case, model))
            if avg is None:
                continue
            rows.append(
                {
                    "case": case,
                    "split": CASE_LABELS[case],
                    "model": model,
                    "model_label": DISPLAY_NAMES[model],
                    **avg,
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    case_order = {case: i for i, case in enumerate(CASE_LABELS)}
    model_order = {model: i for i, model in enumerate(MODEL_ORDER)}
    df["case_order"] = df["case"].map(case_order)
    df["model_order"] = df["model"].map(model_order)
    return df.sort_values(["case_order", "model_order"]).drop(columns=["case_order", "model_order"])


def collect_property_metrics(output_root: Path, random_point_root: Path) -> pd.DataFrame:
    rows = []
    for case in CASE_LABELS:
        for model in MODEL_ORDER:
            if model == "mipgraph":
                metrics = load_metric_rows(MIPGRAPH_METRICS[case])
            else:
                metrics = load_metric_rows(metric_path(output_root, random_point_root, case, model))
            if metrics is None:
                continue
            for _, item in metrics.iterrows():
                prop = str(item["property"])
                rows.append(
                    {
                        "case": case,
                        "split": CASE_LABELS[case],
                        "property": prop,
                        "property_label": PROPERTY_DISPLAY.get(prop, prop),
                        "model": model,
                        "model_label": DISPLAY_NAMES[model],
                        "log_MAE": float(item["log_MAE"]),
                        "log_RMSE": float(item["log_RMSE"]),
                        "log_R2": float(item["log_R2"]),
                        "log_NMAE": float(item["log_NMAE"]),
                    }
                )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    case_order = {case: i for i, case in enumerate(CASE_LABELS)}
    prop_order = {prop: i for i, prop in enumerate(PROPERTY_ORDER)}
    model_order = {model: i for i, model in enumerate(MODEL_ORDER)}
    df["case_order"] = df["case"].map(case_order)
    df["property_order"] = df["property"].map(prop_order).fillna(len(PROPERTY_ORDER))
    df["model_order"] = df["model"].map(model_order)
    return df.sort_values(["case_order", "property_order", "model_order"]).drop(
        columns=["case_order", "property_order", "model_order"]
    )


def write_outputs(summary: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "split_baseline_macro_long.csv", index=False)
    rows = []
    for case in CASE_LABELS:
        sub = summary[summary["case"] == case].set_index("model")
        row = {"Split": CASE_LABELS[case]}
        for model in MODEL_ORDER:
            if model in sub.index:
                item = sub.loc[model]
                row[DISPLAY_NAMES[model]] = f"{item['macro_log_R2']:.4f}/{item['macro_log_NMAE']:.4f}"
            else:
                row[DISPLAY_NAMES[model]] = "--"
        rows.append(row)
    wide = pd.DataFrame(rows)
    wide.to_csv(output_root / "split_baseline_macro_wide.csv", index=False)

    columns = ["Split"] + [LATEX_DISPLAY_NAMES[model] for model in MODEL_ORDER]
    with (output_root / "split_baseline_macro_table.tex").open("w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("  \\centering\n")
        f.write(
            "  \\caption{Macro-averaged comparison of baseline models and MIPGraph across split strategies. "
            "Each entry reports log-space $R^2$/NMAE on the test set.}\n"
        )
        f.write("  \\label{tab:split_baseline_macro}\n")
        f.write("  \\scriptsize\n")
        f.write("  \\resizebox{\\textwidth}{!}{%\n")
        f.write("  \\begin{tabular}{l" + "c" * (len(columns) - 1) + "}\n")
        f.write("    \\hline\n")
        f.write("    " + " & ".join(columns) + " \\\\\n")
        f.write("    \\hline\n")
        for _, row in wide.iterrows():
            values = [str(row["Split"])] + [str(row[DISPLAY_NAMES[model]]) for model in MODEL_ORDER]
            f.write("    " + " & ".join(values) + " \\\\\n")
        f.write("    \\hline\n")
        f.write("  \\end{tabular}%\n")
        f.write("  }\n")
        f.write("\\end{table*}\n")


def write_property_outputs(property_summary: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    property_summary.to_csv(output_root / "split_baseline_property_long.csv", index=False)
    rows = []
    for case in CASE_LABELS:
        for prop in PROPERTY_ORDER:
            sub = property_summary[
                (property_summary["case"] == case) & (property_summary["property"] == prop)
            ].set_index("model")
            if sub.empty:
                continue
            selected_metrics = (
                [("NMAE", "log_NMAE", "NMAE")]
                if prop == "Average"
                else [("MAE", "log_MAE", "MAE")]
            )
            for metric_label, metric_column, _ in selected_metrics:
                row = {
                    "Split": CASE_LABELS[case],
                    "Property": PROPERTY_DISPLAY.get(prop, prop),
                    "Metric": metric_label,
                }
                for model in MODEL_ORDER:
                    if model in sub.index:
                        item = sub.loc[model]
                        row[DISPLAY_NAMES[model]] = f"{item[metric_column]:.4f}"
                    else:
                        row[DISPLAY_NAMES[model]] = "--"
                rows.append(row)
    wide = pd.DataFrame(rows)
    wide.to_csv(output_root / "split_baseline_property_wide.csv", index=False)

    columns = ["Split", "Property", "Metric"] + [PROPERTY_TABLE_MODEL_NAMES[model] for model in MODEL_ORDER]
    with (output_root / "split_baseline_property_table.tex").open("w", encoding="utf-8") as f:
        f.write("\\begin{table*}[p]\n")
        f.write("  \\centering\n")
        f.write(
            "  \\caption{Property-level comparison of baseline models and MIPGraph across split strategies. "
            "Rows for individual properties report log-space MAE on the test set, whereas Average rows report macro NMAE over the six properties. "
            "Model abbreviations follow Table~\\ref{tab:comparison_extended}.}\n"
        )
        f.write("  \\label{tab:split_baseline_property}\n")
        f.write("  \\scriptsize\n")
        f.write("  \\setlength{\\tabcolsep}{2pt}\n")
        f.write("  \\renewcommand{\\arraystretch}{0.9}\n")
        f.write("  \\resizebox{\\textwidth}{!}{%\n")
        f.write("  \\begin{tabular}{lll" + "c" * len(MODEL_ORDER) + "}\n")
        f.write("    \\hline\n")
        f.write("    " + " & ".join(columns) + " \\\\\n")
        f.write("    \\hline\n")
        previous_split = None
        previous_property = None
        for _, row in wide.iterrows():
            split_text = str(row["Split"]) if row["Split"] != previous_split else ""
            previous_split = row["Split"]
            property_text = str(row["Property"]) if row["Property"] != previous_property or split_text else ""
            previous_property = row["Property"]
            metric_text = next(
                latex_label for label, _, latex_label in METRIC_ORDER if label == row["Metric"]
            )
            numeric_values = [
                float(row[DISPLAY_NAMES[model]])
                for model in MODEL_ORDER
                if row[DISPLAY_NAMES[model]] != "--"
            ]
            best_value = min(numeric_values) if numeric_values else None
            formatted_values = []
            for model in MODEL_ORDER:
                cell = str(row[DISPLAY_NAMES[model]])
                if best_value is not None and cell != "--" and abs(float(cell) - best_value) < 1e-12:
                    cell = f"\\textbf{{{cell}}}"
                formatted_values.append(cell)
            values = [split_text, property_text, metric_text] + formatted_values
            f.write("    " + " & ".join(values) + " \\\\\n")
            if row["Property"] == PROPERTY_DISPLAY["Average"]:
                f.write("    \\hline\n")
        f.write("  \\end{tabular}%\n")
        f.write("  }\n")
        f.write("\\end{table*}\n")


def main() -> None:
    args = parse_args()
    summary = collect(args.output_root.resolve(), args.random_point_root.resolve())
    if summary.empty:
        raise FileNotFoundError("No split baseline metrics were found.")
    write_outputs(summary, args.output_root.resolve())
    property_summary = collect_property_metrics(args.output_root.resolve(), args.random_point_root.resolve())
    if property_summary.empty:
        raise FileNotFoundError("No property-level split baseline metrics were found.")
    write_property_outputs(property_summary, args.output_root.resolve())
    print({"long": str(args.output_root / "split_baseline_macro_long.csv")})
    print({"table": str(args.output_root / "split_baseline_macro_table.tex")})
    print({"property_table": str(args.output_root / "split_baseline_property_table.tex")})


if __name__ == "__main__":
    main()
