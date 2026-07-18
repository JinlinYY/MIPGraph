"""Publication-style, data-driven Figure 5 generation using Matplotlib only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model_adapter import PROPERTY_UNITS


PANEL_FILES = {
    "a": "panel_a_workflow",
    "b": "panel_b_funnel",
    "c": "panel_c_properties",
    "d": "panel_d_proxies",
    "e": "panel_e_applicability_domain",
    "f": "panel_f_constraints",
    "g": "panel_g_pareto",
    "h": "panel_h_candidates",
}

COLORS = {
    "blue": "#2864A8",
    "orange": "#DB7C26",
    "green": "#2D8A5B",
    "red": "#B8483E",
    "purple": "#7653A6",
    "gray": "#66717E",
    "light": "#E8EDF3",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _empty(ax: mpl.axes.Axes, message: str) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", color=COLORS["gray"], wrap=True)


def _title(ax: mpl.axes.Axes, letter: str, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=6)
    ax.text(
        -0.08,
        1.08,
        letter.upper(),
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def _style_axis(ax: mpl.axes.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#D7DCE2", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)


def _select_curve_candidates(predictions: pd.DataFrame, final: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    if not final.empty:
        selected.extend(final["candidate_id"].astype(str).head(3).tolist())
    references = predictions[predictions["candidate_type"].eq("observed_reference")]
    selected.extend(references["candidate_id"].astype(str).drop_duplicates().head(2).tolist())
    return list(dict.fromkeys(selected))[:5]


def _panel_a(container: Any) -> None:
    ax = container.subplots()
    _title(ax, "a", "Auditable MIPGraph application workflow")
    ax.set_axis_off()
    labels = [
        "Current data\n+ split audit",
        "Supported ion\nrecombination",
        "Six-property\ninference",
        "Proxy + curve\nquality audit",
        "AD + hard\nconstraints",
        "Pareto leads +\nsubstitutions",
    ]
    x = np.linspace(0.08, 0.92, len(labels))
    for index, (position, label) in enumerate(zip(x, labels)):
        color = COLORS["blue"] if index < 3 else COLORS["green"]
        ax.text(
            position,
            0.5,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=7.2,
            bbox={"boxstyle": "round,pad=0.38", "fc": color, "ec": "white", "lw": 0.8},
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x[index + 1] - 0.065, 0.5),
                xytext=(position + 0.065, 0.5),
                arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 1.1},
            )
    ax.text(
        0.5,
        0.17,
        "Every stage writes a schema-checked artefact and a resumable completion marker.",
        ha="center",
        fontsize=7.2,
        color=COLORS["gray"],
    )


def _panel_b(container: Any, paths: dict[str, Path]) -> None:
    ax = container.subplots()
    _title(ax, "b", "Candidate-screening funnel")
    generation = _read_json(paths["steps"] / "candidate_generation.json")
    inference = _read_json(paths["steps"] / "inference.json")
    screening = _read_json(paths["steps"] / "screening.json")
    pareto = _read_json(paths["steps"] / "pareto.json")
    labels = ["Theoretical", "Unseen retained", "Inferred", "Hard-pass", "Pareto-1", "Final"]
    values = [
        generation.get("theoretical_combinations", 0),
        generation.get("unseen_candidates", 0),
        max(0, inference.get("successful_predictions", 0) // max(1, len(_read_csv(paths["data"] / "property_predictions_long.csv")["temperature_K"].unique()) if (paths["data"] / "property_predictions_long.csv").exists() else 1) - generation.get("observed_references", 0)),
        screening.get("hard_constraint_pass", 0),
        pareto.get("pareto_rank_1", 0),
        pareto.get("final_recommendations", 0),
    ]
    if not any(values):
        _empty(ax, "No completed funnel outputs")
        return
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, values, color=[COLORS["gray"], COLORS["blue"], COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]])
    ax.set_yticks(y, labels)
    ax.set_xlabel("Number of ion pairs")
    for bar, value in zip(bars, values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {int(value):,}", va="center", fontsize=7)
    _style_axis(ax)


def _panel_c(container: Any, paths: dict[str, Path]) -> None:
    axes = np.asarray(container.subplots(2, 3, sharex=True)).ravel()
    predictions = _read_csv(paths["data"] / "property_predictions_long.csv")
    final = _read_csv(paths["data"] / "final_prioritized_candidates.csv")
    if predictions.empty:
        for ax in axes:
            _empty(ax, "No predictions")
        return
    if "analysis_window" in predictions:
        predictions = predictions[predictions["analysis_window"].eq("main")]
    selected = _select_curve_candidates(predictions, final)
    flags = _read_csv(paths["data"] / "curve_quality_flags.csv")
    negative: str | None = None
    if not flags.empty:
        severe = flags[flags["severity"].eq("severe")]
        if not severe.empty:
            negative = str(severe["candidate_id"].value_counts().index[0])
            if negative not in selected:
                selected.append(negative)
    palette = [
        COLORS["purple"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["blue"],
        COLORS["gray"],
        COLORS["red"],
    ]
    for property_name, ax in zip(PROPERTY_UNITS, axes):
        for color, candidate_id in zip(palette, selected):
            group = predictions[predictions["candidate_id"].astype(str).eq(candidate_id)].sort_values("temperature_K")
            if group.empty:
                continue
            label = candidate_id if len(candidate_id) <= 16 else candidate_id[:13] + "…"
            if candidate_id == negative:
                label += " (curve-flag example)"
            ax.plot(
                group["temperature_K"],
                group[property_name],
                marker="o",
                ms=2.2,
                lw=1.0,
                ls="--" if candidate_id == negative else "-",
                color=color,
                label=label,
            )
        ax.set_title(property_name, fontsize=7.5)
        ax.set_ylabel(PROPERTY_UNITS[property_name], fontsize=6.3)
        ax.tick_params(labelsize=6.1)
        _style_axis(ax)
    for ax in axes[-3:]:
        ax.set_xlabel("Temperature (K)", fontsize=6.5)
    axes[0].text(-0.31, 1.25, "C", transform=axes[0].transAxes, fontsize=12, fontweight="bold", va="top")
    axes[0].text(-0.16, 1.25, "Six predicted properties across the full window", transform=axes[0].transAxes, fontsize=9.5, fontweight="bold", va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, fontsize=5.3, frameon=False, loc="best")


def _panel_d(container: Any, paths: dict[str, Path]) -> None:
    axes = np.asarray(container.subplots(2, 2, sharex=True)).ravel()
    proxies = _read_csv(paths["data"] / "application_proxies_temperature.csv")
    final = _read_csv(paths["data"] / "final_prioritized_candidates.csv")
    if proxies.empty:
        for ax in axes:
            _empty(ax, "No proxy outputs")
        return
    if "analysis_window" in proxies:
        proxies = proxies[proxies["analysis_window"].eq("main")]
    selected = _select_curve_candidates(proxies, final)[:4]
    definitions = [
        ("transport_favorability", "Transport favorability", "standardized score"),
        ("volumetric_heat_capacity", "Volumetric heat capacity", "J m$^{-3}$ K$^{-1}$"),
        ("thermal_diffusivity", "Thermal diffusivity", "m$^2$ s$^{-1}$"),
        ("interfacial_window_deviation", "Interfacial-window deviation", "reference IQR"),
    ]
    palette = [COLORS["purple"], COLORS["orange"], COLORS["green"], COLORS["blue"]]
    for (column, title, unit), ax in zip(definitions, axes):
        for color, candidate_id in zip(palette, selected):
            group = proxies[proxies["candidate_id"].astype(str).eq(candidate_id)].sort_values("temperature_K")
            ax.plot(group["temperature_K"], group[column], marker="o", ms=2.0, lw=1.0, color=color)
        ax.set_title(title, fontsize=7.3)
        ax.set_ylabel(unit, fontsize=6.3)
        ax.tick_params(labelsize=6.1)
        _style_axis(ax)
    for ax in axes[-2:]:
        ax.set_xlabel("Temperature (K)", fontsize=6.5)
    axes[0].text(-0.31, 1.25, "D", transform=axes[0].transAxes, fontsize=12, fontweight="bold", va="top")
    axes[0].text(-0.16, 1.25, "Application-proxy response", transform=axes[0].transAxes, fontsize=9.5, fontweight="bold", va="top")


def _panel_e(container: Any, paths: dict[str, Path]) -> None:
    ax = container.subplots()
    _title(ax, "e", "Applicability domain and transport risk")
    ad = _read_csv(paths["data"] / "applicability_domain.csv")
    robust = _read_csv(paths["data"] / "candidate_robust_summary.csv")
    if ad.empty or robust.empty:
        _empty(ax, "No applicability-domain outputs")
        return
    merged = ad.merge(robust[["candidate_id", "transport_favorability_worst"]], on="candidate_id", how="inner")
    styles = {
        "in_domain": (COLORS["green"], "o"),
        "borderline": (COLORS["orange"], "^"),
        "out_of_domain": (COLORS["red"], "x"),
    }
    for status, group in merged.groupby("AD_status"):
        color, marker = styles.get(str(status), (COLORS["gray"], "o"))
        ax.scatter(group["descriptor_distance_percentile"], group["transport_favorability_worst"], s=23, alpha=0.75, color=color, marker=marker, label=f"{status} (n={len(group)})")
    ax.axvline(0.90, color=COLORS["gray"], ls="--", lw=0.8)
    ax.axvline(0.95, color=COLORS["red"], ls=":", lw=0.8)
    ax.set_xlabel("Descriptor-distance empirical percentile")
    ax.set_ylabel("Worst-window transport favorability")
    ax.legend(frameon=False, fontsize=6.3)
    _style_axis(ax)


def _panel_f(container: Any, paths: dict[str, Path]) -> None:
    ax = container.subplots()
    _title(ax, "f", "Hard-constraint pass matrix")
    trace = _read_csv(paths["data"] / "screening_trace.csv")
    pass_columns = [column for column in trace if column.startswith("pass_")]
    if trace.empty or not pass_columns:
        _empty(ax, "No screening trace")
        return
    subset = trace[trace["candidate_type"].eq("unseen_pair_recombination")].copy()
    if subset.empty:
        subset = trace.copy()
    subset = subset.sort_values(["final_feasible", "candidate_id"], ascending=[False, True]).head(30)
    matrix = subset[pass_columns].astype(float).to_numpy()
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=mpl.colors.ListedColormap(["#D45B50", "#4C9A6A"]), vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pass_columns)), [column.removeprefix("pass_").replace("_", "\n") for column in pass_columns], rotation=45, ha="right", fontsize=5.7)
    ax.set_yticks(np.arange(len(subset)), subset["candidate_id"].astype(str).str.slice(0, 15), fontsize=5.4)
    ax.set_xlabel("Red = fail; green = pass")


def _panel_g(container: Any, paths: dict[str, Path]) -> None:
    ax = container.subplots()
    _title(ax, "g", "Feasible candidates in transport objective space")
    ranked = _read_csv(paths["data"] / "pareto_candidates.csv")
    trace = _read_csv(paths["data"] / "screening_trace.csv")
    final = _read_csv(paths["data"] / "final_prioritized_candidates.csv")
    if ranked.empty:
        _empty(ax, "No feasible unseen candidates after hard constraints")
        return
    references = trace[trace["candidate_type"].eq("observed_reference")] if not trace.empty else pd.DataFrame()
    if not references.empty:
        ax.scatter(
            references["viscosity_worst"],
            references["conductivity_worst"],
            s=18,
            color=COLORS["gray"],
            marker="s",
            alpha=0.45,
            label=f"observed reference (n={len(references)})",
        )
    size_source = pd.to_numeric(ranked["volumetric_heat_capacity_worst"], errors="coerce")
    size_norm = (size_source - size_source.min()) / max(float(size_source.max() - size_source.min()), 1.0e-12)
    sizes = 25 + 90 * size_norm
    colors = pd.to_numeric(ranked["thermal_diffusivity_worst"], errors="coerce")
    scatter = None
    for status, marker in {"in_domain": "o", "borderline": "^", "out_of_domain": "x"}.items():
        mask = ranked["AD_status"].eq(status)
        if not mask.any():
            continue
        scatter = ax.scatter(
            ranked.loc[mask, "viscosity_worst"],
            ranked.loc[mask, "conductivity_worst"],
            s=np.asarray(sizes)[mask.to_numpy()],
            c=colors[mask],
            cmap="viridis",
            vmin=float(colors.min()),
            vmax=float(colors.max()),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
            marker=marker,
            label=f"unseen {status} (n={int(mask.sum())})",
        )
    front = ranked[ranked["Pareto_rank"].eq(1)].sort_values("viscosity_worst")
    if len(front) > 1:
        ax.plot(front["viscosity_worst"], front["conductivity_worst"], color=COLORS["purple"], lw=1.0, ls="--", label="Pareto rank 1")
    class_colors = {
        "balanced": COLORS["purple"],
        "high_transport": COLORS["red"],
        "thermal_robust": COLORS["orange"],
        "exploratory": COLORS["blue"],
    }
    for recommendation, group in final.groupby("recommendation_class") if not final.empty else []:
        ax.scatter(
            group["viscosity_worst"],
            group["conductivity_worst"],
            s=90,
            facecolors="none",
            edgecolors=class_colors.get(str(recommendation), "black"),
            linewidths=1.3,
            marker="o",
            label=str(recommendation),
        )
    for row in front.head(5).itertuples(index=False):
        ax.annotate(str(row.candidate_id)[:10], (row.viscosity_worst, row.conductivity_worst), xytext=(3, 3), textcoords="offset points", fontsize=5.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Worst-window viscosity (Pa s; lower is better)")
    ax.set_ylabel("Worst-window conductivity (S m$^{-1}$; higher is better)")
    if scatter is not None:
        colorbar = container.colorbar(scatter, ax=ax, fraction=0.05, pad=0.03)
        colorbar.set_label("Worst thermal diffusivity (m$^2$ s$^{-1}$)", fontsize=6)
    ax.legend(frameon=False, fontsize=5.2, loc="best", ncol=2)
    _style_axis(ax)


def _panel_h(container: Any, paths: dict[str, Path]) -> None:
    ax = container.subplots()
    _title(ax, "h", "Prioritized candidate classes and trade-offs")
    final = _read_csv(paths["data"] / "final_prioritized_candidates.csv")
    ax.set_axis_off()
    if final.empty:
        ax.text(0.5, 0.5, "No final candidates satisfied the complete decision rule.", ha="center", va="center", color=COLORS["gray"])
        return
    columns = ["candidate_id", "cation_smiles", "anion_smiles", "candidate_type", "Pareto_rank", "AD_status", "recommendation_class", "main_advantage", "main_limitation", "downstream_priority"]
    labels = ["Candidate", "Cation", "Anion", "Novelty", "Rank", "AD", "Class", "Advantage", "Limitation", "Priority"]
    display = final[columns].head(8).astype(str).copy()
    display["candidate_id"] = display["candidate_id"].str.slice(0, 13)
    display["cation_smiles"] = display["cation_smiles"].str.slice(0, 13)
    display["anion_smiles"] = display["anion_smiles"].str.slice(0, 13)
    display["candidate_type"] = display["candidate_type"].str.replace("unseen_pair_recombination", "unseen pair", regex=False)
    display["downstream_priority"] = display["downstream_priority"].str.slice(0, 24)
    table = ax.table(cellText=display.to_numpy(), colLabels=labels, loc="center", cellLoc="left", colLoc="left", colWidths=[0.09, 0.12, 0.11, 0.09, 0.05, 0.08, 0.10, 0.12, 0.12, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(4.5)
    table.scale(1.0, 1.3)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D7DCE2")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(COLORS["blue"])
            cell.set_text_props(color="white", weight="bold")


def _renderers(paths: dict[str, Path]) -> dict[str, Callable[[Any], None]]:
    return {
        "a": _panel_a,
        "b": lambda container: _panel_b(container, paths),
        "c": lambda container: _panel_c(container, paths),
        "d": lambda container: _panel_d(container, paths),
        "e": lambda container: _panel_e(container, paths),
        "f": lambda container: _panel_f(container, paths),
        "g": lambda container: _panel_g(container, paths),
        "h": lambda container: _panel_h(container, paths),
    }


def _save(fig: mpl.figure.Figure, base: Path, formats: list[str], dpi: int) -> list[str]:
    outputs = []
    for extension in formats:
        target = base.with_suffix(f".{extension}")
        fig.savefig(target, dpi=dpi if extension.lower() == "png" else None, bbox_inches="tight", facecolor="white")
        outputs.append(str(target))
    plt.close(fig)
    return outputs


def generate_figure5(paths: dict[str, Path], config: dict[str, Any]) -> dict[str, Any]:
    """Generate the combined eight-panel Figure 5 and every panel separately."""

    figure_config = config["figures"]
    formats = [str(value).lower() for value in figure_config["formats"]]
    unsupported = sorted(set(formats) - {"png", "pdf"})
    if unsupported:
        raise ValueError(f"Unsupported figure formats: {unsupported}")
    dpi = int(figure_config["dpi"])
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.3,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    renderers = _renderers(paths)
    output_files: list[str] = []
    if bool(figure_config.get("make_individual_panels", True)):
        for letter, renderer in renderers.items():
            size = (7.2, 4.6) if letter in {"c", "d"} else (6.4, 4.1)
            fig = plt.figure(figsize=size, constrained_layout=True)
            renderer(fig)
            output_files.extend(_save(fig, paths["figures"] / PANEL_FILES[letter], formats, dpi))
    if bool(figure_config.get("make_combined_figure", True)):
        fig = plt.figure(figsize=(14.0, 18.0), constrained_layout=True)
        subfigures = np.asarray(fig.subfigures(4, 2)).ravel()
        for subfigure, renderer in zip(subfigures, renderers.values()):
            renderer(subfigure)
        fig.suptitle(
            "Figure 5 | MIPGraph-guided multi-property screening of unseen ionic-liquid pairs",
            fontsize=14,
            fontweight="bold",
        )
        output_files.extend(_save(fig, paths["figures"] / "figure5_computational_application_case", formats, dpi))
    return {"figure_files": output_files, "panel_count": 8, "dpi": dpi, "formats": formats}
