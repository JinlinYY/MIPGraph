from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIG_ROOT = PROJECT_ROOT / "LaTex-MIPGraph" / "Fig"
SOURCE_DIR = FIG_ROOT / "source_data"
OUTPUT_PREFIX = FIG_ROOT / "figure_application_case_agent_merged"

CASE_ORDER = ["[BMIM][NTf2]", "[BMIM][BF4]"]
CASE_COLORS = {"[BMIM][NTf2]": "#0B6E99", "[BMIM][BF4]": "#C77700"}
CASE_SHORT = {"BMIM_NTf2": r"NTf$_2$", "BMIM_BF4": r"BF$_4$"}
CASE_LABEL = {"BMIM_NTf2": r"[BMIM][NTf$_2$]", "BMIM_BF4": r"[BMIM][BF$_4$]"}
VISCOSITY_LIMIT_MPA_S = 10.0
CONDUCTIVITY_LIMIT_S_M = 2.0


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 11.0,
            "axes.labelsize": 10.4,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.8,
            "axes.linewidth": 0.72,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
        }
    )


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15.0,
        fontweight="bold",
    )


def rounded_box(ax, x, y, w, h, title, subtitle="", face="#FFFFFF", edge="#B5B5B5", lw=0.75) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.64, title, transform=ax.transAxes, ha="center", va="center", fontsize=8.6, fontweight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.28, subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=6.9, color="#555555")


def arrow(ax, x0, y0, x1, y1, color="#6E6E6E") -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=10.5,
            linewidth=0.9,
            color=color,
            transform=ax.transAxes,
        )
    )


def load_tables(source_dir: Path = SOURCE_DIR) -> dict[str, pd.DataFrame]:
    return {
        "response": pd.read_csv(source_dir / "figure_application_case_b_response_matrix.csv"),
        "fingerprint": pd.read_csv(source_dir / "figure_application_case_c_fingerprint.csv"),
        "gate_shift": pd.read_csv(source_dir / "figure_application_case_d_gate_shift.csv"),
        "trace": pd.read_csv(source_dir / "figure_agent_operation_panel_a_action_trace.csv"),
        "validation": pd.read_csv(source_dir / "figure_agent_operation_panel_b_query_validation.csv"),
        "ranking": pd.read_csv(source_dir / "figure_agent_operation_panel_c_topk_ranking.csv"),
        "constraints": pd.read_csv(source_dir / "figure_agent_operation_panel_f_constraint_pass_map.csv"),
        "pareto": pd.read_csv(source_dir / "figure_agent_operation_panel_g_tradeoff_pareto.csv"),
        "recommendation": pd.read_csv(source_dir / "figure_agent_operation_panel_h_recommendation_card.csv"),
    }


def write_panel_source_data(tables: dict[str, pd.DataFrame], source_dir: Path = SOURCE_DIR) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    panel_a = pd.concat(
        [
            tables["trace"].assign(panel_source="action_trace"),
            tables["validation"].assign(panel_source="query_validation"),
        ],
        ignore_index=True,
        sort=False,
    )
    outputs = {
        "panel_a_operation_trace_validation": panel_a,
        "panel_b_response_atlas": tables["response"],
        "panel_c_anion_fingerprint": tables["fingerprint"],
        "panel_d_moe_routing_shift": tables["gate_shift"],
        "panel_e_topk_ranking": tables["ranking"],
        "panel_f_constraint_pass_map": tables["constraints"],
        "panel_g_tradeoff_pareto": tables["pareto"],
        "panel_h_recommendation_card": tables["recommendation"],
    }
    for suffix, frame in outputs.items():
        frame.to_csv(source_dir / f"figure_application_case_agent_merged_{suffix}.csv", index=False)


def draw_workflow(ax, trace: pd.DataFrame, validation: pd.DataFrame) -> None:
    ax.set_axis_off()
    panel_label(ax, "a", x=-0.13, y=1.04)

    def operation_box(x, y, w, h, face="#FAFAFA", edge="#D0D0D0", lw=0.65):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            transform=ax.transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
        )
        ax.add_patch(patch)
        return patch

    ax.text(0.05, 0.965, "Agent workflow", transform=ax.transAxes, ha="left", va="top", fontsize=11.5, fontweight="bold")
    ax.text(0.05, 0.910, "Scheduler wrapped around MIPGraph", transform=ax.transAxes, ha="left", va="top", fontsize=7.9, color="#555555")

    operation_box(0.05, 0.765, 0.90, 0.125, "#F7FAFB", "#C9DADF", 0.75)
    ax.text(0.085, 0.845, "Input query", transform=ax.transAxes, ha="left", va="center", fontsize=8.1, fontweight="bold")
    ax.text(0.085, 0.785, r"2 ion pairs; 298-398 K; 6 properties", transform=ax.transAxes, ha="left", va="center", fontsize=7.0, color="#555555")

    stages = [
        ("1", "Parse + validate", r"ion roles, charge balance"),
        ("2", "Build condition grid", "2 ILs x 11 T = 22 records"),
        ("3", "Batch MIPGraph", "22 x 6 property predictions"),
        ("4", "Explain + rank", "MoE shifts, constraints, Pareto"),
    ]
    y_positions = [0.600, 0.440, 0.280, 0.120]
    for (step, title, detail), y in zip(stages, y_positions):
        operation_box(0.05, y, 0.90, 0.120, "#FFFFFF", "#D2D2D2", 0.70)
        operation_box(0.075, y + 0.034, 0.070, 0.052, "#E7F0F3", "#7BAFBE", 0.60)
        ax.text(0.110, y + 0.060, step, transform=ax.transAxes, ha="center", va="center", fontsize=7.2, fontweight="bold")
        ax.text(0.175, y + 0.083, title, transform=ax.transAxes, ha="left", va="center", fontsize=7.8, fontweight="bold")
        ax.text(0.175, y + 0.030, detail, transform=ax.transAxes, ha="left", va="center", fontsize=6.8, color="#555555")

    for y0, y1 in zip([0.765, 0.600, 0.440, 0.280], [0.730, 0.565, 0.405, 0.245]):
        ax.add_patch(
            FancyArrowPatch(
                (0.50, y0),
                (0.50, y1),
                arrowstyle="-|>",
                mutation_scale=8.5,
                linewidth=0.75,
                color="#8A8A8A",
                transform=ax.transAxes,
            )
        )

    operation_box(0.05, 0.005, 0.42, 0.065, "#E9F4EC", "#81B48B", 0.60)
    operation_box(0.53, 0.005, 0.42, 0.065, "#F6F1EA", "#D8C4A3", 0.60)
    ax.text(0.260, 0.038, "5 checks passed", transform=ax.transAxes, ha="center", va="center", fontsize=7.0, fontweight="bold", color="#2F6F3C")
    ax.text(0.740, 0.038, "ranked output", transform=ax.transAxes, ha="center", va="center", fontsize=7.0, fontweight="bold", color="#8A5A00")


def draw_response_atlas(ax_top, ax_bottom, response: pd.DataFrame, fig) -> None:
    prop_order = ["Density", "Viscosity", "Electrical cond.", "Heat capacity", "Surface tension", "Thermal cond."]
    cmap = LinearSegmentedColormap.from_list("response", ["#F8FBFC", "#D6E9ED", "#72B0C1", "#0B6E99"])
    image = None
    for panel_index, (ax, case, title) in enumerate([
        (ax_top, "[BMIM][NTf2]", r"[BMIM][NTf$_2$]"),
        (ax_bottom, "[BMIM][BF4]", r"[BMIM][BF$_4$]"),
    ]):
        temps = sorted(response["Temperature_K"].unique())
        matrix = (
            response[response["IL"] == case]
            .pivot(index="property_label", columns="Temperature_K", values="normalized_response")
            .loc[prop_order, temps]
            .to_numpy()
        )
        image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap, interpolation="nearest")
        ax.set_yticks(np.arange(len(prop_order)))
        ax.set_yticklabels(prop_order)
        ticks = [0, len(temps) // 2, len(temps) - 1]
        ax.set_xticks(ticks)
        if panel_index == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([f"{temps[i]:.0f}" for i in ticks])
        ax.set_title(title, loc="left", color=CASE_COLORS[case], fontsize=9.6, fontweight="bold", pad=2)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, len(temps), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(prop_order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    panel_label(ax_top, "b", x=-0.19, y=1.12)
    ax_bottom.set_xlabel("Temperature (K)")
    cbar = fig.colorbar(image, ax=[ax_top, ax_bottom], fraction=0.045, pad=0.020)
    cbar.ax.tick_params(labelsize=8.0, length=2)


def draw_fingerprint(ax_matrix, ax_delta, fingerprint: pd.DataFrame) -> None:
    panel_label(ax_matrix, "c", x=-0.34, y=1.12)
    ax_matrix.set_title("Anion response\nsignature", loc="left", fontweight="bold", fontsize=9.0, linespacing=0.82, pad=4)
    prop_order = ["Density", "Heat capacity", "Surface tension", "Thermal cond.", "Electrical cond.", "Viscosity"]
    short_labels = ["Density", "Cp", "Surface", "Thermal", "E-cond.", "Viscosity"]
    case_order = ["BMIM_NTf2", "BMIM_BF4"]
    matrix = (
        fingerprint.pivot(index="property_label", columns="case_id", values="normalized_to_pair_max")
        .loc[prop_order, case_order]
    )
    y = np.arange(len(prop_order))
    cmap = LinearSegmentedColormap.from_list("fingerprint", ["#F6FAFB", "#B8D7DE", "#0B6E99"])
    ax_matrix.imshow(matrix.to_numpy(), aspect="auto", vmin=0.45, vmax=1.0, cmap=cmap)
    ax_matrix.set_xticks([0, 1])
    ax_matrix.set_xticklabels([r"NTf$_2$", r"BF$_4$"])
    for tick, color in zip(ax_matrix.get_xticklabels(), [CASE_COLORS[CASE_ORDER[0]], CASE_COLORS[CASE_ORDER[1]]]):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax_matrix.set_yticks(y)
    ax_matrix.set_yticklabels(short_labels)
    ax_matrix.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix.iloc[i, j])
            color = "white" if value > 0.78 else "#2A2A2A"
            ax_matrix.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.4, fontweight="bold", color=color)
    ax_matrix.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax_matrix.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax_matrix.grid(which="minor", color="white", linewidth=0.9)
    ax_matrix.tick_params(which="minor", bottom=False, left=False)
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    deltas = matrix["BMIM_BF4"] - matrix["BMIM_NTf2"]
    colors = [CASE_COLORS[CASE_ORDER[1]] if value >= 0 else CASE_COLORS[CASE_ORDER[0]] for value in deltas]
    ax_delta.barh(y, deltas.to_numpy(), height=0.48, color=colors, alpha=0.90)
    ax_delta.axvline(0.0, color="#808080", linewidth=0.8)
    ax_delta.set_title(r"BF$_4$ - NTf$_2$", loc="left", fontsize=9.4, fontweight="bold", pad=5)
    ax_delta.set_xlim(-0.60, 0.60)
    ax_delta.set_ylim(len(prop_order) - 0.5, -0.5)
    ax_delta.set_yticks(y)
    ax_delta.tick_params(labelleft=False, left=False)
    ax_delta.set_xticks([-0.5, 0.0, 0.5])
    ax_delta.set_xlabel(r"$\Delta$ normalized")
    ax_delta.grid(axis="x", color="#E9E9E9", linewidth=0.55)
    for i, value in enumerate(deltas.to_numpy()):
        x = value + (0.035 if value >= 0 else -0.035)
        ha = "left" if value >= 0 else "right"
        ax_delta.text(x, i, f"{value:+.2f}", va="center", ha=ha, fontsize=7.3, color="#333333")
    for spine in ["top", "right", "left"]:
        ax_delta.spines[spine].set_visible(False)


def draw_gate_shift(ax, gate_shift: pd.DataFrame) -> None:
    panel_label(ax, "d", x=-0.13, y=1.08)
    ax.set_title("Physics-MoE routing shift", loc="left", fontweight="bold", pad=5)
    row_order = ["Average", "Density", "Electrical cond.", "Heat capacity", "Surface tension", "Thermal cond.", "Viscosity"]
    col_order = ["packing", "cohesion", "transport", "thermal"]
    matrix = gate_shift.pivot(index="row_label", columns="mechanism", values="BF4_minus_NTf2_gate").loc[row_order, col_order]
    limit = max(0.42, float(np.nanmax(np.abs(matrix.to_numpy()))) * 1.05)
    cmap = LinearSegmentedColormap.from_list("gate_shift", ["#0B6E99", "#F7F7F7", "#C77700"])
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(["Pack.", "Coh.", "Trans.", "Therm."], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix.iloc[i, j])
            color = "white" if abs(value) > 0.24 else "#2A2A2A"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7.4, color=color)
    ax.set_xticks(np.arange(-0.5, len(col_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_ranking(ax, ranking: pd.DataFrame) -> None:
    panel_label(ax, "e", x=-0.10, y=1.08)
    ax.set_title("Top-k screening", loc="left", fontweight="bold", pad=5)
    data = ranking.sort_values("agent_task_score", ascending=True)
    y = np.arange(len(data))
    colors = ["#0B6E99" if row.case_id == "BMIM_NTf2" else "#C77700" for row in data.itertuples()]
    ax.barh(y, data["agent_task_score"], height=0.50, color=colors, alpha=0.92)
    ax.set_xlim(0.0, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{int(r.agent_rank)} {CASE_SHORT[r.case_id]} {r.Temperature_K:.0f}K" for r in data.itertuples()], fontsize=7.8)
    ax.set_xlabel("Task score", labelpad=1)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.6)
    for yi, row in zip(y, data.itertuples()):
        ax.text(row.agent_task_score + 0.015, yi, f"{row.agent_task_score:.2f}", va="center", ha="left", fontsize=7.8)
    ax.set_ylim(-0.55, len(data) - 0.45)


def draw_constraint_map(ax, constraints: pd.DataFrame) -> None:
    panel_label(ax, "f", x=-0.12, y=1.08)
    ax.set_title("Constraint map", loc="left", fontweight="bold", pad=5)
    row_order = ["BMIM_NTf2", "BMIM_BF4"]
    temps = sorted(constraints["Temperature_K"].unique())
    matrix = (
        constraints.pivot(index="case_id", columns="Temperature_K", values="constraint_state")
        .loc[row_order, temps]
        .to_numpy()
    )
    labels = (
        constraints.pivot(index="case_id", columns="Temperature_K", values="constraint_label")
        .loc[row_order, temps]
        .to_numpy()
    )
    cmap = ListedColormap(["#F2F2F2", "#D9E9ED", "#0B6E99"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(temps)))
    ax.set_xticklabels([f"{t:.0f}" for t in temps], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels([CASE_SHORT[c] for c in row_order])
    ax.set_xlabel("Temperature (K)")
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = {"both pass": "OK", "viscosity only": "V", "conductivity only": "C"}.get(labels[i, j], "-")
            color = "white" if matrix[i, j] == 2 else "#2A2A2A"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.2, fontweight="bold", color=color)
    ax.set_xticks(np.arange(-0.5, len(temps), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_pareto(ax, pareto: pd.DataFrame) -> None:
    panel_label(ax, "g", x=-0.09, y=1.08)
    ax.set_title("Conductivity-viscosity trade-off", loc="left", fontweight="bold", pad=5)
    temp_cmap = LinearSegmentedColormap.from_list("temperature_score", ["#E7EEF0", "#79AEBB", "#0B6E99"])
    for case_id, data in pareto.groupby("case_id", sort=False):
        ax.scatter(
            data["Viscosity_mPa_s"],
            data["ElectricalConductivity"],
            c=data["Temperature_K"],
            cmap=temp_cmap,
            vmin=pareto["Temperature_K"].min(),
            vmax=pareto["Temperature_K"].max(),
            marker="o" if case_id == "BMIM_NTf2" else "s",
            s=26,
            edgecolor="#FFFFFF",
            linewidth=0.40,
            zorder=2,
        )
    front = pareto[pareto["is_pareto_front"]].sort_values("Viscosity_mPa_s")
    ax.plot(front["Viscosity_mPa_s"], front["ElectricalConductivity"], color="#2A2A2A", linewidth=0.9, zorder=1)
    top = pareto.dropna(subset=["agent_rank"])
    ax.scatter(top["Viscosity_mPa_s"], top["ElectricalConductivity"], s=46, facecolor="none", edgecolor="#C77700", linewidth=1.0, zorder=3)
    for row in top.itertuples():
        if int(row.agent_rank) <= 4:
            ax.text(row.Viscosity_mPa_s * 1.08, row.ElectricalConductivity, f"#{int(row.agent_rank)}", fontsize=7.6, va="center")
    ax.axvline(VISCOSITY_LIMIT_MPA_S, color="#8A8A8A", linestyle="--", linewidth=0.8)
    ax.axhline(CONDUCTIVITY_LIMIT_S_M, color="#8A8A8A", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Viscosity (mPa s)")
    ax.set_ylabel(r"Electrical cond. (S m$^{-1}$)", labelpad=1)
    ax.grid(color="#E9E9E9", linewidth=0.55)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#79AEBB", markeredgecolor="white", label=r"NTf$_2$", markersize=5.0),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#79AEBB", markeredgecolor="white", label=r"BF$_4$", markersize=5.0),
            Line2D([0], [0], color="#2A2A2A", label="Pareto", linewidth=0.9),
        ],
        loc="upper right",
        fontsize=7.8,
        handlelength=1.2,
        borderaxespad=0.15,
    )


def draw_recommendation(ax, recommendation: pd.DataFrame) -> None:
    ax.set_axis_off()
    panel_label(ax, "h", x=-0.13, y=1.03)
    values = dict(zip(recommendation["field"], recommendation["value"]))

    def card(x, y, w, h, face="#FAFAFA", edge="#D0D0D0", lw=0.75):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
        )
        ax.add_patch(patch)
        return patch

    ax.text(0.04, 0.980, "Recommendation summary", transform=ax.transAxes, ha="left", va="top", fontsize=10.8, fontweight="bold")
    card(0.04, 0.035, 0.92, 0.905, "#FAFAFA", "#D0D0D0", 0.80)

    ax.text(0.095, 0.895, "Top-ranked condition", transform=ax.transAxes, ha="left", va="center", fontsize=8.0, color="#555555")
    ax.text(0.095, 0.845, values.get("Top condition", ""), transform=ax.transAxes, ha="left", va="center", fontsize=9.6, fontweight="bold")
    ax.plot([0.095, 0.905], [0.790, 0.790], transform=ax.transAxes, color="#D6D6D6", lw=0.8)

    metric_rows = [
        ("Task score", values.get("Agent score", ""), "#EAF3F6"),
        ("Electrical cond.", values.get("Electrical conductivity", ""), "#F3F7F8"),
        ("Viscosity", values.get("Viscosity", ""), "#F6F1EA"),
    ]
    y0 = 0.720
    for label, value, face in metric_rows:
        card(0.095, y0 - 0.040, 0.810, 0.074, face, "#D8D8D8", 0.60)
        ax.text(0.135, y0, label, transform=ax.transAxes, ha="left", va="center", fontsize=7.7, color="#555555")
        ax.text(0.865, y0, value, transform=ax.transAxes, ha="right", va="center", fontsize=8.6, fontweight="bold", color="#222222")
        y0 -= 0.100

    ax.text(0.095, 0.430, "Why selected", transform=ax.transAxes, ha="left", va="center", fontsize=8.5, fontweight="bold")
    card(0.095, 0.335, 0.810, 0.076, "#FFFFFF", "#D8D8D8", 0.60)
    ax.text(0.125, 0.383, "highest score among 22 records", transform=ax.transAxes, ha="left", va="center", fontsize=7.0, color="#555555")
    ax.text(0.125, 0.354, "passes viscosity and conductivity filters", transform=ax.transAxes, ha="left", va="center", fontsize=7.0, color="#555555")

    ax.text(0.095, 0.285, "Decision trace", transform=ax.transAxes, ha="left", va="center", fontsize=8.5, fontweight="bold")
    trace_rows = [
        ("1", "Validate", "22 complete records"),
        ("2", "Screen", r"$\eta \leq 10$, $\sigma \geq 2$"),
        ("3", "Rank", r"BF$_4$ at 398 K selected"),
    ]
    ax.plot([0.133, 0.133], [0.090, 0.235], transform=ax.transAxes, color="#B8C8CE", lw=1.1, zorder=0)
    y0 = 0.235
    for step, label, value in trace_rows:
        card(0.105, y0 - 0.022, 0.056, 0.044, "#E7F0F3", "#7BAFBE", 0.55)
        ax.text(0.133, y0, step, transform=ax.transAxes, ha="center", va="center", fontsize=7.0, fontweight="bold")
        ax.text(0.190, y0 + 0.013, label, transform=ax.transAxes, ha="left", va="center", fontsize=7.7, fontweight="bold")
        ax.text(0.190, y0 - 0.015, value, transform=ax.transAxes, ha="left", va="center", fontsize=7.2, color="#555555")
        y0 -= 0.073



def draw_figure(
    tables: dict[str, pd.DataFrame],
    output_prefix: Path = OUTPUT_PREFIX,
    source_dir: Path = SOURCE_DIR,
    dpi: int = 600,
) -> None:
    apply_style()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_panel_source_data(tables, source_dir)

    fig = plt.figure(figsize=(11.6, 9.20), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        22,
        left=0.065,
        right=0.985,
        bottom=0.060,
        top=0.955,
        wspace=0.90,
        hspace=0.55,
        height_ratios=[1.24, 1.12, 1.12],
    )
    left_grid = gs[:, 0:5].subgridspec(2, 1, hspace=0.32, height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(left_grid[0, 0])

    b_grid = gs[0, 7:13].subgridspec(2, 1, hspace=0.36)
    ax_b1 = fig.add_subplot(b_grid[0, 0])
    ax_b2 = fig.add_subplot(b_grid[1, 0])
    c_grid = gs[0, 15:22].subgridspec(1, 2, width_ratios=[1.12, 1.30], wspace=0.30)
    ax_c_left = fig.add_subplot(c_grid[0, 0])
    ax_c_right = fig.add_subplot(c_grid[0, 1])

    ax_h = fig.add_subplot(left_grid[1, 0])
    ax_d = fig.add_subplot(gs[1, 7:13])
    ax_e = fig.add_subplot(gs[1, 15:22])

    ax_f = fig.add_subplot(gs[2, 7:13])
    ax_g = fig.add_subplot(gs[2, 15:22])

    draw_workflow(ax_a, tables["trace"], tables["validation"])
    draw_response_atlas(ax_b1, ax_b2, tables["response"], fig)
    draw_fingerprint(ax_c_left, ax_c_right, tables["fingerprint"])
    draw_gate_shift(ax_d, tables["gate_shift"])
    draw_ranking(ax_e, tables["ranking"])
    draw_constraint_map(ax_f, tables["constraints"])
    draw_pareto(ax_g, tables["pareto"])
    draw_recommendation(ax_h, tables["recommendation"])

    fig.savefig(output_prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the merged application-case and design-agent figure.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = load_tables(args.source_dir)
    draw_figure(tables, args.output_prefix, args.source_dir, args.dpi)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote merged panel source data to {args.source_dir}")


if __name__ == "__main__":
    main()
