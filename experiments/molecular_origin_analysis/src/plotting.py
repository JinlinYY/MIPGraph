"""Publication-grade Matplotlib figures with source-data export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .utils import MODULE_ROOT, ensure_within, write_table


BLUE = "#2166AC"
RED = "#B2182B"
TEAL = "#1B9E77"
ORANGE = "#D95F02"
PURPLE = "#7570B3"
GREY = "#6B7280"

PROPERTY_ORDER = [
    "Density",
    "Viscosity",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
]

PROPERTY_LABELS = {
    "Density": "Density",
    "Viscosity": "Viscosity",
    "ElectricalConductivity": "Electrical conductivity",
    "HeatCapacity": "Heat capacity",
    "SurfaceTension": "Surface tension",
    "ThermalConductivity": "Thermal conductivity",
}


def _property_label(property_name: str) -> str:
    return PROPERTY_LABELS.get(property_name, property_name)


class PublicationPlotter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        figure_config = config["figures"]
        requested = str(figure_config.get("font_family", "Arial"))
        available = {font.name for font in font_manager.fontManager.ttflist}
        family = requested if requested in available else "DejaVu Sans"
        mpl.rcParams.update(
            {
                "font.family": family,
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 9,
                "axes.linewidth": 0.8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "lines.linewidth": 1.2,
                "savefig.bbox": "tight",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        self.formats = list(figure_config.get("formats", ["png", "pdf", "svg"]))
        self.dpi = int(figure_config.get("dpi", 600))

    def _save(self, figure: plt.Figure, stem: Path, source: pd.DataFrame) -> list[Path]:
        stem = ensure_within(stem, MODULE_ROOT)
        stem.parent.mkdir(parents=True, exist_ok=True)
        source_dir = stem.parent.parent / "tables" / "figure_source_data"
        write_table(source, source_dir / f"{stem.name}_source_data.csv")
        outputs = []
        for suffix in self.formats:
            path = stem.with_suffix(f".{suffix}")
            figure.savefig(
                path,
                dpi=self.dpi if suffix.lower() == "png" else None,
                facecolor="white",
            )
            outputs.append(path)
        plt.close(figure)
        return outputs

    def evidence_map(self, rules: pd.DataFrame, stem: Path) -> list[Path]:
        if rules.empty:
            raise ValueError("Design-rule table is empty")
        source = rules.copy()
        source["abs_evidence"] = source["confidence_level"].map(
            {"Level A": 3.0, "Level B": 2.0, "Level C": 1.0}
        )
        source = (
            source.sort_values(
                ["property", "abs_evidence", "family_consistency", "structural_factor"],
                ascending=[True, False, False, True],
            )
            .groupby("property", group_keys=False, sort=True)
            .head(4)
            .sort_values(
                ["abs_evidence", "property", "structural_factor"],
                ascending=[False, True, True],
            )
        )
        factors = source["structural_factor"].drop_duplicates().tolist()
        properties = [
            property_name
            for property_name in PROPERTY_ORDER
            if property_name in set(source["property"])
        ]
        factor_y = {
            factor: value
            for factor, value in zip(factors, np.linspace(0.95, 0.05, len(factors)))
        }
        property_y = {
            prop: value
            for prop, value in zip(properties, np.linspace(0.88, 0.12, len(properties)))
        }
        figure, axis = plt.subplots(figsize=(7.2, 5.0))
        for row in source.itertuples(index=False):
            color = RED if row.effect_direction == "positive" else BLUE
            axis.plot(
                [0.34, 0.66],
                [factor_y[row.structural_factor], property_y[row.property]],
                color=color,
                alpha=0.28 + 0.2 * row.abs_evidence,
                linewidth=0.7 + 0.5 * row.abs_evidence,
                zorder=1,
            )
        for factor, y_value in factor_y.items():
            axis.text(
                0.32,
                y_value,
                factor.replace("_", " "),
                ha="right",
                va="center",
                fontsize=6.6,
            )
        for prop, y_value in property_y.items():
            axis.text(
                0.68,
                y_value,
                _property_label(prop),
                ha="left",
                va="center",
                fontsize=7.5,
                fontweight="bold",
            )
        axis.text(0.20, 1.01, "Audited structural factors", ha="center", fontweight="bold")
        axis.text(0.80, 1.01, "Thermophysical properties", ha="center", fontweight="bold")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.legend(
            handles=[
                Line2D([0], [0], color=RED, lw=2, label="positive association"),
                Line2D([0], [0], color=BLUE, lw=2, label="negative association"),
            ],
            loc="lower center",
            ncol=2,
            frameon=False,
        )
        axis.set_title(
            "Microstructure–property evidence map\n"
            "Line width encodes evidence level; links are non-causal",
            loc="left",
            pad=12,
        )
        return self._save(figure, stem, source)

    def association_heatmap(
        self,
        associations: pd.DataFrame,
        stem: Path,
    ) -> list[Path]:
        source = associations.loc[
            associations["data_type"] == "experimental"
        ].copy()
        source["abs_effect"] = source["partial_correlation"].abs()
        selected_features = (
            source.groupby("feature")["abs_effect"]
            .max()
            .nlargest(20)
            .index
        )
        source = source.loc[source["feature"].isin(selected_features)]
        pivot = source.pivot(
            index="feature",
            columns="property",
            values="partial_correlation",
        )
        property_order = PROPERTY_ORDER
        property_order = [name for name in property_order if name in pivot.columns]
        pivot = pivot[property_order].loc[selected_features]
        figure, axis = plt.subplots(figsize=(7.2, 5.5))
        image = axis.imshow(
            pivot.to_numpy(dtype=float),
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6),
            aspect="auto",
        )
        axis.set_xticks(
            range(len(property_order)),
            [_property_label(name) for name in property_order],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(
            range(len(pivot)),
            [name.replace("_", " ") for name in pivot.index],
        )
        q_values = source.pivot(index="feature", columns="property", values="fdr_q")
        q_values = q_values[property_order].loc[pivot.index]
        for row_index in range(len(pivot)):
            for column_index in range(len(property_order)):
                q_value = q_values.iloc[row_index, column_index]
                if np.isfinite(q_value) and q_value <= 0.05:
                    axis.text(column_index, row_index, "•", ha="center", va="center")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
        colorbar.set_label("Condition-controlled partial correlation")
        axis.set_title(
            "Property-specific structural associations\n"
            "Dots indicate BH-FDR q ≤ 0.05",
            loc="left",
        )
        figure.tight_layout()
        return self._save(figure, stem, source)

    def response_curves(self, nonlinear: pd.DataFrame, stem: Path) -> list[Path]:
        if nonlinear.empty:
            raise ValueError("Nonlinear trend table is empty")
        candidate_combinations = nonlinear[
            ["property", "feature", "monotonic_bin_spearman"]
        ].drop_duplicates(["property", "feature"])
        candidate_combinations["selection_score"] = (
            candidate_combinations["monotonic_bin_spearman"].abs().fillna(-np.inf)
        )
        candidate_combinations["property_order"] = (
            candidate_combinations["property"]
            .map({name: index for index, name in enumerate(PROPERTY_ORDER)})
            .fillna(len(PROPERTY_ORDER))
        )
        combinations = (
            candidate_combinations.sort_values(
                ["property_order", "selection_score", "feature"],
                ascending=[True, False, True],
            )
            .groupby("property", sort=False)
            .head(1)[["property", "feature"]]
            .head(6)
        )
        source = nonlinear.merge(combinations, on=["property", "feature"], how="inner")
        figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), squeeze=False)
        for axis, combination in zip(axes.flat, combinations.itertuples(index=False)):
            subset = source.loc[
                (source["property"] == combination.property)
                & (source["feature"] == combination.feature)
            ].sort_values("feature_mean")
            axis.errorbar(
                subset["feature_mean"],
                subset["response_log_mean"],
                yerr=subset["response_log_sem"],
                marker="o",
                markersize=3,
                color=TEAL,
                capsize=2,
            )
            axis.set_title(_property_label(combination.property), loc="left")
            axis.set_xlabel(combination.feature.replace("_", " "))
            axis.set_ylabel("Mean ln(property)")
            axis.text(
                0.02,
                0.97,
                f"n={int(subset['sample_count'].sum())}",
                transform=axis.transAxes,
                va="top",
                fontsize=6.5,
            )
        for axis in axes.flat[len(combinations) :]:
            axis.axis("off")
        figure.suptitle(
            "Binned observed responses for selected robust factors",
            x=0.01,
            ha="left",
            fontweight="bold",
        )
        figure.tight_layout()
        return self._save(figure, stem, source)

    def counterfactual_trends(
        self,
        predictions: pd.DataFrame,
        matched_pairs: pd.DataFrame,
        stem: Path,
    ) -> list[Path]:
        if predictions.empty:
            raise ValueError("Counterfactual prediction table is empty")
        properties = ["Viscosity", "ElectricalConductivity"]
        properties = [name for name in properties if name in predictions]
        prediction_source = predictions.copy()
        candidates = prediction_source["candidate_id"].drop_duplicates().head(8)
        prediction_source = prediction_source.loc[
            prediction_source["candidate_id"].isin(candidates)
        ]
        figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), squeeze=False)
        colors = plt.get_cmap("tab10")(np.linspace(0, 0.8, len(candidates)))
        for axis, property_name in zip(axes[0], properties):
            for color, (candidate_id, group) in zip(
                colors,
                prediction_source.groupby("candidate_id"),
            ):
                group = group.sort_values("temperature_K")
                axis.plot(
                    group["temperature_K"],
                    group[property_name],
                    marker="o",
                    markersize=2.5,
                    color=color,
                    label=candidate_id,
                )
            axis.set_title(_property_label(property_name), loc="left")
            axis.set_xlabel("Temperature (K)")
            axis.set_ylabel(_property_label(property_name))
            if property_name in {"Viscosity", "ElectricalConductivity"}:
                axis.set_yscale("log")
        axes[0, 1].legend(frameon=False, fontsize=5.5, ncol=2)
        role_order = ["anion_fixed", "cation_fixed"]
        role_labels = [
            "Cation substitution\n(anion fixed)",
            "Anion substitution\n(cation fixed)",
        ]
        matched_columns = [
            "fixed_role",
            "left_sample_id",
            "right_sample_id",
            *[
                f"observed_abs_log_difference_{property_name}"
                for property_name in properties
            ],
        ]
        available_columns = [
            column for column in matched_columns if column in matched_pairs.columns
        ]
        matched_source = matched_pairs[available_columns].copy()
        for axis, property_name in zip(axes[1], properties):
            value_column = f"observed_abs_log_difference_{property_name}"
            if value_column not in matched_source:
                axis.text(
                    0.5,
                    0.5,
                    "Observed matched-pair labels unavailable",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.axis("off")
                continue
            groups = [
                matched_source.loc[
                    matched_source["fixed_role"] == role,
                    value_column,
                ]
                .dropna()
                .to_numpy(dtype=float)
                for role in role_order
            ]
            axis.boxplot(
                groups,
                tick_labels=role_labels,
                patch_artist=True,
                widths=0.55,
                showfliers=True,
                boxprops={"facecolor": "#D9EAF3", "edgecolor": BLUE},
                medianprops={"color": RED, "linewidth": 1.2},
                whiskerprops={"color": BLUE},
                capprops={"color": BLUE},
                flierprops={
                    "marker": "o",
                    "markersize": 1.0,
                    "markerfacecolor": GREY,
                    "markeredgecolor": "none",
                    "alpha": 0.15,
                },
            )
            axis.set_title(
                f"Observed matched-pair |Δ ln {_property_label(property_name).lower()}|",
                loc="left",
            )
            axis.set_ylabel("|Δ ln(observed property)|")
            axis.tick_params(axis="x", labelsize=6)
            for index, values in enumerate(groups, start=1):
                axis.text(
                    index,
                    0.97,
                    f"n={len(values)}",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=6,
                )
        prediction_export = prediction_source.copy()
        prediction_export.insert(0, "source_kind", "virtual_counterfactual")
        matched_export = matched_source.copy()
        matched_export.insert(0, "source_kind", "observed_matched_pair")
        source = pd.concat(
            [prediction_export, matched_export],
            ignore_index=True,
            sort=False,
        )
        figure.suptitle(
            "Matched molecular pairs and valid-SMILES counterfactual responses\n"
            "Predictions are conditional and require applicability-domain qualification",
            x=0.01,
            ha="left",
            fontweight="bold",
        )
        figure.tight_layout()
        return self._save(figure, stem, source)

    def cross_ion_profile(
        self,
        contrasts: pd.DataFrame,
        stem: Path,
    ) -> list[Path]:
        if contrasts.empty:
            raise ValueError("Cross-ion contrast table is empty")
        source = contrasts.copy()
        pivot = source.pivot(
            index="interaction_category",
            columns="property",
            values="high_minus_low",
        )
        figure, axis = plt.subplots(figsize=(7.2, 3.8))
        maximum = max(float(np.nanmax(np.abs(pivot.to_numpy()))), 1e-12)
        image = axis.imshow(
            pivot.to_numpy(),
            cmap="PuOr_r",
            norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
            aspect="auto",
        )
        axis.set_xticks(
            range(len(pivot.columns)),
            [_property_label(name) for name in pivot.columns],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(
            range(len(pivot.index)),
            [value.replace("-", " ") for value in pivot.index],
        )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
        colorbar.set_label("High − low attention per pair")
        axis.set_title(
            "Shared cross-ion attention profile by condition-controlled property quartile\n"
            "Attention is a model focus pattern, not interaction energy",
            loc="left",
        )
        figure.tight_layout()
        return self._save(figure, stem, source)

    def screening_design_map(
        self,
        trajectory: pd.DataFrame,
        top8: pd.DataFrame,
        stem: Path,
    ) -> list[Path]:
        if trajectory.empty:
            raise ValueError("Candidate trajectory table is empty")
        source = trajectory.copy()
        top_ids = set(top8["candidate_id"].astype(str))
        source["formal_shortlist"] = source["candidate_id"].astype(str).isin(top_ids)
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        heat = np.log10(
            source["thermal_diffusivity_worst"].to_numpy(dtype=float).clip(min=1e-12)
        )
        capacity = source["volumetric_heat_capacity_worst"].to_numpy(dtype=float)
        sizes = 8 + 34 * (capacity - np.nanmin(capacity)) / max(
            np.nanmax(capacity) - np.nanmin(capacity),
            1e-12,
        )
        scatter = axis.scatter(
            source["viscosity_worst"],
            source["conductivity_worst"],
            c=heat,
            s=sizes,
            cmap="viridis",
            alpha=0.42,
            linewidths=0,
        )
        selected = source.loc[source["formal_shortlist"]]
        axis.scatter(
            selected["viscosity_worst"],
            selected["conductivity_worst"],
            facecolors="none",
            edgecolors=RED,
            s=70,
            linewidths=1.2,
            label="formal Top-8",
        )
        selected_for_labels = selected.sort_values(
            ["conductivity_worst", "candidate_id"],
            ascending=[False, True],
        )
        compact_cluster = selected_for_labels.loc[
            selected_for_labels["viscosity_worst"] >= 0.01
        ]
        isolated = selected_for_labels.loc[
            selected_for_labels["viscosity_worst"] < 0.01
        ]
        if not compact_cluster.empty:
            label_x = float(compact_cluster["viscosity_worst"].max()) * 1.7
            label_y = np.geomspace(
                float(compact_cluster["conductivity_worst"].max()) * 1.18,
                float(compact_cluster["conductivity_worst"].min()) * 0.82,
                len(compact_cluster),
            )
            for text_y, row in zip(
                label_y,
                compact_cluster.itertuples(index=False),
            ):
                axis.annotate(
                    row.candidate_id.replace("UPR-", ""),
                    (row.viscosity_worst, row.conductivity_worst),
                    xytext=(label_x, text_y),
                    textcoords="data",
                    fontsize=5.5,
                    arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.45},
                    bbox={
                        "boxstyle": "round,pad=0.08",
                        "fc": "white",
                        "ec": "none",
                        "alpha": 0.75,
                    },
                )
        isolated_offsets = [
            (6, 8 if index % 2 == 0 else -15)
            for index in range(len(isolated))
        ]
        for offset, row in zip(
            isolated_offsets,
            isolated.sort_values(
                ["conductivity_worst", "candidate_id"],
                ascending=[False, True],
            ).itertuples(index=False),
        ):
            axis.annotate(
                row.candidate_id.replace("UPR-", ""),
                (row.viscosity_worst, row.conductivity_worst),
                xytext=offset,
                textcoords="offset points",
                fontsize=5.5,
                arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.45},
                bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.75},
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Worst-window viscosity (Pa s; lower is favourable)")
        axis.set_ylabel("Worst-window conductivity (S m$^{-1}$; higher is favourable)")
        colorbar = figure.colorbar(scatter, ax=axis, fraction=0.035, pad=0.02)
        colorbar.set_label("log$_{10}$(worst-window thermal diffusivity / m$^2$ s$^{-1}$)")
        axis.legend(frameon=False)
        axis.set_title(
            "Structure-informed context for thermophysical candidate pre-screening\n"
            "Point size represents worst-window volumetric heat capacity",
            loc="left",
        )
        figure.tight_layout()
        return self._save(figure, stem, source)
