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
                "svg.fonttype": "none",
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

    def _save_composite(
        self,
        figure: plt.Figure,
        stem: Path,
        panel_sources: dict[str, pd.DataFrame],
    ) -> list[Path]:
        """Save an editable multi-panel figure and one source table per panel."""

        stem = ensure_within(stem, MODULE_ROOT)
        stem.parent.mkdir(parents=True, exist_ok=True)
        source_dir = stem.parent.parent / "tables" / "figure_source_data"
        for panel, source in panel_sources.items():
            write_table(
                source,
                source_dir / f"{stem.name}_panel_{panel}_source_data.csv",
            )
        formats = list(dict.fromkeys([*self.formats, "tiff"]))
        outputs = []
        for suffix in formats:
            path = stem.with_suffix(f".{suffix}")
            extra_save_options = (
                {"pil_kwargs": {"compression": "tiff_lzw"}}
                if suffix.lower() == "tiff"
                else {}
            )
            figure.savefig(
                path,
                dpi=self.dpi if suffix.lower() in {"png", "tiff"} else None,
                facecolor="white",
                **extra_save_options,
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

    def composite_results_figure(
        self,
        rules: pd.DataFrame,
        associations: pd.DataFrame,
        nonlinear: pd.DataFrame,
        predictions: pd.DataFrame,
        matched_pairs: pd.DataFrame,
        contrasts: pd.DataFrame,
        trajectory: pd.DataFrame,
        top8: pd.DataFrame,
        stem: Path,
    ) -> list[Path]:
        """Build the manuscript-facing a–f evidence figure from audited tables."""

        required_frames = {
            "rules": rules,
            "associations": associations,
            "nonlinear": nonlinear,
            "predictions": predictions,
            "matched_pairs": matched_pairs,
            "contrasts": contrasts,
            "trajectory": trajectory,
            "top8": top8,
        }
        empty = [name for name, frame in required_frames.items() if frame.empty]
        if empty:
            raise ValueError(
                "Composite figure inputs are empty: " + ", ".join(empty)
            )

        def compact_feature_label(value: str) -> str:
            is_functional_group = value.endswith("_functional_group")
            label = value.replace("_functional_group", "").replace("_", " ")
            return f"{label} (FG)" if is_functional_group else label

        figure = plt.figure(figsize=(7.2, 9.0), facecolor="white")
        outer = figure.add_gridspec(
            4,
            2,
            height_ratios=[1.18, 1.05, 1.05, 1.25],
            width_ratios=[0.92, 1.08],
            hspace=0.78,
            wspace=0.46,
        )
        panel_sources: dict[str, pd.DataFrame] = {}

        # a | One strongest evidence-gated association per property.
        axis_a = figure.add_subplot(outer[0, 0])
        source_a = rules.copy()
        source_a["partial_r"] = pd.to_numeric(
            source_a["statistical_evidence"].str.extract(
                r"partial r=([+-]?[0-9]*\.?[0-9]+)"
            )[0],
            errors="coerce",
        )
        source_a["level_order"] = source_a["confidence_level"].map(
            {"Level A": 0, "Level B": 1, "Level C": 2}
        )
        source_a["property_order"] = source_a["property"].map(
            {name: index for index, name in enumerate(PROPERTY_ORDER)}
        )
        eligible = source_a.loc[
            source_a["confidence_level"].isin(["Level A", "Level B"])
            & source_a["partial_r"].notna()
        ].copy()
        if eligible.empty:
            eligible = source_a.loc[source_a["partial_r"].notna()].copy()
        eligible["abs_partial_r"] = eligible["partial_r"].abs()
        source_a = (
            eligible.sort_values(
                [
                    "property_order",
                    "level_order",
                    "abs_partial_r",
                    "family_consistency",
                    "structural_factor",
                ],
                ascending=[True, True, False, False, True],
            )
            .groupby("property", sort=False)
            .head(1)
            .sort_values("property_order")
        )
        y_positions = np.arange(len(source_a))[::-1]
        axis_a.axvline(0, color="#B8BDC5", linewidth=0.7, zorder=0)
        for y_value, row in zip(y_positions, source_a.itertuples(index=False)):
            color = RED if row.partial_r > 0 else BLUE
            axis_a.plot(
                [0, row.partial_r],
                [y_value, y_value],
                color=color,
                linewidth=1.4,
                alpha=0.8,
            )
            axis_a.scatter(
                [row.partial_r],
                [y_value],
                s=24,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        factor_labels = [
            compact_feature_label(row.structural_factor)
            for row in source_a.itertuples(index=False)
        ]
        tick_labels = [
            f"{_property_label(row.property)}\n{factor_label}"
            for row, factor_label in zip(
                source_a.itertuples(index=False),
                factor_labels,
            )
        ]
        axis_a.set_yticks(y_positions, tick_labels)
        axis_a.tick_params(axis="y", labelsize=5.2, length=0)
        axis_a.set_xlim(-0.9, 0.9)
        axis_a.set_xlabel("Condition-controlled partial $r$", fontsize=6)
        axis_a.set_title(
            "Evidence-gated structural associations",
            loc="left",
            fontsize=7,
            fontweight="bold",
        )
        axis_a.text(
            -0.16,
            1.09,
            "a",
            transform=axis_a.transAxes,
            fontsize=8.5,
            fontweight="bold",
        )
        axis_a.spines[["top", "right", "left"]].set_visible(False)
        panel_sources["a"] = source_a.drop(
            columns=["level_order", "property_order", "abs_partial_r"],
            errors="ignore",
        )

        # b | Compact condition-controlled association matrix.
        axis_b = figure.add_subplot(outer[0, 1])
        source_b = associations.loc[
            associations["data_type"] == "experimental"
        ].copy()
        source_b["abs_effect"] = source_b["partial_correlation"].abs()
        selected_features = (
            source_b.groupby("feature")["abs_effect"]
            .max()
            .nlargest(12)
            .index
        )
        source_b = source_b.loc[source_b["feature"].isin(selected_features)]
        pivot_b = source_b.pivot(
            index="feature",
            columns="property",
            values="partial_correlation",
        )
        property_order_b = [
            name for name in PROPERTY_ORDER if name in pivot_b.columns
        ]
        pivot_b = pivot_b[property_order_b].loc[selected_features]
        image_b = axis_b.imshow(
            pivot_b.to_numpy(dtype=float),
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6),
            aspect="auto",
        )
        short_properties = {
            "Density": "Density",
            "Viscosity": "Viscosity",
            "ElectricalConductivity": "Conductivity",
            "HeatCapacity": "Heat capacity",
            "SurfaceTension": "Surface tension",
            "ThermalConductivity": "Thermal cond.",
        }
        axis_b.set_xticks(
            range(len(property_order_b)),
            [short_properties[name] for name in property_order_b],
            rotation=38,
            ha="right",
        )
        axis_b.set_yticks(
            range(len(pivot_b)),
            [compact_feature_label(value) for value in pivot_b.index],
        )
        axis_b.tick_params(axis="both", labelsize=4.9)
        q_values = source_b.pivot(
            index="feature",
            columns="property",
            values="fdr_q",
        )[property_order_b].loc[pivot_b.index]
        for row_index in range(len(pivot_b)):
            for column_index in range(len(property_order_b)):
                q_value = q_values.iloc[row_index, column_index]
                if np.isfinite(q_value) and q_value <= 0.05:
                    axis_b.text(
                        column_index,
                        row_index,
                        "•",
                        ha="center",
                        va="center",
                        fontsize=5,
                    )
        colorbar_b = figure.colorbar(
            image_b,
            ax=axis_b,
            fraction=0.035,
            pad=0.025,
        )
        colorbar_b.ax.tick_params(labelsize=4.8, length=2)
        colorbar_b.set_label("Partial $r$", fontsize=5.5)
        axis_b.set_title(
            "Property-specific association matrix",
            loc="left",
            fontsize=7,
            fontweight="bold",
        )
        axis_b.text(
            -0.15,
            1.09,
            "b",
            transform=axis_b.transAxes,
            fontsize=8.5,
            fontweight="bold",
        )
        panel_sources["b"] = source_b

        # c | Six observed response curves, one representative factor per property.
        grid_c = outer[1, :].subgridspec(1, 6, wspace=0.62)
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
        source_c = nonlinear.merge(
            combinations,
            on=["property", "feature"],
            how="inner",
        )
        axes_c = []
        for panel_index, combination in enumerate(
            combinations.itertuples(index=False)
        ):
            axis = figure.add_subplot(grid_c[0, panel_index])
            axes_c.append(axis)
            subset = source_c.loc[
                (source_c["property"] == combination.property)
                & (source_c["feature"] == combination.feature)
            ].sort_values("feature_mean")
            axis.errorbar(
                subset["feature_mean"],
                subset["response_log_mean"],
                yerr=subset["response_log_sem"],
                marker="o",
                markersize=2.1,
                color=TEAL,
                capsize=1.3,
                linewidth=0.9,
            )
            axis.set_title(
                short_properties[combination.property],
                loc="left",
                fontsize=5.5,
                fontweight="bold",
            )
            axis.set_xlabel(
                compact_feature_label(combination.feature),
                fontsize=4.8,
                labelpad=1.5,
            )
            if panel_index == 0:
                axis.set_ylabel("Mean ln(property)", fontsize=5)
            axis.tick_params(labelsize=4.8, length=2)
            axis.text(
                0.03,
                0.94,
                f"n={int(subset['sample_count'].sum())}",
                transform=axis.transAxes,
                va="top",
                fontsize=4.7,
            )
            axis.spines[["top", "right"]].set_visible(False)
        axes_c[0].text(
            -0.38,
            1.20,
            "c",
            transform=axes_c[0].transAxes,
            fontsize=8.5,
            fontweight="bold",
        )
        axes_c[0].text(
            0,
            1.20,
            "Observed property responses across structural-factor quantiles",
            transform=axes_c[0].transAxes,
            fontsize=7,
            fontweight="bold",
        )
        panel_sources["c"] = source_c

        # d | Conditional counterfactual temperature responses and observed pairs.
        grid_d = outer[2, :].subgridspec(1, 4, wspace=0.55)
        properties_d = ["Viscosity", "ElectricalConductivity"]
        axes_d = [figure.add_subplot(grid_d[0, index]) for index in range(4)]
        prediction_source = predictions.copy()
        for axis, property_name in zip(axes_d[:2], properties_d):
            response = prediction_source.pivot(
                index="candidate_id",
                columns="temperature_K",
                values=property_name,
            ).sort_index(axis=1)
            reference_temperature = float(response.columns.min())
            relative = np.log(response).subtract(
                np.log(response[reference_temperature]),
                axis=0,
            )
            for row in relative.itertuples(index=False):
                axis.plot(
                    relative.columns,
                    np.asarray(row, dtype=float),
                    color="#A9AFB7",
                    alpha=0.34,
                    linewidth=0.55,
                )
            median = relative.median(axis=0)
            lower = relative.quantile(0.25, axis=0)
            upper = relative.quantile(0.75, axis=0)
            axis.fill_between(
                relative.columns.to_numpy(dtype=float),
                lower.to_numpy(dtype=float),
                upper.to_numpy(dtype=float),
                color=TEAL,
                alpha=0.18,
                linewidth=0,
            )
            axis.plot(
                relative.columns,
                median,
                color=TEAL,
                marker="o",
                markersize=2.1,
                linewidth=1.2,
            )
            axis.axhline(0, color="#B8BDC5", linewidth=0.6)
            axis.set_title(
                f"{short_properties[property_name]}\ncounterfactual",
                loc="left",
                fontsize=5.5,
                fontweight="bold",
            )
            axis.set_xlabel("Temperature (K)", fontsize=5.0)
            axis.set_ylabel(
                f"Δln({short_properties[property_name].lower()})",
                fontsize=5.0,
            )
            axis.tick_params(labelsize=4.8, length=2)
            axis.text(
                0.03,
                0.93,
                f"n={len(relative)}",
                transform=axis.transAxes,
                va="top",
                fontsize=4.7,
            )
            axis.spines[["top", "right"]].set_visible(False)
        role_order = ["anion_fixed", "cation_fixed"]
        role_labels = ["Cation\nsubstitution", "Anion\nsubstitution"]
        matched_columns = [
            "fixed_role",
            "left_sample_id",
            "right_sample_id",
            *[
                f"observed_abs_log_difference_{property_name}"
                for property_name in properties_d
            ],
        ]
        matched_source = matched_pairs[
            [column for column in matched_columns if column in matched_pairs]
        ].copy()
        for axis, property_name in zip(axes_d[2:], properties_d):
            value_column = f"observed_abs_log_difference_{property_name}"
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
                boxprops={"facecolor": "#D9EAF3", "edgecolor": BLUE, "linewidth": 0.7},
                medianprops={"color": RED, "linewidth": 1.0},
                whiskerprops={"color": BLUE, "linewidth": 0.7},
                capprops={"color": BLUE, "linewidth": 0.7},
                flierprops={
                    "marker": "o",
                    "markersize": 0.55,
                    "markerfacecolor": GREY,
                    "markeredgecolor": "none",
                    "alpha": 0.08,
                },
            )
            axis.set_title(
                f"{short_properties[property_name]}\nmatched pairs",
                loc="left",
                fontsize=5.5,
                fontweight="bold",
            )
            axis.set_ylabel("|Δln(observed property)|", fontsize=5.0)
            axis.tick_params(labelsize=4.8, length=2)
            for index, values in enumerate(groups, start=1):
                axis.text(
                    index,
                    0.95,
                    f"n={len(values)}",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=4.7,
                )
            axis.spines[["top", "right"]].set_visible(False)
        axes_d[0].text(
            -0.38,
            1.20,
            "d",
            transform=axes_d[0].transAxes,
            fontsize=8.5,
            fontweight="bold",
        )
        axes_d[0].text(
            0,
            1.20,
            "Matched-pair and valid-SMILES counterfactual evidence",
            transform=axes_d[0].transAxes,
            fontsize=7,
            fontweight="bold",
        )
        prediction_export = prediction_source.copy()
        prediction_export.insert(0, "source_kind", "virtual_counterfactual")
        matched_export = matched_source.copy()
        matched_export.insert(0, "source_kind", "observed_matched_pair")
        panel_sources["d"] = pd.concat(
            [prediction_export, matched_export],
            ignore_index=True,
            sort=False,
        )

        # e | Shared-attention contrasts, explicitly presented as diagnostics.
        axis_e = figure.add_subplot(outer[3, 0])
        source_e = contrasts.copy()
        selected_categories = (
            source_e.assign(abs_contrast=source_e["high_minus_low"].abs())
            .groupby("interaction_category")["abs_contrast"]
            .max()
            .nlargest(8)
            .index
        )
        source_e = source_e.loc[
            source_e["interaction_category"].isin(selected_categories)
        ]
        pivot_e = source_e.pivot(
            index="interaction_category",
            columns="property",
            values="high_minus_low",
        )
        property_order_e = [
            name for name in PROPERTY_ORDER if name in pivot_e.columns
        ]
        pivot_e = pivot_e[property_order_e].loc[selected_categories]
        maximum_e = max(
            float(np.nanmax(np.abs(pivot_e.to_numpy(dtype=float)))),
            1e-12,
        )
        image_e = axis_e.imshow(
            pivot_e.to_numpy(dtype=float),
            cmap="PuOr_r",
            norm=TwoSlopeNorm(
                vmin=-maximum_e,
                vcenter=0.0,
                vmax=maximum_e,
            ),
            aspect="auto",
        )
        interaction_labels = {
            "H-bond-compatible pairs": "H-bond compatible",
            "alkyl-carbon--anion-polar-site pairs": "Alkyl C–anion polar",
            "aromatic-contact pairs": "Aromatic contact",
            "carboxylate-site-containing pairs": "Carboxylate site",
            "cation-aromatic-core--anion-polar-site pairs": "Cation aromatic–anion polar",
            "charge-complementary pairs": "Charge complement",
            "fluorine-containing pairs": "F-containing",
            "high-charge-magnitude pairs": "High |charge|",
            "non-fluorine-halogen-containing pairs": "Non-F halogen",
            "other atom pairs": "Other atom pairs",
            "sulfonyl-site-containing pairs": "Sulfonyl site",
        }
        axis_e.set_xticks(
            range(len(property_order_e)),
            [short_properties[name] for name in property_order_e],
            rotation=38,
            ha="right",
        )
        axis_e.set_yticks(
            range(len(pivot_e)),
            [interaction_labels.get(value, value) for value in pivot_e.index],
        )
        axis_e.tick_params(axis="both", labelsize=4.8)
        colorbar_e = figure.colorbar(
            image_e,
            ax=axis_e,
            fraction=0.04,
            pad=0.025,
        )
        colorbar_e.ax.tick_params(labelsize=4.7, length=2)
        colorbar_e.set_label("High − low attention per pair", fontsize=5)
        axis_e.set_title(
            "Shared cross-ion attention diagnostics",
            loc="left",
            fontsize=7,
            fontweight="bold",
        )
        axis_e.text(
            -0.16,
            1.08,
            "e",
            transform=axis_e.transAxes,
            fontsize=8.5,
            fontweight="bold",
        )
        axis_e.text(
            0.0,
            -0.36,
            "Model focus pattern; not interaction energy",
            transform=axis_e.transAxes,
            fontsize=5.0,
            color=GREY,
        )
        panel_sources["e"] = source_e

        # f | Post hoc connection to the unchanged audited Top-8.
        axis_f = figure.add_subplot(outer[3, 1])
        source_f = trajectory.copy()
        top_ids = set(top8["candidate_id"].astype(str))
        source_f["formal_shortlist"] = source_f["candidate_id"].astype(str).isin(
            top_ids
        )
        heat = np.log10(
            source_f["thermal_diffusivity_worst"]
            .to_numpy(dtype=float)
            .clip(min=1e-12)
        )
        capacity = source_f["volumetric_heat_capacity_worst"].to_numpy(dtype=float)
        sizes = 4 + 18 * (capacity - np.nanmin(capacity)) / max(
            np.nanmax(capacity) - np.nanmin(capacity),
            1e-12,
        )
        scatter_f = axis_f.scatter(
            source_f["viscosity_worst"],
            source_f["conductivity_worst"],
            c=heat,
            s=sizes,
            cmap="viridis",
            alpha=0.35,
            linewidths=0,
            rasterized=False,
        )
        selected_f = source_f.loc[source_f["formal_shortlist"]].copy()
        axis_f.scatter(
            selected_f["viscosity_worst"],
            selected_f["conductivity_worst"],
            facecolors="none",
            edgecolors=RED,
            s=35,
            linewidths=0.9,
            label="Formal Top-8",
        )
        selected_for_labels = selected_f.sort_values(
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
            label_x = float(compact_cluster["viscosity_worst"].max()) * 1.8
            label_y = np.geomspace(
                float(compact_cluster["conductivity_worst"].max()) * 1.30,
                float(compact_cluster["conductivity_worst"].min()) * 0.68,
                len(compact_cluster),
            )
            for text_y, row in zip(
                label_y,
                compact_cluster.itertuples(index=False),
            ):
                axis_f.annotate(
                    row.candidate_id.replace("UPR-", ""),
                    (row.viscosity_worst, row.conductivity_worst),
                    xytext=(label_x, text_y),
                    textcoords="data",
                    fontsize=4.7,
                    arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.35},
                    bbox={
                        "boxstyle": "round,pad=0.05",
                        "fc": "white",
                        "ec": "none",
                        "alpha": 0.76,
                    },
                )
        isolated_offsets = [
            (5, 7 if index % 2 == 0 else -11)
            for index in range(len(isolated))
        ]
        for offset, row in zip(
            isolated_offsets,
            isolated.sort_values(
                ["conductivity_worst", "candidate_id"],
                ascending=[False, True],
            ).itertuples(index=False),
        ):
            axis_f.annotate(
                row.candidate_id.replace("UPR-", ""),
                (row.viscosity_worst, row.conductivity_worst),
                xytext=offset,
                textcoords="offset points",
                fontsize=4.7,
                arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.35},
                bbox={
                    "boxstyle": "round,pad=0.05",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.72,
                },
            )
        axis_f.set_xscale("log")
        axis_f.set_yscale("log")
        axis_f.set_xlabel(
            "Worst-window viscosity (Pa s; lower is favourable)",
            fontsize=5,
        )
        axis_f.set_ylabel(
            "Worst-window conductivity (S m$^{-1}$)",
            fontsize=5,
        )
        axis_f.tick_params(labelsize=4.8, length=2)
        colorbar_f = figure.colorbar(
            scatter_f,
            ax=axis_f,
            fraction=0.04,
            pad=0.025,
        )
        colorbar_f.ax.tick_params(labelsize=4.7, length=2)
        colorbar_f.set_label(
            "log$_{10}$(thermal diffusivity / m$^2$ s$^{-1}$)",
            fontsize=5,
        )
        axis_f.legend(
            frameon=False,
            fontsize=4.9,
            loc="upper right",
            handletextpad=0.3,
        )
        axis_f.set_title(
            "Post hoc context for the unchanged shortlist",
            loc="left",
            fontsize=7,
            fontweight="bold",
        )
        axis_f.text(
            -0.15,
            1.08,
            "f",
            transform=axis_f.transAxes,
            fontsize=8.5,
            fontweight="bold",
        )
        axis_f.spines[["top", "right"]].set_visible(False)
        panel_sources["f"] = source_f

        figure.subplots_adjust(
            left=0.105,
            right=0.975,
            bottom=0.055,
            top=0.975,
        )
        return self._save_composite(figure, stem, panel_sources)
