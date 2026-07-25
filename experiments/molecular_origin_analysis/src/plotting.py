"""Publication-grade Matplotlib figures with source-data export."""

from __future__ import annotations

from pathlib import Path
import textwrap
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
        *,
        fixed_canvas: bool = False,
        print_width_preview: bool = False,
    ) -> list[Path]:
        """Save an editable multi-panel figure and one source table per panel."""

        stem = ensure_within(stem, MODULE_ROOT)
        stem.parent.mkdir(parents=True, exist_ok=True)
        source_dir = stem.parent.parent / "tables" / "figure_source_data"
        expected_source_names = {
            f"{stem.name}_panel_{panel}_source_data.csv"
            for panel in panel_sources
        }
        for existing_source in source_dir.glob(
            f"{stem.name}_panel_*_source_data.csv"
        ):
            if existing_source.name not in expected_source_names:
                existing_source.unlink()
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
                bbox_inches=figure.bbox_inches if fixed_canvas else "tight",
                pad_inches=0 if fixed_canvas else 0.1,
                **extra_save_options,
            )
            outputs.append(path)
        if print_width_preview:
            preview_path = stem.with_name(f"{stem.name}_17p8cm").with_suffix(".png")
            figure.savefig(
                preview_path,
                dpi=600,
                facecolor="white",
                bbox_inches=figure.bbox_inches,
                pad_inches=0,
            )
            outputs.append(preview_path)
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

    @staticmethod
    def _rule_evidence_source(
        rules: pd.DataFrame,
        nonlinear: pd.DataFrame,
        top_k: int = 1,
    ) -> pd.DataFrame:
        """Return deterministic curve-supported structural priors per property."""

        if rules.empty or nonlinear.empty:
            raise ValueError("Rule-evidence inputs are empty")
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        source = rules.copy()
        source["partial_r"] = pd.to_numeric(
            source["statistical_evidence"].str.extract(
                r"partial r=([+-]?[0-9]*\.?[0-9]+)"
            )[0],
            errors="coerce",
        )
        source["bootstrap_ci_low"] = pd.to_numeric(
            source["statistical_evidence"].str.extract(
                r"bootstrap CI=\[([+-]?[0-9]*\.?[0-9]+)"
            )[0],
            errors="coerce",
        )
        source["bootstrap_ci_high"] = pd.to_numeric(
            source["statistical_evidence"].str.extract(
                r"bootstrap CI=\[[+-]?[0-9]*\.?[0-9]+,\s*"
                r"([+-]?[0-9]*\.?[0-9]+)\]"
            )[0],
            errors="coerce",
        )
        source["fdr_q"] = pd.to_numeric(
            source["statistical_evidence"].str.extract(
                r"(?:^|;\s*)q=([0-9.eE+-]+)"
            )[0],
            errors="coerce",
        )
        source["attribution_rank"] = pd.to_numeric(
            source["attribution_evidence"].str.extract(r"rank=([0-9]+)")[0],
            errors="coerce",
        )
        source["level_order"] = source["confidence_level"].map(
            {"Level A": 0, "Level B": 1, "Level C": 2}
        )
        source["property_order"] = source["property"].map(
            {name: index for index, name in enumerate(PROPERTY_ORDER)}
        )
        source["abs_partial_r"] = source["partial_r"].abs()
        curve_pairs = nonlinear[["property", "feature"]].drop_duplicates().rename(
            columns={"feature": "structural_factor"}
        )
        curve_pairs["curve_available"] = True
        source = source.merge(
            curve_pairs,
            on=["property", "structural_factor"],
            how="left",
        )
        eligible = source.loc[
            source["partial_r"].notna()
            & source["curve_available"].eq(True)
        ].copy()
        if eligible.empty:
            raise ValueError("No rule has an auditable nonlinear response curve")
        primary = (
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
            .head(top_k)
            .sort_values(
                [
                    "property_order",
                    "level_order",
                    "abs_partial_r",
                    "family_consistency",
                    "structural_factor",
                ],
                ascending=[True, True, False, False, True],
            )
        )
        primary["selection_rank"] = (
            primary.groupby("property", sort=False).cumcount() + 1
        )
        primary["selection_rule"] = (
            "curve-supported evidence ordering: confidence level, absolute "
            "partial correlation, family consistency, then factor name"
        )
        maximum_rank = (
            source.groupby("property")["attribution_rank"]
            .max()
            .rename("maximum_attribution_rank")
        )
        primary = primary.merge(
            maximum_rank,
            left_on="property",
            right_index=True,
            how="left",
        )
        denominator = (primary["maximum_attribution_rank"] - 1).clip(
            lower=1.0
        )
        primary["attribution_percentile"] = (
            1.0 - (primary["attribution_rank"] - 1.0) / denominator
        ).clip(0.0, 1.0)

        trend_records: list[dict[str, Any]] = []
        for (property_name, feature), group in nonlinear.groupby(
            ["property", "feature"],
            sort=False,
        ):
            values = group[
                ["feature_mean", "response_log_mean"]
            ].apply(pd.to_numeric, errors="coerce").dropna()
            if len(values) < 2:
                coefficient = np.nan
            else:
                x_rank = values["feature_mean"].rank().to_numpy(dtype=float)
                y_rank = values["response_log_mean"].rank().to_numpy(dtype=float)
                coefficient = (
                    float(np.corrcoef(x_rank, y_rank)[0, 1])
                    if np.std(x_rank) > 0 and np.std(y_rank) > 0
                    else np.nan
                )
            trend_records.append(
                {
                    "property": property_name,
                    "structural_factor": feature,
                    "binned_response_spearman": coefficient,
                }
            )
        primary = primary.merge(
            pd.DataFrame.from_records(trend_records),
            on=["property", "structural_factor"],
            how="left",
        )
        primary["abs_binned_response_spearman"] = (
            primary["binned_response_spearman"].abs().clip(0.0, 1.0)
        )
        primary["response_direction_consistent"] = (
            np.sign(primary["binned_response_spearman"])
            == np.sign(primary["partial_r"])
        )
        primary["interpretation_scope"] = (
            "qualitative structure-based prior; not a property prediction "
            "or electrolyte-suitability label"
        )
        return primary.drop(
            columns=["level_order", "property_order"],
            errors="ignore",
        )

    @staticmethod
    def _draw_rule_evidence_profile(
        figure: plt.Figure,
        axis: plt.Axes,
        source: pd.DataFrame,
        compact: bool,
    ) -> None:
        metric_columns = [
            "abs_partial_r",
            "family_consistency",
            "attribution_percentile",
            "abs_binned_response_spearman",
        ]
        metric_labels = [
            "|partial r|",
            "Family\nconsistency",
            "Attribution\npercentile",
            "Response\nmonotonicity",
        ]
        matrix = source[metric_columns].to_numpy(dtype=float)
        image = axis.imshow(
            matrix,
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        compact_factor_labels = {
            "anion_fluorine_fraction": "anion F fraction",
            "cation_hetero_aromatic_atom": "cation heteroaromatic atom",
            "pair_alkyl_chain_length_sum": "alkyl-chain-length sum",
            "pair_total_molecular_weight_scaled": "total molecular weight",
            "cation_aromatic_atom_fraction_functional_group": (
                "cation aromatic fraction (FG)"
            ),
            "anion_molecular_weight_scaled": "anion molecular weight",
        }
        compact_property_labels = {
            "ElectricalConductivity": "Conductivity",
            "ThermalConductivity": "Thermal cond.",
        }
        feature_labels = []
        for row in source.itertuples(index=False):
            factor = compact_factor_labels.get(
                row.structural_factor,
                row.structural_factor
                .replace("_functional_group", " (FG)")
                .replace("_", " "),
            )
            sign = "+" if row.partial_r > 0 else "−"
            property_label = (
                compact_property_labels.get(
                    row.property,
                    _property_label(row.property),
                )
                if compact
                else _property_label(row.property)
            )
            feature_labels.append(
                f"{property_label}\n{sign} {factor}"
            )
        axis.set_yticks(range(len(source)), feature_labels)
        axis.set_xticks(
            range(len(metric_labels)),
            metric_labels,
            rotation=28 if compact else 0,
            ha="right" if compact else "center",
        )
        axis.tick_params(
            axis="both",
            labelsize=4.5 if compact else 6.5,
            length=0,
        )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=4.2 if compact else 6.0,
                        color="white" if value >= 0.58 else "#263238",
                    )
        colorbar = figure.colorbar(
            image,
            ax=axis,
            fraction=0.04 if compact else 0.025,
            pad=0.025,
        )
        colorbar.ax.tick_params(
            labelsize=4.4 if compact else 6.0,
            length=2,
        )
        colorbar.set_label(
            "Normalized evidence support",
            fontsize=4.8 if compact else 6.5,
        )
        axis.set_title(
            "Evidence profile of structure-based priors",
            loc="left",
            fontsize=7 if compact else 9,
            fontweight="bold",
        )
        axis.text(
            0.0,
            -0.31 if compact else -0.20,
            "Qualitative tendency cues; not property predictions or suitability labels",
            transform=axis.transAxes,
            fontsize=4.6 if compact else 6.2,
            color=GREY,
        )

    @staticmethod
    def _draw_rule_evidence_forest(
        axis: plt.Axes,
        source: pd.DataFrame,
        compact: bool,
    ) -> None:
        """Draw pooled associations alongside family/model evidence diagnostics."""

        compact_factor_labels = {
            "anion_fluorine_fraction": "anion F fraction",
            "cation_hetero_aromatic_atom": "cation heteroaromatic atom",
            "pair_alkyl_chain_length_sum": "alkyl-chain-length sum",
            "pair_total_molecular_weight_scaled": "total molecular weight",
            "cation_aromatic_atom_fraction_functional_group": (
                "cation aromatic fraction (FG)"
            ),
            "anion_molecular_weight_scaled": "anion molecular weight",
        }
        compact_property_labels = {
            "ElectricalConductivity": "Conductivity",
            "ThermalConductivity": "Thermal cond.",
        }
        feature_labels: list[str] = []
        for row in source.itertuples(index=False):
            factor = compact_factor_labels.get(
                row.structural_factor,
                row.structural_factor
                .replace("_functional_group", " (FG)")
                .replace("_", " "),
            )
            property_label = compact_property_labels.get(
                row.property,
                _property_label(row.property),
            )
            sign = "+" if row.partial_r > 0 else "−"
            feature_labels.append(f"{property_label}\n{sign} {factor}")

        y_positions = np.arange(len(source))[::-1]
        axis.axvspan(1.02, 1.66, color="#F3F4F6", zorder=0)
        axis.axvline(0.0, color="#9CA3AF", linewidth=0.65, zorder=0)
        axis.axvline(1.02, color="#D1D5DB", linewidth=0.5, zorder=0)
        for index, (y_value, row) in enumerate(
            zip(y_positions, source.itertuples(index=False))
        ):
            if index % 2 == 0:
                axis.axhspan(
                    y_value - 0.48,
                    y_value + 0.48,
                    color="#F8FAFC",
                    zorder=-1,
                )
            point_color = RED if row.partial_r > 0 else BLUE
            ci_low = getattr(row, "bootstrap_ci_low", np.nan)
            ci_high = getattr(row, "bootstrap_ci_high", np.nan)
            if np.isfinite(ci_low) and np.isfinite(ci_high):
                axis.plot(
                    [ci_low, ci_high],
                    [y_value, y_value],
                    color="#4B5563",
                    linewidth=2.3 if compact else 3.0,
                    solid_capstyle="round",
                    zorder=2,
                )
                axis.plot(
                    [ci_low, ci_high],
                    [y_value, y_value],
                    color="white",
                    linewidth=0.65 if compact else 0.9,
                    solid_capstyle="round",
                    zorder=3,
                )
            axis.scatter(
                [row.partial_r],
                [y_value],
                s=19 if compact else 32,
                marker="D",
                color=point_color,
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
            diagnostics = [
                row.family_consistency,
                row.attribution_percentile,
                row.abs_binned_response_spearman,
            ]
            for x_value, metric_value in zip(
                [1.16, 1.37, 1.58],
                diagnostics,
            ):
                valid_value = (
                    float(np.clip(metric_value, 0.0, 1.0))
                    if np.isfinite(metric_value)
                    else np.nan
                )
                axis.scatter(
                    [x_value],
                    [y_value],
                    s=54 if compact else 86,
                    color=(
                        mpl.colormaps["YlGnBu"](0.20 + 0.72 * valid_value)
                        if np.isfinite(valid_value)
                        else "#E5E7EB"
                    ),
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=3,
                )
                axis.text(
                    x_value,
                    y_value,
                    f"{valid_value:.2f}" if np.isfinite(valid_value) else "NA",
                    ha="center",
                    va="center",
                    fontsize=3.6 if compact else 5.1,
                    color=(
                        "white"
                        if np.isfinite(valid_value) and valid_value >= 0.57
                        else "#263238"
                    ),
                    zorder=4,
                )

        axis.set_yticks(y_positions, feature_labels)
        axis.set_xlim(-1.03, 1.68)
        axis.set_ylim(-0.62, len(source) - 0.28)
        axis.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        axis.set_xlabel(
            "Condition-controlled partial $r$",
            fontsize=4.8 if compact else 6.5,
            labelpad=2,
        )
        axis.tick_params(
            axis="both",
            labelsize=4.35 if compact else 6.2,
            length=2,
        )
        for x_value, label in zip(
            [1.16, 1.37, 1.58],
            ["Fam.", "Attr.", "Mono."],
        ):
            axis.text(
                x_value,
                len(source) - 0.13,
                label,
                ha="center",
                va="bottom",
                fontsize=4.0 if compact else 5.7,
                fontweight="bold",
            )
        axis.text(
            0.0,
            len(source) - 0.13,
            "pooled $r$ (diamond); family-bootstrap range (bar)",
            ha="center",
            va="bottom",
            fontsize=3.8 if compact else 5.4,
            color=GREY,
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(
            "Primary-rule evidence forest",
            loc="left",
            fontsize=7 if compact else 9,
            fontweight="bold",
        )
        axis.text(
            0.0,
            -0.27 if compact else -0.18,
            "Fam., family consistency; Attr., attribution percentile; "
            "Mono., response monotonicity",
            transform=axis.transAxes,
            fontsize=4.3 if compact else 6.0,
            color=GREY,
        )

    @staticmethod
    def _draw_rule_evidence_flow(
        axis: plt.Axes,
        source: pd.DataFrame,
        compact: bool,
    ) -> None:
        """Draw four auditable evidence steps for each primary structural rule."""

        compact_factor_labels = {
            "anion_fluorine_fraction": "anion F fraction",
            "cation_hetero_aromatic_atom": "cation heteroaromatic atom",
            "pair_alkyl_chain_length_sum": "alkyl-chain-length sum",
            "pair_total_molecular_weight_scaled": "total molecular weight",
            "cation_aromatic_atom_fraction_functional_group": (
                "cation aromatic fraction (FG)"
            ),
            "anion_molecular_weight_scaled": "anion molecular weight",
        }
        compact_property_labels = {
            "ElectricalConductivity": "Conductivity",
            "ThermalConductivity": "Thermal cond.",
        }
        metric_columns = [
            "abs_partial_r",
            "family_consistency",
            "attribution_percentile",
            "abs_binned_response_spearman",
        ]
        x_positions = np.arange(len(metric_columns), dtype=float)
        y_positions = np.arange(len(source))[::-1]
        row_labels: list[str] = []

        for row_index, (y_value, row) in enumerate(
            zip(y_positions, source.itertuples(index=False))
        ):
            factor = compact_factor_labels.get(
                row.structural_factor,
                row.structural_factor
                .replace("_functional_group", " (FG)")
                .replace("_", " "),
            )
            property_label = compact_property_labels.get(
                row.property,
                _property_label(row.property),
            )
            sign = "↑" if row.partial_r > 0 else "↓"
            row_labels.append(f"{property_label}\n{sign} {factor}")
            if row_index % 2 == 0:
                axis.axhspan(
                    y_value - 0.45,
                    y_value + 0.45,
                    color="#F8FAFC",
                    zorder=0,
                )
            values = np.asarray(
                [getattr(row, column) for column in metric_columns],
                dtype=float,
            )
            axis.plot(
                x_positions,
                np.full_like(x_positions, y_value),
                color="#CBD5E1",
                linewidth=0.75,
                zorder=1,
            )
            for metric_index, (x_value, metric_value) in enumerate(
                zip(x_positions, values)
            ):
                valid_value = (
                    float(np.clip(metric_value, 0.0, 1.0))
                    if np.isfinite(metric_value)
                    else np.nan
                )
                direction_colour = RED if row.partial_r > 0 else BLUE
                axis.scatter(
                    [x_value],
                    [y_value],
                    s=68 if compact else 105,
                    color=(
                        mpl.colormaps["YlGnBu"](0.18 + 0.76 * valid_value)
                        if np.isfinite(valid_value)
                        else "#E5E7EB"
                    ),
                    edgecolor=direction_colour if metric_index == 0 else "white",
                    linewidth=1.15 if metric_index == 0 else 0.55,
                    zorder=3,
                )
                axis.text(
                    x_value,
                    y_value,
                    f"{valid_value:.2f}" if np.isfinite(valid_value) else "NA",
                    ha="center",
                    va="center",
                    fontsize=3.8 if compact else 5.2,
                    color=(
                        "white"
                        if np.isfinite(valid_value) and valid_value >= 0.52
                        else "#263238"
                    ),
                    zorder=4,
                )

        axis.set_yticks(y_positions, row_labels)
        axis.set_xticks(
            x_positions,
            [
                "Association\n$|r_p|$",
                "Across-family\nconsistency",
                "Model\nattribution",
                "Observed-trend\nmonotonicity",
            ],
        )
        axis.xaxis.tick_top()
        axis.tick_params(
            axis="both",
            labelsize=4.1 if compact else 6.1,
            length=0,
            pad=2,
        )
        axis.set_xlim(-0.42, 3.42)
        axis.set_ylim(-0.56, len(source) - 0.42)
        axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
        axis.set_title(
            "Four-source evidence convergence",
            loc="left",
            fontsize=7 if compact else 9,
            fontweight="bold",
            pad=25 if compact else 30,
        )
        axis.text(
            0.0,
            -0.17 if compact else -0.13,
            "Stronger agreement across all four steps supports the prior. "
            "First-circle rim: red, positive; blue, negative. Not a suitability score.",
            transform=axis.transAxes,
            fontsize=4.0 if compact else 5.8,
            color=GREY,
        )

    def rule_evidence_profile(
        self,
        rules: pd.DataFrame,
        nonlinear: pd.DataFrame,
        stem: Path,
    ) -> list[Path]:
        source = self._rule_evidence_source(rules, nonlinear)
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        self._draw_rule_evidence_flow(
            axis,
            source,
            compact=False,
        )
        figure.tight_layout()
        return self._save(figure, stem, source)

    def composite_results_figure(
        self,
        rules: pd.DataFrame,
        associations: pd.DataFrame,
        nonlinear: pd.DataFrame,
        matched_pairs: pd.DataFrame,
        contrasts: pd.DataFrame,
        stem: Path,
    ) -> list[Path]:
        """Build the manuscript-facing a–d evidence figure from audited tables."""

        required_frames = {
            "rules": rules,
            "associations": associations,
            "nonlinear": nonlinear,
            "matched_pairs": matched_pairs,
            "contrasts": contrasts,
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

        figure = plt.figure(figsize=(7.2, 9.8), facecolor="white")
        outer = figure.add_gridspec(
            4,
            2,
            height_ratios=[1.58, 1.38, 1.02, 1.42],
            width_ratios=[0.92, 1.08],
            hspace=0.68,
            wspace=0.46,
        )
        panel_sources: dict[str, pd.DataFrame] = {}

        # a | Compact multi-link evidence map with quantitative effect widths.
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
            .head(3)
            .sort_values("property_order")
        )
        source_a["property_rank"] = (
            source_a.groupby("property", sort=False).cumcount() + 1
        )
        source_a["line_width_pt"] = (
            0.35 + 2.4 * source_a["abs_partial_r"].clip(0.0, 1.0)
        )
        source_a["selection_rule"] = (
            "top three Level A/B links per property; deterministic evidence, "
            "|partial r|, family-consistency and name ordering"
        )

        properties_a = [
            property_name
            for property_name in PROPERTY_ORDER
            if property_name in set(source_a["property"])
        ]
        property_y_a = {
            property_name: y_value
            for property_name, y_value in zip(
                properties_a,
                np.linspace(0.86, 0.18, len(properties_a)),
            )
        }
        factor_score_records = source_a.assign(
            property_position=source_a["property"].map(
                {name: index for index, name in enumerate(properties_a)}
            ),
            association_weight=source_a["abs_partial_r"].clip(lower=1e-12),
        )
        factor_score_records["weighted_position"] = (
            factor_score_records["property_position"]
            * factor_score_records["association_weight"]
        )
        factor_scores = (
            factor_score_records.groupby("structural_factor", as_index=False)
            .agg(
                weighted_position=("weighted_position", "sum"),
                association_weight=("association_weight", "sum"),
            )
        )
        factor_scores["weighted_property_position"] = (
            factor_scores["weighted_position"]
            / factor_scores["association_weight"]
        )
        factor_scores = factor_scores.sort_values(
            ["weighted_property_position", "structural_factor"],
            ascending=[True, True],
        )
        factors_a = factor_scores["structural_factor"].tolist()
        factor_y_a = {
            factor: y_value
            for factor, y_value in zip(
                factors_a,
                np.linspace(0.89, 0.15, len(factors_a)),
            )
        }

        for row in source_a.sort_values(
            ["abs_partial_r", "property", "structural_factor"]
        ).itertuples(index=False):
            color = RED if row.partial_r > 0 else BLUE
            axis_a.plot(
                [0.47, 0.69],
                [
                    factor_y_a[row.structural_factor],
                    property_y_a[row.property],
                ],
                color=color,
                linewidth=row.line_width_pt,
                alpha=0.72,
                solid_capstyle="round",
                transform=axis_a.transAxes,
                clip_on=False,
                zorder=1,
            )

        for factor, y_value in factor_y_a.items():
            axis_a.text(
                0.45,
                y_value,
                compact_feature_label(factor),
                ha="right",
                va="center",
                fontsize=4.15,
                transform=axis_a.transAxes,
                zorder=3,
            )
        for property_name, y_value in property_y_a.items():
            axis_a.text(
                0.72,
                y_value,
                _property_label(property_name),
                ha="left",
                va="center",
                fontsize=4.8,
                fontweight="bold",
                transform=axis_a.transAxes,
                zorder=3,
            )

        axis_a.text(
            0.24,
            0.98,
            "Audited structural factors",
            ha="center",
            va="bottom",
            fontsize=5.0,
            fontweight="bold",
            transform=axis_a.transAxes,
        )
        axis_a.text(
            0.85,
            0.98,
            "Properties",
            ha="center",
            va="bottom",
            fontsize=5.0,
            fontweight="bold",
            transform=axis_a.transAxes,
        )
        legend_handles = [
            Line2D([0], [0], color=RED, lw=1.2, label="+"),
            Line2D([0], [0], color=BLUE, lw=1.2, label="−"),
            *[
                Line2D(
                    [0],
                    [0],
                    color=GREY,
                    lw=0.35 + 2.4 * strength,
                    label=f"|r|={strength:.1f}",
                )
                for strength in (0.3, 0.6, 0.9)
            ],
        ]
        axis_a.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.53, -0.075),
            ncol=5,
            frameon=False,
            fontsize=3.9,
            handlelength=1.45,
            handletextpad=0.25,
            columnspacing=0.55,
            borderaxespad=0,
        )
        axis_a.set_xlim(0, 1)
        axis_a.set_ylim(0, 1)
        axis_a.axis("off")
        axis_a.set_title(
            "Audited multi-link evidence map",
            loc="left",
            fontsize=7,
            fontweight="bold",
            pad=12,
        )
        axis_a.text(
            0,
            1.035,
            "Top three eligible links per property; width scales with |partial r|; non-causal",
            transform=axis_a.transAxes,
            fontsize=4.5,
            color=GREY,
            va="bottom",
        )
        axis_a.text(
            -0.16,
            1.12,
            "a",
            transform=axis_a.transAxes,
            fontsize=8.5,
            fontweight="bold",
        )
        panel_sources["a"] = source_a.drop(
            columns=["level_order", "property_order"],
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

        # c | Observed response curves for the six primary structural priors.
        grid_c = outer[1, :].subgridspec(1, 6, wspace=0.62)
        combinations = (
            source_a.loc[source_a["property_rank"] == 1]
            [["property", "structural_factor"]]
            .rename(columns={"structural_factor": "feature"})
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

        # d | Condition-matched ion substitutions across all six properties.
        grid_d = outer[2, :].subgridspec(1, 6, wspace=0.62)
        properties_d = [
            property_name
            for property_name in PROPERTY_ORDER
            if f"observed_abs_log_difference_{property_name}" in matched_pairs
        ]
        axes_d = [
            figure.add_subplot(grid_d[0, index])
            for index in range(len(properties_d))
        ]
        role_order = ["anion_fixed", "cation_fixed"]
        role_labels = ["Cation\nchange", "Anion\nchange"]
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
        property_value_columns = [
            f"observed_abs_log_difference_{property_name}"
            for property_name in properties_d
        ]
        matched_source = matched_source.dropna(
            subset=property_value_columns,
            how="all",
        )
        for panel_index, (axis, property_name) in enumerate(
            zip(axes_d, properties_d)
        ):
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
                short_properties[property_name],
                loc="left",
                fontsize=5.5,
                fontweight="bold",
            )
            if panel_index == 0:
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
            "Condition-matched ion-substitution responses",
            transform=axes_d[0].transAxes,
            fontsize=7,
            fontweight="bold",
        )
        matched_source.insert(0, "source_kind", "observed_matched_pair")
        panel_sources["d"] = matched_source

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

        # f | Evidence convergence of the primary structure-based priors.
        axis_f = figure.add_subplot(outer[3, 1])
        source_f = self._rule_evidence_source(rules, nonlinear)
        self._draw_rule_evidence_profile(
            figure,
            axis_f,
            source_f,
            compact=True,
        )
        axis_f.text(
            -0.15,
            1.08,
            "f",
            transform=axis_f.transAxes,
            fontsize=8.5,
            fontweight="bold",
        )
        panel_sources["f"] = source_f

        figure.subplots_adjust(
            left=0.105,
            right=0.975,
            bottom=0.055,
            top=0.975,
        )
        return self._save_composite(figure, stem, panel_sources)

    def composite_results_figure_v2(
        self,
        rules: pd.DataFrame,
        associations: pd.DataFrame,
        nonlinear: pd.DataFrame,
        matched_pairs: pd.DataFrame,
        contrasts: pd.DataFrame,
        stem: Path,
    ) -> list[Path]:
        """Submission-grade identity-balanced four-panel manuscript figure."""

        property_labels = {
            "Density": "Density",
            "Viscosity": "Viscosity",
            "ElectricalConductivity": "Electrical cond.",
            "HeatCapacity": "Heat capacity",
            "SurfaceTension": "Surface tension",
            "ThermalConductivity": "Thermal cond.",
        }
        curve_property_labels = {
            **property_labels,
            "ElectricalConductivity": "Electrical conductivity",
            "ThermalConductivity": "Thermal conductivity",
        }
        font = {
            "panel_label": 10.0,
            "main_title": 10.0,
            "subtitle": 7.5,
            "column_header": 8.0,
            "node": 7.0,
            "property_node": 7.5,
            "panel_title": 8.2,
            "axis_label": 7.5,
            "tick": 7.5,
            "legend": 7.0,
            "feature_key": 7.0,
            "annotation": 7.0,
        }
        scope_colours = {
            "Cation": "#4477AA",
            "Anion": "#CC6677",
            "Ion pair": "#7A6FA8",
        }
        rank_styles = {
            1: ("-", "o"),
            2: ("--", "s"),
            3: (":", "^"),
        }

        def full_feature(value: str) -> str:
            exact_labels = {
                "cation_longest_aliphatic_carbon_chain": (
                    "Cation longest alkyl chain"
                ),
                "pair_cation_positive_anion_negative_fraction": (
                    "Pair charge-complementarity fraction"
                ),
                "pair_cation_positive_anion_negative": (
                    "Pair charge-complementarity fraction"
                ),
                "pair_total_heavy_atom_count": "Pair heavy-atom count",
                "pair_total_heavy_atom_count_scaled": "Pair heavy-atom count",
                "cation_hetero_aromatic_atom": (
                    "Cation heteroaromatic fraction"
                ),
                "cation_positive_atom_fraction": (
                    "Cation positive-atom fraction"
                ),
                "anion_exact_molecular_weight": "Anion exact mass",
                "anion_exact_molecular_weight_scaled": "Anion exact mass",
                "cation_logp": "Cation RDKit MolLogP",
                "cation_logp_scaled": "Cation RDKit MolLogP",
                "anion_longest_aliphatic_carbon_chain": (
                    "Anion longest alkyl chain"
                ),
                "pair_alkyl_chain_length_sum": (
                    "Pair alkyl-chain length sum"
                ),
                "pair_anion_to_cation_heavy_atom_ratio": (
                    "Anion/cation heavy-atom ratio"
                ),
                "cation_radius_of_gyration_scaled": (
                    "Cation radius of gyration"
                ),
                "anion_heavy_atom_count_scaled": "Anion heavy-atom count",
                "cation_heavy_atom_count_scaled": "Cation heavy-atom count",
                "pair_total_molecular_weight_scaled": "Pair molecular weight",
                "cation_charged_atom_fraction": (
                    "Cation charged-atom fraction"
                ),
                "pair_radius_of_gyration_difference": (
                    "Pair radius-of-gyration difference"
                ),
                "anion_fluorine_fraction": "Anion fluorine fraction",
                "cation_carbon_fraction": "Cation carbon fraction",
                "anion_fluorinated_carbon": "Anion fluorinated carbon",
                "anion_hbond_donor": "Anion H-bond donor",
            }
            raw = str(value)
            if raw in exact_labels:
                return exact_labels[raw]
            text = (
                raw.replace("_functional_group", " functional group")
                .replace("_scaled", "")
                .replace("_", " ")
                .replace("logp", "RDKit MolLogP")
            )
            return text[:1].upper() + text[1:]

        motif_labels = {
            "fluorine-containing pairs": "F-containing pairs",
            "other atom pairs": "Other atom pairs",
            "H-bond-compatible pairs": "H-bond-compatible pairs",
            "alkyl-carbon--anion-polar-site pairs": (
                "Alkyl-carbon–anion-polar-site pairs"
            ),
            "aromatic-contact pairs": "Aromatic-contact pairs",
            "charge-complementary pairs": "Charge-complementary pairs",
            "high-charge-magnitude pairs": "High-charge-magnitude pairs",
            "sulfonyl-site-containing pairs": "Sulfonyl-site pairs",
        }

        if {
            "property",
            "structural_factor",
            "confidence_level",
        }.issubset(rules.columns):
            evidence_lookup = (
                rules[
                    ["property", "structural_factor", "confidence_level"]
                ]
                .drop_duplicates(["property", "structural_factor"])
                .rename(columns={"structural_factor": "feature"})
            )
        else:
            evidence_lookup = (
                associations[["property", "feature"]]
                .drop_duplicates()
                .assign(confidence_level="Level C")
            )

        def label(
            axis: plt.Axes,
            value: str,
            x: float = -0.13,
            y: float = 1.08,
        ) -> None:
            axis.text(
                x,
                y,
                value,
                transform=axis.transAxes,
                fontsize=font["panel_label"],
                fontweight="bold",
                va="top",
            )

        figure = plt.figure(
            figsize=(178.0 / 25.4, 228.0 / 25.4),
            facecolor="white",
        )
        outer = figure.add_gridspec(
            2,
            2,
            height_ratios=[1.08, 1.56],
            width_ratios=[0.43, 0.57],
            hspace=0.27,
            wspace=0.31,
        )
        left = outer[1, 0].subgridspec(2, 1, hspace=0.68)
        curves = outer[1, 1].subgridspec(3, 2, hspace=0.68, wspace=0.36)
        panel_sources: dict[str, pd.DataFrame] = {}

        # a | Integrated, evidence-gated association and attention map.
        axis_a = figure.add_subplot(outer[0, :])
        association_a = associations.loc[
            associations["main_figure_eligible"].fillna(False).astype(bool)
        ].copy().merge(
            evidence_lookup,
            on=["property", "feature"],
            how="left",
            validate="many_to_one",
        )
        association_a["confidence_level"] = association_a[
            "confidence_level"
        ].fillna("Level C")
        association_a["line_alpha"] = association_a[
            "confidence_level"
        ].map({"Level B": 0.96, "Level C": 0.56}).fillna(0.56)
        association_a["abs_effect"] = association_a[
            "partial_correlation"
        ].abs()
        association_a["line_width_pt"] = (
            0.55 + 2.15 * association_a["abs_effect"].clip(0, 1)
        )
        attention_a = contrasts.loc[
            contrasts.get(
                "main_figure_eligible",
                pd.Series(False, index=contrasts.index),
            )
            .fillna(False)
            .astype(bool)
        ].copy()
        attention_a["abs_effect"] = attention_a["high_minus_low"].abs()
        attention_a = (
            attention_a.sort_values(
                ["property", "abs_effect", "interaction_category"],
                ascending=[True, False, True],
            )
            .groupby("property", sort=False)
            .head(3)
        )
        attention_max = max(
            float(attention_a["abs_effect"].max())
            if not attention_a.empty
            else 0.0,
            1e-12,
        )
        attention_a["line_width_pt"] = (
            0.55 + 2.15 * attention_a["abs_effect"] / attention_max
        )
        properties_a = [
            name
            for name in PROPERTY_ORDER
            if name in set(association_a["property"])
        ]
        y_property = dict(
            zip(properties_a, np.linspace(0.86, 0.18, len(properties_a)))
        )
        association_a["property_order"] = association_a["property"].map(
            {name: index for index, name in enumerate(properties_a)}
        )
        factors_a = (
            association_a.sort_values(
                ["property_order", "selection_rank", "feature"]
            )["feature"]
            .drop_duplicates()
            .tolist()
        )
        y_factor = dict(
            zip(factors_a, np.linspace(0.90, 0.13, len(factors_a)))
        )
        attention_a["property_order"] = attention_a["property"].map(
            {name: index for index, name in enumerate(properties_a)}
        )
        motifs_a = (
            attention_a.sort_values(
                ["property_order", "abs_effect", "interaction_category"],
                ascending=[True, False, True],
            )["interaction_category"]
            .drop_duplicates()
            .tolist()
        )
        y_motif = dict(
            zip(motifs_a, np.linspace(0.90, 0.13, max(len(motifs_a), 1)))
        )
        for row in association_a.itertuples(index=False):
            axis_a.plot(
                [0.31, 0.47],
                [y_factor[row.feature], y_property[row.property]],
                color=RED if row.partial_correlation > 0 else BLUE,
                lw=row.line_width_pt,
                alpha=row.line_alpha,
                ls="-" if row.confidence_level == "Level B" else "--",
                solid_capstyle="round",
                transform=axis_a.transAxes,
            )
        for row in attention_a.itertuples(index=False):
            axis_a.plot(
                [0.53, 0.665],
                [y_property[row.property], y_motif[row.interaction_category]],
                color=ORANGE if row.high_minus_low > 0 else PURPLE,
                lw=row.line_width_pt,
                alpha=0.72,
                solid_capstyle="round",
                transform=axis_a.transAxes,
            )
        for feature, y_value in y_factor.items():
            axis_a.text(
                0.30,
                y_value,
                full_feature(feature),
                ha="right",
                va="center",
                fontsize=font["node"],
                transform=axis_a.transAxes,
            )
        for property_name, y_value in y_property.items():
            axis_a.text(
                0.50,
                y_value,
                property_labels[property_name],
                ha="center",
                va="center",
                fontsize=font["property_node"],
                fontweight="bold",
                transform=axis_a.transAxes,
                bbox={
                    "boxstyle": "round,pad=0.17",
                    "facecolor": "#F1F5F9",
                    "edgecolor": "#CBD5E1",
                    "linewidth": 0.35,
                },
            )
        for motif, y_value in y_motif.items():
            axis_a.text(
                0.675,
                y_value,
                motif_labels.get(motif, str(motif)),
                ha="left",
                va="center",
                fontsize=font["node"],
                transform=axis_a.transAxes,
            )
        axis_a.text(
            0.15,
            0.99,
            "Condition-adjusted structural associations",
            ha="center",
            va="bottom",
            fontsize=font["column_header"],
            fontweight="bold",
            transform=axis_a.transAxes,
        )
        axis_a.text(
            0.50,
            0.99,
            "Thermophysical properties",
            ha="center",
            va="bottom",
            fontsize=font["column_header"],
            fontweight="bold",
            transform=axis_a.transAxes,
        )
        axis_a.text(
            0.825,
            0.99,
            "Shared cross-ion atom-pair\nattention contrasts",
            ha="center",
            va="bottom",
            fontsize=font["column_header"],
            fontweight="bold",
            transform=axis_a.transAxes,
        )
        direction_legend = axis_a.legend(
            handles=[
                Line2D(
                    [0], [0], color=RED, lw=1.8, label="positive partial $r$"
                ),
                Line2D(
                    [0], [0], color=BLUE, lw=1.8, label="negative partial $r$"
                ),
                Line2D(
                    [0], [0], color=ORANGE, lw=1.8,
                    label=r"$\Delta$attention $> 0$",
                ),
                Line2D(
                    [0], [0], color=PURPLE, lw=1.8,
                    label=r"$\Delta$attention $< 0$",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.50, -0.045),
            ncol=4,
            frameon=False,
            fontsize=font["legend"],
            handlelength=1.7,
            columnspacing=0.95,
        )
        axis_a.add_artist(direction_legend)
        axis_a.legend(
            handles=[
                Line2D(
                    [0], [0], color="#374151", lw=3.0,
                    label="line width: effect magnitude",
                ),
                Line2D(
                    [0], [0], color="#374151", lw=1.8, ls="-",
                    marker="o", markerfacecolor="#374151",
                    markeredgecolor="#374151", label="Level B",
                ),
                Line2D(
                    [0], [0], color="#6B7280", lw=1.8, ls="--",
                    marker="o", markerfacecolor="white",
                    markeredgecolor="#6B7280", label="Level C",
                ),
                Line2D(
                    [0], [0], color="#9CA3AF", lw=1.8,
                    label="Exploratory",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.50, -0.135),
            ncol=4,
            frameon=False,
            fontsize=font["legend"],
            handlelength=1.9,
            columnspacing=0.95,
        )
        figure.text(
            0.04,
            0.985,
            "Integrated molecular-structure–property evidence map",
            fontsize=font["main_title"],
            fontweight="bold",
            ha="left",
            va="top",
        )
        figure.text(
            0.04,
            0.957,
            "Identity-balanced associations and confidence-supported "
            "attention contrasts",
            fontsize=font["subtitle"],
            color=GREY,
            ha="left",
            va="top",
        )
        axis_a.set_xlim(0, 1)
        axis_a.set_ylim(0, 1)
        axis_a.axis("off")
        label(axis_a, "a", x=-0.075, y=1.02)
        panel_sources["a"] = pd.concat(
            [
                association_a.assign(
                    evidence_family="identity_balanced_association",
                    signed_effect=association_a["partial_correlation"],
                ),
                attention_a.assign(
                    evidence_family="shared_attention_contrast",
                    signed_effect=attention_a["high_minus_low"],
                ),
            ],
            ignore_index=True,
            sort=False,
        )

        # b | Strongest eligible association within each descriptor scope.
        axis_b = figure.add_subplot(left[0, 0])
        source_b = associations.loc[
            associations["eligibility_status"].eq("eligible")
        ].copy().merge(
            evidence_lookup,
            on=["property", "feature"],
            how="left",
            validate="many_to_one",
        )
        source_b["confidence_level"] = source_b[
            "confidence_level"
        ].fillna("Level C")
        source_b = source_b.loc[
            source_b["structural_scope"].isin(scope_colours)
        ]
        source_b["abs_effect"] = source_b["partial_correlation"].abs()
        source_b = (
            source_b.sort_values(
                ["property", "structural_scope", "abs_effect", "feature"],
                ascending=[True, True, False, True],
            )
            .groupby(["property", "structural_scope"], sort=False)
            .head(1)
        )
        ordered_b = [name for name in PROPERTY_ORDER if name in set(source_b.property)]
        y_base = {name: len(ordered_b) - index - 1 for index, name in enumerate(ordered_b)}
        offsets = {"Cation": 0.20, "Anion": 0.0, "Ion pair": -0.20}
        markers = {"Cation": "o", "Anion": "s", "Ion pair": "D"}
        for scope in scope_colours:
            subset = source_b.loc[source_b["structural_scope"] == scope]
            for row in subset.itertuples(index=False):
                y_value = y_base[row.property] + offsets[scope]
                is_level_b = row.confidence_level == "Level B"
                axis_b.plot(
                    [0, row.partial_correlation],
                    [y_value, y_value],
                    color=scope_colours[scope],
                    lw=1.35,
                    alpha=0.95 if is_level_b else 0.58,
                    ls="-" if is_level_b else "--",
                    zorder=1,
                )
                axis_b.scatter(
                    [row.partial_correlation],
                    [y_value],
                    facecolor=scope_colours[scope] if is_level_b else "white",
                    edgecolor=scope_colours[scope],
                    marker=markers[scope],
                    s=28,
                    linewidth=0.8,
                    alpha=0.98 if is_level_b else 0.72,
                    zorder=3,
                )
        axis_b.axvline(0, color="#94A3B8", lw=0.65)
        axis_b.set_yticks(
            [y_base[name] for name in ordered_b],
            [property_labels[name] for name in ordered_b],
        )
        axis_b.set_xlim(-1.02, 1.02)
        axis_b.set_xlabel(
            "Identity-balanced partial $r$",
            fontsize=font["axis_label"],
        )
        axis_b.set_title(
            "Strongest association by structural scope",
            loc="left",
            fontsize=font["panel_title"],
            fontweight="bold",
        )
        axis_b.legend(
            handles=[
                Line2D(
                    [0], [0], color=scope_colours[scope], marker=markers[scope],
                    lw=0, markerfacecolor=scope_colours[scope],
                    markeredgecolor=scope_colours[scope], label=scope,
                )
                for scope in scope_colours
            ],
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.43),
            frameon=False,
            fontsize=font["legend"],
            handletextpad=0.2,
            columnspacing=0.7,
        )
        axis_b.tick_params(labelsize=font["tick"], length=2)
        axis_b.grid(axis="x", color="#E5E7EB", lw=0.45)
        axis_b.spines[["top", "right", "left"]].set_visible(False)
        label(axis_b, "b", x=-0.23)
        panel_sources["b"] = source_b

        # c | Three nonredundant Q1–Q5 response shapes per property.
        source_c = nonlinear.loc[
            nonlinear["main_figure_eligible"].fillna(False).astype(bool)
        ].copy().merge(
            evidence_lookup,
            on=["property", "feature"],
            how="left",
            validate="many_to_one",
        )
        source_c["confidence_level"] = source_c[
            "confidence_level"
        ].fillna("Level C")
        source_c["line_alpha"] = source_c["confidence_level"].map(
            {"Level B": 0.96, "Level C": 0.62}
        ).fillna(0.62)
        source_c["full_feature_label"] = source_c["feature"].map(full_feature)
        source_c["exploratory"] = source_c["property"].eq(
            "ThermalConductivity"
        )
        for index, property_name in enumerate(PROPERTY_ORDER):
            axis = figure.add_subplot(curves[index // 2, index % 2])
            subset = source_c.loc[source_c["property"] == property_name]
            key_rows: list[tuple[int, str, str]] = []
            for (feature, rank), group in subset.groupby(
                ["feature", "selection_rank"],
                sort=True,
            ):
                group = group.sort_values("quantile_bin")
                rank_int = int(rank)
                scope = str(group["structural_scope"].iloc[0])
                _, marker = rank_styles[rank_int]
                is_exploratory = property_name == "ThermalConductivity"
                colour = (
                    "#8A9099"
                    if is_exploratory
                    else scope_colours.get(scope, GREY)
                )
                evidence_level = str(group["confidence_level"].iloc[0])
                is_level_b = evidence_level == "Level B"
                line_alpha = float(group["line_alpha"].iloc[0])
                x = group["quantile_bin"].to_numpy(dtype=float)
                y = group["response_log_mean"].to_numpy(dtype=float)
                lower = group["bootstrap_ci_low"].to_numpy(dtype=float)
                upper = group["bootstrap_ci_high"].to_numpy(dtype=float)
                axis.fill_between(
                    x,
                    lower,
                    upper,
                    color=colour,
                    alpha=0.11 if evidence_level == "Level B" else 0.055,
                    linewidth=0,
                )
                axis.plot(
                    x,
                    y,
                    color=colour,
                    ls="-" if is_level_b else "--",
                    marker=marker,
                    markerfacecolor=colour if is_level_b else "white",
                    markeredgecolor=colour,
                    markeredgewidth=0.85,
                    ms=4.0,
                    lw=1.35,
                    alpha=line_alpha,
                )
                axis.text(
                    x[-1] + 0.10,
                    y[-1],
                    str(rank_int),
                    color=colour,
                    alpha=line_alpha,
                    fontsize=font["annotation"],
                    fontweight="bold",
                    va="center",
                    clip_on=False,
                )
                key_rows.append(
                    (
                        rank_int,
                        full_feature(feature),
                        evidence_level,
                    )
                )
            n_unique = (
                int(subset["n_unique_ils"].max()) if not subset.empty else 0
            )
            axis.axhline(0, color="#CBD5E1", lw=0.45)
            axis.set_xticks(range(1, 6), [f"Q{i}" for i in range(1, 6)])
            axis.set_xlim(0.82, 5.52)
            y_low, y_high = axis.get_ylim()
            axis.set_ylim(y_low, y_high + 0.46 * (y_high - y_low))
            curve_title = curve_property_labels[property_name]
            if property_name in {
                "ElectricalConductivity",
                "ThermalConductivity",
            }:
                if property_name == "ThermalConductivity":
                    curve_title = (
                        "Thermal conductivity\n"
                        f"(Exploratory; $n_{{IL}}$={n_unique})"
                    )
                else:
                    curve_title = (
                        f"{curve_title}\n"
                        f"($n_{{IL}}$={n_unique})"
                    )
            else:
                curve_title = (
                    f"{curve_title}\n"
                    f"($n_{{IL}}$={n_unique})"
                )
            axis.set_title(
                curve_title,
                loc="left",
                fontsize=font["panel_title"],
                fontweight="bold",
                color=(
                    "#6B7280"
                    if property_name == "ThermalConductivity"
                    else "#111827"
                ),
            )
            axis.tick_params(labelsize=font["tick"], length=1.8)
            axis.grid(axis="y", color="#EEF2F7", lw=0.4)
            axis.spines[["top", "right"]].set_visible(
                property_name == "ThermalConductivity"
            )
            if property_name == "ThermalConductivity":
                for spine in axis.spines.values():
                    spine.set_color("#9CA3AF")
                    spine.set_linewidth(0.75)
            key_text = "\n".join(
                textwrap.fill(
                    f"{rank_int}  {feature_label}",
                    width=25,
                    subsequent_indent="   ",
                )
                for rank_int, feature_label, level in sorted(key_rows)
            )
            axis.text(
                0.98,
                0.96,
                key_text,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=font["feature_key"],
                linespacing=1.18,
                color=(
                    "#6B7280"
                    if property_name == "ThermalConductivity"
                    else "#374151"
                ),
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": (
                        "#9CA3AF"
                        if property_name == "ThermalConductivity"
                        else "none"
                    ),
                    "linewidth": 0.45,
                    "alpha": 0.86,
                },
                zorder=6,
            )
            if index % 2 == 0:
                axis.set_ylabel(
                    "Condition-adjusted\nresidual in ln(property)",
                    fontsize=font["axis_label"],
                )
            if index >= 4:
                axis.set_xlabel(
                    "Structural-factor quantile",
                    fontsize=font["axis_label"],
                )
            if index == 0:
                axis.text(
                    -0.26,
                    1.39,
                    "c",
                    transform=axis.transAxes,
                    fontsize=font["panel_label"],
                    fontweight="bold",
                )
                axis.text(
                    0,
                    1.46,
                    "Top-ranked condition-adjusted response shapes",
                    transform=axis.transAxes,
                    fontsize=font["panel_title"],
                    fontweight="bold",
                )
        panel_sources["c"] = source_c

        # d | Unique substitution-pair distributions.
        axis_d = figure.add_subplot(left[1, 0])
        source_d = matched_pairs.copy()
        source_d["substitution_scope"] = source_d["fixed_role"].map(
            {
                "anion_fixed": "Cation substitution",
                "cation_fixed": "Anion substitution",
            }
        )
        ordered_d = [
            name for name in PROPERTY_ORDER if name in set(source_d["property"])
        ]
        source_d["exploratory"] = source_d["property"].eq(
            "ThermalConductivity"
        )
        positions: list[float] = []
        values: list[np.ndarray] = []
        colours: list[str] = []
        thermal_position = (
            len(ordered_d) - ordered_d.index("ThermalConductivity") - 1
            if "ThermalConductivity" in ordered_d
            else None
        )
        if thermal_position is not None:
            axis_d.axhspan(
                thermal_position - 0.36,
                thermal_position + 0.36,
                facecolor="#F3F4F6",
                edgecolor="#9CA3AF",
                linewidth=0.7,
                zorder=-3,
            )
        for property_index, property_name in enumerate(ordered_d):
            base = len(ordered_d) - property_index - 1
            for scope, offset, colour in (
                ("Cation substitution", 0.18, "#4477AA"),
                ("Anion substitution", -0.18, "#CC6677"),
            ):
                group = source_d.loc[
                    (source_d["property"] == property_name)
                    & (source_d["substitution_scope"] == scope),
                    "observed_abs_log_difference",
                ].dropna()
                if group.empty:
                    continue
                positions.append(base + offset)
                values.append(group.to_numpy(dtype=float) + 1e-4)
                colours.append("#A8ADB4" if property_name == "ThermalConductivity" else colour)
        boxes = axis_d.boxplot(
            values,
            positions=positions,
            vert=False,
            widths=0.27,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#263238", "linewidth": 0.75},
            whiskerprops={"linewidth": 0.6, "color": "#64748B"},
            capprops={"linewidth": 0.6, "color": "#64748B"},
            boxprops={"linewidth": 0.55, "edgecolor": "#64748B"},
        )
        for patch, colour in zip(boxes["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.68)
        for property_name in ordered_d:
            base = len(ordered_d) - ordered_d.index(property_name) - 1
            counts = source_d.loc[
                source_d["property"] == property_name
            ].groupby("substitution_scope")["substitution_pair_id"].nunique()
            count_text = "/".join(
                str(int(counts.get(scope, 0)))
                for scope in ("Cation substitution", "Anion substitution")
            )
            axis_d.text(
                0.985,
                base,
                f"C/A pairs = {count_text}",
                ha="right",
                va="center",
                fontsize=font["annotation"],
                color=(
                    "#6B7280"
                    if property_name == "ThermalConductivity"
                    else "#4B5563"
                ),
                transform=axis_d.get_yaxis_transform(),
                bbox={
                    "boxstyle": "square,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
                zorder=5,
            )
        axis_d.set_xscale("log")
        all_box_values = np.concatenate(values)
        axis_d.set_xlim(
            max(float(np.nanpercentile(all_box_values, 0.25)) * 0.70, 1e-4),
            float(np.nanpercentile(all_box_values, 99.75)) * 6.0,
        )
        axis_d.set_yticks(
            [len(ordered_d) - index - 1 for index in range(len(ordered_d))],
            [
                (
                    "Thermal cond.\n(Exploratory)"
                    if name == "ThermalConductivity"
                    else property_labels[name]
                )
                for name in ordered_d
            ],
        )
        axis_d.set_xlabel(
            r"Absolute substitution contrast, $|\Delta\ln y|$",
            fontsize=font["axis_label"],
        )
        axis_d.set_title(
            "Condition-matched ion-substitution sensitivity",
            loc="left",
            fontsize=font["panel_title"],
            fontweight="bold",
        )
        axis_d.legend(
            handles=[
                Line2D([0], [0], color="#4477AA", lw=5, label="Cation"),
                Line2D([0], [0], color="#CC6677", lw=5, label="Anion"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.47),
            ncol=2,
            frameon=False,
            fontsize=font["legend"],
            handlelength=0.9,
            columnspacing=1.0,
        )
        axis_d.tick_params(labelsize=font["tick"], length=2)
        axis_d.grid(axis="x", which="both", color="#E5E7EB", lw=0.4)
        axis_d.spines[["top", "right", "left"]].set_visible(False)
        label(axis_d, "d", x=-0.23)
        panel_sources["d"] = source_d

        figure.subplots_adjust(
            left=0.12,
            right=0.985,
            bottom=0.145,
            top=0.925,
        )
        return self._save_composite(
            figure,
            stem,
            panel_sources,
            fixed_canvas=True,
            print_width_preview=True,
        )

    def heat_capacity_size_control_figure(
        self,
        identity_data: pd.DataFrame,
        stem: Path,
    ) -> list[Path]:
        """SI comparison of molar, mass-specific and MW-adjusted heat capacity."""

        if identity_data.empty:
            raise ValueError("Heat-capacity size-control identity table is empty")
        panels = [
            (
                "molar_condition_adjusted_log_response",
                "Molar $C_{p,m}$",
                "#4477AA",
            ),
            (
                "mass_specific_condition_adjusted_log_response",
                "Mass-specific $c_p$",
                "#228833",
            ),
            (
                "molecular_weight_adjusted_log_response",
                "MW-adjusted molar $C_{p,m}$",
                "#AA3377",
            ),
        ]
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(7.205, 2.35),
            sharex=True,
        )
        x = np.log(identity_data["molar_mass_g_mol"].to_numpy(dtype=float))
        source_parts = []
        for index, (column, title, colour) in enumerate(panels):
            axis = axes[index]
            y = identity_data[column].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            axis.scatter(
                x[finite],
                y[finite],
                s=8,
                alpha=0.48,
                color=colour,
                edgecolor="none",
            )
            if finite.sum() >= 4:
                slope, intercept = np.polyfit(x[finite], y[finite], 1)
                x_line = np.linspace(x[finite].min(), x[finite].max(), 100)
                axis.plot(x_line, intercept + slope * x_line, color="#263238", lw=1.0)
                rho = np.corrcoef(x[finite], y[finite])[0, 1]
                axis.text(
                    0.04,
                    0.94,
                    f"$r$={rho:+.2f}; $n_{{IL}}$={finite.sum()}",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=6.0,
                )
            axis.axhline(0, color="#CBD5E1", lw=0.5)
            axis.set_title(title, loc="left", fontsize=7.0, fontweight="bold")
            axis.set_xlabel("ln(molar mass / g mol$^{-1}$)", fontsize=6.5)
            axis.tick_params(labelsize=6.0, length=2)
            axis.grid(color="#EEF2F7", lw=0.4)
            axis.spines[["top", "right"]].set_visible(False)
            axis.text(
                -0.14,
                1.08,
                chr(ord("a") + index),
                transform=axis.transAxes,
                fontsize=8.5,
                fontweight="bold",
            )
            if index == 0:
                axis.set_ylabel(
                    "Condition-adjusted response residual",
                    fontsize=6.5,
                )
            source_parts.append(
                identity_data[
                    ["il_identity_key", "molar_mass_g_mol", column]
                ].rename(columns={column: "condition_adjusted_response"}).assign(
                    analysis_panel=title
                )
            )
        figure.tight_layout(w_pad=1.2)
        return self._save(
            figure,
            stem,
            pd.concat(source_parts, ignore_index=True),
        )

    def composite_results_figure(
        self,
        rules: pd.DataFrame,
        associations: pd.DataFrame,
        nonlinear: pd.DataFrame,
        matched_pairs: pd.DataFrame,
        contrasts: pd.DataFrame,
        stem: Path,
    ) -> list[Path]:
        """Build the dense manuscript-facing a–d evidence figure."""

        required_frames = {
            "rules": rules,
            "associations": associations,
            "nonlinear": nonlinear,
            "matched_pairs": matched_pairs,
            "contrasts": contrasts,
        }
        empty = [name for name, frame in required_frames.items() if frame.empty]
        if empty:
            raise ValueError(
                "Composite figure inputs are empty: " + ", ".join(empty)
            )

        short_properties = {
            "Density": "Density",
            "Viscosity": "Viscosity",
            "ElectricalConductivity": "Conductivity",
            "HeatCapacity": "Heat capacity",
            "SurfaceTension": "Surface tension",
            "ThermalConductivity": "Thermal cond.",
        }
        compact_factors = {
            "anion_fluorine_fraction": "anion F fraction",
            "cation_hetero_aromatic_atom": "cation heteroaromatic atom",
            "pair_alkyl_chain_length_sum": "alkyl-chain-length sum",
            "pair_total_molecular_weight_scaled": "total molecular weight",
            "cation_aromatic_atom_fraction_functional_group": (
                "cation aromatic fraction (FG)"
            ),
            "anion_molecular_weight_scaled": "anion molecular weight",
        }
        interaction_labels = {
            "H-bond-compatible pairs": "H-bond compatible",
            "alkyl-carbon--anion-polar-site pairs": "Alkyl C–anion polar",
            "aromatic-contact pairs": "Aromatic contact",
            "carboxylate-site-containing pairs": "Carboxylate site",
            "cation-aromatic-core--anion-polar-site pairs": (
                "Cation aromatic–anion polar"
            ),
            "charge-complementary pairs": "Charge complement",
            "fluorine-containing pairs": "F-containing",
            "high-charge-magnitude pairs": "High |charge|",
            "non-fluorine-halogen-containing pairs": "Non-F halogen",
            "other atom pairs": "Other atom pairs",
            "sulfonyl-site-containing pairs": "Sulfonyl site",
        }

        def feature_label(value: str) -> str:
            if value in compact_factors:
                return compact_factors[value]
            is_group = value.endswith("_functional_group")
            text = value.replace("_functional_group", "").replace("_", " ")
            return f"{text} (FG)" if is_group else text

        def panel_label(axis: plt.Axes, label: str, x: float = -0.15) -> None:
            axis.text(
                x,
                1.09,
                label,
                transform=axis.transAxes,
                fontsize=10.0,
                fontweight="bold",
            )

        figure = plt.figure(figsize=(7.2, 6.8), facecolor="white")
        outer = figure.add_gridspec(
            2,
            2,
            height_ratios=[1.10, 1.65],
            width_ratios=[0.45, 0.55],
            hspace=0.20,
            wspace=0.25,
        )
        lower_left = outer[1, 0].subgridspec(2, 1, hspace=0.76)
        lower_right = outer[1, 1].subgridspec(
            3,
            2,
            hspace=0.78,
            wspace=0.30,
        )
        panel_sources: dict[str, pd.DataFrame] = {}

        # Prepare the attention half of the integrated hero panel.
        source_attention_a = contrasts.copy()
        source_attention_a["abs_contrast"] = pd.to_numeric(
            source_attention_a["high_minus_low"],
            errors="coerce",
        ).abs()
        source_attention_a["property_order"] = source_attention_a[
            "property"
        ].map(
            {name: index for index, name in enumerate(PROPERTY_ORDER)}
        )
        source_attention_a = (
            source_attention_a.sort_values(
                ["property_order", "abs_contrast", "interaction_category"],
                ascending=[True, False, True],
            )
            .groupby("property", sort=False)
            .head(3)
            .sort_values("property_order")
        )
        source_attention_a["within_property_rank"] = (
            source_attention_a.groupby("property", sort=False).cumcount() + 1
        )
        maximum_attention_a = max(
            float(source_attention_a["abs_contrast"].max()),
            1e-12,
        )
        source_attention_a["line_width_pt"] = (
            0.35
            + 2.9
            * source_attention_a["abs_contrast"]
            / maximum_attention_a
        )
        source_attention_a["selection_rule"] = (
            "three largest absolute high-minus-low attention contrasts per "
            "property"
        )

        # a | Integrated association and model-attention hero network.
        axis_a = figure.add_subplot(outer[0, :])
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
        eligible_a = source_a.loc[
            source_a["confidence_level"].isin(["Level A", "Level B"])
            & source_a["partial_r"].notna()
        ].copy()
        if eligible_a.empty:
            eligible_a = source_a.loc[source_a["partial_r"].notna()].copy()
        eligible_a["abs_partial_r"] = eligible_a["partial_r"].abs()
        source_a = (
            eligible_a.sort_values(
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
            .head(3)
            .sort_values("property_order")
        )
        source_a["property_rank"] = (
            source_a.groupby("property", sort=False).cumcount() + 1
        )
        source_a["line_width_pt"] = (
            0.35 + 2.4 * source_a["abs_partial_r"].clip(0.0, 1.0)
        )
        source_a["selection_rule"] = (
            "top three Level A/B links per property; deterministic evidence, "
            "|partial r|, family-consistency and name ordering"
        )
        properties_a = [
            name for name in PROPERTY_ORDER if name in set(source_a["property"])
        ]
        property_y_a = dict(
            zip(properties_a, np.linspace(0.87, 0.17, len(properties_a)))
        )
        factor_order_a = (
            source_a.groupby("structural_factor", as_index=False)
            .agg(
                property_position=(
                    "property",
                    lambda values: np.mean(
                        [properties_a.index(value) for value in values]
                    ),
                ),
                strength=("abs_partial_r", "max"),
            )
            .sort_values(
                ["property_position", "strength", "structural_factor"],
                ascending=[True, False, True],
            )["structural_factor"]
            .tolist()
        )
        factor_y_a = dict(
            zip(factor_order_a, np.linspace(0.90, 0.14, len(factor_order_a)))
        )
        category_records_a = source_attention_a.assign(
            property_position=source_attention_a["property"].map(
                {name: index for index, name in enumerate(properties_a)}
            ),
            contrast_weight=source_attention_a["abs_contrast"].clip(
                lower=1e-12
            ),
        )
        category_records_a["weighted_position"] = (
            category_records_a["property_position"]
            * category_records_a["contrast_weight"]
        )
        category_order_a = (
            category_records_a.groupby(
                "interaction_category",
                as_index=False,
            )
            .agg(
                weighted_position=("weighted_position", "sum"),
                contrast_weight=("contrast_weight", "sum"),
            )
        )
        category_order_a["weighted_property_position"] = (
            category_order_a["weighted_position"]
            / category_order_a["contrast_weight"]
        )
        categories_a = category_order_a.sort_values(
            ["weighted_property_position", "interaction_category"]
        )["interaction_category"].tolist()
        category_y_a = dict(
            zip(categories_a, np.linspace(0.90, 0.14, len(categories_a)))
        )
        for row in source_a.sort_values("abs_partial_r").itertuples(
            index=False
        ):
            axis_a.plot(
                [0.24, 0.46],
                [
                    factor_y_a[row.structural_factor],
                    property_y_a[row.property],
                ],
                color=RED if row.partial_r > 0 else BLUE,
                linewidth=row.line_width_pt,
                alpha=0.72,
                solid_capstyle="round",
                transform=axis_a.transAxes,
                zorder=1,
            )
        for row in source_attention_a.sort_values(
            "abs_contrast"
        ).itertuples(index=False):
            axis_a.plot(
                [0.54, 0.76],
                [
                    property_y_a[row.property],
                    category_y_a[row.interaction_category],
                ],
                color=ORANGE if row.high_minus_low > 0 else PURPLE,
                linewidth=row.line_width_pt,
                alpha=0.68,
                solid_capstyle="round",
                transform=axis_a.transAxes,
                zorder=1,
            )
        for factor, y_value in factor_y_a.items():
            axis_a.text(
                0.22,
                y_value,
                feature_label(factor),
                ha="right",
                va="center",
                fontsize=5.8,
                transform=axis_a.transAxes,
            )
        for property_name, y_value in property_y_a.items():
            axis_a.text(
                0.50,
                y_value,
                short_properties[property_name],
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold",
                transform=axis_a.transAxes,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "#F1F5F9",
                    "edgecolor": "none",
                },
                zorder=3,
            )
        for category, y_value in category_y_a.items():
            axis_a.text(
                0.78,
                y_value,
                interaction_labels.get(
                    category,
                    category.replace("-", " "),
                ),
                ha="left",
                va="center",
                fontsize=5.8,
                transform=axis_a.transAxes,
            )
        axis_a.text(
            0.12,
            0.99,
            "Condition-controlled structural factors",
            ha="center",
            va="bottom",
            fontsize=6.6,
            fontweight="bold",
            transform=axis_a.transAxes,
        )
        axis_a.text(
            0.50,
            0.99,
            "Properties",
            ha="center",
            va="bottom",
            fontsize=6.6,
            fontweight="bold",
            transform=axis_a.transAxes,
        )
        axis_a.text(
            0.88,
            0.99,
            "Cross-ion attention motifs",
            ha="center",
            va="bottom",
            fontsize=6.6,
            fontweight="bold",
            transform=axis_a.transAxes,
        )
        axis_a.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=RED,
                    lw=1.4,
                    label="association +",
                ),
                Line2D(
                    [0],
                    [0],
                    color=BLUE,
                    lw=1.4,
                    label="association −",
                ),
                Line2D(
                    [0],
                    [0],
                    color=ORANGE,
                    lw=1.4,
                    label="attention high > low",
                ),
                Line2D(
                    [0],
                    [0],
                    color=PURPLE,
                    lw=1.4,
                    label="attention high < low",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.50, 0.00),
            ncol=4,
            frameon=False,
            fontsize=5.5,
            handlelength=1.55,
            handletextpad=0.22,
            columnspacing=0.75,
        )
        axis_a.set_xlim(0, 1)
        axis_a.set_ylim(0, 1)
        axis_a.axis("off")
        axis_a.set_title(
            "Integrated microstructure–property evidence map",
            loc="left",
            fontsize=9.0,
            fontweight="bold",
            pad=12,
        )
        axis_a.text(
            0,
            1.035,
            "Left: partial correlations; right: residual-quartile attention contrasts; "
            "widths scale within evidence family",
            transform=axis_a.transAxes,
            fontsize=5.8,
            color=GREY,
            va="bottom",
        )
        panel_label(axis_a, "a", x=-0.08)
        association_source_a = source_a.drop(
            columns=["level_order"],
            errors="ignore",
        ).assign(
            evidence_family="condition_controlled_association",
            node_label=source_a["structural_factor"],
            signed_effect=source_a["partial_r"],
            absolute_effect=source_a["abs_partial_r"],
        )
        attention_source_a = source_attention_a.assign(
            evidence_family="cross_ion_attention_contrast",
            node_label=source_attention_a["interaction_category"],
            signed_effect=source_attention_a["high_minus_low"],
            absolute_effect=source_attention_a["abs_contrast"],
        )
        panel_sources["a"] = pd.concat(
            [association_source_a, attention_source_a],
            ignore_index=True,
            sort=False,
        )

        # b | Ion-level dominance of the strongest association per scope.
        axis_b = figure.add_subplot(lower_left[0, 0])
        source_b = associations.loc[
            associations["data_type"] == "experimental"
        ].copy()
        source_b["partial_correlation"] = pd.to_numeric(
            source_b["partial_correlation"],
            errors="coerce",
        )
        source_b["fdr_q"] = pd.to_numeric(
            source_b["fdr_q"] if "fdr_q" in source_b else np.nan,
            errors="coerce",
        )
        source_b["abs_effect"] = source_b["partial_correlation"].abs()
        source_b["ion_scope"] = np.select(
            [
                source_b["feature"].str.startswith("cation_"),
                source_b["feature"].str.startswith("anion_"),
                source_b["feature"].str.startswith("pair_"),
            ],
            ["Cation", "Anion", "Ion pair"],
            default="Other",
        )
        source_b = source_b.loc[
            source_b["ion_scope"].isin(["Cation", "Anion", "Ion pair"])
            & source_b["partial_correlation"].notna()
        ].copy()
        source_b = (
            source_b.sort_values(
                ["property", "ion_scope", "abs_effect", "feature"],
                ascending=[True, True, False, True],
            )
            .groupby(["property", "ion_scope"], sort=False)
            .head(1)
            .copy()
        )
        source_b["fdr_significant"] = source_b["fdr_q"] <= 0.05
        source_b["within_property_strength_rank"] = source_b.groupby(
            "property"
        )["abs_effect"].rank(
            method="first",
            ascending=False,
        )
        source_b["selection_rule"] = (
            "single descriptor with maximum absolute partial correlation "
            "within each property and cation/anion/ion-pair scope"
        )
        properties_b = [
            name for name in PROPERTY_ORDER if name in set(source_b["property"])
        ]
        property_y_b = {
            name: len(properties_b) - 1 - index
            for index, name in enumerate(properties_b)
        }
        scope_style_b = {
            "Cation": ("#4C78A8", "o", 0.20),
            "Anion": ("#D97762", "s", 0.00),
            "Ion pair": ("#8276B5", "D", -0.20),
        }
        for property_index, property_name in enumerate(properties_b):
            base_y_b = property_y_b[property_name]
            if property_index % 2 == 0:
                axis_b.axhspan(
                    base_y_b - 0.43,
                    base_y_b + 0.43,
                    color="#F8FAFC",
                    zorder=0,
                )
        for scope, (scope_colour, scope_marker, offset_b) in (
            scope_style_b.items()
        ):
            subset_b = source_b.loc[source_b["ion_scope"] == scope].copy()
            y_b = subset_b["property"].map(property_y_b) + offset_b
            for x_value, y_value in zip(
                subset_b["partial_correlation"],
                y_b,
            ):
                axis_b.plot(
                    [0.0, x_value],
                    [y_value, y_value],
                    color=scope_colour,
                    linewidth=1.05,
                    alpha=0.70,
                    zorder=2,
                )
            axis_b.scatter(
                subset_b["partial_correlation"],
                y_b,
                s=29 + 35 * subset_b["abs_effect"].clip(0.0, 1.0),
                marker=scope_marker,
                color=scope_colour,
                edgecolor=np.where(
                    subset_b["fdr_significant"],
                    "#263238",
                    "#CBD5E1",
                ),
                linewidth=0.45,
                zorder=3,
            )
            for row, y_value in zip(
                subset_b.itertuples(index=False),
                y_b,
            ):
                alignment_b = "left" if row.partial_correlation >= 0 else "right"
                x_shift_b = 0.035 if row.partial_correlation >= 0 else -0.035
                axis_b.text(
                    row.partial_correlation + x_shift_b,
                    y_value,
                    f"{row.partial_correlation:+.2f}",
                    ha=alignment_b,
                    va="center",
                    fontsize=5.3,
                    color=scope_colour,
                )
        axis_b.axvline(
            0.0,
            color="#64748B",
            linewidth=0.65,
            zorder=1,
        )
        axis_b.set_yticks(
            [property_y_b[name] for name in properties_b],
            [short_properties[name] for name in properties_b],
        )
        axis_b.set_xlim(-1.02, 1.02)
        axis_b.set_ylim(-0.48, len(properties_b) - 0.52)
        axis_b.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        axis_b.set_xlabel(
            "Strongest signed partial $r$ within each structural scope",
            fontsize=6.0,
        )
        axis_b.tick_params(axis="both", labelsize=5.8, length=2)
        axis_b.grid(axis="x", color="#E5E7EB", linewidth=0.5, zorder=0)
        axis_b.spines[["top", "right", "left"]].set_visible(False)
        axis_b.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=colour,
                    marker=marker,
                    markersize=4.0,
                    linewidth=1.0,
                    label=scope,
                )
                for scope, (colour, marker, _) in scope_style_b.items()
            ],
            loc="lower center",
            bbox_to_anchor=(0.50, -0.43),
            ncol=3,
            frameon=False,
            fontsize=5.6,
            handletextpad=0.28,
            columnspacing=0.75,
        )
        axis_b.set_title(
            "Ion-level dominance",
            loc="left",
            fontsize=8.5,
            fontweight="bold",
            pad=8,
        )
        axis_b.text(
            0,
            1.015,
            "Dark rim: BH–FDR q≤0.05",
            transform=axis_b.transAxes,
            fontsize=5.6,
            color=GREY,
            va="bottom",
        )
        panel_label(axis_b, "b", x=-0.24)
        panel_sources["b"] = source_b

        # Three curve-supported rules per property preserve deterministic linkage.
        source_f = self._rule_evidence_source(
            rules,
            nonlinear,
            top_k=3,
        )

        # c | Three ranked structural response curves for each property.
        combinations_c = source_f[
            [
                "property",
                "structural_factor",
                "partial_r",
                "binned_response_spearman",
                "confidence_level",
                "selection_rank",
                "selection_rule",
            ]
        ].rename(columns={"structural_factor": "feature"})
        combinations_c["property_order"] = combinations_c["property"].map(
            {name: index for index, name in enumerate(PROPERTY_ORDER)}
        )
        combinations_c = combinations_c.sort_values(
            ["property_order", "selection_rank", "feature"]
        ).drop(columns=["property_order"])
        source_c = nonlinear.merge(
            combinations_c,
            on=["property", "feature"],
            how="inner",
        )
        source_c["curve_bin"] = source_c.groupby(
            ["property", "feature"],
            sort=False,
        ).cumcount() + 1
        source_c["response_log_centered"] = source_c[
            "response_log_mean"
        ] - source_c.groupby(["property", "feature"])[
            "response_log_mean"
        ].transform(
            "mean"
        )
        source_c["normalized_quantile_position"] = source_c.groupby(
            ["property", "feature"],
            sort=False,
        )["curve_bin"].transform(
            lambda values: np.linspace(0.0, 1.0, len(values))
        )
        source_c["response_log_span"] = source_c.groupby(
            ["property", "feature"]
        )["response_log_mean"].transform(
            lambda values: float(values.max() - values.min())
        )
        source_c["curve_sample_total"] = source_c.groupby(
            ["property", "feature"]
        )["sample_count"].transform("sum")
        scope_style_c = {
            "Cation": "#4C78A8",
            "Anion": "#D97762",
            "Ion pair": "#8276B5",
        }
        rank_style_c = {
            1: ("-", "o", 1.20, 0.96),
            2: ((0, (4.0, 1.8)), "s", 1.00, 0.84),
            3: ((0, (1.2, 1.4)), "^", 0.95, 0.76),
        }
        compact_curve_labels_c = {
            "anion_fluorine_fraction": "anion F fraction",
            "pair_alkyl_chain_length_sum": "alkyl-chain sum",
            "pair_total_fluorinated_carbon": "fluorinated-C total",
            "cation_longest_aliphatic_carbon_chain": "cation longest chain",
            "cation_radius_of_gyration_scaled": "cation $R_g$",
            "pair_total_molecular_weight_scaled": "total mol. weight",
            "pair_total_heavy_atom_count_scaled": "total heavy atoms",
            "cation_heavy_atom_count_scaled": "cation heavy atoms",
            "cation_aromatic_atom_fraction_functional_group": (
                "cation aromatic fraction"
            ),
            "cation_hetero_aromatic_atom": "cation heteroaromatic",
            "cation_charged_atom_fraction_functional_group": (
                "cation charged fraction"
            ),
            "cation_charged_atom_fraction": "cation charged atoms",
            "anion_molecular_weight_scaled": "anion mol. weight",
            "anion_exact_molecular_weight_scaled": "anion exact MW",
        }
        source_c["structural_scope"] = np.select(
            [
                source_c["feature"].str.startswith("cation_"),
                source_c["feature"].str.startswith("anion_"),
                source_c["feature"].str.startswith("pair_"),
            ],
            ["Cation", "Anion", "Ion pair"],
            default="Ion pair",
        )
        source_c["plot_color_hex"] = source_c["structural_scope"].map(
            scope_style_c
        )
        source_c["plot_line_style"] = source_c["selection_rank"].map(
            {
                1: "solid",
                2: "dashed",
                3: "dotted",
            }
        )
        source_c["plot_marker"] = source_c["selection_rank"].map(
            {
                1: "circle",
                2: "square",
                3: "triangle",
            }
        )
        grid_c = lower_right
        axes_c: list[plt.Axes] = []
        properties_c = [
            name
            for name in PROPERTY_ORDER
            if name in set(combinations_c["property"])
        ]
        for panel_index, property_name in enumerate(properties_c):
            axis_c = figure.add_subplot(
                grid_c[panel_index // 2, panel_index % 2]
            )
            axes_c.append(axis_c)
            property_rules_c = combinations_c.loc[
                combinations_c["property"] == property_name
            ].sort_values("selection_rank")
            for rule_c in property_rules_c.itertuples(index=False):
                subset_c = source_c.loc[
                    (source_c["property"] == property_name)
                    & (source_c["feature"] == rule_c.feature)
                ].sort_values("curve_bin")
                if subset_c.empty:
                    continue
                x_c = subset_c[
                    "normalized_quantile_position"
                ].to_numpy(dtype=float)
                y_c = subset_c["response_log_centered"].to_numpy(dtype=float)
                sem_c = subset_c["response_log_sem"].to_numpy(dtype=float)
                scope_c = str(subset_c["structural_scope"].iloc[0])
                colour_c = scope_style_c[scope_c]
                line_style_c, marker_c, width_c, alpha_c = rank_style_c[
                    int(rule_c.selection_rank)
                ]
                axis_c.fill_between(
                    x_c,
                    y_c - sem_c,
                    y_c + sem_c,
                    color=colour_c,
                    alpha=0.08 + 0.03 * (4 - int(rule_c.selection_rank)),
                    linewidth=0,
                )
                axis_c.plot(
                    x_c,
                    y_c,
                    marker=marker_c,
                    markersize=2.0,
                    color=colour_c,
                    linewidth=width_c,
                    linestyle=line_style_c,
                    alpha=alpha_c,
                )
                factor_text_c = compact_curve_labels_c.get(
                    rule_c.feature,
                    feature_label(rule_c.feature),
                )
                axis_c.text(
                    0.03,
                    0.96 - 0.105 * (int(rule_c.selection_rank) - 1),
                    (
                        f"{int(rule_c.selection_rank)}  {factor_text_c}  "
                        f"$r_p$={rule_c.partial_r:+.2f}"
                    ),
                    transform=axis_c.transAxes,
                    ha="left",
                    va="top",
                    fontsize=5.3,
                    color=colour_c,
                    fontweight=(
                        "bold" if int(rule_c.selection_rank) == 1 else "normal"
                    ),
                    bbox={
                        "boxstyle": "round,pad=0.08",
                        "facecolor": (1, 1, 1, 0.78),
                        "edgecolor": "none",
                    },
                    zorder=5,
                )
            axis_c.axhline(
                0.0,
                color="#CBD5E1",
                linewidth=0.45,
                zorder=0,
            )
            axis_c.set_xlim(-0.03, 1.03)
            axis_c.set_xticks([0.0, 0.5, 1.0], ["Q1", "mid", "Qmax"])
            axis_c.tick_params(axis="both", labelsize=5.5, length=1.8)
            axis_c.grid(axis="y", color="#EEF2F7", linewidth=0.42)
            axis_c.spines[["top", "right"]].set_visible(False)
            axis_c.spines[["bottom", "left"]].set_linewidth(0.6)
            axis_c.set_title(
                short_properties[property_name],
                loc="left",
                fontsize=6.8,
                fontweight="bold",
                pad=2.5,
            )
            axis_c.set_xlabel(
                "Structural-factor quantile",
                fontsize=5.5,
                labelpad=1.4,
            )
            if panel_index % 2 == 0:
                axis_c.set_ylabel(
                    "Centered ln(property)",
                    fontsize=5.5,
                    labelpad=2.0,
                )
            else:
                axis_c.set_ylabel("")
        if axes_c:
            scope_handles_c = [
                Line2D(
                    [0],
                    [0],
                    color=scope_style_c[scope],
                    marker="o",
                    markersize=3.2,
                    linewidth=1.5,
                    label=scope,
                )
                for scope in ["Cation", "Anion", "Ion pair"]
            ]
            rank_handles_c = [
                Line2D(
                    [0],
                    [0],
                    color="#64748B",
                    marker=rank_style_c[rank][1],
                    markersize=3.2,
                    linewidth=1.25,
                    linestyle=rank_style_c[rank][0],
                    label=f"Rank {rank}",
                )
                for rank in [1, 2, 3]
            ]
            axes_c[0].text(
                -0.18,
                1.52,
                "c",
                transform=axes_c[0].transAxes,
                fontsize=10.0,
                fontweight="bold",
            )
            axes_c[0].text(
                0.0,
                1.52,
                "Top-three structural-prior response shapes",
                transform=axes_c[0].transAxes,
                fontsize=8.8,
                fontweight="bold",
            )
            axes_c[0].legend(
                handles=scope_handles_c + rank_handles_c,
                loc="lower left",
                bbox_to_anchor=(0.0, 1.12, 2.23, 0.18),
                mode="expand",
                ncol=6,
                frameon=False,
                borderaxespad=0.0,
                fontsize=5.3,
                handlelength=1.45,
                handletextpad=0.30,
                columnspacing=0.70,
            )
        panel_sources["c"] = source_c

        # d | Effect-distribution forest for condition-matched substitutions.
        axis_d = figure.add_subplot(lower_left[1, 0])
        properties_d = [
            name
            for name in PROPERTY_ORDER
            if f"observed_abs_log_difference_{name}" in matched_pairs
        ]
        summary_records_d: list[dict[str, Any]] = []
        role_metadata_d = {
            "anion_fixed": ("Cation change", "#4C78A8", 0.13),
            "cation_fixed": ("Anion change", "#D97762", -0.13),
        }
        for property_index, property_name in enumerate(properties_d):
            base_y_d = len(properties_d) - 1 - property_index
            if property_index % 2 == 0:
                axis_d.axhspan(
                    base_y_d - 0.43,
                    base_y_d + 0.43,
                    color="#F8FAFC",
                    zorder=0,
                )
            value_column_d = (
                f"observed_abs_log_difference_{property_name}"
            )
            medians_d: list[float] = []
            for fixed_role, (
                role_label,
                role_colour,
                offset_d,
            ) in role_metadata_d.items():
                values_d = pd.to_numeric(
                    matched_pairs.loc[
                        matched_pairs["fixed_role"] == fixed_role,
                        value_column_d,
                    ],
                    errors="coerce",
                ).dropna()
                if values_d.empty:
                    continue
                quantiles_d = values_d.quantile(
                    [0.10, 0.25, 0.50, 0.75, 0.90]
                )
                record_d = {
                    "property": property_name,
                    "fixed_role": fixed_role,
                    "changed_ion": role_label,
                    "sample_count": int(len(values_d)),
                    "p10": float(quantiles_d.loc[0.10]),
                    "q25": float(quantiles_d.loc[0.25]),
                    "median": float(quantiles_d.loc[0.50]),
                    "q75": float(quantiles_d.loc[0.75]),
                    "p90": float(quantiles_d.loc[0.90]),
                    "source_kind": "observed_matched_pair_summary",
                }
                summary_records_d.append(record_d)
                y_d = base_y_d + offset_d
                axis_d.plot(
                    [record_d["p10"], record_d["p90"]],
                    [y_d, y_d],
                    color=role_colour,
                    linewidth=0.75,
                    alpha=0.65,
                    zorder=2,
                )
                axis_d.plot(
                    [record_d["q25"], record_d["q75"]],
                    [y_d, y_d],
                    color=role_colour,
                    linewidth=3.0,
                    solid_capstyle="round",
                    zorder=3,
                )
                axis_d.scatter(
                    [record_d["median"]],
                    [y_d],
                    s=23,
                    color=role_colour,
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=4,
                )
                axis_d.text(
                    0.995,
                    y_d,
                    (
                        "C  "
                        if role_label == "Cation change"
                        else "A  "
                    )
                    + f"n={record_d['sample_count']:,}",
                    transform=axis_d.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=5.4,
                    color=role_colour,
                )
                medians_d.append(record_d["median"])
            if len(medians_d) == 2:
                axis_d.plot(
                    medians_d,
                    [base_y_d + 0.13, base_y_d - 0.13],
                    color="#94A3B8",
                    linewidth=0.45,
                    linestyle=":",
                    zorder=1,
                )
        summary_d = pd.DataFrame.from_records(summary_records_d)
        positive_d = summary_d[
            ["p10", "q25", "median", "q75", "p90"]
        ].to_numpy(dtype=float)
        positive_d = positive_d[np.isfinite(positive_d) & (positive_d > 0)]
        if len(positive_d):
            axis_d.set_xscale("log")
            axis_d.set_xlim(
                max(float(np.min(positive_d)) * 0.75, 1e-5),
                float(np.max(positive_d)) * 1.50,
            )
        axis_d.set_yticks(
            np.arange(len(properties_d))[::-1],
            [short_properties[name] for name in properties_d],
        )
        axis_d.set_xlabel(
            "|Δ ln(observed property)|  (log scale)",
            fontsize=6.0,
        )
        axis_d.tick_params(axis="both", labelsize=5.8, length=2)
        axis_d.grid(axis="x", color="#E5E7EB", linewidth=0.5)
        axis_d.spines[["top", "right", "left"]].set_visible(False)
        axis_d.set_title(
            "Matched substitution effects",
            loc="left",
            fontsize=8.5,
            fontweight="bold",
        )
        axis_d.text(
            0,
            1.015,
            "Thin: 10–90%; thick: IQR; point: median",
            transform=axis_d.transAxes,
            fontsize=5.5,
            color=GREY,
        )
        panel_label(axis_d, "d", x=-0.24)
        panel_sources["d"] = summary_d

        figure.subplots_adjust(
            left=0.105,
            right=0.975,
            bottom=0.055,
            top=0.975,
        )
        return self._save_composite(figure, stem, panel_sources)

        # Legacy standalone attention layout retained below for source history;
        # the manuscript figure returns above after folding this evidence into a.
        axis_e = figure.add_subplot(outer[2, :])
        source_e = contrasts.copy()
        source_e["abs_contrast"] = pd.to_numeric(
            source_e["high_minus_low"],
            errors="coerce",
        ).abs()
        source_e["property_order"] = source_e["property"].map(
            {name: index for index, name in enumerate(PROPERTY_ORDER)}
        )
        source_e = (
            source_e.sort_values(
                ["property_order", "abs_contrast", "interaction_category"],
                ascending=[True, False, True],
            )
            .groupby("property", sort=False)
            .head(3)
            .sort_values("property_order")
        )
        source_e["within_property_rank"] = (
            source_e.groupby("property", sort=False).cumcount() + 1
        )
        maximum_e = max(float(source_e["abs_contrast"].max()), 1e-12)
        source_e["line_width_pt"] = (
            0.35 + 2.9 * source_e["abs_contrast"] / maximum_e
        )
        source_e["selection_rule"] = (
            "three largest absolute high-minus-low attention contrasts per "
            "property"
        )
        properties_e = [
            name for name in PROPERTY_ORDER if name in set(source_e["property"])
        ]
        property_y_e = dict(
            zip(properties_e, np.linspace(0.86, 0.18, len(properties_e)))
        )
        category_records_e = source_e.assign(
            property_position=source_e["property"].map(
                {name: index for index, name in enumerate(properties_e)}
            ),
            contrast_weight=source_e["abs_contrast"].clip(lower=1e-12),
        )
        category_records_e["weighted_position"] = (
            category_records_e["property_position"]
            * category_records_e["contrast_weight"]
        )
        category_order_e = (
            category_records_e.groupby(
                "interaction_category",
                as_index=False,
            )
            .agg(
                weighted_position=("weighted_position", "sum"),
                contrast_weight=("contrast_weight", "sum"),
            )
        )
        category_order_e["weighted_property_position"] = (
            category_order_e["weighted_position"]
            / category_order_e["contrast_weight"]
        )
        categories_e = category_order_e.sort_values(
            ["weighted_property_position", "interaction_category"]
        )["interaction_category"].tolist()
        category_y_e = dict(
            zip(categories_e, np.linspace(0.89, 0.15, len(categories_e)))
        )
        for row in source_e.sort_values("abs_contrast").itertuples(
            index=False
        ):
            axis_e.plot(
                [0.39, 0.62],
                [
                    category_y_e[row.interaction_category],
                    property_y_e[row.property],
                ],
                color=ORANGE if row.high_minus_low > 0 else PURPLE,
                linewidth=row.line_width_pt,
                alpha=0.70,
                solid_capstyle="round",
                transform=axis_e.transAxes,
                zorder=1,
            )
        for category, y_value in category_y_e.items():
            axis_e.text(
                0.37,
                y_value,
                interaction_labels.get(category, category.replace("-", " ")),
                ha="right",
                va="center",
                fontsize=3.95,
                transform=axis_e.transAxes,
            )
        for property_name, y_value in property_y_e.items():
            axis_e.text(
                0.65,
                y_value,
                short_properties[property_name],
                ha="left",
                va="center",
                fontsize=4.45,
                fontweight="bold",
                transform=axis_e.transAxes,
            )
        axis_e.legend(
            handles=[
                Line2D([0], [0], color=ORANGE, lw=1.4, label="high > low"),
                Line2D([0], [0], color=PURPLE, lw=1.4, label="high < low"),
                *[
                    Line2D(
                        [0],
                        [0],
                        color=GREY,
                        lw=0.35 + 2.9 * fraction,
                        label=f"{fraction:.0%} max |ΔA|",
                    )
                    for fraction in (0.35, 0.70)
                ],
            ],
            loc="lower center",
            bbox_to_anchor=(0.51, -0.08),
            ncol=4,
            frameon=False,
            fontsize=3.55,
            handlelength=1.35,
            handletextpad=0.20,
            columnspacing=0.40,
        )
        axis_e.set_xlim(0, 1)
        axis_e.set_ylim(0, 1)
        axis_e.axis("off")
        axis_e.set_title(
            "Cross-ion attention contrast network",
            loc="left",
            fontsize=7,
            fontweight="bold",
            pad=10,
        )
        axis_e.text(
            0,
            1.025,
            "Three largest residual-quartile contrasts per property",
            transform=axis_e.transAxes,
            fontsize=4.15,
            color=GREY,
        )
        axis_e.text(
            0.0,
            -0.16,
            "Model-focus diagnostic; not interaction energy",
            transform=axis_e.transAxes,
            fontsize=4.35,
            color=GREY,
        )
        panel_label(axis_e, "e", x=-0.08)
        panel_sources["e"] = source_e.drop(
            columns=["property_order"],
            errors="ignore",
        )

        figure.subplots_adjust(
            left=0.105,
            right=0.975,
            bottom=0.055,
            top=0.975,
        )
        return self._save_composite(figure, stem, panel_sources)
