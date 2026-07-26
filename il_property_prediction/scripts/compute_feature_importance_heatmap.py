from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.chem.functional_groups import (  # noqa: E402
    ION_FUNCTIONAL_GROUP_NAMES,
    PAIR_FUNCTIONAL_GROUP_NAMES,
)
from src.data.dataset import ILPropertyDataset, PROPERTY_NAMES  # noqa: E402
from src.data.scaler import fit_scalers  # noqa: E402
from src.models.factory import build_model  # noqa: E402


PROPERTY_SHORT = {
    "Density": "Density",
    "ElectricalConductivity": "EC",
    "HeatCapacity": "HC",
    "SurfaceTension": "ST",
    "ThermalConductivity": "TC",
    "Viscosity": "Visc.",
}

NODE_CATEGORIES = [
    ("charged atoms", lambda chem: chem[..., 0].abs() > 0),
    ("H-bond atoms", lambda chem: (chem[..., 1] > 0) | (chem[..., 2] > 0)),
    ("aromatic atoms", lambda chem: chem[..., 3] > 0),
    ("hetero atoms", lambda chem: chem[..., 4] != 6),
    ("halogen/F atoms", lambda chem: (chem[..., 4] == 9) | (chem[..., 4] == 17) | (chem[..., 4] == 35) | (chem[..., 4] == 53)),
    ("carbon atoms", lambda chem: chem[..., 4] == 6),
]

EDGE_CATEGORIES = [
    "charge-complementary pairs",
    "H-bond compatible pairs",
    "aromatic-contact pairs",
    "charge-magnitude pairs",
    "other attended pairs",
]


def safe_torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_path(path: str | Path, base: Path = PROJECT_DIR) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (base / value).resolve()


def load_split(path: str | Path) -> dict[str, list[int]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {key: [int(item) for item in value] for key, value in payload.items()}


def make_functional_group_names() -> list[str]:
    return (
        [f"Cat: {name}" for name in ION_FUNCTIONAL_GROUP_NAMES]
        + [f"Ani: {name}" for name in ION_FUNCTIONAL_GROUP_NAMES]
        + [f"Pair: {name}" for name in PAIR_FUNCTIONAL_GROUP_NAMES]
    )


def pretty_feature_name(name: str) -> str:
    replacements = {
        "hbond": "H-bond",
        "NTf2": "NTf2",
        "fsi": "FSI",
        "anion": "anion",
        "cation": "cation",
    }
    out = name.replace("_", " ")
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = out.replace("Cat:", "Cat:").replace("Ani:", "Ani:").replace("Pair:", "Pair:")
    return out


def choose_test_indices(
    split_indices: list[int],
    eval_mask: np.ndarray,
    samples_per_property: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    split_array = np.asarray(split_indices, dtype=np.int64)
    for prop_idx in range(eval_mask.shape[1]):
        candidates = split_array[eval_mask[split_array, prop_idx] > 0]
        if candidates.size == 0:
            continue
        rng.shuffle(candidates)
        selected.extend(candidates[:samples_per_property].tolist())
    return list(dict.fromkeys(selected))


def normalize_rows(df: pd.DataFrame, value_col: str = "importance") -> pd.DataFrame:
    df = df.copy()
    totals = df.groupby("property")[value_col].transform(lambda x: float(np.nansum(x)))
    df["normalized_importance"] = np.where(totals > 0, df[value_col] / totals, 0.0)
    return df


def add_category_score(store: dict[tuple[str, str], float], prop: str, category: str, value: torch.Tensor) -> None:
    key = (prop, category)
    store[key] = store.get(key, 0.0) + float(value.detach().sum().cpu())


def accumulate_node_scores(
    store: dict[tuple[str, str], float],
    prop: str,
    c_score: torch.Tensor,
    a_score: torch.Tensor,
    c_chem: torch.Tensor,
    a_chem: torch.Tensor,
) -> None:
    for category, mask_fn in NODE_CATEGORIES:
        c_mask = mask_fn(c_chem).to(c_score.dtype)
        a_mask = mask_fn(a_chem).to(a_score.dtype)
        add_category_score(store, prop, category, (c_score * c_mask).sum() + (a_score * a_mask).sum())


def edge_masks(c_chem: torch.Tensor, a_chem: torch.Tensor, valid_pair_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    c_charge = c_chem[..., 0].unsqueeze(2)
    a_charge = a_chem[..., 0].unsqueeze(1)
    charge_product = c_charge * a_charge
    hbond = (
        c_chem[..., 1].unsqueeze(2) * a_chem[..., 2].unsqueeze(1)
        + c_chem[..., 2].unsqueeze(2) * a_chem[..., 1].unsqueeze(1)
    ) > 0
    aromatic = (c_chem[..., 3].unsqueeze(2) * a_chem[..., 3].unsqueeze(1)) > 0
    charge_complementary = charge_product < 0
    charge_magnitude = charge_product.abs() > 0
    explained = charge_complementary | hbond | aromatic | charge_magnitude
    return {
        "charge-complementary pairs": valid_pair_mask & charge_complementary,
        "H-bond compatible pairs": valid_pair_mask & hbond,
        "aromatic-contact pairs": valid_pair_mask & aromatic,
        "charge-magnitude pairs": valid_pair_mask & charge_magnitude,
        "other attended pairs": valid_pair_mask & (~explained),
    }


def accumulate_edge_scores(
    store: dict[tuple[str, str], float],
    prop: str,
    pair_score: torch.Tensor,
    c_chem: torch.Tensor,
    a_chem: torch.Tensor,
    valid_pair_mask: torch.Tensor,
) -> None:
    masks = edge_masks(c_chem, a_chem, valid_pair_mask)
    for category in EDGE_CATEGORIES:
        add_category_score(store, prop, category, pair_score * masks[category].to(pair_score.dtype))


def make_records(
    store: dict[tuple[str, str], float],
    feature_type: str,
    label_counts: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for prop in PROPERTY_NAMES:
        for (_, feature), value in sorted((key, val) for key, val in store.items() if key[0] == prop):
            rows.append(
                {
                    "feature_type": feature_type,
                    "property": prop,
                    "property_short": PROPERTY_SHORT[prop],
                    "feature": feature,
                    "importance": value / max(label_counts.get(prop, 0), 1),
                    "n_labels": label_counts.get(prop, 0),
                }
            )
    return normalize_rows(pd.DataFrame(rows))


def build_loader(config: dict, split: dict, indices: list[int], batch_size: int):
    base = Path(config.get("_base_dir", PROJECT_DIR))
    arrays_path = resolve_path(config["data"]["arrays_path"], base)
    clean_csv = resolve_path(config["data"]["clean_csv"], base)
    graph_cache = resolve_path(config["data"]["graph_cache_path"], base)
    with np.load(arrays_path, allow_pickle=True) as arrays:
        arrays_dict = {key: arrays[key] for key in arrays.files}
    loss_cfg = config.get("loss", {})
    _, _, y_scaled, condition, error_weights = fit_scalers(
        arrays_dict,
        split["train"],
        loss_cfg.get("error_weight_clip_min", 0.1),
        loss_cfg.get("error_weight_clip_max", 10.0),
        loss_cfg.get("target_scaler_mask", "mask"),
    )
    ds = ILPropertyDataset(
        clean_csv,
        arrays_path,
        graph_cache,
        indices,
        condition,
        y_scaled,
        error_weights,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), arrays_dict


def compute_importance(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    checkpoint = safe_torch_load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    config["_base_dir"] = str(PROJECT_DIR)
    config.setdefault("training", {})["device"] = args.device
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    split = load_split(args.split)

    loader_for_indices, arrays = build_loader(config, split, split["test"], args.batch_size)
    eval_mask = arrays.get("evaluation_mask", arrays["mask"])
    selected_indices = choose_test_indices(split["test"], eval_mask, args.samples_per_property, args.seed)
    loader, _ = build_loader(config, split, selected_indices, args.batch_size)

    model = build_model(config).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    activation_cache: dict[str, torch.Tensor] = {}
    atom_cache: dict[str, torch.Tensor] = {}

    def cation_hook(_module, _inputs, output):
        output.retain_grad()
        activation_cache["cation_base"] = output

    def anion_hook(_module, _inputs, output):
        output.retain_grad()
        activation_cache["anion_base"] = output

    def atom_interaction_hook(_module, inputs, _output):
        atom_cache["cation_mask"] = inputs[1].detach()
        atom_cache["cation_chemistry"] = inputs[2].detach()
        atom_cache["anion_mask"] = inputs[4].detach()
        atom_cache["anion_chemistry"] = inputs[5].detach()

    handles = [
        model.atom_interaction.cation_atom_projection.register_forward_hook(cation_hook),
        model.atom_interaction.anion_atom_projection.register_forward_hook(anion_hook),
        model.atom_interaction.register_forward_hook(atom_interaction_hook),
    ]

    node_store: dict[tuple[str, str], float] = {}
    edge_store: dict[tuple[str, str], float] = {}
    fg_store: dict[tuple[str, str], float] = {}
    label_counts = {prop: 0 for prop in PROPERTY_NAMES}
    fg_names = make_functional_group_names()

    try:
        for batch in loader:
            batch = batch.to(device)
            batch.functional_group_desc = batch.functional_group_desc.detach().clone().requires_grad_(True)
            prediction, aux = model(batch)
            eval_mask_batch = batch.eval_mask.view(-1, len(PROPERTY_NAMES))

            c_base = activation_cache["cation_base"]
            a_base = activation_cache["anion_base"]
            c_mask = atom_cache["cation_mask"]
            a_mask = atom_cache["anion_mask"]
            c_chem = atom_cache["cation_chemistry"]
            a_chem = atom_cache["anion_chemistry"]
            attention = 0.5 * (
                aux["cation_to_anion_attention"] + aux["anion_to_cation_attention"].transpose(1, 2)
            )
            valid_pair_mask = c_mask.unsqueeze(2) & a_mask.unsqueeze(1)

            active_props = [
                prop_idx
                for prop_idx, prop in enumerate(PROPERTY_NAMES)
                if bool((eval_mask_batch[:, prop_idx] > 0).any().detach().cpu())
            ]
            for order, prop_idx in enumerate(active_props):
                prop = PROPERTY_NAMES[prop_idx]
                active = eval_mask_batch[:, prop_idx] > 0
                label_counts[prop] += int(active.sum().detach().cpu())
                model.zero_grad(set_to_none=True)
                if batch.functional_group_desc.grad is not None:
                    batch.functional_group_desc.grad.zero_()
                if c_base.grad is not None:
                    c_base.grad.zero_()
                if a_base.grad is not None:
                    a_base.grad.zero_()
                target = prediction[active, prop_idx].sum()
                retain_graph = order < len(active_props) - 1
                target.backward(retain_graph=retain_graph)

                row_weight = active.to(c_base.dtype)
                c_node_score = (c_base.grad * c_base).abs().sum(dim=-1) * c_mask.to(c_base.dtype) * row_weight[:, None]
                a_node_score = (a_base.grad * a_base).abs().sum(dim=-1) * a_mask.to(a_base.dtype) * row_weight[:, None]
                total_node = c_node_score.sum(dim=1, keepdim=True) + a_node_score.sum(dim=1, keepdim=True).clamp_min(0)
                total_node = total_node.clamp_min(1e-12)
                c_node_score = c_node_score / total_node
                a_node_score = a_node_score / total_node
                accumulate_node_scores(node_store, prop, c_node_score, a_node_score, c_chem, a_chem)

                pair_score = attention * 0.5 * (
                    c_node_score.unsqueeze(2) + a_node_score.unsqueeze(1)
                ) * row_weight[:, None, None]
                pair_total = pair_score.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
                pair_score = pair_score / pair_total
                accumulate_edge_scores(edge_store, prop, pair_score, c_chem, a_chem, valid_pair_mask)

                fg_grad = batch.functional_group_desc.grad.view(prediction.size(0), -1)
                fg_value = batch.functional_group_desc.view(prediction.size(0), -1)
                fg_score = (fg_grad * fg_value).abs() * active.to(fg_grad.dtype)[:, None]
                for fg_idx, name in enumerate(fg_names):
                    fg_store[(prop, name)] = fg_store.get((prop, name), 0.0) + float(
                        fg_score[:, fg_idx].detach().sum().cpu()
                    )
    finally:
        for handle in handles:
            handle.remove()

    node_df = make_records(node_store, "atom-node category", label_counts)
    edge_df = make_records(edge_store, "cross-ion edge category", label_counts)
    fg_df = make_records(fg_store, "functional-group descriptor", label_counts)
    if not fg_df.empty:
        fg_df["feature_pretty"] = fg_df["feature"].map(pretty_feature_name)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": str(Path(args.split).resolve()),
        "selected_test_rows": len(selected_indices),
        "samples_per_property": args.samples_per_property,
        "batch_size": args.batch_size,
        "device": str(device),
        "label_counts": label_counts,
        "load_state_dict": {
            "missing_keys": list(missing_keys),
            "unexpected_keys": list(unexpected_keys),
        },
        "attribution": {
            "nodes": "absolute gradient-times-activation on projected atom representations, aggregated by atom category",
            "edges": "symmetric cross-ion attention weighted by property-specific node attribution, aggregated by pair category",
            "functional_groups": "absolute gradient-times-input on the 80-dimensional functional-group descriptor",
        },
    }
    return node_df, edge_df, fg_df, summary


def plot_heatmap(node_df: pd.DataFrame, edge_df: pd.DataFrame, fg_df: pd.DataFrame, out_prefix: Path, top_k: int) -> None:
    """Render attribution as ranked dot-bar summaries instead of dense heatmaps."""
    import matplotlib.pyplot as plt
    from textwrap import fill

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 11.6,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.1,
            "legend.fontsize": 9.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
        }
    )

    prop_order = list(PROPERTY_NAMES)
    prop_colors = {
        "Density": "#4C78A8",
        "ElectricalConductivity": "#59A14F",
        "HeatCapacity": "#E3AE3D",
        "SurfaceTension": "#A879B2",
        "ThermalConductivity": "#62A8A8",
        "Viscosity": "#D96B5F",
    }
    prop_markers = {
        "Density": "o",
        "ElectricalConductivity": "s",
        "HeatCapacity": "^",
        "SurfaceTension": "D",
        "ThermalConductivity": "P",
        "Viscosity": "X",
    }

    def ranked_frame(frame: pd.DataFrame, feature_order: list[str] | None = None, top_n: int | None = None) -> pd.DataFrame:
        data = frame.copy()
        if "feature_pretty" not in data.columns:
            data["feature_pretty"] = data["feature"]
        if feature_order is None:
            order = (
                data.groupby("feature_pretty")["normalized_importance"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
        else:
            pretty_lookup = data.drop_duplicates("feature").set_index("feature")["feature_pretty"].to_dict()
            order = [pretty_lookup.get(feature, feature) for feature in feature_order]
        if top_n is not None:
            order = order[:top_n]
        data = data[data["feature_pretty"].isin(order)].copy()
        data["feature_pretty"] = pd.Categorical(data["feature_pretty"], order[::-1], ordered=True)
        data["property"] = pd.Categorical(data["property"], prop_order, ordered=True)
        return data.sort_values(["feature_pretty", "property"])

    def draw_rank_panel(
        ax: plt.Axes,
        frame: pd.DataFrame,
        title: str,
        panel: str,
        wrap_width: int,
        xlabel: str = "normalized attribution",
    ) -> pd.DataFrame:
        data = frame.copy()
        order = data["feature_pretty"].cat.categories.tolist()
        y_lookup = {feature: idx for idx, feature in enumerate(order)}
        summary = (
            data.groupby("feature_pretty", observed=False)["normalized_importance"]
            .mean()
            .reindex(order)
            .reset_index(name="mean_normalized_importance")
        )
        y = np.arange(len(order))
        ax.barh(
            y,
            summary["mean_normalized_importance"].to_numpy(),
            height=0.74,
            color="#D8DEE6",
            edgecolor="white",
            linewidth=0.65,
            zorder=1,
        )
        offsets = np.linspace(-0.22, 0.22, len(prop_order))
        for prop_idx, prop in enumerate(prop_order):
            sub = data[data["property"] == prop]
            ys = [y_lookup[str(feature)] + offsets[prop_idx] for feature in sub["feature_pretty"]]
            ax.scatter(
                sub["normalized_importance"],
                ys,
                s=44,
                marker=prop_markers[prop],
                color=prop_colors[prop],
                edgecolor="white",
                linewidth=0.65,
                alpha=0.95,
                zorder=3,
                label=PROPERTY_SHORT[prop],
            )
        ax.set_yticks(y)
        ax.set_yticklabels([fill(str(feature), wrap_width) for feature in order])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", color="#E6E8EB", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#3F3F3F")
            ax.spines[side].set_linewidth(0.55)
        ax.tick_params(direction="out", pad=1.4)
        ax.text(
            -0.16,
            1.06,
            panel,
            transform=ax.transAxes,
            fontsize=12.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        return summary

    def panel_plot_data(frame: pd.DataFrame, panel: str, group: str) -> pd.DataFrame:
        data = frame.copy()
        data["feature_pretty"] = data["feature_pretty"].astype(str)
        data["property"] = data["property"].astype(str)
        order = frame["feature_pretty"].cat.categories.tolist()
        y_lookup = {str(feature): idx for idx, feature in enumerate(order)}
        offsets = {
            prop: offset
            for prop, offset in zip(prop_order, np.linspace(-0.22, 0.22, len(prop_order)))
        }
        summary = (
            data.groupby("feature_pretty", observed=False)["normalized_importance"]
            .mean()
            .reset_index(name="mean_normalized_importance")
        )
        out = data.merge(summary, on="feature_pretty", how="left")
        out["panel"] = panel
        out["feature_group"] = group
        out["property_short"] = out["property"].map(PROPERTY_SHORT)
        out["marker"] = out["property"].map(prop_markers)
        out["color"] = out["property"].map(prop_colors)
        out["mean_bar_color"] = "#D8DEE6"
        out["plot_y_base"] = out["feature_pretty"].map(y_lookup)
        out["plot_y_offset"] = out["property"].map(offsets)
        out["plot_y"] = out["plot_y_base"] + out["plot_y_offset"]
        out["plot_order_bottom_to_top"] = out["plot_y_base"] + 1
        out["plot_order_top_to_bottom"] = len(order) - out["plot_y_base"]
        preferred = [
            "panel",
            "feature_group",
            "feature_type",
            "feature",
            "feature_pretty",
            "plot_order_top_to_bottom",
            "plot_order_bottom_to_top",
            "plot_y_base",
            "plot_y_offset",
            "plot_y",
            "property",
            "property_short",
            "importance",
            "normalized_importance",
            "mean_normalized_importance",
            "n_labels",
            "marker",
            "color",
            "mean_bar_color",
        ]
        return out[[col for col in preferred if col in out.columns]]

    node_rank = ranked_frame(node_df)
    edge_rank = ranked_frame(edge_df)
    fg_source = fg_df.copy()
    if "feature_pretty" not in fg_source.columns:
        fg_source["feature_pretty"] = fg_source["feature"].map(pretty_feature_name)
    top_features = (
        fg_source.groupby("feature_pretty")["normalized_importance"]
        .mean()
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )
    fg_rank = ranked_frame(fg_source[fg_source["feature_pretty"].isin(top_features)], top_n=top_k)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.1, 5.15),
        gridspec_kw={"width_ratios": [1.04, 1.12, 1.84], "wspace": 0.50},
        constrained_layout=False,
    )
    node_summary = draw_rank_panel(axes[0], node_rank, "Atom-node attribution", "a", 16)
    edge_summary = draw_rank_panel(axes[1], edge_rank, "Cross-ion interaction attribution", "b", 19)
    fg_summary = draw_rank_panel(axes[2], fg_rank, "Top functional-group descriptors", "c", 25)
    axes[0].set_xlim(0, max(0.34, float(node_rank["normalized_importance"].max()) * 1.12))
    axes[1].set_xlim(0, 1.0)
    axes[2].set_xlim(0, max(0.22, float(fg_rank["normalized_importance"].max()) * 1.14))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.018),
        ncol=6,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.15,
        markerscale=1.0,
    )
    fig.subplots_adjust(left=0.073, right=0.992, top=0.920, bottom=0.205)

    fig.savefig(out_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    source = pd.concat(
        [
            node_summary.assign(panel="A", group="atom-node"),
            edge_summary.assign(panel="B", group="cross-ion"),
            fg_summary.assign(panel="C", group="functional-group"),
        ],
        ignore_index=True,
    )
    source.to_csv(out_prefix.with_name(out_prefix.name + "_ranked_summary.csv"), index=False)
    plot_data = pd.concat(
        [
            panel_plot_data(node_rank, "A", "atom-node"),
            panel_plot_data(edge_rank, "B", "cross-ion"),
            panel_plot_data(fg_rank, "C", "functional-group"),
        ],
        ignore_index=True,
    )
    plot_data.to_csv(out_prefix.with_name(out_prefix.name + "_plot_data.csv"), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MIPGraph node, edge, and functional-group attribution heatmaps.")
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/fg_transformer_random_point_seed42_noamp/checkpoints/"
            "unimol2_fg_transformer_random_point_seed42_noamp_resume56/"
            "best_model_pid73748_epoch088.pt"
        ),
        help="Checkpoint path, relative to il_property_prediction/ unless absolute.",
    )
    parser.add_argument(
        "--split",
        default="data/processed/splits/row_level_seed42.json",
        help="Split JSON path, relative to il_property_prediction/ unless absolute.",
    )
    parser.add_argument("--samples-per-property", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-functional-groups", type=int, default=12)
    parser.add_argument(
        "--out-prefix",
        default="../LaTex-MIPGraph/Fig/feature_importance_heatmap",
        help="Output figure prefix, relative to il_property_prediction/ unless absolute.",
    )
    parser.add_argument(
        "--source-data-dir",
        default=(
            "../experiments/manuscript_figure_source_data/"
            "interpretability_feature_importance_4x3"
        ),
        help=(
            "Authoritative CSV output directory, relative to "
            "il_property_prediction/ unless absolute."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = resolve_path(args.checkpoint, PROJECT_DIR)
    args.split = resolve_path(args.split, PROJECT_DIR)
    out_prefix = resolve_path(args.out_prefix, PROJECT_DIR)
    source_data_dir = resolve_path(args.source_data_dir, PROJECT_DIR)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    source_data_dir.mkdir(parents=True, exist_ok=True)

    node_df, edge_df, fg_df, summary = compute_importance(args)
    node_path = source_data_dir / f"{out_prefix.name}_source_data_nodes.csv"
    edge_path = source_data_dir / f"{out_prefix.name}_source_data_edges.csv"
    functional_group_path = (
        source_data_dir
        / f"{out_prefix.name}_source_data_functional_groups.csv"
    )
    node_df.to_csv(node_path, index=False)
    edge_df.to_csv(edge_path, index=False)
    fg_df.to_csv(functional_group_path, index=False)
    with out_prefix.with_name(out_prefix.name + "_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot_heatmap(node_df, edge_df, fg_df, out_prefix, args.top_functional_groups)
    print(
        {
            "figure": str(out_prefix.with_suffix(".png")),
            "nodes": str(node_path),
            "edges": str(edge_path),
            "functional_groups": str(functional_group_path),
            "summary": summary,
        }
    )


if __name__ == "__main__":
    main()
