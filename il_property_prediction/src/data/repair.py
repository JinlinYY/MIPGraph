from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]


def property_names_from_arrays(arrays: dict[str, Any]) -> list[str]:
    names = arrays.get("property_names")
    if names is None:
        return DEFAULT_PROPERTY_NAMES
    return [str(item) for item in list(names)]


def apply_max_train_value_repair(
    arrays: dict[str, Any],
    train_indices: list[int],
    property_name: str,
    max_value: float,
    action: str,
    downweight: float = 0.05,
) -> dict[str, Any]:
    """Drop or downweight high-value labels in the training split only."""
    action = action.lower().strip()
    if action not in {"none", "drop", "downweight"}:
        raise ValueError("--repair-viscosity-action must be one of: none, drop, downweight")
    if action == "none":
        return {"enabled": False, "action": action}
    if max_value <= 0:
        raise ValueError("--repair-viscosity-max-train must be positive")
    if action == "downweight" and not (0.0 <= downweight <= 1.0):
        raise ValueError("--repair-viscosity-downweight must be in [0, 1]")

    property_names = property_names_from_arrays(arrays)
    if property_name not in property_names:
        raise ValueError(f"Unknown repair property {property_name!r}. Valid names: {property_names}")
    prop_idx = property_names.index(property_name)
    train_idx = np.asarray(train_indices, dtype=np.int64)
    y = np.asarray(arrays["y"])
    mask = np.asarray(arrays["mask"])
    selected = train_idx[
        (mask[train_idx, prop_idx] > 0)
        & np.isfinite(y[train_idx, prop_idx])
        & (y[train_idx, prop_idx] > float(max_value))
    ]

    report = {
        "enabled": True,
        "property": property_name,
        "max_train_value": float(max_value),
        "action": action,
        "downweight": float(downweight) if action == "downweight" else None,
        "affected_labels": int(selected.size),
        "affected_sample_ids_preview": [int(item) for item in selected[:20]],
    }
    if selected.size == 0:
        return report

    if action == "drop":
        repaired_mask = np.asarray(arrays["mask"], dtype=np.float32).copy()
        repaired_mask[selected, prop_idx] = 0.0
        arrays["mask"] = repaired_mask
        if "error_mask" in arrays:
            repaired_error_mask = np.asarray(arrays["error_mask"], dtype=np.float32).copy()
            repaired_error_mask[selected, prop_idx] = 0.0
            arrays["error_mask"] = repaired_error_mask
    else:
        label_weight = np.asarray(
            arrays.get("label_weight", np.ones_like(arrays["mask"], dtype=np.float32)),
            dtype=np.float32,
        ).copy()
        label_weight[selected, prop_idx] *= float(downweight)
        arrays["label_weight"] = label_weight

    return report
