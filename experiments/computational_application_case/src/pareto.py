"""Deterministic non-dominated sorting and rank-one utopia distances."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _objective_columns(objectives: Mapping[str, Sequence[str]]) -> tuple[list[str], list[float]]:
    maximize = list(objectives.get("maximize", []))
    minimize = list(objectives.get("minimize", []))
    columns = maximize + minimize
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Pareto objectives must be non-empty and unique")
    return columns, [1.0] * len(maximize) + [-1.0] * len(minimize)


def _dominates(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left >= right) and np.any(left > right))


def non_dominated_sort(
    candidates: pd.DataFrame,
    objectives: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Assign Pareto ranks plus initial domination diagnostics."""

    columns, direction = _objective_columns(objectives)
    missing = [column for column in columns if column not in candidates.columns]
    if missing:
        raise ValueError(f"Candidate table lacks Pareto objectives: {missing}")
    output = candidates.copy().reset_index(drop=True)
    if output.empty:
        output["Pareto_rank"] = pd.Series(dtype="int64")
        output["domination_count"] = pd.Series(dtype="int64")
        output["dominated_set_size"] = pd.Series(dtype="int64")
        return output
    raw = output[columns].to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Pareto objectives contain NaN or infinity")
    values = raw * np.asarray(direction, dtype=float)[None, :]
    count = len(output)
    dominated_sets: list[list[int]] = [[] for _ in range(count)]
    initial_domination = np.zeros(count, dtype=int)
    for left in range(count):
        for right in range(left + 1, count):
            if _dominates(values[left], values[right]):
                dominated_sets[left].append(right)
                initial_domination[right] += 1
            elif _dominates(values[right], values[left]):
                dominated_sets[right].append(left)
                initial_domination[left] += 1
    remaining_domination = initial_domination.copy()
    ranks = np.zeros(count, dtype=int)
    front = [index for index in range(count) if remaining_domination[index] == 0]
    rank = 1
    assigned = 0
    while front:
        next_front: list[int] = []
        for index in front:
            if ranks[index] != 0:
                continue
            ranks[index] = rank
            assigned += 1
            for dominated in dominated_sets[index]:
                remaining_domination[dominated] -= 1
                if remaining_domination[dominated] == 0:
                    next_front.append(dominated)
        front = sorted(set(next_front))
        rank += 1
    if assigned != count:
        raise RuntimeError("Non-dominated sorting did not assign every candidate")
    output["Pareto_rank"] = ranks
    output["domination_count"] = initial_domination
    output["dominated_set_size"] = [len(items) for items in dominated_sets]
    return output


def add_utopia_distance(
    ranked_candidates: pd.DataFrame,
    objectives: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Add min-max rank-one distance to the all-objective utopia point."""

    columns, direction = _objective_columns(objectives)
    output = ranked_candidates.copy().reset_index(drop=True)
    if "Pareto_rank" not in output:
        raise ValueError("Pareto_rank is required before utopia-distance calculation")
    output["utopia_distance"] = np.nan
    output["utopia_rank_one_order"] = pd.Series([pd.NA] * len(output), dtype="Int64")
    if output.empty:
        return output
    rank_one_indices = output.index[output["Pareto_rank"].eq(1)].to_numpy(dtype=int)
    if rank_one_indices.size == 0:
        return output
    raw = output.loc[rank_one_indices, columns].to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Rank-one objectives contain NaN or infinity")
    normalized = np.ones_like(raw, dtype=float)
    for column_index, sense in enumerate(direction):
        values = raw[:, column_index]
        low = float(np.min(values))
        high = float(np.max(values))
        if np.isclose(high, low, rtol=0.0, atol=1.0e-15):
            normalized[:, column_index] = 1.0
        elif sense > 0:
            normalized[:, column_index] = (values - low) / (high - low)
        else:
            normalized[:, column_index] = (high - values) / (high - low)
    distances = np.sqrt(np.mean((1.0 - normalized) ** 2, axis=1))
    output.loc[rank_one_indices, "utopia_distance"] = distances
    ordered = sorted(
        zip(rank_one_indices.tolist(), distances.tolist()),
        key=lambda item: (item[1], item[0]),
    )
    for order, (index, _) in enumerate(ordered, start=1):
        output.at[index, "utopia_rank_one_order"] = order
    return output

