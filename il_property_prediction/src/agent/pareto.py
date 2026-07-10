from __future__ import annotations

import math

import pandas as pd


def _finite(value: object, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _dominates(a: pd.Series, b: pd.Series, objectives: dict[str, str]) -> bool:
    better_or_equal = True
    strictly_better = False
    for column, direction in objectives.items():
        if direction == "max":
            av = _finite(a.get(column), -float("inf"))
            bv = _finite(b.get(column), -float("inf"))
            better_or_equal = better_or_equal and av >= bv
            strictly_better = strictly_better or av > bv
        else:
            av = _finite(a.get(column), float("inf"))
            bv = _finite(b.get(column), float("inf"))
            better_or_equal = better_or_equal and av <= bv
            strictly_better = strictly_better or av < bv
    return better_or_equal and strictly_better


def assign_pareto_ranks(df: pd.DataFrame, objectives: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["pareto_rank"] = []
        return out
    out = df.copy().reset_index(drop=True)
    remaining = set(out.index.tolist())
    ranks = pd.Series(index=out.index, dtype="int64")
    rank = 0
    while remaining:
        front = []
        for idx in remaining:
            row = out.loc[idx]
            dominated = any(_dominates(out.loc[other], row, objectives) for other in remaining if other != idx)
            if not dominated:
                front.append(idx)
        for idx in front:
            ranks.loc[idx] = rank
        remaining.difference_update(front)
        rank += 1
    out["pareto_rank"] = ranks.astype(int)
    return out
