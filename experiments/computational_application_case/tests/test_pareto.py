from __future__ import annotations

import pandas as pd
import pytest

from experiments.computational_application_case.src.pareto import (
    add_utopia_distance,
    non_dominated_sort,
)


OBJECTIVES = {"maximize": ["benefit"], "minimize": ["cost"]}


def test_dominated_and_non_dominated_candidates_are_identified() -> None:
    frame = pd.DataFrame(
        {"candidate_id": ["A", "B", "C"], "benefit": [3.0, 2.0, 1.0], "cost": [1.0, 2.0, 0.5]}
    )
    result = non_dominated_sort(frame, OBJECTIVES)
    ranks = result.set_index("candidate_id")["Pareto_rank"].to_dict()
    assert ranks["A"] == 1
    assert ranks["C"] == 1
    assert ranks["B"] == 2


def test_maximize_and_minimize_directions_are_respected() -> None:
    frame = pd.DataFrame(
        {"candidate_id": ["low_cost", "high_cost"], "benefit": [1.0, 1.0], "cost": [1.0, 2.0]}
    )
    result = non_dominated_sort(frame, OBJECTIVES).set_index("candidate_id")
    assert result.loc["low_cost", "Pareto_rank"] == 1
    assert result.loc["high_cost", "Pareto_rank"] == 2
    assert result.loc["low_cost", "dominated_set_size"] == 1
    assert result.loc["high_cost", "domination_count"] == 1


def test_non_dominated_sort_builds_more_than_one_rank() -> None:
    frame = pd.DataFrame(
        {"candidate_id": ["A", "B", "C"], "benefit": [3.0, 2.0, 1.0], "cost": [1.0, 2.0, 3.0]}
    )
    assert non_dominated_sort(frame, OBJECTIVES)["Pareto_rank"].tolist() == [1, 2, 3]


def test_utopia_distance_prefers_balanced_rank_one_candidate() -> None:
    frame = pd.DataFrame(
        {"candidate_id": ["transport", "balanced", "thermal"], "benefit": [10.0, 8.0, 6.0], "cost": [5.0, 2.0, 1.0], "Pareto_rank": [1, 1, 1]}
    )
    result = add_utopia_distance(frame, OBJECTIVES).set_index("candidate_id")
    assert result.loc["balanced", "utopia_distance"] < result.loc["transport", "utopia_distance"]
    assert result.loc["balanced", "utopia_distance"] < result.loc["thermal", "utopia_distance"]


def test_tied_candidates_remain_on_the_same_front() -> None:
    frame = pd.DataFrame(
        {"candidate_id": ["A", "B"], "benefit": [2.0, 2.0], "cost": [1.0, 1.0]}
    )
    result = non_dominated_sort(frame, OBJECTIVES)
    assert result["Pareto_rank"].tolist() == [1, 1]
    assert result["domination_count"].tolist() == [0, 0]


def test_single_candidate_has_rank_one_and_zero_utopia_distance() -> None:
    frame = pd.DataFrame({"candidate_id": ["A"], "benefit": [2.0], "cost": [1.0]})
    result = add_utopia_distance(non_dominated_sort(frame, OBJECTIVES), OBJECTIVES)
    assert result.loc[0, "Pareto_rank"] == 1
    assert result.loc[0, "utopia_distance"] == pytest.approx(0.0)


def test_empty_candidate_set_returns_typed_result_columns() -> None:
    frame = pd.DataFrame(columns=["candidate_id", "benefit", "cost"])
    result = add_utopia_distance(non_dominated_sort(frame, OBJECTIVES), OBJECTIVES)
    assert result.empty
    assert {"Pareto_rank", "domination_count", "dominated_set_size", "utopia_distance"}.issubset(result.columns)

