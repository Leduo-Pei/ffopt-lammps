from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.feasible_clusters import (
    FeasibleClusterError,
    cluster_feasible_candidates,
    select_cluster_balanced_candidates,
)


BOUNDS_2D = {"p": (0.0, 1.0), "q": (-2.0, 2.0)}


def _three_basin_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "p": [0.02, 0.04, 0.06, 0.46, 0.48, 0.50, 0.91],
            "q": [-1.82, -1.78, -1.75, -0.05, 0.00, 0.04, 1.74],
            "loss": [0.10, 0.12, 0.14, 0.20, 0.22, 0.24, 99.0],
            "candidate_id": ["a", "b", "c", "d", "e", "f", "tiny"],
        }
    )


def test_cluster_deduplicates_exact_keys_and_preserves_full_precision() -> None:
    half_next = np.nextafter(0.5, 1.0)
    frame = pd.DataFrame(
        {
            "p": [0.5, 0.5, half_next],
            "q": [0.0, 0.0, 0.0],
            "loss": [2.0, 1.0, 3.0],
            "source": ["worse_duplicate", "better_duplicate", "adjacent_float"],
        }
    )

    clustered = cluster_feasible_candidates(
        frame,
        parameter_columns=["p", "q"],
        bounds=BOUNDS_2D,
        quality_column="loss",
    )

    assert len(clustered) == 2
    assert clustered["parameter_key"].nunique() == 2
    exact = clustered.loc[clustered["p"].eq(0.5)]
    assert exact.iloc[0]["source"] == "better_duplicate"
    assert set(clustered["p"]) == {0.5, half_next}


def test_cluster_labels_are_invariant_to_row_order_and_detect_three_basins() -> None:
    frame = _three_basin_frame()
    original = cluster_feasible_candidates(
        frame,
        parameter_columns=["p", "q"],
        bounds=BOUNDS_2D,
        maximum_clusters=6,
    )
    shuffled = cluster_feasible_candidates(
        frame.sample(frac=1.0, random_state=912).reset_index(drop=True),
        parameter_columns=["p", "q"],
        bounds=BOUNDS_2D,
        maximum_clusters=6,
    )

    assert original.attrs["feasible_cluster_count"] == 3
    assert shuffled.attrs["feasible_cluster_count"] == 3
    original_map = dict(zip(original["parameter_key"], original["feasible_cluster"], strict=True))
    shuffled_map = dict(zip(shuffled["parameter_key"], shuffled["feasible_cluster"], strict=True))
    assert original_map == shuffled_map
    assert sorted(original["feasible_cluster_size"].unique()) == [1, 3]


def test_automatic_cluster_count_respects_configured_cap() -> None:
    clustered = cluster_feasible_candidates(
        _three_basin_frame(),
        parameter_columns=["p", "q"],
        bounds=BOUNDS_2D,
        maximum_clusters=2,
    )

    assert clustered.attrs["feasible_cluster_count"] == 2


def test_selection_gives_even_a_poor_small_basin_its_minimum_quota() -> None:
    frame = _three_basin_frame()
    selected = select_cluster_balanced_candidates(
        frame,
        parameter_columns=["p", "q"],
        bounds=BOUNDS_2D,
        quality_column="loss",
        budget=5,
        minimum_per_cluster=1,
        quality_elite_fraction=0.40,
        maximum_clusters=6,
    )
    shuffled = select_cluster_balanced_candidates(
        frame.sample(frac=1.0, random_state=48).reset_index(drop=True),
        parameter_columns=["p", "q"],
        bounds=BOUNDS_2D,
        quality_column="loss",
        budget=5,
        minimum_per_cluster=1,
        quality_elite_fraction=0.40,
        maximum_clusters=6,
    )

    assert len(selected) == 5
    assert list(selected["parameter_key"]) == list(shuffled["parameter_key"])
    assert list(selected["cluster_selection_role"]) == list(
        shuffled["cluster_selection_role"]
    )
    assert list(selected["cluster_selection_rank"]) == [1, 2, 3, 4, 5]
    assert selected["parameter_key"].is_unique
    assert selected["feasible_cluster"].nunique() == 3
    assert "tiny" in set(selected["candidate_id"])
    assert selected.loc[
        selected["candidate_id"].eq("tiny"), "cluster_selection_role"
    ].item() == "cluster_quota"
    # The best-quality member of every basin is the one that pays its quota.
    for _cluster, rows in selected.groupby("feasible_cluster"):
        source_rows = _three_basin_frame()
        member_keys = set(rows["parameter_key"])
        clustered_source = cluster_feasible_candidates(
            source_rows,
            parameter_columns=["p", "q"],
            bounds=BOUNDS_2D,
            quality_column="loss",
            maximum_clusters=6,
        )
        cluster = rows.iloc[0]["feasible_cluster"]
        best_key = clustered_source.loc[
            clustered_source["feasible_cluster"].eq(cluster)
        ].sort_values(["loss", "parameter_key"]).iloc[0]["parameter_key"]
        assert best_key in member_keys


def test_required_candidate_is_retained_and_counts_toward_cluster_quota() -> None:
    clustered = cluster_feasible_candidates(
        _three_basin_frame(),
        parameter_columns=["p", "q"],
        bounds=BOUNDS_2D,
        quality_column="loss",
    )
    required_key = clustered.loc[clustered["candidate_id"].eq("f"), "parameter_key"].item()
    selected = select_cluster_balanced_candidates(
        _three_basin_frame(),
        parameter_columns=["p", "q"],
        bounds=BOUNDS_2D,
        quality_column="loss",
        budget=4,
        minimum_per_cluster=1,
        required_parameter_keys=[required_key],
    )

    required = selected.loc[selected["parameter_key"].eq(required_key)]
    assert len(required) == 1
    assert required.iloc[0]["cluster_selection_role"] == "required"


def test_single_basin_selection_has_exact_budget_and_maximin_fill() -> None:
    frame = pd.DataFrame(
        {
            "p": [0.0, 0.25, 0.50, 0.75, 1.0],
            "loss": [4.0, 3.0, 0.0, 2.0, 1.0],
        }
    )
    selected = select_cluster_balanced_candidates(
        frame,
        parameter_columns=["p"],
        bounds={"p": (0.0, 1.0)},
        quality_column="loss",
        budget=4,
        minimum_per_cluster=1,
        quality_elite_fraction=0.50,
    )

    assert selected.attrs["feasible_cluster_count"] == 1
    assert len(selected) == 4
    assert "maximin_fill" in set(selected["cluster_selection_role"])
    assert 0.50 in set(selected["p"])


@pytest.mark.parametrize(
    ("frame", "kwargs", "message"),
    [
        (
            pd.DataFrame(columns=["p", "loss"]),
            {"budget": 1},
            "no feasible candidates",
        ),
        (
            pd.DataFrame({"p": [0.2], "loss": [1.0]}),
            {"budget": 0},
            "budget must be",
        ),
        (
            pd.DataFrame({"p": [0.2], "loss": [1.0]}),
            {"budget": 2},
            "exceeds 1 unique",
        ),
    ],
)
def test_invalid_or_empty_selection_requests_fail_clearly(
    frame: pd.DataFrame,
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(FeasibleClusterError, match=message):
        select_cluster_balanced_candidates(
            frame,
            parameter_columns=["p"],
            bounds={"p": (0.0, 1.0)},
            quality_column="loss",
            **kwargs,
        )


def test_impossible_cluster_quota_and_bad_bounds_fail_clearly() -> None:
    with pytest.raises(FeasibleClusterError, match="too small.*cluster quota"):
        select_cluster_balanced_candidates(
            _three_basin_frame(),
            parameter_columns=["p", "q"],
            bounds=BOUNDS_2D,
            quality_column="loss",
            budget=2,
            minimum_per_cluster=1,
        )
    with pytest.raises(FeasibleClusterError, match="positive span"):
        cluster_feasible_candidates(
            pd.DataFrame({"p": [0.2]}),
            parameter_columns=["p"],
            bounds={"p": (1.0, 1.0)},
        )


def test_mismatched_recorded_parameter_key_is_rejected() -> None:
    with pytest.raises(FeasibleClusterError, match="recorded parameter_key"):
        cluster_feasible_candidates(
            pd.DataFrame({"p": [0.2], "parameter_key": ["named:sha256:" + "0" * 64]}),
            parameter_columns=["p"],
            bounds={"p": (0.0, 1.0)},
        )
