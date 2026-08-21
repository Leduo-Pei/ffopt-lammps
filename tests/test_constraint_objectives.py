import math

import pytest

from engine.constraint_objectives import (
    assess_constraint,
    constrained_minimax_rank_key,
    minimax_relative_errors,
    relative_error_percent,
    summarize_constraints,
)


def test_constraint_dead_zone_retains_margin_and_continuous_violation():
    inside = assess_constraint("density", 7.82, 7.874, 1.0)
    outside = assess_constraint("surface", 2.50, 2.34, 5.0)
    summary = summarize_constraints([inside, outside])

    assert inside.passed is True
    assert inside.margin > 0.0
    assert outside.passed is False
    assert outside.margin < 0.0
    assert summary["feasible"] is False
    assert summary["limiting_constraint"] == "surface"
    assert summary["constraint_violation"] == pytest.approx(
        (max(0.0, outside.ratio - 1.0) ** 2 / 2.0) ** 0.5
    )


def test_absolute_angle_constraint_and_invalid_observation_fail_closed():
    angle = assess_constraint("alpha", 89.4, 90.0, 1.0, mode="absolute")
    missing = assess_constraint("beta", math.nan, 90.0, 1.0, mode="absolute")

    assert angle.error == pytest.approx(0.6)
    assert angle.passed is True
    assert missing.passed is False
    assert summarize_constraints([angle, missing])["constraint_violation"] == math.inf


def test_minimax_uses_independent_targets_and_reports_limiter():
    metrics = minimax_relative_errors(
        {"B": 190.3, "Cprime": 50.0, "C44": 100.0},
        {"B": {"value": 173.0}, "Cprime": 52.5, "C44": 121.9},
    )

    assert metrics["errors_percent"]["B"] == pytest.approx(10.0)
    assert metrics["limiting_target"] == "C44"
    assert metrics["maximum_error_percent"] == pytest.approx(17.9655455291)
    assert metrics["rmse_percent"] < metrics["maximum_error_percent"]


def test_zero_target_and_missing_value_cannot_become_incumbent():
    assert relative_error_percent(0.0, 0.0) == 0.0
    assert relative_error_percent(0.1, 0.0) == math.inf
    metrics = minimax_relative_errors({"B": None}, {"B": 173.0})
    assert metrics["maximum_error_percent"] == math.inf


def test_hard_constraint_precedes_better_mechanical_score():
    passed = summarize_constraints([assess_constraint("density", 7.874, 7.874, 1.0)])
    failed = summarize_constraints([assess_constraint("density", 8.1, 7.874, 1.0)])
    worse_mechanics = {"maximum_error_percent": 30.0, "rmse_percent": 20.0}
    better_mechanics = {"maximum_error_percent": 5.0, "rmse_percent": 4.0}

    eligible_key = constrained_minimax_rank_key(
        constraint_summary=passed,
        objective_summary=worse_mechanics,
        born_stable=True,
        fit_r2=0.99,
        minimum_fit_r2=0.98,
    )
    rejected_key = constrained_minimax_rank_key(
        constraint_summary=failed,
        objective_summary=better_mechanics,
        born_stable=True,
        fit_r2=0.99,
        minimum_fit_r2=0.98,
    )

    assert eligible_key < rejected_key


def test_invalid_constraint_mode_and_tolerance_are_rejected():
    with pytest.raises(ValueError, match="Unsupported constraint mode"):
        assess_constraint("x", 1.0, 1.0, 1.0, mode="fraction")
    with pytest.raises(ValueError, match="positive tolerance"):
        assess_constraint("x", 1.0, 1.0, 0.0)
