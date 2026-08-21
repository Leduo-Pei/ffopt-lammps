from __future__ import annotations

import numpy as np
import pandas as pd

from engine.structural_coverage import (
    assess_structural_properties,
    build_coverage_archives,
    coverage_role_counts,
    select_coverage_batch,
    select_coverage_representative,
    structural_constraint_specs,
)


def _config() -> dict:
    return {
        "targets": {
            "a": {"value": 2.0, "tolerance": 0.02},
            "b": {"value": 2.0, "tolerance": 0.02},
            "c": {"value": 2.0, "tolerance": 0.02},
            "alpha": {"value": 90.0, "tolerance": 1.0},
            "density": {"value": 8.0, "tolerance": 0.08},
            "surf_energy": {"value": 2.4, "tolerance": 0.12},
        },
        "elasticity": {
            "selection": {
                "structural_gates": {
                    "lattice": {"maximum_relative_error_percent": 1.0},
                    "angles": {"maximum_absolute_error_degree": 1.0},
                    "density": {"maximum_relative_error_percent": 1.0},
                    "surface": {"maximum_relative_error_percent": 5.0},
                }
            }
        },
    }


def test_structural_gate_is_satisficing_and_continuous_outside_band() -> None:
    exact = {
        "a": 2.0, "b": 2.0, "c": 2.0, "alpha": 90.0,
        "density": 8.0, "surf_energy": 2.4,
    }
    inside = dict(exact, a=2.019, surf_energy=2.51)
    outside = dict(exact, a=2.03)

    centre_score = assess_structural_properties(exact, _config())
    inside_score = assess_structural_properties(inside, _config())
    outside_score = assess_structural_properties(outside, _config())

    assert centre_score["structural_feasible"] is True
    assert inside_score["structural_feasible"] is True
    assert centre_score["acquisition_objective"] == 0.0
    assert inside_score["acquisition_objective"] == 0.0
    assert outside_score["structural_feasible"] is False
    assert outside_score["structural_constraint_violation"] > 0.0
    assert outside_score["structural_limiting_constraint"] == "a"


def test_specs_use_group_modes_and_target_fallback() -> None:
    specs = {item["name"]: item for item in structural_constraint_specs(_config())}
    assert specs["a"]["mode"] == "relative_percent"
    assert specs["alpha"]["mode"] == "absolute"
    assert specs["surf_energy"]["tolerance"] == 5.0


def test_coverage_selector_preserves_roles_and_exact_batch_size() -> None:
    rng = np.random.default_rng(17)
    pool = rng.random((500, 3))
    probability = np.clip(pool[:, 0], 0.0, 1.0)
    uncertainty = pool[:, 1]
    fractions = {
        "feasible_interior": 0.5,
        "constraint_boundary": 0.25,
        "model_uncertainty": 0.15,
        "global_novelty": 0.10,
    }
    selected, roles = select_coverage_batch(
        pool,
        existing=np.asarray([[0.5, 0.5, 0.5]]),
        feasible_probability=probability,
        model_uncertainty=uncertainty,
        batch_size=20,
        fractions=fractions,
    )

    assert selected.shape == (20, 3)
    assert {role: roles.count(role) for role in set(roles)} == {
        role: count for role, count in coverage_role_counts(20, fractions).items()
        if count
    }
    again, again_roles = select_coverage_batch(
        pool,
        existing=np.asarray([[0.5, 0.5, 0.5]]),
        feasible_probability=probability,
        model_uncertainty=uncertainty,
        batch_size=20,
        fractions=fractions,
    )
    np.testing.assert_allclose(selected, again)
    assert roles == again_roles


def test_archives_keep_strict_science_separate_from_boundary_anchors() -> None:
    frame = pd.DataFrame({
        "p1": [0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
        "p2": [0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
        "success": [True] * 6,
        # The last row deliberately carries an inconsistent surrogate-like
        # ratio.  The exact classifier label remains authoritative.
        "structural_feasible": [True, True, False, False, False, False],
        "structural_band_max_ratio": [0.2, 0.9, 1.1, 1.8, 4.0, 0.8],
        "fit_objective": [0.2, 0.1, 0.01, 0.4, 0.3, 0.001],
    })
    strict, anchors = build_coverage_archives(
        frame,
        parameter_names=["p1", "p2"],
        parameter_bounds=np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        archive_target=4,
        anchor_max_band_ratio=2.0,
    )

    assert strict["structural_band_max_ratio"].max() <= 1.0
    assert set(anchors["anchor_class"]) == {"strict_feasible", "near_boundary"}
    assert anchors["structural_band_max_ratio"].max() <= 2.0
    assert anchors.loc[anchors["p1"] == 0.95, "anchor_class"].item() == (
        "near_boundary"
    )
    assert strict.iloc[0]["p1"] == 0.2
    assert strict.iloc[0]["archive_selection_role"] == (
        "best_fit_structural_feasible"
    )
    assert set(strict["archive_selection_role"]) == {
        "best_fit_structural_feasible",
        "maximin_diversity",
    }


def test_report_representative_prioritizes_exact_gate_then_fit_objective() -> None:
    frame = pd.DataFrame([
        {
            "p": 0.0,
            "success": True,
            "structural_feasible": False,
            "structural_constraint_violation": 0.01,
            "structural_band_max_ratio": 1.01,
            "fit_objective": 0.001,
        },
        {
            "p": 0.4,
            "success": True,
            "structural_feasible": True,
            "structural_constraint_violation": 0.0,
            "structural_band_max_ratio": 0.95,
            "fit_objective": 0.20,
        },
        {
            "p": 0.8,
            "success": True,
            "structural_feasible": True,
            "structural_constraint_violation": 0.0,
            "structural_band_max_ratio": 0.10,
            "fit_objective": 0.10,
        },
    ])

    representative = select_coverage_representative(
        frame, parameter_names=["p"]
    )

    # The infeasible near miss is ineligible.  Among the tied-zero feasible
    # rows, fit objective—not row order or distance to the target centre—is
    # the deterministic report tie-breaker.
    assert representative["p"] == 0.8


def test_small_batch_retains_every_positive_coverage_role() -> None:
    fractions = {
        "feasible_interior": 0.50,
        "constraint_boundary": 0.25,
        "model_uncertainty": 0.15,
        "global_novelty": 0.10,
    }

    assert coverage_role_counts(4, fractions) == {
        "feasible_interior": 1,
        "constraint_boundary": 1,
        "model_uncertainty": 1,
        "global_novelty": 1,
    }
