from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from engine.constrained_refinement import (
    AcquisitionContext,
    MechanicalObjectiveSpec,
    RefinementError,
    RefinementSpec,
    StructuralConstraintSpec,
    _with_parameter_keys,
    run_refinement_round,
)
from engine.gp_structural_acquisition import (
    GaussianProcessStructuralAcquisition,
    _improvement_feasibility_tiers,
    constrained_expected_improvement,
)


PARAMETERS = ("x",)
CONSTRAINTS = (
    StructuralConstraintSpec.from_mapping({
        "name": "density",
        "column": "density",
        "target": 10.0,
        "tolerance": 1.0,
        "mode": "absolute",
    }),
)
OBJECTIVES = (
    MechanicalObjectiveSpec.from_mapping({
        "name": "B",
        "column": "B",
        "target": 100.0,
    }),
)


def _keyed(rows) -> pd.DataFrame:
    return _with_parameter_keys(pd.DataFrame(rows), PARAMETERS)


def _context(
    *,
    structural: pd.DataFrame,
    mechanical: pd.DataFrame,
    pool: pd.DataFrame,
    seed: int = 2718,
    proposal_count: int = 6,
) -> AcquisitionContext:
    return AcquisitionContext(
        candidate_pool=pool,
        structural_observations=structural,
        mechanical_observations=mechanical,
        parameter_names=PARAMETERS,
        round_number=1,
        proposal_count=proposal_count,
        seed=seed,
        structural_constraints=CONSTRAINTS,
        mechanical_objectives=OBJECTIVES,
    )


def test_saturated_structural_probability_preserves_mechanical_ei():
    expected, constrained = constrained_expected_improvement(
        feasible_probability=np.ones(3),
        predicted_mean=np.asarray([8.0, 10.0, 12.0]),
        predicted_std=np.ones(3),
        incumbent=10.0,
    )

    assert expected[0] > expected[1] > expected[2] > 0.0
    assert constrained == pytest.approx(expected)


def test_improvement_feasibility_tiers_expand_before_relaxing_floor():
    local, global_, relaxed, diagnostics = _improvement_feasibility_tiers(
        local_indices=[0, 1],
        all_indices=range(6),
        probability=np.asarray([0.90, 0.80, 0.70, 0.60, 0.20, 0.10]),
        constrained_improvement=np.asarray([1.0, 2.0, 3.0, 4.0, 100.0, 200.0]),
        count=3,
        nominal_floor=0.50,
    )

    assert local == [0, 1]
    assert global_ == [2, 3]
    assert relaxed == []
    assert diagnostics["effective_floor"] == pytest.approx(0.50)
    assert not diagnostics["floor_relaxed"]


def test_improvement_feasibility_tiers_relax_by_probability_not_huge_ei():
    local, global_, relaxed, diagnostics = _improvement_feasibility_tiers(
        local_indices=[0],
        all_indices=range(5),
        probability=np.asarray([0.80, 0.40, 0.30, 0.10, np.nan]),
        constrained_improvement=np.asarray([1.0, 2.0, 3.0, 1.0e6, 1.0e9]),
        count=3,
        nominal_floor=0.50,
    )

    assert local == [0]
    assert global_ == []
    assert relaxed == [1, 2]
    assert diagnostics["effective_floor"] == pytest.approx(0.30)
    assert diagnostics["floor_relaxed"]
    assert diagnostics["finite_capacity_shortfall"] == 0


def test_structural_failure_with_perfect_mechanics_never_becomes_incumbent():
    structural = _keyed([
        {"x": 0.0, "density": 12.0},  # exact structural failure
        {"x": 0.3, "density": 10.0},
        {"x": 0.6, "density": 10.2},
        {"x": 0.9, "density": 11.5},
    ])
    mechanical = _keyed([
        {"x": 0.0, "B": 100.0},  # zero error, but structurally invalid
        {"x": 0.3, "B": 130.0},
        {"x": 0.9, "B": 101.0},  # also structurally invalid
    ])
    pool = _keyed([{"x": value} for value in (0.1, 0.2, 0.4, 0.5, 0.7, 0.8)])
    backend = GaussianProcessStructuralAcquisition(
        minimum_structural_observations=3,
        minimum_mechanical_observations=3,
    )

    result = backend.propose(_context(
        structural=structural, mechanical=mechanical, pool=pool
    ))

    expected_key = structural.loc[structural["x"].eq(0.3), "parameter_key"].iloc[0]
    assert result.diagnostics["incumbent"] == {
        "available": True,
        "source": "exact_observed",
        "parameter_key": expected_key,
        "mechanical_minimax_error_percent": pytest.approx(30.0),
    }
    assert set(result.proposals["evidence_kind"]) == {"surrogate_proposal"}
    assert not result.proposals["surrogate_is_exact_observation"].any()
    assert "structural_feasible" not in result.proposals


def test_seed_reproduces_coverage_fallback_batch_exactly():
    structural = _keyed([
        {"x": 0.0, "density": 12.0},
        {"x": 0.5, "density": 10.0},
        {"x": 1.0, "density": 12.0},
    ])
    mechanical = pd.DataFrame(columns=["x", "B", "parameter_key"])
    pool = _keyed([
        *({"x": value} for value in np.linspace(0.025, 0.975, 20)),
        {"x": 0.025},  # duplicate physical coordinate is removed before selection
    ])
    backend = GaussianProcessStructuralAcquisition(
        minimum_structural_observations=3,
        minimum_mechanical_observations=3,
    )
    context = _context(
        structural=structural,
        mechanical=mechanical,
        pool=pool,
        seed=1234,
        proposal_count=7,
    )

    first = backend.propose(context)
    second = backend.propose(context)

    pdt.assert_frame_equal(first.proposals, second.proposals)
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics["mode"] == "coverage_fallback"
    assert not first.scientific_convergence_capable
    assert first.proposals["parameter_key"].is_unique


def _smooth_context(*, mechanical: bool = True) -> AcquisitionContext:
    x_observed = np.linspace(0.0, 1.0, 31)
    structural = _keyed([
        {"x": float(value), "density": 10.0 + 3.0 * (value - 0.5)}
        for value in x_observed
    ])
    if mechanical:
        mechanics = _keyed([
            {"x": float(value), "B": 100.0 + 40.0 * value}
            for value in x_observed
        ])
    else:
        mechanics = pd.DataFrame(columns=["x", "B", "parameter_key"])
    pool = _keyed([
        {"x": float(value)} for value in np.linspace(0.0125, 0.9875, 40)
    ])
    return _context(
        structural=structural,
        mechanical=mechanics,
        pool=pool,
        seed=8675309,
        proposal_count=6,
    )


def test_reliable_gp_batch_has_local_boundary_and_global_roles():
    backend = GaussianProcessStructuralAcquisition(
        improvement_fraction=0.5,
        boundary_fraction=1.0 / 3.0,
        global_fraction=1.0 / 6.0,
        trust_region_radius=0.2,
        minimum_structural_observations=6,
        minimum_mechanical_observations=6,
        minimum_holdout_r2=-100.0,
    )

    result = backend.propose(_smooth_context())
    proposals = result.proposals

    assert result.diagnostics["mode"] == "constrained_gp"
    assert result.scientific_convergence_capable
    assert set(proposals["selection_role"]) == {
        "constrained_mechanical_improvement_local",
        "structural_boundary_entropy_local",
        "global_coverage",
    }
    local = proposals["selection_role"].str.endswith("_local")
    assert proposals.loc[local, "surrogate_inside_trust_region"].all()
    assert not proposals.loc[
        proposals["selection_role"].eq("global_coverage"),
        "surrogate_inside_trust_region",
    ].any()
    assert np.isfinite(
        proposals["surrogate_structural_boundary_entropy"].to_numpy(float)
    ).all()
    assert np.isfinite(
        proposals["surrogate_mechanical_expected_improvement_percent"].to_numpy(float)
    ).all()


def test_unreliable_gate_gp_does_not_veto_reliable_mechanical_improvement():
    rng = np.random.default_rng(20260829)
    observed_x = np.linspace(0.0, 1.0, 61)
    gate_values = np.where(rng.random(len(observed_x)) < 0.70, 0.5, 1.5)
    constraint = StructuralConstraintSpec.from_mapping({
        "name": "beta_gate_ratio_proxy",
        "column": "beta_error",
        "target": 0.0,
        "tolerance": 1.0,
        "mode": "absolute",
    })
    structural = _keyed([
        {"x": float(value), "beta_error": float(gate)}
        for value, gate in zip(observed_x, gate_values)
    ])
    mechanical = _keyed([
        {"x": float(value), "B": 100.0 + 30.0 * float(value)}
        for value, gate in zip(observed_x, gate_values)
        if gate <= 1.0
    ])
    pool = _keyed([
        {"x": float(value)} for value in np.linspace(0.0075, 0.9925, 80)
    ])
    context = AcquisitionContext(
        candidate_pool=pool,
        structural_observations=structural,
        mechanical_observations=mechanical,
        parameter_names=PARAMETERS,
        round_number=1,
        proposal_count=10,
        seed=314159,
        structural_constraints=(constraint,),
        mechanical_objectives=OBJECTIVES,
    )
    backend = GaussianProcessStructuralAcquisition(
        improvement_fraction=0.6,
        boundary_fraction=0.2,
        global_fraction=0.2,
        minimum_structural_observations=10,
        minimum_mechanical_observations=10,
        minimum_holdout_r2=0.90,
    )

    result = backend.propose(context)

    assert not result.diagnostics["quality"]["structural_models_reliable"]
    assert result.diagnostics["quality"]["mechanical_model_reliable"]
    assert result.diagnostics["mode"] == "partial_constraint_fallback"
    assert not result.scientific_convergence_capable
    assert (
        result.proposals["selection_role"]
        .eq("mechanical_improvement_partial_constraints_local")
        .any()
    )
    assert np.isfinite(
        result.proposals["surrogate_mechanical_expected_improvement_percent"]
    ).all()


def test_partial_fallback_retains_reliable_gate_model_alongside_noisy_gate():
    rng = np.random.default_rng(90210)
    observed_x = np.linspace(0.0, 1.0, 81)
    density = 10.0 + 2.0 * (observed_x - 0.5)
    beta_error = np.where(rng.random(len(observed_x)) < 0.75, 0.4, 1.4)
    constraints = (
        StructuralConstraintSpec.from_mapping({
            "name": "density",
            "column": "density",
            "target": 10.0,
            "tolerance": 0.75,
            "mode": "absolute",
        }),
        StructuralConstraintSpec.from_mapping({
            "name": "beta",
            "column": "beta_error",
            "target": 0.0,
            "tolerance": 1.0,
            "mode": "absolute",
        }),
    )
    structural = _keyed([
        {"x": float(x), "density": float(rho), "beta_error": float(beta)}
        for x, rho, beta in zip(observed_x, density, beta_error)
    ])
    mechanical = _keyed([
        {"x": float(x), "B": 100.0 + 30.0 * float(x)}
        for x, rho, beta in zip(observed_x, density, beta_error)
        if abs(float(rho) - 10.0) <= 0.75 and beta <= 1.0
    ])
    pool = _keyed([
        {"x": float(value)} for value in np.linspace(0.006, 0.994, 100)
    ])
    context = AcquisitionContext(
        candidate_pool=pool,
        structural_observations=structural,
        mechanical_observations=mechanical,
        parameter_names=PARAMETERS,
        round_number=1,
        proposal_count=12,
        seed=112358,
        structural_constraints=constraints,
        mechanical_objectives=OBJECTIVES,
    )
    backend = GaussianProcessStructuralAcquisition(
        improvement_fraction=0.5,
        boundary_fraction=0.25,
        global_fraction=0.25,
        minimum_structural_observations=10,
        minimum_mechanical_observations=10,
        minimum_holdout_r2=0.75,
    )

    result = backend.propose(context)

    modes = result.diagnostics["constraint_guidance_modes"]
    assert result.diagnostics["mode"] == "partial_constraint_fallback"
    assert modes["density"] == "reliable_gp"
    assert modes["beta"] == "shrunk_gp_fallback"
    assert result.diagnostics["quality"]["mechanical_model_reliable"]
    assert not result.scientific_convergence_capable
    improvement = result.proposals["selection_role"].str.startswith(
        "mechanical_improvement_partial_constraints"
    )
    gate = result.diagnostics["improvement_feasibility_gate"]
    assert not gate["floor_relaxed"]
    assert (
        result.proposals.loc[
            improvement, "surrogate_structural_feasibility_probability"
        ]
        >= gate["nominal_floor"]
    ).all()


def test_zero_mechanical_labels_fails_safe_to_coverage_without_convergence_claim():
    backend = GaussianProcessStructuralAcquisition(
        minimum_structural_observations=6,
        minimum_mechanical_observations=6,
        minimum_holdout_r2=-100.0,
    )

    result = backend.propose(_smooth_context(mechanical=False))

    assert result.diagnostics["mode"] == "coverage_fallback"
    assert "mechanical_gp_quality_insufficient" in result.diagnostics[
        "fallback_reasons"
    ]
    assert not result.scientific_convergence_capable
    assert result.proposals[
        "surrogate_mechanical_expected_improvement_percent"
    ].isna().all()
    assert "global_coverage" in set(result.proposals["selection_role"])


def test_candidate_pool_cannot_masquerade_property_values_as_exact_evidence():
    context = _smooth_context()
    ambiguous_pool = context.candidate_pool.copy()
    ambiguous_pool["density"] = 10.0
    backend = GaussianProcessStructuralAcquisition(
        minimum_structural_observations=6,
        minimum_mechanical_observations=6,
    )

    with pytest.raises(RefinementError, match="not exact property evidence"):
        backend.propose(AcquisitionContext(
            candidate_pool=ambiguous_pool,
            structural_observations=context.structural_observations,
            mechanical_observations=context.mechanical_observations,
            parameter_names=context.parameter_names,
            round_number=context.round_number,
            proposal_count=context.proposal_count,
            seed=context.seed,
            structural_constraints=context.structural_constraints,
            mechanical_objectives=context.mechanical_objectives,
        ))


def test_round_state_records_per_evidence_capability_and_diagnostics(tmp_path):
    structural_path = tmp_path / "structural.csv"
    mechanical_path = tmp_path / "mechanical.csv"
    pool_path = tmp_path / "pool.csv"
    pd.DataFrame([
        {"x": 0.0, "density": 12.0},
        {"x": 0.3, "density": 10.0},
        {"x": 0.6, "density": 10.2},
        {"x": 0.9, "density": 11.5},
    ]).to_csv(structural_path, index=False)
    pd.DataFrame([{"x": 0.3, "B": 130.0}]).to_csv(mechanical_path, index=False)
    pd.DataFrame([
        {"x": value} for value in (0.1, 0.2, 0.4, 0.5, 0.7, 0.8)
    ]).to_csv(pool_path, index=False)
    spec = RefinementSpec.from_mapping({
        "parameter_names": ["x"],
        "structural_constraints": [
            {
                "name": "density",
                "column": "density",
                "target": 10.0,
                "tolerance": 1.0,
                "mode": "absolute",
            }
        ],
        "mechanical_objectives": [
            {"name": "B", "column": "B", "target": 100.0}
        ],
        "require_stability": False,
        "structural_proposals_per_round": 3,
    })
    backend = GaussianProcessStructuralAcquisition(
        minimum_structural_observations=3,
        minimum_mechanical_observations=3,
    )

    result = run_refinement_round(
        spec=spec,
        structural_paths=[structural_path],
        mechanical_paths=[mechanical_path],
        candidate_pool_path=pool_path,
        output_dir=tmp_path / "round_01",
        acquisition=backend,
        seed=99,
    )

    assert not result.state["acquisition_scientific_convergence_capable"]
    assert result.state["acquisition_diagnostics"]["mode"] == "coverage_fallback"
    assert result.state["acquisition_diagnostics"]["seed"] == 99
