from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import qmc

from engine.active_learning import ActiveLearner
from engine.lammps_interface import LARGE_PENALTY, LAMMPSRunner
from engine.optimizer import ForceFieldOptimizer
from engine.parameter_space import parameter_difference_error


DIFFERENCES = [{
    "parameter": "sigma",
    "types": ["Fe_corner", "Fe_body"],
    "max_abs": 0.75,
    "unit": "A",
}]


def _difference_error(params):
    return parameter_difference_error(params, DIFFERENCES)


@pytest.mark.parametrize("delta", [0.75, 0.75 + 2.0e-12])
def test_difference_constraint_accepts_boundary_float_noise(delta):
    assert _difference_error({
        "Fe_corner_sigma": 2.0,
        "Fe_body_sigma": 2.0 + delta,
    }) is None


def test_difference_constraint_rejects_value_beyond_boundary_tolerance():
    error = _difference_error({
        "Fe_corner_sigma": 2.0,
        "Fe_body_sigma": 2.75000001,
    })

    assert error is not None
    assert "parameter difference constraint #1 violated" in error
    assert "|Fe_corner_sigma - Fe_body_sigma|" in error
    assert "max_abs 0.75" in error


@pytest.mark.parametrize(
    ("params", "differences", "message"),
    [
        (
            {"Fe_corner_sigma": 2.0},
            DIFFERENCES,
            "missing resolved parameter 'Fe_body_sigma'",
        ),
        (
            {"Fe_corner_sigma": np.nan, "Fe_body_sigma": 2.0},
            DIFFERENCES,
            "'Fe_corner_sigma' is not finite",
        ),
        (
            {"Fe_corner_sigma": 2.0, "Fe_body_sigma": 2.1},
            [{**DIFFERENCES[0], "max_abs": float("inf")}],
            "'max_abs' must be a finite non-negative number",
        ),
        (
            {"Fe_corner_sigma": 2.0, "Fe_body_sigma": 2.1},
            {"parameter": "sigma"},
            "expected a list of difference rules",
        ),
    ],
)
def test_difference_constraint_malformed_or_nonfinite_inputs_fail_closed(
    params, differences, message
):
    assert message in parameter_difference_error(params, differences)


def test_lammps_runner_rejects_difference_before_property_execution(tmp_path):
    class MustNotExecute:
        def execute(self, *_args, **_kwargs):
            raise AssertionError("property execution must not start")

    runner = object.__new__(LAMMPSRunner)
    runner.atom_types = []
    runner.derived_params_cfg = []
    runner.use_charge = False
    runner.parameter_difference_constraints = DIFFERENCES
    runner.property_plan = MustNotExecute()

    result = runner._run_single(
        {"Fe_corner_sigma": 1.0, "Fe_body_sigma": 2.0}, str(tmp_path)
    )

    assert result.success is False
    assert result.objective == LARGE_PENALTY
    assert "parameter difference constraint #1 violated" in result.error_msg
    assert list(tmp_path.iterdir()) == []


def test_pair_coefficient_writer_has_defensive_difference_gate(tmp_path):
    runner = object.__new__(LAMMPSRunner)
    runner.parameter_difference_constraints = DIFFERENCES

    with pytest.raises(ValueError, match="refusing to write LAMMPS pair coefficients"):
        runner._build_pair_coeffs(
            {"Fe_corner_sigma": 1.0, "Fe_body_sigma": 2.0}, str(tmp_path)
        )

    assert list(tmp_path.iterdir()) == []


class _ConstraintRunner:
    parameter_difference_constraints = DIFFERENCES

    @staticmethod
    def feasibility_error(params):
        return _difference_error(params)


def _bare_optimizer(runner):
    optimizer = object.__new__(ForceFieldOptimizer)
    optimizer.n_params = 2
    optimizer.param_space = [
        ("Fe_corner_sigma", 0.0, 5.0),
        ("Fe_body_sigma", 0.0, 5.0),
    ]
    optimizer.param_names = [item[0] for item in optimizer.param_space]
    optimizer.seed = 123
    optimizer.current_round = 0
    optimizer.runner = runner
    return optimizer


def test_bo_lhs_rejection_sampling_is_feasible_and_reproducible():
    first = _bare_optimizer(_ConstraintRunner())._generate_lhs(64)
    second = _bare_optimizer(_ConstraintRunner())._generate_lhs(64)

    assert first == second
    assert len(first) == 64
    assert all(_difference_error(point) is None for point in first)


def test_bo_replaces_acquisition_violations_before_lammps():
    optimizer = _bare_optimizer(_ConstraintRunner())
    valid = {"Fe_corner_sigma": 2.0, "Fe_body_sigma": 2.5}
    invalid = {"Fe_corner_sigma": 1.0, "Fe_body_sigma": 4.0}

    prepared = optimizer._replace_hard_constraint_violations([valid, invalid])

    assert len(prepared) == 2
    assert prepared[0] is valid
    assert all(_difference_error(point) is None for point in prepared)


def test_bo_lhs_without_difference_rule_keeps_legacy_sequence():
    class LegacyRunner:
        parameter_difference_constraints = []

        @staticmethod
        def feasibility_error(_params):
            raise AssertionError("legacy LHS must not invoke constraint filtering")

    actual = _bare_optimizer(LegacyRunner())._generate_lhs(8)
    unit = qmc.LatinHypercube(d=2, seed=123).random(n=8)
    expected = qmc.scale(unit, [0.0, 0.0], [5.0, 5.0])

    assert np.allclose(
        [[row["Fe_corner_sigma"], row["Fe_body_sigma"]] for row in actual],
        expected,
    )


def _bare_active_learner(runner):
    learner = object.__new__(ActiveLearner)
    learner.candidate_sampling = "random"
    learner.seed = 456
    learner.n_params = 2
    learner.param_names = ["Fe_corner_sigma", "Fe_body_sigma"]
    learner.sample_lo = np.asarray([0.0, 0.0])
    learner.sample_hi = np.asarray([5.0, 5.0])
    learner.runner = runner
    return learner


def test_al_envelope_rejection_sampling_is_feasible_and_reproducible():
    first = _bare_active_learner(_ConstraintRunner())._sample_candidates(64, 7)
    second = _bare_active_learner(_ConstraintRunner())._sample_candidates(64, 7)

    assert np.array_equal(first, second)
    assert first.shape == (64, 2)
    assert np.all(np.abs(first[:, 0] - first[:, 1]) <= 0.75 + 1.0e-12)
