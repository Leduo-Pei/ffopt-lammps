from __future__ import annotations

import math

import pytest

from engine.property_evaluators import (
    PropertyEvaluationContext,
    build_property_plan,
)


class FakeRunner:
    def __init__(self) -> None:
        self.bulk_required = True
        self.ads_enabled = False
        self.sub_enabled = True
        self.compute_surface = False
        self.bulk_col_map = ["a", "pe"]
        self.bulk_targets = {"a": "a"}
        self.calls: list[str] = []

    def _run_bulk(self, resolved, eval_dir, save_traj, seed_override):
        self.calls.append("bulk")
        return {"a": 4.2, "pe": -100.0}

    def _bulk_sanity_check(self, properties):
        return None

    def _compute_sublimation_proxy_from_bulk_pe(
        self, resolved, eval_dir, bulk_pe, save_traj
    ):
        self.calls.append("sublimation")
        assert bulk_pe == -100.0
        return {
            "esub_proxy_kj_mol": 98.5,
            "esub_proxy_kcal_mol": 23.54,
            "E_single_kcal": -10.0,
            "E_bulk_per_molecule_kcal": -33.54,
        }


def test_bulk_precedes_sublimation_and_composes_properties() -> None:
    runner = FakeRunner()
    config = {
        "targets": {
            "a": {"value": 4.2, "weight": 1.0},
            "esub_proxy": {"value": 98.5, "weight": 0.3},
        }
    }
    plan = build_property_plan(config, runner)
    assert plan.names == ("bulk", "sublimation")
    result = plan.execute(
        runner,
        PropertyEvaluationContext({}, "evaluation", seed_override=202),
    )
    assert result.success
    assert runner.calls == ["bulk", "sublimation"]
    assert result.properties["a"] == 4.2
    assert result.properties["esub_proxy"] == 98.5


def test_adsorption_only_plan_does_not_add_bulk() -> None:
    runner = FakeRunner()
    runner.bulk_required = False
    runner.sub_enabled = False
    runner.ads_enabled = True
    runner._run_adsorption = lambda *args: {"ead": -3.5}
    plan = build_property_plan(
        {"targets": {"ead": {"value": -3.0, "weight": 1.0}}}, runner
    )
    assert plan.names == ("adsorption",)
    result = plan.execute(runner, PropertyEvaluationContext({}, "evaluation"))
    assert result.success
    assert result.properties == {"ead": -3.5}


def test_surface_evaluator_uses_cleavage_energy_conversion() -> None:
    runner = FakeRunner()
    runner.bulk_required = False
    runner.sub_enabled = False
    runner.compute_surface = True
    runner.cutoff = 8.0
    runner.surf_npt_seed = 101
    runner.surf_equil = 10
    runner.surf_prod = 20
    runner.use_charge = False
    runner.surf_complete = "complete.data"
    runner.surf_split = "split.data"
    runner.surf_E_key = "energy"
    runner.surf_A_key = "area"
    runner.surf_energy_min = 0.1
    runner.surf_energy_max = 10.0
    runner._run_surf = lambda resolved, eval_dir, tag, path, variables: (
        {"energy": 0.0, "area": 10.0}
        if tag == "complete" else {"energy": 10.0, "area": 10.0}
    )
    plan = build_property_plan(
        {"targets": {"surf_energy": {"value": 0.35, "weight": 1.0}}},
        runner,
    )
    result = plan.execute(runner, PropertyEvaluationContext({}, "evaluation"))
    assert result.success
    assert math.isclose(result.properties["surf_energy"], 0.347385)


def test_missing_target_provider_fails_preflight() -> None:
    runner = FakeRunner()
    runner.bulk_required = False
    runner.sub_enabled = False
    with pytest.raises(ValueError, match="unknown_property"):
        build_property_plan(
            {"targets": {"unknown_property": {"value": 1.0, "weight": 1.0}}},
            runner,
        )

