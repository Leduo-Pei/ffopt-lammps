from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from engine.constrained_refinement import (
    AcquisitionResult,
    ExplicitPoolAcquisition,
    RefinementError,
    RefinementSpec,
    assess_and_rank_candidates,
    main,
    run_refinement_round,
)


def _spec(**overrides):
    raw = {
        "schema": "ffopt-constrained-refinement-v1",
        "parameter_names": ["epsilon", "sigma"],
        "structural_constraints": [
            {
                "name": "density",
                "column": "calc_density",
                "target": 10.0,
                "tolerance": 10.0,
                "mode": "relative_percent",
            }
        ],
        "mechanical_objectives": [
            {"name": "B", "column": "B", "target": 100.0},
            {"name": "Cprime", "column": "Cprime", "target": 100.0},
        ],
        "require_stability": False,
        "report_threshold_percent": 10.0,
        "maximum_rounds": 4,
        "patience": 1,
        "minimum_improvement_percent_points": 0.1,
        "mechanical_proposals_per_round": 2,
        "structural_proposals_per_round": 3,
    }
    raw.update(overrides)
    return RefinementSpec.from_mapping(raw)


def _keyed(frame: pd.DataFrame, spec: RefinementSpec) -> pd.DataFrame:
    from engine.constrained_refinement import _with_parameter_keys

    return _with_parameter_keys(frame, spec.parameter_names)


def test_exact_structure_gate_precedes_minimax_and_rmse_breaks_ties():
    spec = _spec()
    structural = _keyed(pd.DataFrame([
        {"epsilon": 1.0, "sigma": 1.0, "calc_density": 10.0},
        {"epsilon": 2.0, "sigma": 2.0, "calc_density": 13.0},
        {"epsilon": 3.0, "sigma": 3.0, "calc_density": 10.0},
    ]), spec)
    mechanical = _keyed(pd.DataFrame([
        # Same M-infinity as candidate 3, but a larger RMSE.
        {"epsilon": 1.0, "sigma": 1.0, "B": 120.0, "Cprime": 120.0},
        # Perfect mechanics cannot rescue a failed structural constraint.
        {"epsilon": 2.0, "sigma": 2.0, "B": 100.0, "Cprime": 100.0},
        {"epsilon": 3.0, "sigma": 3.0, "B": 120.0, "Cprime": 100.0},
    ]), spec)

    assessed, ranking = assess_and_rank_candidates(structural, mechanical, spec)

    assert list(ranking["epsilon"]) == [3.0, 1.0]
    assert list(ranking["mechanical_max_error_percent"]) == pytest.approx([20.0, 20.0])
    assert ranking.iloc[0]["mechanical_rmse_percent"] < ranking.iloc[1][
        "mechanical_rmse_percent"
    ]
    failed = assessed.loc[assessed["epsilon"] == 2.0].iloc[0]
    assert not bool(failed["structural_feasible"])
    assert not bool(failed["mechanical_eligible"])
    # The 10% threshold is reporting-only; both 20% best-effort rows survive.
    assert set(ranking["mechanical_quality_tier"]) == {"best_effort_mechanical"}


def _write_round_inputs(tmp_path: Path, *, all_structural_fail: bool = False):
    structural = tmp_path / "structural.csv"
    mechanical = tmp_path / "mechanical.csv"
    pd.DataFrame([
        {
            "epsilon": 1.0,
            "sigma": 1.0,
            "calc_density": 14.0 if all_structural_fail else 10.0,
            "success": True,
        },
        {
            "epsilon": 2.0,
            "sigma": 2.0,
            "calc_density": 10.0,
            "success": True,
        },
    ]).to_csv(structural, index=False)
    pd.DataFrame([
        {
            "epsilon": 1.0,
            "sigma": 1.0,
            "B": 150.0,
            "Cprime": 130.0,
            "success": True,
        }
    ]).to_csv(mechanical, index=False)
    return structural, mechanical


def test_one_round_is_manifest_resumable_and_preserves_best_effort(tmp_path: Path):
    structural, mechanical = _write_round_inputs(tmp_path)
    output = tmp_path / "round_01"
    first = run_refinement_round(
        spec=_spec(),
        structural_paths=[structural],
        mechanical_paths=[mechanical],
        output_dir=output,
    )
    second = run_refinement_round(
        spec=_spec(),
        structural_paths=[structural],
        mechanical_paths=[mechanical],
        output_dir=output,
    )

    assert not first.reused
    assert second.reused
    assert first.state["status"] == "active"
    assert first.state["best_effort_retained"]
    assert first.state["best_mechanical_max_error_percent"] == pytest.approx(50.0)
    assert first.state["within_mechanical_report_threshold"] == 0
    ranking = pd.read_csv(output / "exact_structural_mechanical_ranking.csv")
    assert len(ranking) == 1
    assert ranking.iloc[0]["mechanical_quality_tier"] == "best_effort_mechanical"
    assert (output / "artifact_manifest.json").is_file()


def test_manifest_refuses_changed_explicit_input(tmp_path: Path):
    structural, mechanical = _write_round_inputs(tmp_path)
    output = tmp_path / "round_01"
    run_refinement_round(
        spec=_spec(),
        structural_paths=[structural],
        mechanical_paths=[mechanical],
        output_dir=output,
    )
    frame = pd.read_csv(mechanical)
    frame.loc[0, "B"] = 99.0
    frame.to_csv(mechanical, index=False)

    with pytest.raises(RefinementError, match="cannot be reused safely"):
        run_refinement_round(
            spec=_spec(),
            structural_paths=[structural],
            mechanical_paths=[mechanical],
            output_dir=output,
        )


def test_zero_eligible_round_is_valid_and_budget_state_is_machine_readable(tmp_path: Path):
    structural, mechanical = _write_round_inputs(tmp_path, all_structural_fail=True)
    # The second structural row has no mechanical label, so there is still a
    # valid promotion proposal but no jointly assessed eligible candidate.
    output = tmp_path / "round_01"
    result = run_refinement_round(
        spec=_spec(maximum_rounds=1),
        structural_paths=[structural],
        mechanical_paths=[mechanical],
        output_dir=output,
    )

    assert result.state["eligible_mechanical_candidates"] == 0
    assert result.state["status"] == "budget_exhausted"
    assert result.state["convergence_status"] == "budget_exhausted"
    assert result.state["best_effort_retained"] is False
    assert pd.read_csv(output / "exact_structural_mechanical_ranking.csv").empty
    assert len(pd.read_csv(output / "mechanical_proposals.csv")) == 1


def test_default_acquisition_contract_selects_only_unobserved_explicit_pool(tmp_path: Path):
    structural, mechanical = _write_round_inputs(tmp_path)
    pool = tmp_path / "pool.csv"
    pd.DataFrame([
        # Already measured structurally: must not be emitted.
        {"epsilon": 1.0, "sigma": 1.0, "p_structural_feasible": 1.0},
        {"epsilon": 3.0, "sigma": 1.2, "p_structural_feasible": 0.9},
        {"epsilon": 4.0, "sigma": 1.3, "structural_entropy": 0.9},
        {"epsilon": 5.0, "sigma": 1.4, "global_distance": 0.8},
        {"epsilon": 6.0, "sigma": 1.5, "global_distance": 0.7},
    ]).to_csv(pool, index=False)
    output = tmp_path / "round_01"

    result = run_refinement_round(
        spec=_spec(),
        structural_paths=[structural],
        mechanical_paths=[mechanical],
        candidate_pool_path=pool,
        output_dir=output,
        acquisition=ExplicitPoolAcquisition(
            feasible_fraction=1 / 3, boundary_fraction=1 / 3
        ),
    )
    proposals = pd.read_csv(output / "structural_proposals.csv")

    assert len(proposals) == 3
    assert 1.0 not in set(proposals["epsilon"])
    assert set(proposals["selection_role"]) == {
        "predicted_feasible",
        "structural_boundary",
        "global_diversity",
    }
    assert proposals["parameter_key"].str.startswith("named:sha256:").all()
    assert result.state["acquisition_backend"] == "explicit_pool"
    assert result.state["acquisition_capability"] == "selection_only"
    assert not result.state["acquisition_scientific_convergence_capable"]


def test_selection_only_backend_cannot_claim_scientific_convergence(tmp_path: Path):
    structural, mechanical = _write_round_inputs(tmp_path)
    first_dir = tmp_path / "round_01"
    first = run_refinement_round(
        spec=_spec(),
        structural_paths=[structural],
        mechanical_paths=[mechanical],
        output_dir=first_dir,
    )
    assert first.state["no_improvement_rounds"] == 0
    second = run_refinement_round(
        spec=_spec(),
        structural_paths=[structural],
        mechanical_paths=[mechanical],
        previous_state_path=first_dir / "refinement_state.json",
        output_dir=tmp_path / "round_02",
    )

    assert second.state["no_improvement_rounds"] == 1
    assert second.state["status"] == "active"
    assert second.state["convergence_status"] == "incomplete_capability"


class _ConvergenceCapableBackend:
    name = "test_surrogate"
    capability = "surrogate_generation"
    scientific_convergence_capable = True

    def scientific_identity(self):
        return {"backend": self.name, "version": 1}

    def propose(self, context):
        columns = [
            "candidate_id",
            "selection_role",
            "parameter_key",
            *context.parameter_names,
        ]
        return AcquisitionResult(
            pd.DataFrame(columns=columns),
            self.name,
            self.capability,
            self.scientific_convergence_capable,
        )


def test_capable_backend_may_report_patience_convergence(tmp_path: Path):
    structural, mechanical = _write_round_inputs(tmp_path)
    backend = _ConvergenceCapableBackend()
    first_dir = tmp_path / "round_01"
    run_refinement_round(
        spec=_spec(),
        structural_paths=[structural],
        mechanical_paths=[mechanical],
        output_dir=first_dir,
        acquisition=backend,
    )
    second = run_refinement_round(
        spec=_spec(),
        structural_paths=[structural],
        mechanical_paths=[mechanical],
        previous_state_path=first_dir / "refinement_state.json",
        output_dir=tmp_path / "round_02",
        acquisition=backend,
    )

    assert second.state["status"] == "converged"
    assert second.state["convergence_status"] == "static_search_converged"
    assert second.state["requires_finalist_validation"] is True


def test_cubic_derivation_reuses_shared_elasticity_core():
    raw = _spec().scientific_identity()
    raw["mechanical_objectives"] = [
        {"name": "B", "column": "derived_B", "target": 100.0},
        {"name": "Cprime", "column": "derived_Cprime", "target": 50.0},
        {"name": "C44", "column": "derived_C44", "target": 80.0},
    ]
    raw["require_stability"] = True
    raw["stability_column"] = "derived_born_stable"
    raw["derivation"] = {"kind": "cubic"}
    spec = RefinementSpec.from_mapping(raw)
    structural = _keyed(pd.DataFrame([
        {"epsilon": 1.0, "sigma": 1.0, "calc_density": 10.0}
    ]), spec)
    mechanical = _keyed(pd.DataFrame([
        {"epsilon": 1.0, "sigma": 1.0, "C11": 166.6666666667, "C12": 66.6666666667, "C44": 80.0}
    ]), spec)
    from engine.constrained_refinement import _enrich_cubic_mechanics

    mechanical = _enrich_cubic_mechanics(mechanical, spec.derivation)
    _assessed, ranking = assess_and_rank_candidates(structural, mechanical, spec)

    assert len(ranking) == 1
    assert ranking.iloc[0]["mechanical_max_error_percent"] == pytest.approx(0.0, abs=1e-10)
    assert bool(ranking.iloc[0]["mechanical_stability_gate_pass"])


def test_state_is_plain_json_without_non_finite_values(tmp_path: Path):
    structural, mechanical = _write_round_inputs(tmp_path, all_structural_fail=True)
    output = tmp_path / "round_01"
    run_refinement_round(
        spec=_spec(),
        structural_paths=[structural],
        mechanical_paths=[mechanical],
        output_dir=output,
    )
    text = (output / "refinement_state.json").read_text(encoding="utf-8")
    state = json.loads(text)
    assert "NaN" not in text and "Infinity" not in text
    assert state["status"] in {"active", "converged", "budget_exhausted"}


def test_first_round_accepts_no_mechanical_evidence_yet(tmp_path: Path):
    structural, _mechanical = _write_round_inputs(tmp_path)
    output = tmp_path / "round_01"
    result = run_refinement_round(
        spec=_spec(),
        structural_paths=[structural],
        mechanical_paths=[],
        output_dir=output,
    )

    assert result.state["mechanical_observations"] == 0
    assert result.state["eligible_mechanical_candidates"] == 0
    assert len(pd.read_csv(output / "mechanical_proposals.csv")) == 2


def test_module_cli_checks_explicit_round_and_config_artifact(tmp_path: Path, capsys):
    structural, mechanical = _write_round_inputs(tmp_path)
    config = tmp_path / "refinement.json"
    config.write_text(
        json.dumps({"constrained_refinement": _spec().scientific_identity()}),
        encoding="utf-8",
    )
    output = tmp_path / "round_01"
    arguments = [
        "--config",
        str(config),
        "--structural-observations",
        str(structural),
        "--static-observations",
        str(mechanical),
        "--output-dir",
        str(output),
        "--round",
        "1",
    ]

    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["reused"] is False
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["reused"] is True
    with pytest.raises(RefinementError, match="previous state requires round 1"):
        main([*arguments[:-1], "2"])
