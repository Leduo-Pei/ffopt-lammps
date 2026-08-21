from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from engine.constrained_refinement import AcquisitionResult
from engine.gp_structural_acquisition import GaussianProcessStructuralAcquisition
from engine.lammps_interface import EvalResult
from engine.material_al_round import MaterialALRoundError, run_material_al_round
from workflow.artifact_manifest import canonical_parameter_key
from workflow.pipeline import PipelineRunner
from workflow.project import Project
from workflow.state import WorkflowState


def _config() -> dict:
    return {
        "material": {"kind": "elemental", "name": "X"},
        "crystal": {"family": "cubic"},
        "manifest": {"system_name": "x", "data_files": {"bulk": "unused.data"}},
        "lammps": {"bulk": {"npt_seed": 101}},
        "parallel": {"max_workers": 2, "cores_per_worker": 1},
        "charge": {
            "enabled": False,
            "neutrality_constraint": {"enabled": False, "derive_from_type": None},
        },
        "pair_params": {
            "mixing_rule": "default",
            "derived_params": [],
            "explicit_pairs": [],
        },
        "parameter_constraints": {"ties": [], "differences": []},
        "atom_types": [{
            "type": 1,
            "label": "X",
            "params": {
                "epsilon": {"min": 0.1, "max": 1.0, "init": 0.5},
                "sigma": {"min": 1.0, "max": 2.0, "init": 1.5},
            },
        }],
        "targets": {
            "a": {"value": 3.0},
            "b": {"value": 3.0},
            "c": {"value": 3.0},
            "alpha": {"value": 90.0},
            "beta": {"value": 90.0},
            "gamma_ang": {"value": 90.0},
            "density": {"value": 7.0},
            "surf_energy": {"value": 2.0},
        },
        "active_learning": {
            "n_candidate_pool": 8,
            "n_candidates_per_round": 2,
            "static_points": 2,
            "improvement_fraction": 0.5,
            "boundary_fraction": 0.3,
            "global_fraction": 0.2,
            "local_radii": [0.05],
            "minimum_structural_observations": 12,
            "minimum_mechanical_observations": 10,
        },
        "elasticity": {
            "enabled": True,
            "selection": {
                "structural_gates": {
                    "lattice": {"maximum_relative_error_percent": 5.0},
                    "angles": {"maximum_absolute_error_degree": 2.0},
                    "density": {"maximum_relative_error_percent": 5.0},
                    "surface": {"maximum_relative_error_percent": 10.0},
                },
                "fit_quality": {"minimum_r2": 0.9},
                "born_stability": {"required": True},
            },
            "reporting": {"mechanical_tier_percent": 20.0},
            "modules": {
                "static": {
                    "targets": {
                        "B": {"value": 100.0},
                        "Cprime": {"value": 50.0},
                        "C44": {"value": 75.0},
                    }
                }
            },
        },
    }


def _spec() -> dict:
    constraints = []
    for name, target, tolerance, mode in (
        ("a", 3.0, 5.0, "relative_percent"),
        ("b", 3.0, 5.0, "relative_percent"),
        ("c", 3.0, 5.0, "relative_percent"),
        ("alpha", 90.0, 2.0, "absolute"),
        ("beta", 90.0, 2.0, "absolute"),
        ("gamma_ang", 90.0, 2.0, "absolute"),
        ("density", 7.0, 5.0, "relative_percent"),
        ("surf_energy", 2.0, 10.0, "relative_percent"),
    ):
        constraints.append({
            "name": name,
            "column": f"calc_{name}",
            "target": target,
            "tolerance": tolerance,
            "mode": mode,
            "role": "constraint",
        })
    return {
        "schema": "ffopt-constrained-refinement-v1",
        "parameter_names": ["X_epsilon", "X_sigma"],
        "structural_constraints": constraints,
        "mechanical_objectives": [
            {"name": "B", "column": "B_gpa", "target": 100.0, "role": "objective"},
            {
                "name": "Cprime",
                "column": "Cprime_gpa",
                "target": 50.0,
                "role": "objective",
            },
            {"name": "C44", "column": "C44_gpa", "target": 75.0, "role": "objective"},
        ],
        "maximum_rounds": 3,
        "patience": 1,
        "minimum_improvement_percent_points": 0.1,
        "mechanical_proposals_per_round": 2,
        "structural_proposals_per_round": 2,
        "report_threshold_percent": 20.0,
        "require_stability": True,
        "stability_column": "born_stability_pass",
        "minimum_fit_quality": 0.9,
        "fit_quality_column": "minimum_fit_r2",
    }


def _structural_row(epsilon: float, sigma: float) -> dict:
    params = {"X_epsilon": epsilon, "X_sigma": sigma}
    return {
        **params,
        "parameter_key": canonical_parameter_key(params),
        "success": True,
        "objective": abs(epsilon - 0.5) + abs(sigma - 1.5),
        "calc_a": 3.0 + 0.01 * (epsilon - 0.5),
        "calc_b": 3.0,
        "calc_c": 3.0,
        "calc_alpha": 90.0,
        "calc_beta": 90.0,
        "calc_gamma_ang": 90.0,
        "calc_density": 7.0,
        "calc_surf_energy": 2.0,
    }


def _static_row(row: dict) -> dict:
    return {
        **row,
        "B_gpa": 110.0,
        "Cprime_gpa": 55.0,
        "C44_gpa": 82.5,
        "born_stability_pass": True,
        "minimum_fit_r2": 0.99,
    }


class _FakeStructuralRunner:
    calls = 0

    def feasibility_error(self, _params):
        return None

    def evaluate_replicates(self, params, _work_dir, seeds, **_kwargs):
        type(self).calls += 1
        properties = _structural_row(params["X_epsilon"], params["X_sigma"])
        return [
            EvalResult(
                params=params,
                properties={key.removeprefix("calc_"): value for key, value in properties.items() if key.startswith("calc_")},
                objective=0.01,
                success=True,
            )
            for _seed in seeds
        ]


def _fake_static_batch(**kwargs):
    proposals = pd.read_csv(
        kwargs["parameters_path"], float_precision="round_trip"
    )
    results = pd.DataFrame([_static_row(row) for row in proposals.to_dict("records")])
    output = Path(kwargs["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "static_results.csv", index=False)
    (output / "batch_summary.json").write_text(
        json.dumps({"failed_work_units": 0}), encoding="utf-8"
    )
    (output / "stage_manifest.json").write_text("{}\n", encoding="utf-8")
    return results


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps(_config()), encoding="utf-8")
    spec = tmp_path / "refinement.json"
    spec.write_text(json.dumps(_spec()), encoding="utf-8")
    rows = [
        _structural_row(0.2 + 0.1 * index, 1.1 + 0.12 * index)
        for index in range(6)
    ]
    structural = tmp_path / "structural.csv"
    pd.DataFrame(rows).to_csv(structural, index=False)
    static = tmp_path / "static.csv"
    pd.DataFrame([_static_row(rows[0]), _static_row(rows[1])]).to_csv(static, index=False)
    return config, spec, structural, static


def test_round_uses_gp_coverage_fallback_and_round2_consumes_round1_labels(tmp_path):
    config, spec, structural, static = _write_inputs(tmp_path)
    _FakeStructuralRunner.calls = 0
    first = run_material_al_round(
        runtime_config_path=config,
        refinement_config_path=spec,
        structural_paths=[structural],
        mechanical_paths=[static],
        candidate_pool_paths=[],
        output_dir=tmp_path / "round_1",
        round_number=1,
        maximum_rounds=3,
        structural_seeds=[101],
        available_cores=2,
        cores_per_state=1,
        structural_runner_factory=lambda _config: _FakeStructuralRunner(),
        elasticity_batch_runner=_fake_static_batch,
    )
    evidence_1 = json.loads(
        (first.output_dir / "round_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence_1["acquisition"]["backend"] == "gp_structural_constraints"
    assert evidence_1["acquisition"]["diagnostics"]["mode"] == "coverage_fallback"
    planned = pd.read_csv(first.output_dir / "planned_structural_proposals.csv")
    exact = pd.read_csv(first.output_dir / "new_structural_observations.csv")
    assert not planned["surrogate_is_exact_observation"].astype(bool).any()
    assert exact["is_exact_observation"].astype(bool).all()
    assert exact["surrogate_is_exact_observation"].astype(bool).all()
    assert evidence_1["training_counts"] == {
        "pre_round_structural": 6,
        "post_round_structural": 8,
        "pre_round_static": 2,
        "post_round_static": 4,
    }
    calls_after_first = _FakeStructuralRunner.calls
    reused = run_material_al_round(
        runtime_config_path=config,
        refinement_config_path=spec,
        structural_paths=[structural],
        mechanical_paths=[static],
        candidate_pool_paths=[],
        output_dir=tmp_path / "round_1",
        round_number=1,
        maximum_rounds=3,
        structural_seeds=[101],
        available_cores=2,
        cores_per_state=1,
        structural_runner_factory=lambda _config: _FakeStructuralRunner(),
        elasticity_batch_runner=_fake_static_batch,
    )
    assert reused.reused
    assert _FakeStructuralRunner.calls == calls_after_first

    second = run_material_al_round(
        runtime_config_path=config,
        refinement_config_path=spec,
        structural_paths=[],
        mechanical_paths=[],
        candidate_pool_paths=[],
        previous_state_path=first.output_dir / "refinement_state.json",
        previous_evidence_path=first.output_dir / "round_evidence.json",
        output_dir=tmp_path / "round_2",
        round_number=2,
        maximum_rounds=3,
        structural_seeds=[101],
        available_cores=2,
        cores_per_state=1,
        structural_runner_factory=lambda _config: _FakeStructuralRunner(),
        elasticity_batch_runner=_fake_static_batch,
    )
    evidence_2 = json.loads(
        (second.output_dir / "round_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence_2["training_counts"]["pre_round_structural"] == 8
    assert evidence_2["training_counts"]["post_round_structural"] == 10


class _ArtifactStructuralRunner(_FakeStructuralRunner):
    def __init__(self, input_path: Path):
        self.input_path = input_path

    def structural_input_artifacts(self):
        return {"bulk_data": self.input_path}


def test_round_reuse_fails_closed_when_structural_input_content_changes(tmp_path):
    config, spec, structural, static = _write_inputs(tmp_path)
    runner_input = tmp_path / "bulk.data"
    runner_input.write_text("version one\n", encoding="utf-8")
    kwargs = {
        "runtime_config_path": config,
        "refinement_config_path": spec,
        "structural_paths": [structural],
        "mechanical_paths": [static],
        "candidate_pool_paths": [],
        "output_dir": tmp_path / "round",
        "round_number": 1,
        "maximum_rounds": 3,
        "available_cores": 2,
        "cores_per_state": 1,
        "structural_runner_factory": lambda _cfg: _ArtifactStructuralRunner(
            runner_input
        ),
        "elasticity_batch_runner": _fake_static_batch,
    }
    run_material_al_round(**kwargs)
    runner_input.write_text("version two\n", encoding="utf-8")
    with pytest.raises(MaterialALRoundError, match="not reusable"):
        run_material_al_round(**kwargs)


class _ReplicateAuditRunner(_FakeStructuralRunner):
    def evaluate_replicates(self, params, _work_dir, seeds, **_kwargs):
        rows = []
        for index, _seed in enumerate(seeds):
            properties = _structural_row(params["X_epsilon"], params["X_sigma"])
            measured = {
                key.removeprefix("calc_"): value
                for key, value in properties.items()
                if key.startswith("calc_")
            }
            if index == len(seeds) - 1:
                measured["a"] = 4.0
            rows.append(EvalResult(
                params=params,
                properties=measured,
                objective=0.01,
                success=True,
            ))
        return rows


def test_multiseed_structural_label_requires_every_replicate_to_pass(tmp_path):
    config, spec, structural, static = _write_inputs(tmp_path)
    result = run_material_al_round(
        runtime_config_path=config,
        refinement_config_path=spec,
        structural_paths=[structural],
        mechanical_paths=[static],
        candidate_pool_paths=[],
        output_dir=tmp_path / "audit_round",
        round_number=1,
        maximum_rounds=3,
        structural_seeds=[101, 202],
        available_cores=2,
        cores_per_state=1,
        structural_runner_factory=lambda _cfg: _ReplicateAuditRunner(),
        elasticity_batch_runner=_fake_static_batch,
    )
    new = pd.read_csv(result.output_dir / "new_structural_observations.csv")
    assert not new["replicate_gate_all_pass"].astype(bool).any()
    assert not new["audit_certified"].astype(bool).any()
    assert not new["structural_gate_pass"].astype(bool).any()
    promoted = pd.read_csv(result.output_dir / "executed_static_proposals.csv")
    assert not set(new["parameter_key"]) & set(promoted["parameter_key"])


class _CapableDeterministicGP(GaussianProcessStructuralAcquisition):
    def propose(self, context):
        pool = context.candidate_pool.head(context.proposal_count).copy()
        pool.insert(0, "selection_role", "mock_gp_improvement")
        pool.insert(0, "candidate_id", range(1, len(pool) + 1))
        return AcquisitionResult(
            pool,
            self.name,
            self.capability,
            True,
            {
                "schema": "test-gp-v1",
                "mode": "constrained_gp",
                "scientific_convergence_capable_this_round": True,
            },
        )


def capable_gp() -> GaussianProcessStructuralAcquisition:
    return _CapableDeterministicGP(
        improvement_fraction=0.5,
        boundary_fraction=0.3,
        global_fraction=0.2,
    )


def test_public_pipeline_advances_two_exact_rounds_then_traceably_skips(
    tmp_path, monkeypatch
):
    config = _config()
    config.update({
        "machine": {"backend": "local"},
        "optimization": {"method": "turbo"},
        "checkpoint": {"enabled": True},
        "nn": {"enabled": True},
    })
    config["active_learning"]["enabled"] = True
    project_file = tmp_path / "ffopt.in"
    project_file.write_text("ffopt 1\nproject x\n", encoding="utf-8")
    project = Project(project_file, {
        "project": {"name": "x", "run_root": "runs/x"},
        "modules": {},
        "pipeline": {
            "stages": ["bo", "audit", "screen", "nn", "constrained_al"],
            "stage_repetitions": {"constrained_al": 3},
            "constrained_al": {
                "patience": 1,
                "minimum_improvement_percent_points": 0.1,
                "seed": 77,
            },
            "graph_mode": "linear",
        },
        "stages": {
            "screen": {"minimum": 2, "maximum": 2},
            "static": {"available_cores": 2, "cores_per_state": 1},
        },
    })
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(project=project, machine="local", run_id="two-rounds")
    rows = [
        _structural_row(0.2 + 0.1 * index, 1.1 + 0.12 * index)
        for index in range(6)
    ]
    initial_static = pd.DataFrame([_static_row(rows[0]), _static_row(rows[1])])
    executed: list[str] = []

    def write_upstream(spec):
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        for artifact in spec.artifacts:
            if artifact.name in {"all_results.csv", "stable_results.csv"}:
                pd.DataFrame(rows).to_csv(artifact, index=False)
            elif artifact.name in {"feasible_archive.csv", "coverage_anchors.csv"}:
                pd.DataFrame(rows[:3]).to_csv(artifact, index=False)
            elif artifact.name == "stability_replicates.csv":
                pd.DataFrame(rows[:2]).to_csv(artifact, index=False)
            elif artifact.name in {"candidates.csv", "static_screen_candidates.csv"}:
                pd.DataFrame(rows if artifact.name == "candidates.csv" else rows[:2]).to_csv(
                    artifact, index=False
                )
            elif artifact.name in {"candidate_pool.csv", "initial_proposals.csv"}:
                pd.DataFrame(columns=["parameter_key", "X_epsilon", "X_sigma"]).to_csv(
                    artifact, index=False
                )
            elif artifact.name in {"static_results.csv", "finalists_selected.csv"}:
                initial_static.to_csv(artifact, index=False)
            elif artifact.name == "best_candidate.json":
                artifact.write_text(
                    json.dumps({"raw_free_parameters": {
                        "X_epsilon": rows[0]["X_epsilon"],
                        "X_sigma": rows[0]["X_sigma"],
                    }}),
                    encoding="utf-8",
                )
            elif artifact.name == "best_parameters.txt":
                artifact.write_text(
                    "X_epsilon = 0.2\nX_sigma = 1.1\n", encoding="utf-8"
                )
            elif artifact.suffix == ".json":
                artifact.write_text("{}\n", encoding="utf-8")
            else:
                artifact.write_text("test\n", encoding="utf-8")

    def fake_run_local(self, state, spec):
        executed.append(spec.name)
        state.transition(spec.name, "running", increment_attempt=True)
        if spec.kind == "constrained_al":
            round_number = self._constrained_round_number(spec.name)
            kwargs = {
                "runtime_config_path": self.config_path,
                "refinement_config_path": self.refinement_spec_path,
                "candidate_pool_paths": [self.root / "nn" / "candidate_pool.csv"],
                "output_dir": spec.output_dir,
                "round_number": round_number,
                "maximum_rounds": self.refinement_maximum_rounds,
                "seed": 77,
                "structural_seeds": [101],
                "available_cores": 2,
                "cores_per_state": 1,
                "structural_runner_factory": lambda _cfg: _FakeStructuralRunner(),
                "elasticity_batch_runner": _fake_static_batch,
                "acquisition_backend": capable_gp(),
            }
            if round_number == 1:
                kwargs.update({
                    "structural_paths": [self.root / "candidates" / "candidates.csv"],
                    "mechanical_paths": [self.root / "static" / "static_results.csv"],
                })
            else:
                previous = self._previous_constrained_name(spec.name)
                kwargs.update({
                    "structural_paths": [],
                    "mechanical_paths": [],
                    "previous_state_path": self.root / previous / "refinement_state.json",
                    "previous_evidence_path": self.root / previous / "round_evidence.json",
                })
            run_material_al_round(**kwargs)
        else:
            write_upstream(spec)
        missing = [str(path) for path in spec.artifacts if not path.is_file()]
        assert not missing
        state.transition(spec.name, "completed")

    monkeypatch.setattr(PipelineRunner, "_run_local", fake_run_local)
    assert runner.run() == "completed"
    assert "constrained_al_01" in executed
    assert "constrained_al_02" in executed
    assert "constrained_al_03" not in executed
    evidence_1 = json.loads((
        runner.root / "constrained_al_01" / "round_evidence.json"
    ).read_text(encoding="utf-8"))
    evidence_2 = json.loads((
        runner.root / "constrained_al_02" / "round_evidence.json"
    ).read_text(encoding="utf-8"))
    assert evidence_2["training_counts"]["pre_round_structural"] == evidence_1[
        "training_counts"
    ]["post_round_structural"]
    assert evidence_2["training_counts"]["pre_round_static"] == evidence_1[
        "training_counts"
    ]["post_round_static"]
    assert json.loads((
        runner.root / "constrained_al_02" / "refinement_state.json"
    ).read_text(encoding="utf-8"))["status"] == "converged"
    with WorkflowState(runner.state_path) as state:
        assert state.get("constrained_al_03").status == "skipped"

    # A missing outer manifest forces the public runner back through the
    # stage command (which safely reconstructs it from candidate manifests).
    outer_manifest = runner.root / "constrained_al_01" / "material_al_manifest.json"
    outer_manifest.unlink()
    executed.clear()
    resumed = PipelineRunner(
        project=project, machine="local", run_id="two-rounds", resume=True
    )
    assert resumed.run() == "completed"
    assert executed == ["constrained_al_01"]

    # Corrupt provenance is never accepted or overwritten silently.
    outer_manifest.write_text("{}\n", encoding="utf-8")
    executed.clear()
    corrupt_resume = PipelineRunner(
        project=project, machine="local", run_id="two-rounds", resume=True
    )
    with pytest.raises(MaterialALRoundError, match="not reusable"):
        corrupt_resume.run()
    assert executed == ["constrained_al_01"]
