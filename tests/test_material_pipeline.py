from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from workflow.material_candidates import collect_material_candidates
from workflow.material_pipeline import build_refinement_spec
from workflow.pipeline import PipelineRunner
from workflow.project import Project
from workflow.state import WorkflowState


def _material_config() -> dict:
    return {
        "machine": {"backend": "local"},
        "manifest": {"system_name": "elemental_demo"},
        "lammps": {"executable": "lmp", "mpiexec": "mpiexec"},
        "parallel": {"max_workers": 8},
        "optimization": {"method": "turbo"},
        "checkpoint": {"enabled": True},
        "nn": {"enabled": True},
        "active_learning": {"enabled": True},
        "charge": {
            "enabled": False,
            "neutrality_constraint": {"enabled": False, "derive_from_type": None},
        },
        "pair_params": {
            "mixing_rule": "default",
            "derived_params": [],
            "explicit_pairs": [],
        },
        "atom_types": [
            {
                "type": 1,
                "label": "FeA",
                "params": {
                    "epsilon": {"min": 0.1, "max": 10.0},
                    "sigma": {"min": 0.1, "max": 5.0},
                },
            },
            {
                "type": 2,
                "label": "FeB",
                "params": {
                    "epsilon": 1.0,
                    "sigma": {"min": 0.1, "max": 5.0},
                },
            },
        ],
        "targets": {
            "a": {"value": 2.86},
            "b": {"value": 2.86},
            "c": {"value": 2.86},
            "alpha": {"value": 90.0},
            "beta": {"value": 90.0},
            "gamma_ang": {"value": 90.0},
            "density": {"value": 7.87},
            "surf_energy": {"value": 2.34},
        },
        "elasticity": {
            "enabled": True,
            "selection": {
                "structural_gates": {
                    "lattice": {"maximum_relative_error_percent": 1.0},
                    "angles": {"maximum_absolute_error_degree": 1.0},
                    "density": {"maximum_relative_error_percent": 1.0},
                    "surface": {"maximum_relative_error_percent": 5.0},
                },
                "fit_quality": {"minimum_r2": 0.98},
                "born_stability": {"required": True},
            },
            "reporting": {"mechanical_tier_percent": 20.0},
            "modules": {
                "static": {
                    "targets": {
                        "B": {"value": 173.0},
                        "Cprime": {"value": 52.5},
                        "C44": {"value": 121.9},
                    }
                },
                "dynamic": {
                    "targets": {
                        "B": {"value": 166.2},
                        "Cprime": {"value": 48.15},
                        "C44": {"value": 115.87},
                    }
                },
            },
        },
    }


def _project(tmp_path: Path) -> Project:
    path = tmp_path / "ffopt.in"
    path.write_text("ffopt 1\nproject elemental_demo\n", encoding="utf-8")
    return Project(path, {
        "project": {"name": "elemental_demo", "run_root": "runs/elemental_demo"},
        "modules": {},
        "pipeline": {
            "stages": [
                "bo",
                "sample",
                "audit",
                "screen",
                "nn",
                "constrained_al",
                "finalists",
                "validate",
            ],
            "stage_repetitions": {"constrained_al": 3},
            "constrained_al": {"patience": 1, "seed": 77},
            "graph_mode": "linear",
        },
        "stages": {
            "sample": {
                "n_points": 12,
                "elite_centers": 2,
                "radii": [0.01],
                "global_fraction": 0.0,
                "seeds": [101],
            },
            "audit": {"top_k": 4, "seeds": [101]},
            "static": {"available_cores": 8, "cores_per_state": 2},
            "finalists": {
                "available_cores": 8,
                "cores_per_state": 2,
                "top_n": 5,
            },
        },
    })


def test_compiled_elasticity_maps_to_generic_refinement_contract():
    spec = build_refinement_spec(
        _material_config(),
        maximum_rounds=6,
        settings={"patience": 2},
    )

    assert spec["parameter_names"] == ["FeA_epsilon", "FeA_sigma", "FeB_sigma"]
    assert spec["maximum_rounds"] == 6
    assert spec["patience"] == 2
    assert [item["column"] for item in spec["mechanical_objectives"]] == [
        "B_gpa",
        "Cprime_gpa",
        "C44_gpa",
    ]
    assert spec["stability_column"] == "born_stability_pass"
    assert spec["fit_quality_column"] == "minimum_fit_r2"
    assert next(
        item for item in spec["structural_constraints"] if item["name"] == "alpha"
    )["mode"] == "absolute"


def test_material_pipeline_installs_commands_and_explicit_coverage_sources(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
    config = _material_config()
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="local", run_id="material", dry_run=True
    )
    specs = {spec.name: spec for spec in runner.build_specs()}

    assert runner.material_workflow
    assert list(specs) == [
        "bo",
        "sample",
        "audit",
        "candidates",
        "static",
        "nn",
        "constrained_al_01",
        "constrained_al_02",
        "constrained_al_03",
        "finalists",
        "validate",
    ]
    sample = specs["sample"].command
    assert Path(sample[sample.index("--source") + 1]) == (
        runner.root / "bo" / "coverage_anchors.csv"
    )
    candidates = specs["candidates"].command
    sources = [
        candidates[index + 1]
        for index, token in enumerate(candidates)
        if token == "--source"
    ]
    assert sources == [
        str(runner.root / "bo" / "all_results.csv"),
        str(runner.root / "bo" / "feasible_archive.csv"),
        str(runner.root / "bo" / "coverage_anchors.csv"),
        str(runner.root / "sample" / "local_results.csv"),
        str(runner.root / "audit" / "stable_results.csv"),
    ]
    assert specs["static"].command[2] == "engine.cubic_elastic_batch"
    assert specs["static"].command[
        specs["static"].command.index("--parameters") + 1
    ] == str(runner.root / "candidates" / "static_screen_candidates.csv")
    for option, expected in (
        ("--available-cores", "8"),
        ("--cores-per-state", "1"),
        ("--omp-threads-per-state", "1"),
        ("--candidate-workers", "8"),
    ):
        assert specs["static"].command[
            specs["static"].command.index(option) + 1
        ] == expected
    assert specs["nn"].command[2] == "engine.material_surrogate_preflight"
    second = specs["constrained_al_02"].command
    assert second[2] == "engine.material_al_round"
    assert second[second.index("--previous-state") + 1] == str(
        runner.root / "constrained_al_01" / "refinement_state.json"
    )
    assert second[second.index("--previous-evidence") + 1] == str(
        runner.root / "constrained_al_01" / "round_evidence.json"
    )
    assert second[second.index("--round") + 1] == "2"
    finalists = specs["finalists"].command
    assert finalists[finalists.index("--parameters") + 1] == str(
        runner.root
        / "constrained_al_03"
        / "exact_structural_mechanical_ranking.csv"
    )
    validate = specs["validate"].command
    assert validate[2] == "engine.material_validation"
    assert validate[validate.index("--parameters") + 1] == str(
        runner.root / "finalists" / "best_candidate.json"
    )


def test_material_elastic_resources_come_from_full_machine_allocation(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
    config = _material_config()
    config.update({
        "machine": {"backend": "slurm"},
        "parallel": {
            "max_workers": 38,
            "cores_per_worker": 2,
            "omp_threads_per_worker": 1,
            "use_mpi": True,
            "scheduler_launcher": "srun",
        },
        "cluster": {
            "al": {
                "nodes": 1,
                "cores": 76,
                "tasks": 38,
                "cpus_per_task": 2,
                "distributed_steps": True,
                "time": "24:00:00",
            },
            # This legacy profile must not throttle material validation.
            "validate": {
                "nodes": 1,
                "cores": 2,
                "tasks": 1,
                "cpus_per_task": 2,
                "distributed_steps": True,
                "time": "24:00:00",
            },
        },
    })
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="cluster", run_id="resources", dry_run=True
    )
    specs = {spec.name: spec for spec in runner.build_specs()}

    for stage in ("static", "constrained_al_01", "finalists", "validate"):
        command = specs[stage].command
        expected = {
            "--available-cores": "76",
            "--cores-per-state": "2",
            "--omp-threads-per-state": "1",
            "--candidate-workers": "38",
        }
        for option, value in expected.items():
            assert command[command.index(option) + 1] == value
        assert "--state-workers" not in command

    validate_profile = runner._slurm_profile("validate")
    assert validate_profile["cores"] == 76
    assert validate_profile["tasks"] == 76
    assert validate_profile["cpus_per_task"] == 1


def test_material_resource_command_accounts_for_omp_without_oversubscription(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
    config = _material_config()
    config.update({
        "machine": {"backend": "slurm"},
        "parallel": {
            "max_workers": 4,
            "cores_per_worker": 2,
            "omp_threads_per_worker": 2,
            "use_mpi": True,
            "scheduler_launcher": "srun",
        },
        "cluster": {
            "al": {
                "nodes": 1,
                "cores": 32,
                "tasks": 4,
                "cpus_per_task": 4,
                "distributed_steps": True,
                "time": "24:00:00",
            },
        },
    })
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="cluster", run_id="omp", dry_run=True
    )
    command = {spec.name: spec for spec in runner.build_specs()}["static"].command
    assert command[command.index("--available-cores") + 1] == "32"
    assert command[command.index("--cores-per-state") + 1] == "2"
    assert command[command.index("--omp-threads-per-state") + 1] == "2"
    assert command[command.index("--candidate-workers") + 1] == "4"
    profile = runner._slurm_profile("static")
    assert profile["tasks"] == 16
    assert profile["cpus_per_task"] == 2


def test_material_validate_only_uses_explicit_initial_parameters(tmp_path, monkeypatch):
    project = _project(tmp_path)
    project.data["pipeline"] = {"stages": ["validate"], "graph_mode": "linear"}
    config = _material_config()
    config["material"] = {"kind": "elemental", "name": "Fe"}
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)

    runner = PipelineRunner(
        project=project, machine="local", run_id="validate-only", dry_run=True
    )
    spec = runner.build_specs()[0]

    assert runner.material_workflow
    assert spec.command[2] == "engine.material_validation"
    assert "--parameters" not in spec.command
    supplied = [
        spec.command[index + 1]
        for index, token in enumerate(spec.command)
        if token == "--parameter"
    ]
    assert {item.split("=", 1)[0] for item in supplied} == {
        "FeA_epsilon",
        "FeA_sigma",
        "FeB_sigma",
    }
    assert "model_adequacy.json" in {path.name for path in spec.artifacts}


def test_structural_only_elemental_bo_validate_uses_material_registry(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
    project.data["pipeline"] = {
        "stages": ["bo", "validate"],
        "graph_mode": "linear",
    }
    config = _material_config()
    config["material"] = {"kind": "elemental", "name": "Fe"}
    config.pop("elasticity")
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)

    runner = PipelineRunner(
        project=project, machine="local", run_id="structure-only", dry_run=True
    )
    specs = {spec.name: spec for spec in runner.build_specs()}

    assert runner.material_workflow
    assert "coverage_anchors.csv" in {path.name for path in specs["bo"].artifacts}
    validate = specs["validate"].command
    assert validate[2] == "engine.material_validation"
    assert validate[validate.index("--parameters") + 1] == str(
        runner.root / "bo" / "best_parameters.txt"
    )


@pytest.mark.parametrize("terminal_status", ["converged", "budget_exhausted"])
def test_terminal_al_rounds_are_traceably_skipped_and_reused(
    tmp_path, monkeypatch, terminal_status
):
    project = _project(tmp_path)
    if terminal_status == "budget_exhausted":
        project.data["pipeline"]["constrained_al"]["maximum_rounds"] = 1
    config = _material_config()
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    calls: list[str] = []

    def fake_run_local(self, state, spec):
        calls.append(spec.name)
        state.transition(spec.name, "running", increment_attempt=True)
        for artifact in spec.artifacts:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            if artifact.name == "refinement_state.json":
                artifact.write_text(
                    json.dumps({
                        "schema": "ffopt-constrained-refinement-state-v1",
                        "status": terminal_status,
                        "convergence_status": (
                            "scientifically_converged"
                            if terminal_status == "converged"
                            else "budget_exhausted"
                        ),
                        "completed_round": 1,
                    }),
                    encoding="utf-8",
                )
            elif artifact.name == "exact_structural_mechanical_ranking.csv":
                artifact.write_text(
                    "candidate_rank,parameter_key,FeA_epsilon,FeA_sigma,FeB_sigma\n"
                    "1,named:sha256:" + "0" * 64 + ",1,2,3\n",
                    encoding="utf-8",
                )
            elif artifact.name in {
                "structural_proposals.csv",
                "mechanical_proposals.csv",
            }:
                artifact.write_text(
                    "candidate_id,selection_role,parameter_key,FeA_epsilon,FeA_sigma,FeB_sigma\n",
                    encoding="utf-8",
                )
            elif artifact.name == "stage_disposition.json":
                artifact.write_text(
                    json.dumps({"disposition": "executed"}), encoding="utf-8"
                )
            elif artifact.name == "round_evidence.json":
                artifact.write_text(
                    json.dumps({
                        "schema": "ffopt-material-al-round-evidence-v1",
                        "disposition": "executed",
                        "round": 1,
                        "candidate_manifests": [],
                    }),
                    encoding="utf-8",
                )
            elif artifact.suffix == ".json":
                artifact.write_text("{}\n", encoding="utf-8")
            else:
                artifact.write_text("test\n", encoding="utf-8")
        state.transition(spec.name, "completed")

    monkeypatch.setattr(PipelineRunner, "_run_local", fake_run_local)
    # This test isolates terminal-round propagation; manifest corruption and
    # forced rerun behaviour are covered by the material AL end-to-end test.
    monkeypatch.setattr(
        "workflow.pipeline.validate_material_al_stage_outputs",
        lambda _path: (True, "fixture manifest accepted"),
    )
    runner = PipelineRunner(project=project, machine="local", run_id="skip")
    assert runner.run() == "completed"

    assert "constrained_al_01" in calls
    assert "constrained_al_02" not in calls
    assert "constrained_al_03" not in calls
    with WorkflowState(runner.state_path) as state:
        assert state.get("constrained_al_02").status == "skipped"
        assert state.get("constrained_al_03").status == "skipped"
    disposition = json.loads((
        runner.root / "constrained_al_02" / "stage_disposition.json"
    ).read_text(encoding="utf-8"))
    assert disposition == {
        "schema": "ffopt-stage-disposition-v1",
        "stage": "constrained_al_02",
        "disposition": "skipped",
        "round": 2,
        "previous_stage": "constrained_al_01",
        "reason": terminal_status,
    }
    copied = pd.read_csv(
        runner.root
        / "constrained_al_03"
        / "exact_structural_mechanical_ranking.csv"
    )
    assert len(copied) == 1

    calls.clear()
    resumed = PipelineRunner(
        project=project, machine="local", run_id="skip", resume=True
    )
    assert resumed.run() == "completed"
    assert calls == []


def test_candidate_collector_uses_only_explicit_sources_and_manifest(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_material_config()), encoding="utf-8")
    source = tmp_path / "source.csv"
    pd.DataFrame([
        {
            "FeA_epsilon": 1.0,
            "FeA_sigma": 2.0,
            "FeB_sigma": 3.0,
            "calc_a": 2.86,
            "success": True,
        }
    ]).to_csv(source, index=False)
    nn = tmp_path / "nn.json"
    nn.write_text(json.dumps({
        "all_candidates": [
            {
                "params": {
                    "FeA_epsilon": 1.5,
                    "FeA_sigma": 2.5,
                    "FeB_sigma": 3.5,
                },
                "predicted_obj": 0.02,
            }
        ],
        "all_lammps": [],
    }), encoding="utf-8")
    output = tmp_path / "candidates"

    first = collect_material_candidates(
        config_path=config,
        source_paths=[source],
        nn_result_path=nn,
        output_dir=output,
    )
    second = collect_material_candidates(
        config_path=config,
        source_paths=[source],
        nn_result_path=nn,
        output_dir=output,
    )

    assert first == second
    assert first["measured_candidates"] == 1
    assert first["unevaluated_candidate_pool"] == 1
    assert (output / "stage_manifest.json").is_file()
    assert pd.read_csv(output / "candidates.csv")["parameter_key"].str.startswith(
        "named:sha256:"
    ).all()


def test_candidate_collector_does_not_replace_audited_evidence_with_nn_row(
    tmp_path: Path,
):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_material_config()), encoding="utf-8")
    source = tmp_path / "audited.csv"
    params = {"FeA_epsilon": 1.0, "FeA_sigma": 2.0, "FeB_sigma": 3.0}
    pd.DataFrame([{
        **params,
        "calc_a": 2.86,
        "audit_certified": True,
        "n_seeds": 3,
    }]).to_csv(source, index=False)
    nn = tmp_path / "nn.json"
    nn.write_text(json.dumps({
        "all_candidates": [],
        "all_lammps": [{
            "params": params,
            "lammps_success": True,
            "lammps_obj": 0.1,
            "lammps_properties": {"a": 9.99},
        }],
    }), encoding="utf-8")

    collect_material_candidates(
        config_path=config,
        source_paths=[source],
        nn_result_path=nn,
        output_dir=tmp_path / "collected",
    )

    observed = pd.read_csv(tmp_path / "collected" / "candidates.csv")
    assert len(observed) == 1
    assert observed.loc[0, "calc_a"] == pytest.approx(2.86)
    assert bool(observed.loc[0, "audit_certified"])
