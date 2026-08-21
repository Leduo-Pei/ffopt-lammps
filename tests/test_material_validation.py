from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pandas as pd

from engine.cubic_elastic_batch import plan_nested_resources
from engine.lammps_interface import EvalResult
from engine.material_results_report import write_top_parameters_report
from engine.material_validation import run_material_validation
from engine.parameter_space import build_parameter_space
from workflow.artifact_manifest import canonical_parameter_key


def _config(tmp_path: Path) -> tuple[Path, dict]:
    config = {
        "project": {"name": "fe_validation"},
        "material": {"kind": "elemental", "name": "Fe", "force_field": "lj/cut"},
        "crystal": {"system": "cubic", "prototype": "bcc"},
        "atom_types": [
            {
                "type": 1,
                "label": "Fe_corner",
                "mass": 55.845,
                "params": {
                    "epsilon": {"min": 0.001, "max": 10.0, "init": 5.0},
                    "sigma": {"min": 0.001, "max": 5.0, "init": 2.3},
                    "charge": 0.0,
                },
            },
            {
                "type": 2,
                "label": "Fe_body",
                "mass": 55.845,
                "params": {
                    "epsilon": 5.0,
                    "sigma": {"min": 0.001, "max": 5.0, "init": 2.35},
                    "charge": 0.0,
                },
            },
        ],
        "charge": {
            "enabled": False,
            "neutrality_constraint": {"enabled": False, "derive_from_type": None},
        },
        "pair_params": {
            "pair_style": "lj/cut",
            "mixing_rule": "default",
            "explicit_pairs": [],
            "derived_params": [
                {
                    "target": "Fe_body_epsilon",
                    "expression": "Fe_corner_epsilon",
                }
            ],
        },
        "parameter_constraints": {
            "ties": [
                {
                    "parameter": "epsilon",
                    "types": ["Fe_corner", "Fe_body"],
                    "source": "Fe_corner",
                }
            ],
            "differences": [
                {
                    "parameter": "sigma",
                    "types": ["Fe_corner", "Fe_body"],
                    "max_abs": 0.75,
                    "unit": "A",
                }
            ],
        },
        "lammps": {"pair_style": "lj/cut"},
        "parallel": {
            "max_workers": 1,
            "cores_per_worker": 2,
            "omp_threads_per_worker": 1,
            "use_mpi": False,
        },
        "targets": {
            "a": {"value": 2.8665, "tolerance": 0.03, "unit": "A"},
            "b": {"value": 2.8665, "tolerance": 0.03, "unit": "A"},
            "c": {"value": 2.8665, "tolerance": 0.03, "unit": "A"},
            "alpha": {"value": 90.0, "tolerance": 1.0, "unit": "degree"},
            "beta": {"value": 90.0, "tolerance": 1.0, "unit": "degree"},
            "gamma_ang": {"value": 90.0, "tolerance": 1.0, "unit": "degree"},
            "density": {"value": 7.874, "tolerance": 0.08, "unit": "g/cm3"},
            "surf_energy": {"value": 2.34, "tolerance": 0.117, "unit": "J/m2"},
        },
        "validation": {"property_evaluators": {}, "property_settings": {}},
        "elasticity": {
            "enabled": True,
            "selection": {
                "structural_gates": {
                    "lattice": {"maximum_relative_error_percent": 1.0},
                    "angles": {"maximum_absolute_error_degree": 1.0},
                    "density": {"maximum_relative_error_percent": 1.0},
                    "surface": {"maximum_relative_error_percent": 5.0},
                },
                "born_stability": {"required": True},
                "fit_quality": {"minimum_r2": 0.98},
            },
            "reporting": {"mechanical_tier_percent": 20.0},
            "modules": {
                "static": {
                    "targets": {
                        "B": {"value": 170.0},
                        "Cprime": {"value": 50.0},
                        "C44": {"value": 120.0},
                    },
                    "protocol": {"strain_magnitudes": [0.01, 0.02]},
                },
                "dynamic": {
                    "targets": {
                        "B": {"value": 170.0},
                        "Cprime": {"value": 50.0},
                        "C44": {"value": 120.0},
                    },
                    "protocol": {
                        "strain_magnitudes": [0.01, 0.02],
                        "temperature_k": 300.0,
                        "seeds": [101, 202],
                    },
                },
            },
        },
    }
    path = tmp_path / "compiled.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path, config


def _parameters(config: dict) -> dict[str, float]:
    names = [name for name, _lower, _upper in build_parameter_space(config)]
    values = [5.0, 2.30, 2.35]
    return dict(zip(names, values))


class FakeValidationRunner:
    def __init__(self, config, *, calls, fail_structure=False, input_file):
        self.config = config
        self.calls = calls
        self.fail_structure = fail_structure
        self.input_file = input_file

    def validation_input_artifacts(self):
        return {"fake_structure_input": self.input_file}

    def _resolve_params(self, raw):
        return {
            **raw,
            "Fe_body_epsilon": raw["Fe_corner_epsilon"],
            "Fe_corner_charge": 0.0,
            "Fe_body_charge": 0.0,
        }

    def _resolved_parameter_feasibility_error(self, _resolved):
        return None

    def _build_pair_coeffs(self, resolved, directory):
        path = Path(directory) / "pair_coeffs.lmp"
        path.write_text(
            "# pair_coeffs.lmp -- generated by FFOpt-LAMMPS\n"
            "# DO NOT EDIT: overwritten before each LAMMPS call\n"
            "pair_modify mix geometric\n"
            f"pair_coeff 1 1 {resolved['Fe_corner_epsilon']} "
            f"{resolved['Fe_corner_sigma']}\n"
            f"pair_coeff 2 2 {resolved['Fe_body_epsilon']} "
            f"{resolved['Fe_body_sigma']}\n",
            encoding="ascii",
        )
        return str(path)

    def evaluate_batch(self, _parameters, _work_dir, save_traj):
        assert save_traj is True
        self.calls["structure"] += 1
        properties = {
            "a": 3.20 if self.fail_structure else 2.8665,
            "b": 2.8665,
            "c": 2.8665,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma_ang": 90.0,
            "density": 7.874,
            "surf_energy": 2.34,
        }
        errors = {
            name: 100.0 * abs(value - self.config["targets"][name]["value"])
            / abs(self.config["targets"][name]["value"])
            for name, value in properties.items()
        }
        return [EvalResult(
            params={},
            properties=properties,
            objective=0.01,
            success=True,
            per_property_error=errors,
        )]


class FakeSchedulerPool:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeElasticBatch:
    def __init__(self, *, calls, mechanical_error, structural_pass):
        self.calls = calls
        self.mechanical_error = mechanical_error
        self.structural_pass = structural_pass
        self.dynamic_seeds = None

    def __call__(self, **kwargs):
        protocol = kwargs["protocol"]
        output = Path(kwargs["output_dir"])
        if protocol == "dynamic":
            runtime_config = json.loads(
                Path(kwargs["config_path"]).read_text(encoding="utf-8")
            )
            self.dynamic_seeds = runtime_config["elasticity"]["modules"][
                "dynamic"
            ]["protocol"]["seeds"]
        output.mkdir(parents=True, exist_ok=True)
        result_name = "static_results.csv" if protocol == "static" else "dynamic_results.csv"
        result_path = output / result_name
        manifest = output / "stage_manifest.json"
        if result_path.exists() and manifest.exists():
            return pd.read_csv(result_path)
        self.calls[protocol] += 1
        source = pd.read_csv(kwargs["parameters_path"])
        if protocol == "dynamic" and not bool(source.iloc[0]["finalist_eligible"]):
            frame = pd.DataFrame(columns=source.columns)
            frame.to_csv(result_path, index=False)
            pd.DataFrame(columns=["parameter_key", "trajectory_seed"]).to_csv(
                output / "dynamic_seed_results.csv", index=False
            )
            manifest.write_text("fake complete zero-eligible\n", encoding="ascii")
            return frame
        row = source.iloc[0].to_dict()
        row.update({
            "calculation_status": "completed",
            "C11_gpa": 250.0,
            "C12_gpa": 130.0,
            "C44_gpa": 100.0,
            "B_gpa": 170.0,
            "Cprime_gpa": 60.0,
            "G_hill_gpa": 90.0,
            "E_hill_gpa": 230.0,
            "nu_hill": 0.28,
            "minimum_fit_r2": 0.995,
            "born_stability_pass": True,
            "fit_quality_pass": True,
            "finite_mechanical_score": True,
            "structural_gate_pass": self.structural_pass,
            "structural_margin": 0.5 if self.structural_pass else -1.0,
            "finalist_eligible": self.structural_pass,
            "mechanical_max_error_percent": self.mechanical_error,
            "mechanical_rmse_percent": self.mechanical_error - 2.0,
            "error_B_percent": self.mechanical_error,
            "error_Cprime_percent": self.mechanical_error - 1.0,
            "error_C44_percent": self.mechanical_error - 2.0,
            "within_quality_tier": self.mechanical_error <= 20.0,
            "quality_status": (
                "within_tier" if self.mechanical_error <= 20.0 else "best_effort"
            ),
        })
        if protocol == "dynamic":
            row.update({
                "all_seeds_complete": True,
                "n_seeds_expected": 2,
                "n_seeds_success": 2,
            })
        frame = pd.DataFrame([row])
        frame.to_csv(result_path, index=False)
        if protocol == "dynamic":
            pd.DataFrame([
                {**row, "trajectory_seed": 101},
                {**row, "trajectory_seed": 202},
            ]).to_csv(output / "dynamic_seed_results.csv", index=False)
        manifest.write_text("fake complete\n", encoding="ascii")
        return frame


def _run_validation(
    tmp_path,
    *,
    fail_structure=False,
    mechanical_error=25.0,
    include_top=False,
    validation_seeds=None,
):
    config_path, config = _config(tmp_path)
    if validation_seeds is not None:
        config["elasticity"]["modules"]["dynamic"]["validation_protocol"] = {
            "seeds": list(validation_seeds)
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    parameter_path = tmp_path / "best_candidate.json"
    parameter_path.write_text(
        json.dumps({"raw_free_parameters": _parameters(config)}), encoding="utf-8"
    )
    fake_input = tmp_path / "fake_structure.in"
    fake_input.write_text("fake structure input\n", encoding="ascii")
    calls = Counter()

    def runner_factory(runtime_config):
        return FakeValidationRunner(
            runtime_config,
            calls=calls,
            fail_structure=fail_structure,
            input_file=fake_input,
        )

    batch = FakeElasticBatch(
        calls=calls,
        mechanical_error=mechanical_error,
        structural_pass=not fail_structure,
    )
    output = tmp_path / "validation"
    static_ranking = None
    dynamic_ranking = None
    if include_top:
        static_ranking = tmp_path / "source_static_ranking.csv"
        dynamic_ranking = tmp_path / "source_dynamic_ranking.csv"
        ranking = _ranking_row(
            _parameters(config),
            maximum=mechanical_error,
            rmse=mechanical_error - 2.0,
        )
        pd.DataFrame([ranking]).to_csv(static_ranking, index=False)
        pd.DataFrame([ranking]).to_csv(dynamic_ranking, index=False)
    summary = run_material_validation(
        config_path=config_path,
        parameters_path=parameter_path,
        output_dir=output,
        resources=plan_nested_resources(
            available_cores=4,
            cores_per_state=2,
            candidate_workers=1,
            state_workers=2,
        ),
        runner_factory=runner_factory,
        elastic_batch_runner=batch,
        elastic_backend_factory=lambda **_kwargs: None,
        static_ranking_path=static_ranking,
        dynamic_ranking_path=dynamic_ranking,
        top_n=1,
    )
    return (
        summary,
        output,
        calls,
        config_path,
        parameter_path,
        runner_factory,
        batch,
        static_ranking,
        dynamic_ranking,
    )


def test_validation_releases_structural_pool_before_elastic_batches(tmp_path):
    config_path, config = _config(tmp_path)
    parameter_path = tmp_path / "best_candidate.json"
    parameter_path.write_text(
        json.dumps({"raw_free_parameters": _parameters(config)}), encoding="utf-8"
    )
    fake_input = tmp_path / "fake_structure.in"
    fake_input.write_text("fake structure input\n", encoding="ascii")
    calls = Counter()
    runner = FakeValidationRunner(
        config,
        calls=calls,
        input_file=fake_input,
    )
    runner.scheduler_pool = FakeSchedulerPool()
    scheduler_pool = runner.scheduler_pool
    batch = FakeElasticBatch(
        calls=calls,
        mechanical_error=10.0,
        structural_pass=True,
    )

    def assert_released_then_run_elastic(**kwargs):
        assert scheduler_pool.closed
        assert runner.scheduler_pool is None
        return batch(**kwargs)

    run_material_validation(
        config_path=config_path,
        parameters_path=parameter_path,
        output_dir=tmp_path / "pooled_validation",
        resources=plan_nested_resources(
            available_cores=4,
            cores_per_state=2,
            candidate_workers=1,
            state_workers=2,
        ),
        runner_factory=lambda _config: runner,
        elastic_batch_runner=assert_released_then_run_elastic,
        elastic_backend_factory=lambda **_kwargs: None,
    )

    assert scheduler_pool.closed
    assert runner.scheduler_pool is None

    reuse_runner = FakeValidationRunner(
        config,
        calls=calls,
        input_file=fake_input,
    )
    reuse_runner.scheduler_pool = FakeSchedulerPool()
    reuse_pool = reuse_runner.scheduler_pool

    def assert_reuse_released(**kwargs):
        assert reuse_pool.closed
        assert reuse_runner.scheduler_pool is None
        return batch(**kwargs)

    run_material_validation(
        config_path=config_path,
        parameters_path=parameter_path,
        output_dir=tmp_path / "pooled_validation",
        resources=plan_nested_resources(
            available_cores=4,
            cores_per_state=2,
            candidate_workers=1,
            state_workers=2,
        ),
        runner_factory=lambda _config: reuse_runner,
        elastic_batch_runner=assert_reuse_released,
        elastic_backend_factory=lambda **_kwargs: None,
    )
    assert reuse_pool.closed
    assert reuse_runner.scheduler_pool is None


def test_validation_releases_structural_pool_when_structure_fails(tmp_path):
    config_path, config = _config(tmp_path)
    parameter_path = tmp_path / "best_candidate.json"
    parameter_path.write_text(
        json.dumps({"raw_free_parameters": _parameters(config)}), encoding="utf-8"
    )
    fake_input = tmp_path / "fake_structure.in"
    fake_input.write_text("fake structure input\n", encoding="ascii")
    runner = FakeValidationRunner(
        config,
        calls=Counter(),
        input_file=fake_input,
    )
    runner.scheduler_pool = FakeSchedulerPool()
    scheduler_pool = runner.scheduler_pool

    def fail_structure(*_args, **_kwargs):
        raise RuntimeError("structural failure")

    runner.evaluate_batch = fail_structure
    batch = FakeElasticBatch(
        calls=runner.calls,
        mechanical_error=10.0,
        structural_pass=False,
    )

    def assert_released_then_run_elastic(**kwargs):
        assert scheduler_pool.closed
        assert runner.scheduler_pool is None
        return batch(**kwargs)

    summary = run_material_validation(
        config_path=config_path,
        parameters_path=parameter_path,
        output_dir=tmp_path / "failed_validation",
        resources=plan_nested_resources(
            available_cores=4,
            cores_per_state=2,
            candidate_workers=1,
            state_workers=2,
        ),
        runner_factory=lambda _config: runner,
        elastic_batch_runner=assert_released_then_run_elastic,
        elastic_backend_factory=lambda **_kwargs: None,
    )

    assert scheduler_pool.closed
    assert runner.scheduler_pool is None
    assert summary["status"] == "rejected"


def test_mechanical_tier_excess_is_manifested_best_effort_and_resumes(tmp_path):
    (
        summary,
        output,
        calls,
        config_path,
        parameter_path,
        runner_factory,
        batch,
        static_ranking,
        dynamic_ranking,
    ) = (
        _run_validation(tmp_path, mechanical_error=25.0, include_top=True)
    )

    assert summary["status"] == "best_effort"
    assert summary["hard_gate_pass"] is True
    assert summary["execution_complete"] is True
    assert calls == {"structure": 1, "static": 1, "dynamic": 1}
    assert (output / "stage_manifest.json").is_file()
    assert json.loads((output / "final_parameters.json").read_text())["status"] == (
        "best_effort"
    )
    adequacy = json.loads((output / "model_adequacy.json").read_text())
    assert adequacy["label_swap_diagnostic"]["status"] == "not_evaluated"
    assert adequacy["physical_transferability"]["status"] == "requires_validation"
    atom_types = pd.read_csv(output / "final_atom_parameters.csv")
    assert list(atom_types["label"]) == ["Fe_corner", "Fe_body"]
    assert (output / "TOP_PARAMETERS.csv").is_file()
    assert (output / "TOP_PARAMETERS.json").is_file()
    assert (output / "TOP_PARAMETERS.md").is_file()
    assert (output / "top_parameters_manifest.json").is_file()
    assert summary["top_parameters_report"]["reported"] == 1
    seed_policy = summary["dynamic_elastic_evidence"]["seed_policy"]
    assert seed_policy["status"] == "reused_promotion_seed_set"
    assert seed_policy["warnings"][0]["code"] == "reused_promotion_seed_set"

    rerun = run_material_validation(
        config_path=config_path,
        parameters_path=parameter_path,
        output_dir=output,
        resources=plan_nested_resources(
            available_cores=4,
            cores_per_state=2,
            candidate_workers=1,
            state_workers=2,
        ),
        runner_factory=runner_factory,
        elastic_batch_runner=batch,
        elastic_backend_factory=lambda **_kwargs: None,
        static_ranking_path=static_ranking,
        dynamic_ranking_path=dynamic_ranking,
        top_n=1,
    )
    assert rerun["status"] == "best_effort"
    assert calls == {"structure": 1, "static": 1, "dynamic": 1}


def test_final_validation_uses_explicit_holdout_dynamic_seeds(tmp_path):
    summary, _output, _calls, *_prefix, batch, _static, _dynamic = _run_validation(
        tmp_path,
        mechanical_error=10.0,
        validation_seeds=[404, 505, 606],
    )

    policy = summary["dynamic_elastic_evidence"]["seed_policy"]
    assert policy == {
        "status": "independent_validation_seed_set",
        "promotion_seeds": [101, 202],
        "validation_seeds": [404, 505, 606],
        "warnings": [],
    }
    assert batch.dynamic_seeds == [404, 505, 606]


def test_structural_failure_is_rejected_but_static_evidence_is_still_run(tmp_path):
    summary, output, calls, *_rest = _run_validation(
        tmp_path, fail_structure=True, mechanical_error=5.0
    )

    assert summary["status"] == "rejected"
    assert summary["hard_gate_pass"] is False
    assert any(reason.startswith("structural_gate") for reason in summary["hard_gate_reasons"])
    assert calls["structure"] == 1
    assert calls["static"] == 1
    assert (output / "elasticity_static" / "static_results.csv").is_file()
    assert (output / "stage_manifest.json").is_file()


def test_zero_eligible_finalists_publish_rejected_terminal_bundle_and_resume(tmp_path):
    config_path, config = _config(tmp_path)
    parameters = _parameters(config)
    outcome = tmp_path / "best_candidate.json"
    outcome.write_text(json.dumps({
        "schema_version": 1,
        "status": "zero_hard_gate_eligible",
        "message": "No candidate passed structure, Born, R2, and finite-score gates.",
    }), encoding="utf-8")
    static_ranking = tmp_path / "static.csv"
    dynamic_ranking = tmp_path / "dynamic.csv"
    static_row = _ranking_row(parameters, maximum=12.0, rmse=10.0)
    dynamic_row = {
        **_ranking_row(parameters, maximum=8.0, rmse=6.0),
        "born_stability_pass": False,
        "finalist_eligible": False,
        "quality_status": "rejected",
    }
    pd.DataFrame([static_row]).to_csv(static_ranking, index=False)
    pd.DataFrame([dynamic_row]).to_csv(dynamic_ranking, index=False)
    output = tmp_path / "validation"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("zero-eligible validation must not launch a simulation")

    arguments = dict(
        config_path=config_path,
        parameters_path=outcome,
        output_dir=output,
        resources=plan_nested_resources(
            available_cores=4,
            cores_per_state=2,
            candidate_workers=1,
            state_workers=2,
        ),
        runner_factory=forbidden,
        elastic_batch_runner=forbidden,
        elastic_backend_factory=forbidden,
        static_ranking_path=static_ranking,
        dynamic_ranking_path=dynamic_ranking,
        top_n=5,
    )
    summary = run_material_validation(**arguments)

    assert summary["status"] == "rejected"
    assert summary["model_status"] == "model_inadequate"
    assert summary["execution_complete"] is True
    assert summary["hard_gate_reasons"] == ["zero_hard_gate_eligible_finalists"]
    assert summary["top_parameters_report"]["reported"] == 0
    assert (output / "stage_manifest.json").is_file()
    assert pd.read_csv(output / "TOP_PARAMETERS.csv").empty
    assert pd.read_csv(output / "computed_properties.csv").empty
    final = json.loads((output / "final_parameters.json").read_text())
    assert final["parameter_availability"] == "none"
    assert final["raw_free_parameters"] == {}
    adequacy = json.loads((output / "model_adequacy.json").read_text())
    assert adequacy["candidate_selection"]["status"] == "model_inadequate"
    assert "no pair_coeff" in (
        output / "final_parameters.lammps"
    ).read_text(encoding="ascii")

    assert run_material_validation(**arguments) == summary


def _ranking_row(parameters, *, maximum, rmse):
    return {
        "parameter_key": canonical_parameter_key(parameters),
        **parameters,
        "structural_gate_pass": True,
        "born_stability_pass": True,
        "fit_quality_pass": True,
        "finite_mechanical_score": True,
        "finalist_eligible": True,
        "mechanical_max_error_percent": maximum,
        "mechanical_rmse_percent": rmse,
        "same_element_parameter_contrast": 0.1,
        "structural_margin": 0.5,
        "minimum_fit_r2": 0.995,
        "within_quality_tier": maximum <= 20.0,
        "quality_status": "within_tier" if maximum <= 20.0 else "best_effort",
    }


def test_top_parameters_reports_rank_reversal_and_explicit_provenance(tmp_path):
    config_path, config = _config(tmp_path)
    first = _parameters(config)
    second = {**first, "Fe_corner_epsilon": 7.0}
    static_path = tmp_path / "static_ranking.csv"
    dynamic_path = tmp_path / "dynamic_ranking.csv"
    pd.DataFrame([
        _ranking_row(first, maximum=5.0, rmse=4.0),
        _ranking_row(second, maximum=10.0, rmse=8.0),
    ]).to_csv(static_path, index=False)
    pd.DataFrame([
        _ranking_row(first, maximum=18.0, rmse=15.0),
        _ranking_row(second, maximum=4.0, rmse=3.0),
    ]).to_csv(dynamic_path, index=False)
    output = tmp_path / "top"

    frame = write_top_parameters_report(
        config_path=config_path,
        static_ranking_path=static_path,
        dynamic_ranking_path=dynamic_path,
        output_dir=output,
        top_n=2,
    )

    assert list(frame["Fe_corner_epsilon"]) == [7.0, 5.0]
    assert list(frame["static_rank"]) == [2, 1]
    assert list(frame["dynamic_rank"]) == [1, 2]
    assert set(frame["evidence"]) == {"static+dynamic"}
    document = json.loads((output / "TOP_PARAMETERS.json").read_text())
    assert document["provenance"]["static_ranking"]["path"] == str(
        static_path.resolve()
    )
    assert document["provenance"]["dynamic_ranking"]["path"] == str(
        dynamic_path.resolve()
    )
    assert "Dynamic evidence is ranked first" in (
        output / "TOP_PARAMETERS.md"
    ).read_text(encoding="utf-8")
    assert (output / "stage_manifest.json").is_file()
