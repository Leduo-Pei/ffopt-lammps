from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import threading

import pandas as pd
import pytest

from engine.cubic_elastic_batch import (
    BatchArtifactError,
    CandidateParameterError,
    build_parser as build_elastic_batch_parser,
    effective_available_cores,
    load_candidates,
    plan_nested_resources,
    rank_elastic_results,
    run_elasticity_batch,
    select_finalists,
)
from engine.cubic_elastic_runner import (
    ATM_TO_GPA,
    KCAL_PER_MOL_ANGSTROM3_TO_GPA,
    STATIC_PROTOCOL,
    StateExecutionResult,
)
from engine.parameter_space import build_parameter_space
from workflow.input_compiler import compile_input
from workflow.input_file import parse_input_file


def _rank_row(key, maximum, rmse, *, structure=True, born=True, fit=True, contrast=0.1):
    return {
        "parameter_key": key,
        "epsilon": float(key[-1]) if key[-1].isdigit() else 1.0,
        "sigma": 2.0,
        "structural_gate_pass": structure,
        "born_stability_pass": born,
        "fit_quality_pass": fit,
        "finite_mechanical_score": True,
        "finalist_eligible": structure and born and fit,
        "mechanical_max_error_percent": maximum,
        "mechanical_rmse_percent": rmse,
        "same_element_parameter_contrast": contrast,
        "structural_margin": 0.5,
        "within_quality_tier": maximum <= 20.0,
        "quality_tier_percent": 20.0,
    }


def test_gate_order_prevents_perfect_mechanics_from_rescuing_bad_structure():
    inconsistent_bad_row = _rank_row(
        "candidate2", 0.0, 0.0, structure=False
    )
    inconsistent_bad_row["finalist_eligible"] = True
    frame = pd.DataFrame([
        _rank_row("candidate1", 30.0, 20.0, structure=True),
        inconsistent_bad_row,
    ])

    ranked = rank_elastic_results(frame)

    assert list(ranked["parameter_key"]) == ["candidate1", "candidate2"]
    assert bool(ranked.iloc[0]["finalist_eligible"])
    assert not bool(ranked.iloc[1]["finalist_eligible"])
    selected = select_finalists(
        frame,
        parameter_space=[("epsilon", 0.0, 2.0), ("sigma", 1.0, 3.0)],
        top_n=2,
        near_optimal_window_percent=100.0,
        diversity_slots=0,
    )
    assert list(selected["parameter_key"]) == ["candidate1"]


def test_dynamic_ranking_can_reverse_static_rank_and_retains_best_effort():
    static = pd.DataFrame([
        _rank_row("candidate1", 10.0, 8.0),
        _rank_row("candidate2", 15.0, 9.0),
    ])
    dynamic = static.copy()
    dynamic.loc[dynamic["parameter_key"] == "candidate1", [
        "mechanical_max_error_percent", "mechanical_rmse_percent"
    ]] = [35.0, 25.0]
    dynamic.loc[dynamic["parameter_key"] == "candidate2", [
        "mechanical_max_error_percent", "mechanical_rmse_percent"
    ]] = [25.0, 18.0]
    dynamic["within_quality_tier"] = False

    assert list(rank_elastic_results(static)["parameter_key"]) == ["candidate1", "candidate2"]
    reranked = rank_elastic_results(dynamic)
    assert list(reranked["parameter_key"]) == ["candidate2", "candidate1"]
    assert bool(reranked.iloc[0]["finalist_eligible"])
    assert not bool(reranked.iloc[0]["within_quality_tier"])


def test_refinement_aliases_are_directly_accepted_by_finalist_selection():
    frame = pd.DataFrame([
        {
            "parameter_key": "candidate1",
            "epsilon": 1.0,
            "sigma": 2.0,
            "structural_feasible": True,
            "structural_minimum_margin": 0.4,
            "mechanical_stability_gate_pass": True,
            "mechanical_fit_gate_pass": True,
            "mechanical_eligible": True,
            "mechanical_max_error_percent": 22.0,
            "mechanical_rmse_percent": 15.0,
        }
    ])

    selected = select_finalists(
        frame,
        parameter_space=[("epsilon", 0.0, 2.0), ("sigma", 1.0, 3.0)],
        top_n=1,
        near_optimal_window_percent=5.0,
        diversity_slots=0,
    )

    assert len(selected) == 1
    assert bool(selected.iloc[0]["finalist_eligible"])
    assert selected.iloc[0]["finalist_selection_role"] == "near_optimal_quality"


def test_near_optimal_selection_keeps_quality_primary_and_adds_diversity():
    frame = pd.DataFrame([
        {**_rank_row(f"candidate{index}", 10.0 + index, 8.0 + index), "epsilon": value}
        for index, value in enumerate((0.1, 0.11, 0.9, 0.95), start=1)
    ])
    selected = select_finalists(
        frame,
        parameter_space=[("epsilon", 0.0, 1.0), ("sigma", 1.0, 3.0)],
        top_n=3,
        near_optimal_window_percent=5.0,
        diversity_slots=1,
    )

    assert selected.iloc[0]["parameter_key"] == "candidate1"
    assert list(selected["finalist_selection_role"]).count("near_optimal_diverse") == 1
    assert float(selected.loc[
        selected["finalist_selection_role"] == "near_optimal_diverse", "epsilon"
    ].iloc[0]) >= 0.9


def test_finalist_minimum_fills_by_quality_outside_window_without_exceeding_maximum():
    frame = pd.DataFrame([
        _rank_row(f"candidate{index}", maximum, maximum - 1.0)
        for index, maximum in enumerate((10.0, 11.0, 20.0, 30.0, 40.0), start=1)
    ])

    selected = select_finalists(
        frame,
        parameter_space=[("epsilon", 0.0, 9.0), ("sigma", 1.0, 3.0)],
        top_n=4,
        minimum=3,
        near_optimal_window_percent=1.0,
        diversity_slots=0,
    )

    assert list(selected["parameter_key"]) == [
        "candidate1",
        "candidate2",
        "candidate3",
    ]
    assert list(selected["finalist_selection_role"]) == [
        "near_optimal_quality",
        "near_optimal_quality",
        "minimum_quality_floor",
    ]


def test_finalist_minimum_never_admits_hard_gate_failure():
    frame = pd.DataFrame([
        _rank_row("candidate1", 10.0, 9.0),
        _rank_row("candidate2", 0.0, 0.0, structure=False),
    ])

    selected = select_finalists(
        frame,
        parameter_space=[("epsilon", 0.0, 9.0), ("sigma", 1.0, 3.0)],
        top_n=2,
        minimum=2,
        near_optimal_window_percent=0.0,
        diversity_slots=0,
    )

    assert list(selected["parameter_key"]) == ["candidate1"]


def test_nested_resource_plan_never_silently_oversubscribes():
    plan = plan_nested_resources(
        available_cores=76,
        cores_per_state=2,
        candidate_workers=2,
    )
    assert plan.state_workers == 19
    assert plan.allocated_cores == 76
    with pytest.raises(ValueError, match="oversubscribe"):
        plan_nested_resources(
            available_cores=76,
            cores_per_state=2,
            candidate_workers=2,
            state_workers=20,
        )
    hybrid = plan_nested_resources(
        available_cores=32,
        cores_per_state=2,
        omp_threads_per_rank=2,
        candidate_workers=4,
    )
    assert hybrid.state_workers == 2
    assert hybrid.allocated_cores == 32


def test_elastic_batch_cli_and_slurm_allocation_include_mpi_and_omp(monkeypatch):
    parser = build_elastic_batch_parser()
    arguments = parser.parse_args([
        "--config", "config.json",
        "--parameters", "parameters.csv",
        "--output-dir", "out",
        "--protocol", "static",
        "--available-cores", "76",
        "--cores-per-state", "2",
        "--omp-threads-per-state", "2",
        "--candidate-workers", "19",
    ])
    assert arguments.cores_per_state == 2
    assert arguments.omp_threads_per_state == 2
    assert arguments.candidate_workers == 19

    monkeypatch.setenv("SLURM_NTASKS", "38")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert effective_available_cores(100) == 76


def _bcc_data() -> str:
    return """BCC Fe test box

2 atoms
2 atom types

0.0 2.8665 xlo xhi
0.0 2.8665 ylo yhi
0.0 2.8665 zlo zhi

Masses

1 55.845 # Fe_corner
2 55.845 # Fe_body

Pair Coeffs # lj/cut

1 6.0 2.3 # Fe_corner
2 6.0 2.3 # Fe_body

Atoms # full

1 1 1 0.0 0.0 0.0 0.0 0 0 0
2 1 2 0.0 1.43325 1.43325 1.43325 0 0 0
"""


def _compiled_config(tmp_path: Path) -> tuple[Path, dict]:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    for name in ("bulk.data", "complete.data", "split.data"):
        (data / name).write_text(_bcc_data(), encoding="utf-8")
    ffopt = tmp_path / "ffopt.in"
    ffopt.write_text(
        """ffopt 1
project fe_batch
material elemental Fe
crystal bcc
workflow bo validate

parameters
    range epsilon absolute 0.001 10
    range sigma absolute 0.001 5
    mixing default
    tie epsilon all
    difference sigma Fe_corner Fe_body max 0.75 A
    type 1 Fe_corner 6.0 2.30
    type 2 Fe_body   6.0 2.35
end

property bulk
    data data/bulk.data
    cells_in_data 1 1 1
    target a 2.8665 A tolerance 0.03
    target b 2.8665 A tolerance 0.03
    target c 2.8665 A tolerance 0.03
    target alpha 90 degree tolerance 1
    target beta 90 degree tolerance 1
    target gamma 90 degree tolerance 1
    target density 7.874 g/cm3 tolerance 0.08
end

property surface
    complete data/complete.data
    split data/split.data
    facet 110
    target 2.34 J/m2 tolerance 0.117
end

property elasticity
    module static objective
    target static B 170 GPa
    target static Cprime 50 GPa
    target static C44 120 GPa
    module dynamic promotion
    target dynamic B 170 GPa
    target dynamic Cprime 50 GPa
    target dynamic C44 120 GPa
    temperature 300 K
    timestep 1 fs
    equilibration 2000
    production 4000
    seeds 101 202
    gate lattice 1 percent
    gate angles 1 degree
    gate density 1 percent
    gate surface 5 percent
    born required
    r2 0.98
    tier 20 percent
    strain 0.01 0.02
    replicate 1 1 1
end
""",
        encoding="utf-8",
    )
    config = compile_input(parse_input_file(ffopt)).config
    config["lammps"].update({"executable": "lmp", "mpiexec": "mpiexec", "timeout": 10})
    config["parallel"] = {
        "max_workers": 1,
        "cores_per_worker": 2,
        "omp_threads_per_worker": 1,
        "use_mpi": False,
    }
    path = tmp_path / "compiled.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path, config


def _candidate_frame(config, *, structural_pass=True):
    names = [name for name, _lower, _upper in build_parameter_space(config)]
    values = [
        {names[0]: 5.0, names[1]: 2.30, names[2]: 2.35},
        {names[0]: 7.0, names[1]: 2.31, names[2]: 2.36},
    ]
    rows = []
    for rank, parameters in enumerate(values, start=1):
        rows.append({
            "candidate_rank": rank,
            **parameters,
            "calc_a": 3.2 if not structural_pass else 2.8665,
            "calc_b": 2.8665,
            "calc_c": 2.8665,
            "calc_alpha": 90.0,
            "calc_beta": 90.0,
            "calc_gamma_ang": 90.0,
            "calc_density": 7.874,
            "calc_surf_energy": 2.34,
        })
    return pd.DataFrame(rows)


class FakeBatchBackend:
    def __init__(self, factory, *, protocol, seed, force_field):
        self.factory = factory
        self.protocol = protocol
        self.seed = seed
        identity = json.loads((force_field.parent / "resolved_parameters.json").read_text())
        self.epsilon = float(identity["raw_free_parameters"]["Fe_corner_epsilon"])

    def input_artifacts(self):
        return {"fake_input": self.factory.input_file}

    def scientific_identity(self):
        return {"backend": "fake_batch", "protocol": self.protocol, "epsilon": self.epsilon}

    def expected_artifacts(self, state):
        result = {"trace": "trace.txt"}
        if state.role == "reference":
            result["restart"] = "elastic_reference.restart"
        return result

    def run_state(self, state, work_dir, reference_artifacts, *, cores, trajectory_seed):
        assert cores == 2
        assert trajectory_seed == self.seed
        key = (self.epsilon, self.protocol, trajectory_seed, state.name)
        with self.factory.lock:
            self.factory.calls[key] += 1
            should_fail = key == self.factory.fail_once and not self.factory.failed
            if should_fail:
                self.factory.failed = True
        if should_fail:
            raise RuntimeError("synthetic interrupted state")
        (work_dir / "trace.txt").write_text(state.name + "\n", encoding="utf-8")
        artifacts = {"trace": "trace.txt"}
        if state.role == "reference":
            (work_dir / "elastic_reference.restart").write_bytes(b"same-reference\n")
            artifacts["restart"] = "elastic_reference.restart"
        record = self._record(state)
        return StateExecutionResult(record=record, artifacts=artifacts)

    def _constants(self):
        if self.protocol == STATIC_PROTOCOL:
            # epsilon=5 wins static; epsilon=7 is promoted but worse.
            return (170.0, 50.0, 120.0) if self.epsilon < 6.0 else (180.0, 55.0, 130.0)
        # Dynamic ordering reverses, and each seed is a real separate run.
        base = (190.0, 60.0, 140.0) if self.epsilon < 6.0 else (170.0, 50.0, 120.0)
        shift = -1.0 if self.seed == 101 else 1.0
        return tuple(value + shift for value in base)

    def _record(self, state):
        bulk, cprime, c44 = self._constants()
        if self.protocol == STATIC_PROTOCOL:
            volume = 1000.0
            reference = -12.5
            if state.role == "reference":
                density = reference
            else:
                quadratic = {
                    "hydro": 4.5 * bulk,
                    "orthorhombic": 2.0 * cprime,
                    "shear": 0.5 * c44,
                }[state.mode]
                density = reference + quadratic * state.strain**2 + 2.0e4 * state.strain**4
            return {
                "potential_energy_kcal_mol": density * volume / KCAL_PER_MOL_ANGSTROM3_TO_GPA,
                "volume_angstrom3": volume,
            }
        stress = {name: 0.0 for name in ("sxx", "syy", "szz", "sxy", "sxz", "syz")}
        if state.role != "reference" and state.direction != "zero":
            c11 = bulk + 4.0 * cprime / 3.0
            c12 = bulk - 2.0 * cprime / 3.0
            slopes = {
                "x": {"sxx": c11, "syy": c12, "szz": c12},
                "y": {"syy": c11, "sxx": c12, "szz": c12},
                "z": {"szz": c11, "sxx": c12, "syy": c12},
                "xy": {"sxy": c44},
                "xz": {"sxz": c44},
                "yz": {"syz": c44},
            }[state.direction]
            for component, slope in slopes.items():
                stress[component] = -(slope * state.strain) / ATM_TO_GPA
        return {**stress, "temperature_k": 300.0, "density_g_cm3": 7.874}


class FakeBackendFactory:
    def __init__(self, tmp_path, *, fail_once=None):
        self.input_file = tmp_path / "fake-elastic.in"
        self.input_file.write_text("fake immutable input\n", encoding="utf-8")
        self.calls = Counter()
        self.lock = threading.Lock()
        self.fail_once = fail_once
        self.failed = False

    def __call__(self, *, protocol, trajectory_seed, force_field_include, **_kwargs):
        return FakeBatchBackend(
            self,
            protocol=protocol,
            seed=trajectory_seed,
            force_field=force_field_include,
        )


def _resources():
    return plan_nested_resources(
        available_cores=4,
        cores_per_state=2,
        candidate_workers=1,
        state_workers=2,
    )


def test_static_then_dynamic_expands_real_seeds_and_allows_rank_reversal(tmp_path):
    config_path, config = _compiled_config(tmp_path)
    candidates = tmp_path / "candidates.csv"
    _candidate_frame(config).to_csv(candidates, index=False)
    static_output = tmp_path / "static"
    static_factory = FakeBackendFactory(tmp_path)

    static = run_elasticity_batch(
        config_path=config_path,
        parameters_path=candidates,
        output_dir=static_output,
        protocol="static",
        resources=_resources(),
        top_n=2,
        diversity_slots=0,
        backend_factory=static_factory,
    )
    assert list(static["Fe_corner_epsilon"]) == [5.0, 7.0]
    assert (static_output / "static_results.csv").is_file()
    assert (static_output / "stage_manifest.json").is_file()
    assert not any("rank_" in str(path) for path in (static_output / "candidate_runs").rglob("*"))

    dynamic_output = tmp_path / "dynamic"
    dynamic_factory = FakeBackendFactory(tmp_path)
    dynamic = run_elasticity_batch(
        config_path=config_path,
        parameters_path=static_output / "static_results.csv",
        output_dir=dynamic_output,
        protocol="dynamic",
        resources=_resources(),
        top_n=2,
        near_optimal_window_percent=15.0,
        diversity_slots=0,
        backend_factory=dynamic_factory,
    )

    assert list(dynamic["Fe_corner_epsilon"]) == [7.0, 5.0]
    assert set(seed for _eps, _protocol, seed, _state in dynamic_factory.calls) == {101, 202}
    assert all(count == 1 for count in dynamic_factory.calls.values())
    assert (dynamic["n_seeds_success"] == 2).all()
    assert (dynamic["B_gpa_std"] > 0.0).all()
    assert (dynamic_output / "dynamic_seed_results.csv").is_file()
    assert (dynamic_output / "stage_manifest.json").is_file()
    before = sum(dynamic_factory.calls.values())
    run_elasticity_batch(
        config_path=config_path,
        parameters_path=static_output / "static_results.csv",
        output_dir=dynamic_output,
        protocol="dynamic",
        resources=_resources(),
        top_n=2,
        near_optimal_window_percent=15.0,
        diversity_slots=0,
        backend_factory=dynamic_factory,
    )
    assert sum(dynamic_factory.calls.values()) == before


def test_partial_resume_retries_only_failed_parameter_key_state(tmp_path):
    config_path, config = _compiled_config(tmp_path)
    candidates = tmp_path / "one.csv"
    _candidate_frame(config).iloc[:1].to_csv(candidates, index=False)
    output = tmp_path / "partial"
    fail_key = (5.0, STATIC_PROTOCOL, None, "hydro_m0p02")
    factory = FakeBackendFactory(tmp_path, fail_once=fail_key)

    first = run_elasticity_batch(
        config_path=config_path,
        parameters_path=candidates,
        output_dir=output,
        protocol="static",
        resources=_resources(),
        top_n=1,
        diversity_slots=0,
        backend_factory=factory,
    )
    assert first.iloc[0]["calculation_status"] == "failed"
    assert not (output / "stage_manifest.json").exists()

    second = run_elasticity_batch(
        config_path=config_path,
        parameters_path=candidates,
        output_dir=output,
        protocol="static",
        resources=_resources(),
        top_n=1,
        diversity_slots=0,
        backend_factory=factory,
    )
    assert second.iloc[0]["calculation_status"] == "completed"
    assert factory.calls[fail_key] == 2
    assert all(count == 1 for key, count in factory.calls.items() if key != fail_key)
    assert (output / "stage_manifest.json").is_file()


def test_completed_batch_rejects_parameter_or_candidate_artifact_tampering(tmp_path):
    config_path, config = _compiled_config(tmp_path)
    candidates = tmp_path / "one.csv"
    _candidate_frame(config).iloc[:1].to_csv(candidates, index=False)
    output = tmp_path / "tamper"
    factory = FakeBackendFactory(tmp_path)
    run_elasticity_batch(
        config_path=config_path,
        parameters_path=candidates,
        output_dir=output,
        protocol="static",
        resources=_resources(),
        top_n=1,
        diversity_slots=0,
        backend_factory=factory,
    )
    resolved = next((output / "candidate_runs").glob("*/inputs/resolved_parameters.json"))
    resolved.write_text("{}\n", encoding="ascii")
    with pytest.raises(BatchArtifactError, match="immutable"):
        run_elasticity_batch(
            config_path=config_path,
            parameters_path=candidates,
            output_dir=output,
            protocol="static",
            resources=_resources(),
            top_n=1,
            diversity_slots=0,
            backend_factory=factory,
        )


def test_zero_structural_eligible_is_successful_manifested_outcome(tmp_path):
    config_path, config = _compiled_config(tmp_path)
    candidates = tmp_path / "failed-structure.csv"
    _candidate_frame(config, structural_pass=False).to_csv(candidates, index=False)
    output = tmp_path / "zero"
    factory = FakeBackendFactory(tmp_path)

    ranked = run_elasticity_batch(
        config_path=config_path,
        parameters_path=candidates,
        output_dir=output,
        protocol="static",
        resources=_resources(),
        top_n=2,
        diversity_slots=0,
        backend_factory=factory,
    )

    assert not ranked["finalist_eligible"].any()
    assert set(ranked["calculation_status"]) == {"skipped_structural_gate"}
    assert not factory.calls
    assert json.loads((output / "best_candidate.json").read_text())["status"] == "zero_hard_gate_eligible"
    assert (output / "stage_manifest.json").is_file()


def test_dynamic_zero_eligible_is_successful_without_launching_seeds(tmp_path):
    config_path, config = _compiled_config(tmp_path)
    candidates = tmp_path / "zero-finalists.csv"
    frame = _candidate_frame(config)
    frame["structural_gate_pass"] = True
    frame["born_stability_pass"] = False
    frame["fit_quality_pass"] = True
    frame["finite_mechanical_score"] = True
    frame["finalist_eligible"] = False
    frame["mechanical_max_error_percent"] = 5.0
    frame["mechanical_rmse_percent"] = 4.0
    frame["same_element_parameter_contrast"] = 0.0
    frame["structural_margin"] = 0.5
    frame.to_csv(candidates, index=False)
    output = tmp_path / "dynamic-zero"
    factory = FakeBackendFactory(tmp_path)

    ranked = run_elasticity_batch(
        config_path=config_path,
        parameters_path=candidates,
        output_dir=output,
        protocol="dynamic",
        resources=_resources(),
        top_n=2,
        diversity_slots=0,
        backend_factory=factory,
    )

    assert ranked.empty
    assert not factory.calls
    assert json.loads((output / "best_candidate.json").read_text())["status"] == (
        "zero_hard_gate_eligible"
    )
    assert (output / "stage_manifest.json").is_file()


def test_recorded_parameter_key_detects_value_tampering(tmp_path):
    config_path, config = _compiled_config(tmp_path)
    del config_path
    space = build_parameter_space(config)
    frame = _candidate_frame(config).iloc[:1]
    path = tmp_path / "keyed.csv"
    frame.to_csv(path, index=False)
    candidate = load_candidates(path, space)[0]
    frame["parameter_key"] = candidate.parameter_key
    frame.loc[frame.index[0], space[0][0]] += 0.1
    frame.to_csv(path, index=False)

    with pytest.raises(CandidateParameterError, match="does not match"):
        load_candidates(path, space)
