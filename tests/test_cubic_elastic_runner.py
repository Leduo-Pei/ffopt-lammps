from __future__ import annotations

from collections import Counter
import json
import threading
import time

import pytest

from engine.cubic_elastic_runner import (
    DYNAMIC_PROTOCOL,
    KCAL_PER_MOL_ANGSTROM3_TO_GPA,
    STATIC_PROTOCOL,
    CandidateIdentityError,
    ResourceBudget,
    SingleCandidateCubicElasticRunner,
    StateArtifactError,
    StateExecutionResult,
    load_lammps_candidate_job,
)


class FakeElasticBackend:
    def __init__(self, tmp_path, protocol, *, fail_once=None):
        self.protocol = protocol
        self.source = tmp_path / f"{protocol}.input"
        self.source.write_text("immutable simulation input\n", encoding="utf-8")
        self.fail_once = fail_once
        self.failed = False
        self.calls = []
        self.reference_paths = []
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def input_artifacts(self):
        return {"simulation_input": self.source}

    def scientific_identity(self):
        return {"backend": "deterministic_fake", "protocol": self.protocol}

    def expected_artifacts(self, state):
        outputs = {"trace": "trace.txt"}
        if state.role == "reference":
            outputs["restart"] = "elastic_reference.restart"
        return outputs

    def _static_record(self, state):
        reference_density = -12.5
        volume = 1000.0
        if state.role == "reference":
            density = reference_density
        else:
            quadratic = {"hydro": 4.5 * 170.0, "orthorhombic": 2.0 * 50.0, "shear": 0.5 * 120.0}
            quartic = {"hydro": 2.0e5, "orthorhombic": -1.0e4, "shear": 3.0e4}
            odd = {"hydro": 300.0, "orthorhombic": -80.0, "shear": 40.0}
            strain = state.strain
            density = (
                reference_density
                + quadratic[state.mode] * strain**2
                + quartic[state.mode] * strain**4
                + odd[state.mode] * strain**3
            )
        return {
            "potential_energy_kcal_mol": (
                density * volume / KCAL_PER_MOL_ANGSTROM3_TO_GPA
            ),
            "volume_angstrom3": volume,
        }

    @staticmethod
    def _dynamic_record(state):
        record = {name: 0.0 for name in ("sxx", "syy", "szz", "sxy", "sxz", "syz")}
        if state.role == "reference" or state.direction == "zero":
            return {**record, "temperature_k": 300.0, "density_g_cm3": 7.87}
        slopes = {
            "x": {"sxx": 240.0, "syy": 140.0, "szz": 140.0},
            "y": {"syy": 240.0, "sxx": 140.0, "szz": 140.0},
            "z": {"szz": 240.0, "sxx": 140.0, "syy": 140.0},
            "xy": {"sxy": 120.0},
            "xz": {"sxz": 120.0},
            "yz": {"syz": 120.0},
        }
        for component, slope in slopes[state.direction].items():
            record[component] = slope * state.strain
        return {**record, "temperature_k": 300.0, "density_g_cm3": 7.87}

    def run_state(self, state, work_dir, reference_artifacts, *, cores, trajectory_seed):
        assert cores == 2
        assert trajectory_seed == (101 if self.protocol == DYNAMIC_PROTOCOL else None)
        with self.lock:
            self.calls.append(state.name)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(0.002)
            if state.role != "reference":
                assert set(reference_artifacts) == {"restart", "trace"}
                self.reference_paths.append(reference_artifacts["restart"])
            if self.fail_once == state.name and not self.failed:
                self.failed = True
                raise RuntimeError("synthetic state failure")
            (work_dir / "trace.txt").write_text(state.name + "\n", encoding="utf-8")
            artifacts = {"trace": "trace.txt"}
            if state.role == "reference":
                (work_dir / "elastic_reference.restart").write_bytes(b"common-reference\n")
                artifacts["restart"] = "elastic_reference.restart"
            record = (
                self._static_record(state)
                if self.protocol == STATIC_PROTOCOL
                else self._dynamic_record(state)
            )
            return StateExecutionResult(record=record, artifacts=artifacts)
        finally:
            with self.lock:
                self.active -= 1


def _runner(tmp_path, backend, *, candidate="candidate-1", parameters=None, workers=2):
    config = {"temperature_k": 300.0}
    if backend.protocol == DYNAMIC_PROTOCOL:
        config.update({"stress_convention": "tension_positive", "stress_to_gpa": 1.0})
    return SingleCandidateCubicElasticRunner(
        candidate_dir=tmp_path / candidate,
        candidate_id=candidate,
        protocol=backend.protocol,
        parameters=parameters or {"epsilon": 0.2, "sigma": 2.3},
        scientific_config=config,
        trajectory_seed=101 if backend.protocol == DYNAMIC_PROTOCOL else None,
        backend=backend,
        resource_budget=ResourceBudget(
            max_workers=workers,
            cores_per_state=2,
            available_cores=4,
        ),
    )


def test_resource_budget_rejects_implicit_oversubscription():
    with pytest.raises(ValueError, match="oversubscribe"):
        ResourceBudget(max_workers=3, cores_per_state=2, available_cores=4)
    with pytest.raises(ValueError, match="positive integer"):
        ResourceBudget(max_workers=0, cores_per_state=2, available_cores=4)
    with pytest.raises(ValueError, match="oversubscribe"):
        ResourceBudget(
            max_workers=2,
            cores_per_state=2,
            available_cores=4,
            omp_threads_per_rank=2,
        )


def test_single_candidate_seed_semantics_cannot_claim_unexecuted_replicates(tmp_path):
    dynamic = FakeElasticBackend(tmp_path, DYNAMIC_PROTOCOL)
    with pytest.raises(ValueError, match="exactly one trajectory_seed"):
        SingleCandidateCubicElasticRunner(
            candidate_dir=tmp_path / "dynamic-no-seed",
            candidate_id="dynamic-no-seed",
            protocol=DYNAMIC_PROTOCOL,
            parameters={"epsilon": 0.2, "sigma": 2.3},
            scientific_config={},
            trajectory_seed=None,
            backend=dynamic,
            resource_budget=ResourceBudget(1, 2, 2),
        )
    with pytest.raises(ValueError, match="one integer or None"):
        SingleCandidateCubicElasticRunner(
            candidate_dir=tmp_path / "dynamic-many-seeds",
            candidate_id="dynamic-many-seeds",
            protocol=DYNAMIC_PROTOCOL,
            parameters={"epsilon": 0.2, "sigma": 2.3},
            scientific_config={},
            trajectory_seed=[101, 202],
            backend=dynamic,
            resource_budget=ResourceBudget(1, 2, 2),
        )
    static = FakeElasticBackend(tmp_path, STATIC_PROTOCOL)
    with pytest.raises(ValueError, match="trajectory_seed=None"):
        SingleCandidateCubicElasticRunner(
            candidate_dir=tmp_path / "static-with-seed",
            candidate_id="static-with-seed",
            protocol=STATIC_PROTOCOL,
            parameters={"epsilon": 0.2, "sigma": 2.3},
            scientific_config={},
            trajectory_seed=101,
            backend=static,
            resource_budget=ResourceBudget(1, 2, 2),
        )


def test_static_candidate_recovers_independent_moduli_and_reuses_every_artifact(tmp_path):
    backend = FakeElasticBackend(tmp_path, STATIC_PROTOCOL)
    runner = _runner(tmp_path, backend)
    strains = [-0.02, -0.01, 0.01, 0.02]

    summary = runner.run_static(strains)

    assert summary["elastic_constants_gpa"] == pytest.approx(
        {"C11": 236.6666666667, "C12": 136.6666666667, "C44": 120.0}
    )
    assert summary["independent_moduli_gpa"] == pytest.approx(
        {"B": 170.0, "Cprime": 50.0, "C44": 120.0}
    )
    assert summary["fit_quality"]["minimum_r2"] == pytest.approx(1.0)
    assert summary["born_stability"]["stable"] is True
    assert summary["eligible"] is True
    assert len(backend.calls) == 13
    assert backend.maximum_active <= 2
    assert len(set(backend.reference_paths)) == 1
    assert (tmp_path / "candidate-1" / "candidate_identity.json").is_file()
    assert (tmp_path / "candidate-1" / "artifact_manifest.json").is_file()

    second = runner.run_static(strains)
    assert second == summary
    assert len(backend.calls) == 13


def test_dynamic_candidate_reports_independent_and_derived_diagnostics(tmp_path):
    backend = FakeElasticBackend(tmp_path, DYNAMIC_PROTOCOL)
    runner = _runner(tmp_path, backend)

    summary = runner.run_dynamic([-0.002, 0.002])

    assert summary["elastic_constants_gpa"] == pytest.approx(
        {"C11": 240.0, "C12": 140.0, "C44": 120.0}
    )
    assert summary["independent_moduli_gpa"] == pytest.approx(
        {"B": 173.3333333333, "Cprime": 50.0, "C44": 120.0}
    )
    assert summary["derived_diagnostics"]["G_hill_gpa"] > 0.0
    assert summary["derived_diagnostics"]["E_hill_gpa"] > 0.0
    assert 0.0 < summary["derived_diagnostics"]["nu_hill"] < 0.5
    assert summary["fit_quality"]["minimum_relevant_r2"] == pytest.approx(1.0)
    assert summary["born_stability"]["stable"] is True
    # reference + zero + six directions times two symmetric strains
    assert len(backend.calls) == 14
    assert len(set(backend.reference_paths)) == 1


def test_tampered_completed_candidate_fails_closed_instead_of_recomputing(tmp_path):
    backend = FakeElasticBackend(tmp_path, STATIC_PROTOCOL)
    runner = _runner(tmp_path, backend)
    strains = [-0.02, -0.01, 0.01, 0.02]
    runner.run_static(strains)
    calls = len(backend.calls)

    (tmp_path / "candidate-1" / "strain_records.json").write_text("[]\n", encoding="ascii")
    with pytest.raises(StateArtifactError, match="not safely reusable"):
        runner.run_static(strains)
    assert len(backend.calls) == calls


def test_candidate_directory_rejects_parameter_or_strain_identity_change(tmp_path):
    backend = FakeElasticBackend(tmp_path, STATIC_PROTOCOL)
    strains = [-0.02, -0.01, 0.01, 0.02]
    _runner(tmp_path, backend).run_static(strains)

    changed_parameter = _runner(
        tmp_path,
        backend,
        parameters={"epsilon": 0.3, "sigma": 2.3},
    )
    with pytest.raises(CandidateIdentityError, match="immutable identity"):
        changed_parameter.run_static(strains)

    changed_strain = _runner(tmp_path, backend)
    with pytest.raises(CandidateIdentityError, match="immutable identity"):
        changed_strain.run_static([-0.03, -0.01, 0.01, 0.03])


def test_interrupted_candidate_reuses_finished_states_and_only_retries_failure(tmp_path):
    backend = FakeElasticBackend(tmp_path, STATIC_PROTOCOL, fail_once="hydro_m0p02")
    runner = _runner(tmp_path, backend, workers=1)
    strains = [-0.02, -0.01, 0.01, 0.02]

    with pytest.raises(Exception, match="synthetic state failure"):
        runner.run_static(strains)
    first_counts = Counter(backend.calls)
    assert first_counts["hydro_m0p02"] == 1

    summary = runner.run_static(strains)
    final_counts = Counter(backend.calls)
    assert summary["eligible"] is True
    assert final_counts["hydro_m0p02"] == 2
    assert all(
        count == 1
        for state, count in final_counts.items()
        if state != "hydro_m0p02"
    )


def test_packaged_templates_make_independent_reference_branches_explicit():
    root = __import__("pathlib").Path(__file__).parents[1] / "lammps" / "inputs" / "elasticity"
    static_reference = (root / "in.cubic.static_reference").read_text(encoding="utf-8")
    static_strain = (root / "in.cubic.static_strain").read_text(encoding="utf-8")
    dynamic_reference = (root / "in.cubic.dynamic_reference").read_text(encoding="utf-8")
    dynamic_strain = (root / "in.cubic.dynamic_strain").read_text(encoding="utf-8")

    assert "write_restart elastic_reference.restart" in static_reference
    assert "read_restart elastic_reference.restart" in static_strain
    assert "write_restart elastic_reference.restart" in dynamic_reference
    assert "read_restart elastic_reference.restart" in dynamic_strain
    assert "FFOPT_STATIC_STRAIN:" in static_strain
    assert "FFOPT_DYNAMIC_STRESS:" in dynamic_strain


def test_internal_cli_job_contract_is_strict_and_resolves_relative_paths(tmp_path):
    for name in ("reference.in", "strain.in", "bulk.data", "force_field.lmp"):
        (tmp_path / name).write_text(name + "\n", encoding="utf-8")
    job = {
        "schema_version": 1,
        "candidate_id": "candidate-cli",
        "candidate_dir": "candidate-cli",
        "protocol": STATIC_PROTOCOL,
        "parameters": {"epsilon": 0.2, "sigma": 2.3},
        "scientific_config": {"temperature_k": 0.0},
        "trajectory_seed": None,
        "strains": [-0.02, -0.01, 0.01, 0.02],
        "resources": {"max_workers": 2, "cores_per_state": 2, "available_cores": 4},
        "lammps": {
            "command": ["srun", "--exclusive", "-n", "{cores}", "lmp"],
            "reference_template": "reference.in",
            "strain_template": "strain.in",
            "bulk_data": "bulk.data",
            "force_field_include": "force_field.lmp",
            "variables": {"units": "real", "atom_style": "atomic", "cutoff": 10.0},
        },
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job), encoding="utf-8")

    runner, strains = load_lammps_candidate_job(path)

    assert runner.candidate_dir == (tmp_path / "candidate-cli").resolve()
    assert runner.protocol == STATIC_PROTOCOL
    assert runner.trajectory_seed is None
    assert strains == (-0.02, -0.01, 0.01, 0.02)
    assert runner.resource_budget.max_workers == 2
    assert runner.backend.reference_template == (tmp_path / "reference.in").resolve()

    job["unexpected"] = True
    path.write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_lammps_candidate_job(path)
