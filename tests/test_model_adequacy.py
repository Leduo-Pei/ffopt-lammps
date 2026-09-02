from __future__ import annotations

import copy
import json
import math

import pytest

from engine.model_adequacy import (
    ModelAdequacyError,
    aggregate_transferability_evidence,
    assess_model_adequacy,
    assess_same_element_relabeling_invariance,
)


def _elemental_config() -> dict:
    return {
        "material": {"kind": "elemental", "name": "Fe"},
        "lammps": {"pair_style": "lj/cut"},
        "charge": {"enabled": False},
        "pair_params": {
            "mixing_rule": "default",
            "derived_params": [
                {
                    "target": "Fe_body_epsilon",
                    "expression": "Fe_corner_epsilon",
                }
            ],
        },
        "atom_types": [
            {
                "type": 1,
                "label": "Fe_corner",
                "mass": 55.845,
                "params": {
                    "epsilon": {"min": 0.001, "max": 10.0, "init": 1.0},
                    "sigma": {"min": 0.001, "max": 5.0, "init": 2.3},
                    "charge": 0.0,
                },
            },
            {
                "type": 2,
                "label": "Fe_body",
                "mass": 55.845,
                "params": {
                    "epsilon": 1.0,
                    "sigma": {"min": 0.001, "max": 5.0, "init": 2.5},
                    "charge": 0.0,
                },
            },
        ],
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
    }


def _parameter(report: dict, name: str) -> dict:
    return next(
        row for row in report["parameter_symmetry"]["parameters"]
        if row["parameter"] == name
    )


def test_multitype_elemental_report_separates_composition_from_transferability() -> None:
    config = _elemental_config()
    untouched = copy.deepcopy(config)

    report = assess_model_adequacy(config)

    assert config == untouched
    assert report["read_only"] is True
    assert report["material"] == {
        "kind": "elemental",
        "name": "Fe",
        "atom_type_count": 2,
        "same_element_representation": "permanent_ordered_sublattice_surrogate",
    }
    assert report["model_form"]["central_lj_pair_style"] is True
    assert report["model_form"]["mixing"] == {
        "configured": "default",
        "uses_lammps_native_default": True,
        "resolved_for_diagnostics": "geometric",
        "resolution_status": "known",
    }
    assert _parameter(report, "epsilon")["relation"] == "tied"
    assert _parameter(report, "epsilon")["can_differ_between_same_element_types"] is False
    assert _parameter(report, "sigma")["relation"] == "difference_limited"
    assert _parameter(report, "sigma")["difference_limits"][0]["max_abs"] == 0.75
    assert _parameter(report, "charge")["relation"] == "fixed_equal"
    assert report["parameter_symmetry"]["parameters_that_can_differ"] == ["sigma"]

    assert report["computational_composability"]["status"] == (
        "computationally_composable"
    )
    transfer = report["physical_transferability"]
    assert transfer["status"] == "requires_validation"
    assert transfer["warning_is_not_impossibility_proof"] is True
    assert [item["id"] for item in transfer["required_validations"]] == [
        "label_swap",
        "surface_registry",
        "vacancy",
        "migration_path_sensitivity",
    ]
    assert [
        item["minimum_cases"] for item in transfer["required_validations"]
    ] == [3, 2, 2, 2]
    assert report["warnings"][0]["code"] == "ordered_sublattice_surrogate"
    assert report["warnings"][0]["implies_fit_is_impossible"] is False
    json.dumps(report, allow_nan=False)


def test_single_type_elemental_does_not_emit_ordered_sublattice_warning() -> None:
    config = _elemental_config()
    config["atom_types"] = config["atom_types"][:1]
    config["parameter_constraints"] = {"ties": [], "differences": []}

    report = assess_model_adequacy(config)

    assert report["material"]["same_element_representation"] == "single_type_elemental"
    assert report["warnings"] == []
    assert report["physical_transferability"]["status"] == "unknown"
    assert report["physical_transferability"]["required_validations"] == []


def test_missing_model_and_diagnostic_evidence_remains_unknown() -> None:
    report = assess_model_adequacy({})

    assert report["material"]["kind"] == "unknown"
    assert report["material"]["same_element_representation"] == "unknown"
    assert report["model_form"]["central_lj_pair_style"] is None
    assert report["model_form"]["mixing"]["resolution_status"] == "unknown"
    assert report["computational_composability"]["status"] == "unknown"
    assert report["physical_transferability"]["status"] == "unknown"
    assert report["elasticity_diagnostics"]["zero_kelvin"]["status"] == "unknown"
    assert report["label_swap_diagnostic"]["status"] == "unknown"
    json.dumps(report, allow_nan=False)


def test_zero_k_cauchy_delta_and_finite_temperature_are_reported_with_scope() -> None:
    report = assess_model_adequacy(
        _elemental_config(),
        elasticity_results={
            "zero_k": {
                "elastic_constants_gpa": {"C12": 138.1, "C44": 121.9}
            },
            "finite_temperature": {
                "temperature_k": 300.0,
                "C12": 134.1,
                "C44": 115.87,
            },
        },
    )

    zero = report["elasticity_diagnostics"]["zero_kelvin"]
    assert zero["status"] == "available"
    assert zero["temperature_k"] == 0.0
    assert zero["cauchy_delta_gpa"] == pytest.approx(16.2)
    assert zero["normalized_cauchy_delta"] == pytest.approx(16.2 / 138.1)
    assert zero["interpretation"] == "model_form_diagnostic"

    finite = report["elasticity_diagnostics"]["finite_temperature"]
    assert finite["status"] == "available"
    assert finite["cauchy_delta_gpa"] == pytest.approx(18.23)
    assert finite["normalized_cauchy_delta"] == pytest.approx(18.23 / 134.1)
    assert finite["interpretation"] == "diagnostic_only"
    assert "thermal" in finite["scope"]


def test_direct_elasticity_mapping_is_explicitly_interpreted_as_zero_k() -> None:
    report = assess_model_adequacy(
        _elemental_config(),
        elasticity_results={"C12": 0.0, "C44": 0.0},
    )

    zero = report["elasticity_diagnostics"]["zero_kelvin"]
    assert zero["status"] == "available"
    assert zero["cauchy_delta_gpa"] == 0.0
    assert zero["normalized_cauchy_delta"] is None
    assert "interpreted as zero kelvin" in zero["assumption"]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_diagnostic_or_parameter_values_fail_strictly(bad: float) -> None:
    with pytest.raises(ModelAdequacyError, match="finite number"):
        assess_model_adequacy(
            _elemental_config(), elasticity_results={"C12": bad, "C44": 1.0}
        )

    config = _elemental_config()
    config["atom_types"][0]["params"]["sigma"]["init"] = bad
    with pytest.raises(ModelAdequacyError, match="finite number"):
        assess_model_adequacy(config)


def test_label_swap_values_are_measured_or_checked_without_claiming_impossibility() -> None:
    passing = assess_model_adequacy(
        _elemental_config(),
        label_swap_result={
            "reference_value": -100.0,
            "swapped_value": -100.5,
            "relative_tolerance": 0.01,
        },
    )
    diagnostic = passing["label_swap_diagnostic"]
    assert diagnostic["status"] == "pass"
    assert diagnostic["absolute_change"] == pytest.approx(0.5)
    assert diagnostic["normalized_absolute_change"] == pytest.approx(0.5 / 100.5)
    # The legacy scalar diagnostic remains visible, but without audited case
    # provenance it cannot establish the independent transferability gate.
    assert passing["physical_transferability"]["required_validations"][0]["status"] == (
        "error"
    )

    failed = assess_model_adequacy(
        _elemental_config(), label_swap_result={"passed": False}
    )
    assert failed["label_swap_diagnostic"]["status"] == "fail"
    assert failed["warnings"][-1]["code"] == "label_swap_sensitivity"
    assert failed["warnings"][-1]["implies_fit_is_impossible"] is False


def test_independent_same_element_parameters_are_visible() -> None:
    config = _elemental_config()
    config["parameter_constraints"] = {"ties": [], "differences": []}

    report = assess_model_adequacy(config)

    assert _parameter(report, "epsilon")["relation"] == "independent"
    assert _parameter(report, "sigma")["relation"] == "independent"
    assert report["parameter_symmetry"]["parameters_that_can_differ"] == [
        "epsilon",
        "sigma",
    ]


def test_conflicting_pair_style_declarations_fail_to_unknown_not_guess() -> None:
    config = _elemental_config()
    config["pair_params"]["pair_style"] = "eam"

    report = assess_model_adequacy(config)

    assert report["model_form"]["pair_style"] is None
    assert report["model_form"]["pair_style_source"] == "conflicting_declarations"
    assert report["model_form"]["central_lj_pair_style"] is None
    assert report["computational_composability"]["status"] == "unknown"


def _resolved(*, sigma_corner: float = 2.3, sigma_body: float = 2.5) -> dict:
    return {
        "Fe_corner_epsilon": 6.0,
        "Fe_corner_sigma": sigma_corner,
        "Fe_corner_charge": 0.0,
        "Fe_body_epsilon": 6.0,
        "Fe_body_sigma": sigma_body,
        "Fe_body_charge": 0.0,
    }


def _attested_record(test_id: str, case_count: int) -> dict:
    digest = "a" * 64
    return {
        "status": "pass",
        "cases": [
            {"id": f"{test_id}_{index}", "status": "pass"}
            for index in range(case_count)
        ],
        "provenance_attestation": {
            "schema_version": 1,
            "content_audit_attested": True,
            "producer": "ffopt-transfer-audit-v1",
            "manifest_sha256": digest,
            "protocol_sha256": digest,
            "metrics_sha256": digest,
        },
    }


def test_formal_relabel_invariance_expands_complete_matrix_and_rejects_a11_form() -> None:
    result = assess_same_element_relabeling_invariance(
        _elemental_config(),
        resolved_parameters=_resolved(
            sigma_corner=2.5736461048490398,
            sigma_body=2.0442767275844203,
        ),
    )

    assert result["formal_invariance_status"] == "fail"
    assert result["pair_matrix_complete"] is True
    assert len(result["pair_coefficients"]) == 4
    cross = next(
        row for row in result["pair_coefficients"]
        if row["type_i"] == 1 and row["type_j"] == 2
    )
    assert cross["epsilon"] == pytest.approx(6.0)
    assert cross["sigma"] == pytest.approx(
        math.sqrt(2.5736461048490398 * 2.0442767275844203)
    )
    assert any(
        item["kind"] == "pair_coefficient" and item["parameter"] == "sigma"
        for item in result["violations"]
    )
    assert result["not_a_matrix_permutation_test"] is True


def test_equal_complete_pair_rows_pass_formal_but_need_empirical_evidence() -> None:
    report = assess_model_adequacy(
        _elemental_config(),
        resolved_parameters=_resolved(sigma_corner=2.3, sigma_body=2.3),
    )

    assert report["formal_invariance_status"] == "pass"
    assert report["empirical_transfer_status"] == "not_evaluated"
    assert report["elemental_transferability_claim"] == (
        "requires_empirical_validation"
    )
    assert report["physical_transferability"][
        "ordered_sublattice_bulk_validation_affected"
    ] is False


def test_transferability_evidence_schema_is_fail_closed_and_kept_separate() -> None:
    evidence = aggregate_transferability_evidence({
        "label_swap": "pass",
        "surface_registry": {
            "cases": [
                {"id": "termination_a", "status": "pass"},
                {"id": "termination_b", "status": "fail"},
            ]
        },
        "vacancy": "error",
        "migration_path_sensitivity": "not_applicable",
    })

    assert evidence["status"] == "fail"
    assert evidence["tests"]["surface_registry"]["status"] == "fail"
    assert evidence["tests"]["vacancy"]["status"] == "error"
    assert evidence["ordered_sublattice_bulk_validation_affected"] is False
    json.dumps(evidence, allow_nan=False)


def test_attested_empirical_pass_still_requires_real_content_verifier() -> None:
    report = assess_model_adequacy(
        _elemental_config(),
        resolved_parameters=_resolved(sigma_corner=2.3, sigma_body=2.3),
        transferability_evidence={
            "label_swap": _attested_record("label_swap", 3),
            "surface_registry": _attested_record("surface_registry", 2),
            "vacancy": _attested_record("vacancy", 2),
            "migration_path_sensitivity": _attested_record(
                "migration_path_sensitivity", 2
            ),
        },
    )

    assert report["formal_invariance_status"] == "pass"
    assert report["empirical_transfer_status"] == "attested_pass"
    assert report["elemental_transferability_claim"] == "not_established"
    assert report["physical_transferability"]["status"] == (
        "requires_verified_runner"
    )
    assert report["transferability_evidence"]["complete"] is False


def test_explicit_cross_pair_must_complete_rows_and_columns() -> None:
    config = _elemental_config()
    config["pair_params"]["mixing_rule"] = "none"
    config["pair_params"]["explicit_pairs"] = [{
        "types": [1, 2],
        "epsilon": {"min": 0.1, "max": 10.0},
        "sigma": {"min": 0.1, "max": 5.0},
    }]
    missing = assess_same_element_relabeling_invariance(
        config,
        resolved_parameters=_resolved(sigma_corner=2.3, sigma_body=2.3),
    )
    assert missing["formal_invariance_status"] == "not_evaluated"
    assert missing["missing_pairs"] == ["1-2"]

    resolved = _resolved(sigma_corner=2.3, sigma_body=2.3)
    resolved.update({"cross_1_2_epsilon": 6.0, "cross_1_2_sigma": 2.3})
    complete = assess_same_element_relabeling_invariance(
        config, resolved_parameters=resolved
    )
    assert complete["formal_invariance_status"] == "pass"


def test_transferability_evidence_rejects_inconsistent_case_summary() -> None:
    with pytest.raises(ModelAdequacyError, match="conflicts with its cases"):
        aggregate_transferability_evidence({
            "label_swap": {"status": "pass", "cases": ["fail"]}
        })


def test_missing_mass_prevents_a_formal_pass() -> None:
    config = _elemental_config()
    config["atom_types"][1].pop("mass", None)

    result = assess_same_element_relabeling_invariance(
        config,
        resolved_parameters=_resolved(sigma_corner=2.3, sigma_body=2.3),
    )

    assert result["formal_invariance_status"] == "not_evaluated"
    assert result["missing_type_attributes"] == ["Fe_body.mass"]


def test_empirical_execution_error_is_incomplete_not_nontransferable() -> None:
    report = assess_model_adequacy(
        _elemental_config(),
        resolved_parameters=_resolved(sigma_corner=2.3, sigma_body=2.3),
        transferability_evidence={
            "label_swap": "pass",
            "surface_registry": "pass",
            "vacancy": "error",
            "migration_path_sensitivity": "pass",
        },
    )

    assert report["empirical_transfer_status"] == "error"
    assert report["physical_transferability"]["status"] == "incomplete"
    assert report["elemental_transferability_claim"] == "not_established"


@pytest.mark.parametrize(
    "declaration",
    [
        {"lammps": {"pair_style": "eam"}},
        {"lammps": {"pair_style": "lj/cut/coul/long"}},
        {
            "lammps": {"pair_style": "lj/cut"},
            "pair_params": {"pair_style": "eam"},
        },
    ],
)
def test_formal_pass_is_impossible_for_unsupported_or_conflicting_pair_styles(
    declaration: dict,
) -> None:
    config = _elemental_config()
    config["lammps"] = declaration["lammps"]
    if "pair_params" in declaration:
        config["pair_params"]["pair_style"] = declaration["pair_params"][
            "pair_style"
        ]
    else:
        config["pair_params"].pop("pair_style", None)
    config["material"].pop("force_field", None)

    result = assess_same_element_relabeling_invariance(
        config,
        resolved_parameters=_resolved(sigma_corner=2.3, sigma_body=2.3),
    )

    assert result["formal_invariance_status"] == "not_evaluated"
    assert result["pair_matrix_complete"] is False
    assert result["pair_coefficients"] == []


def test_charged_lj_requires_every_final_resolved_charge() -> None:
    config = _elemental_config()
    config["charge"]["enabled"] = True
    resolved = _resolved(sigma_corner=2.3, sigma_body=2.3)
    resolved.pop("Fe_body_charge")

    result = assess_same_element_relabeling_invariance(
        config, resolved_parameters=resolved
    )

    assert result["formal_invariance_status"] == "not_evaluated"
    assert result["missing_parameters"] == ["Fe_body_charge"]
    assert result["charge_enabled"] is True


def test_missing_charge_policy_does_not_silently_assume_zero() -> None:
    config = _elemental_config()
    config.pop("charge")
    resolved = _resolved(sigma_corner=2.3, sigma_body=2.3)
    resolved.pop("Fe_corner_charge")
    resolved.pop("Fe_body_charge")

    result = assess_same_element_relabeling_invariance(
        config, resolved_parameters=resolved
    )

    assert result["formal_invariance_status"] == "not_evaluated"
    assert result["charge_enabled"] is None
    assert result["missing_parameters"] == [
        "Fe_body_charge",
        "Fe_corner_charge",
    ]


def test_naked_passes_and_too_few_cases_cannot_support_transferability() -> None:
    naked = assess_model_adequacy(
        _elemental_config(),
        resolved_parameters=_resolved(sigma_corner=2.3, sigma_body=2.3),
        transferability_evidence={test_id: "pass" for test_id in (
            "label_swap",
            "surface_registry",
            "vacancy",
            "migration_path_sensitivity",
        )},
    )
    assert naked["empirical_transfer_status"] == "error"
    assert naked["elemental_transferability_claim"] == "not_established"
    assert all(
        record["verification_status"] == "rejected_unverified_pass"
        for record in naked["transferability_evidence"]["tests"].values()
    )

    too_few = aggregate_transferability_evidence({
        "surface_registry": _attested_record("surface_registry", 1)
    })
    surface = too_few["tests"]["surface_registry"]
    assert surface["status"] == "error"
    assert surface["minimum_cases"] == 2
    assert "at least 2" in surface["verification_errors"][0]

    bad_provenance = _attested_record("vacancy", 2)
    bad_provenance["provenance_attestation"]["producer"] = "manual-claim"
    bad_provenance["provenance_attestation"]["metrics_sha256"] = "not-a-hash"
    rejected = aggregate_transferability_evidence({
        "vacancy": bad_provenance
    })["tests"]["vacancy"]
    assert rejected["status"] == "error"
    assert rejected["provenance_attestation"] is None
    assert len(rejected["verification_errors"]) == 2
