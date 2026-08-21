from __future__ import annotations

import copy
import json
import math

import pytest

from engine.model_adequacy import ModelAdequacyError, assess_model_adequacy


def _elemental_config() -> dict:
    return {
        "material": {"kind": "elemental", "name": "Fe"},
        "lammps": {"pair_style": "lj/cut"},
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
                "params": {
                    "epsilon": {"min": 0.001, "max": 10.0, "init": 1.0},
                    "sigma": {"min": 0.001, "max": 5.0, "init": 2.3},
                    "charge": 0.0,
                },
            },
            {
                "type": 2,
                "label": "Fe_body",
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
        "surface_terminations",
        "vacancy",
        "diffusion",
    ]
    assert transfer["required_validations"][1]["minimum_cases"] == 2
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
    assert passing["physical_transferability"]["required_validations"][0]["status"] == (
        "pass"
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
