from engine.validate_final_parameters import _assess_acceptance


def _config():
    return {
        "targets": {
            "a": {"value": 4.0, "tolerance": 0.1},
            "density": {"value": 1.0, "tolerance": 0.03},
        },
        "validation": {
            "require_tolerances": True,
            "objective_max": 0.03,
            "max_error_percent": 3.0,
        },
    }


def test_acceptance_passes_all_configured_thresholds():
    result = _assess_acceptance(
        _config(),
        objective=0.01,
        properties={"a": 4.05, "density": 1.01},
        percent_errors={"a": 1.25, "density": 1.0},
    )
    assert result["passed"] is True
    assert result["tolerances"]["a"]["passed"] is True


def test_acceptance_reports_each_failed_rule():
    result = _assess_acceptance(
        _config(),
        objective=0.05,
        properties={"a": 4.2, "density": 1.01},
        percent_errors={"a": 5.0, "density": 1.0},
    )
    assert result["passed"] is False
    assert len(result["reasons"]) == 3
