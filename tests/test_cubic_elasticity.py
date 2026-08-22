from __future__ import annotations

import math

import pytest

from engine.cubic_elasticity import (
    cubic_minimax_report,
    derive_cubic_properties,
    fit_cubic_energy_strain,
    fit_cubic_static_stress_strain,
    fit_cubic_stress_strain,
)


def test_derive_cubic_properties_and_born_margins() -> None:
    result = derive_cubic_properties(243.1, 138.1, 121.9)

    assert result["independent_targets_gpa"] == pytest.approx(
        {"B": 173.1, "Cprime": 52.5, "C44": 121.9}
    )
    assert result["polycrystalline_moduli_gpa"]["G_voigt"] == pytest.approx(94.14)
    assert result["zener_anisotropy"] == pytest.approx(121.9 / 52.5)
    assert result["born_stability"]["margins_gpa"] == pytest.approx(
        {"C11_minus_C12": 105.0, "C11_plus_2C12": 519.3, "C44": 121.9}
    )
    assert result["born_stability"]["minimum_margin_gpa"] == pytest.approx(105.0)
    assert result["born_stability"]["stable"] is True
    assert result["eligible"] is True
    assert result["units"]["elastic_moduli"] == "GPa"
    assert result["units"]["strain"] == "dimensionless"


def test_derive_reports_unstable_and_singular_result_fail_closed() -> None:
    unstable = derive_cubic_properties(100.0, 120.0, 50.0)
    assert unstable["born_stability"]["stable"] is False
    assert unstable["born_stability"]["minimum_margin_gpa"] < 0.0
    assert unstable["eligible"] is False

    singular = derive_cubic_properties(100.0, 100.0, 0.0)
    assert singular["zener_anisotropy"] is None
    assert singular["all_derived_finite"] is False
    assert singular["eligible"] is False


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "not-a-number"])
def test_derive_rejects_nonfinite_inputs(bad: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        derive_cubic_properties(bad, 100.0, 50.0)


def _energy_records(
    *,
    bulk: float = 170.0,
    cprime: float = 50.0,
    c44: float = 120.0,
) -> list[dict[str, float | str]]:
    quadratic = {
        "hydro": 4.5 * bulk,
        "orthorhombic": 2.0 * cprime,
        "shear": 0.5 * c44,
    }
    quartic = {"hydro": 2.0e5, "orthorhombic": -1.0e4, "shear": 3.0e4}
    odd = {"hydro": 300.0, "orthorhombic": -80.0, "shear": 40.0}
    reference = -12.5
    records: list[dict[str, float | str]] = []
    for mode in ("hydro", "orthorhombic", "shear"):
        for magnitude in (0.005, 0.010, 0.015):
            for sign in (-1.0, 1.0):
                strain = sign * magnitude
                energy = (
                    reference
                    + quadratic[mode] * strain**2
                    + quartic[mode] * strain**4
                    + odd[mode] * strain**3
                )
                records.append({"mode": mode, "strain": strain, "energy_density_gpa": energy})
    return records


def test_energy_strain_fit_recovers_constants_and_audits_odd_terms() -> None:
    result = fit_cubic_energy_strain(_energy_records(), reference_energy_density_gpa=-12.5)

    constants = result["elasticity"]["elastic_constants_gpa"]
    assert constants == pytest.approx({"C11": 236.6666666667, "C12": 136.6666666667, "C44": 120.0})
    assert result["elasticity"]["independent_targets_gpa"] == pytest.approx(
        {"B": 170.0, "Cprime": 50.0, "C44": 120.0}
    )
    assert result["fit_quality"]["minimum_r2"] == pytest.approx(1.0)
    assert result["fit_quality"]["maximum_absolute_odd_energy_density_gpa"] > 0.0
    assert all(item["fit_rank"] == 2 for item in result["fits"])
    assert all(item["degrees_of_freedom"] == 1 for item in result["fits"])


def test_energy_strain_fit_rejects_missing_pair_and_underdetermined_mode() -> None:
    records = _energy_records()
    missing_pair = [
        row
        for row in records
        if not (row["mode"] == "shear" and math.isclose(float(row["strain"]), -0.015))
    ]
    with pytest.raises(ValueError, match="missing a symmetric pair"):
        fit_cubic_energy_strain(missing_pair, reference_energy_density_gpa=-12.5)

    one_shell = [
        row
        for row in records
        if row["mode"] != "hydro" or math.isclose(abs(float(row["strain"])), 0.005)
    ]
    with pytest.raises(ValueError, match="at least two"):
        fit_cubic_energy_strain(one_shell, reference_energy_density_gpa=-12.5)


def _stress_records(
    *,
    c11: float = 240.0,
    c12: float = 140.0,
    c44: float = 120.0,
    compression_positive: bool = False,
) -> list[dict[str, float | str]]:
    baseline = {
        "sxx": 0.4,
        "syy": -0.2,
        "szz": 0.1,
        "sxy": 0.05,
        "sxz": -0.03,
        "syz": 0.02,
    }
    sign = -1.0 if compression_positive else 1.0
    records: list[dict[str, float | str]] = [
        {
            "direction": "zero",
            "strain": 0.0,
            **{key: sign * value for key, value in baseline.items()},
        }
    ]
    slopes = {
        "x": {"sxx": c11, "syy": c12, "szz": c12},
        "y": {"syy": c11, "sxx": c12, "szz": c12},
        "z": {"szz": c11, "sxx": c12, "syy": c12},
        "xy": {"sxy": c44},
        "xz": {"sxz": c44},
        "yz": {"syz": c44},
    }
    for direction, responses in slopes.items():
        for strain in (-0.003, -0.001, 0.001, 0.003):
            row: dict[str, float | str] = {"direction": direction, "strain": strain}
            for component, slope in responses.items():
                row[component] = sign * (baseline[component] + slope * strain)
            records.append(row)
    return records


@pytest.mark.parametrize(
    ("compression_positive", "convention"),
    [(False, "tension_positive"), (True, "compression_positive")],
)
def test_stress_strain_fit_recovers_cubic_constants_for_explicit_convention(
    compression_positive: bool,
    convention: str,
) -> None:
    result = fit_cubic_stress_strain(
        _stress_records(compression_positive=compression_positive),
        stress_convention=convention,
    )

    constants = result["elasticity"]["elastic_constants_gpa"]
    assert constants == pytest.approx({"C11": 240.0, "C12": 140.0, "C44": 120.0})
    assert result["elasticity"]["independent_targets_gpa"] == pytest.approx(
        {"B": 173.3333333333, "Cprime": 50.0, "C44": 120.0}
    )
    assert result["fit_quality"]["minimum_relevant_r2"] == pytest.approx(1.0)
    assert len(result["fits"]) == 12
    assert all(item["n_symmetric_magnitudes"] == 2 for item in result["fits"])


def test_stress_strain_fit_retains_noisy_fit_diagnostics() -> None:
    records = _stress_records()
    for row in records:
        if row["direction"] == "x":
            strain = float(row["strain"])
            row["sxx"] = float(row["sxx"]) + 1000.0 * strain**2

    result = fit_cubic_stress_strain(records)
    x_c11 = next(
        item
        for item in result["fits"]
        if item["direction"] == "x" and item["stress_component"] == "sxx"
    )
    assert x_c11["slope_gpa"] == pytest.approx(240.0)
    assert 0.0 < x_c11["r2"] < 1.0
    assert x_c11["rmse_gpa"] > 0.0
    assert x_c11["degrees_of_freedom"] == 3
    assert result["fit_quality"]["minimum_relevant_r2"] == x_c11["r2"]


def test_stress_strain_fit_rejects_missing_direction_and_nonfinite_response() -> None:
    records = [row for row in _stress_records() if row["direction"] != "yz"]
    with pytest.raises(ValueError, match="missing direction yz"):
        fit_cubic_stress_strain(records)

    records = _stress_records()
    for row in records:
        if row["direction"] == "x":
            row["sxx"] = math.nan
            break
    with pytest.raises(ValueError, match="finite"):
        fit_cubic_stress_strain(records)


def test_stress_strain_fit_rejects_asymmetric_or_too_small_design() -> None:
    records = _stress_records()
    records = [row for row in records if row["direction"] != "xy" or float(row["strain"]) > 0.0]
    with pytest.raises(ValueError, match="symmetric strain pair"):
        fit_cubic_stress_strain(records)


def _static_stress_records(
    *,
    bulk: float = 170.0,
    cprime: float = 50.0,
    c44: float = 120.0,
    stress_to_gpa: float = 1.0e-4,
) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    residual_gpa = {"pxx": 0.30, "pyy": -0.10, "pzz": 0.15, "pxy": 0.04}
    zero_state: dict[str, float | str] = {
        "mode": "zero",
        "strain": 0.0,
        **{name: value / stress_to_gpa for name, value in residual_gpa.items()},
    }
    records: list[dict[str, float | str]] = []
    for mode in ("hydro", "orthorhombic", "shear"):
        for strain in (-0.010, -0.007, -0.004, 0.004, 0.007, 0.010):
            pressure = dict(residual_gpa)
            if mode == "hydro":
                for component in ("pxx", "pyy", "pzz"):
                    pressure[component] += -3.0 * bulk * strain
            elif mode == "orthorhombic":
                pressure["pxx"] += -2.0 * cprime * strain
                pressure["pyy"] += 2.0 * cprime * strain
            else:
                pressure["pxy"] += -c44 * strain
            records.append(
                {
                    "mode": mode,
                    "strain": strain,
                    **{name: value / stress_to_gpa for name, value in pressure.items()},
                }
            )
    return records, zero_state


def test_static_three_mode_stress_fit_recovers_exact_moduli_and_residual() -> None:
    records, zero_state = _static_stress_records()
    pressure_only_zero = {
        component: zero_state[component] for component in ("pxx", "pyy", "pzz", "pxy")
    }

    result = fit_cubic_static_stress_strain(
        records,
        zero_state=pressure_only_zero,
        stress_convention="compression_positive",
        stress_to_gpa=1.0e-4,
    )

    assert result["elasticity"]["independent_targets_gpa"] == pytest.approx(
        {"B": 170.0, "Cprime": 50.0, "C44": 120.0}
    )
    assert result["elasticity"]["elastic_constants_gpa"] == pytest.approx(
        {"C11": 236.6666666667, "C12": 136.6666666667, "C44": 120.0}
    )
    assert result["zero_state"]["residual_pressure_compression_positive_gpa"] == (
        pytest.approx({"pxx": 0.30, "pyy": -0.10, "pzz": 0.15, "pxy": 0.04})
    )
    assert result["fit_quality"]["minimum_r2"] == pytest.approx(1.0)
    assert result["fit_quality"]["maximum_rmse_gpa"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["fit_quality"]["maximum_central_difference_std_gpa"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    expected_intercepts = {
        "hydro": (0.30 - 0.10 + 0.15) / 3.0,
        "orthorhombic": 0.30 - (-0.10),
        "shear": 0.04,
    }
    assert {item["mode"]: item["intercept_gpa"] for item in result["fits"]} == (
        pytest.approx(expected_intercepts)
    )
    assert all(item["n_symmetric_magnitudes"] == 3 for item in result["fits"])
    assert all(
        item["zero_strain_extrapolation"]["canonical_strain_magnitudes"]
        == pytest.approx([0.004, 0.007])
        for item in result["fits"]
    )


def test_static_three_mode_stress_fit_cancels_even_terms() -> None:
    records, zero_state = _static_stress_records(stress_to_gpa=1.0)
    even_coefficients = {"hydro": 9000.0, "orthorhombic": -3000.0, "shear": 5000.0}
    for row in records:
        strain = float(row["strain"])
        mode = str(row["mode"])
        even = even_coefficients[mode] * strain**2
        if mode == "hydro":
            for component in ("pxx", "pyy", "pzz"):
                row[component] = float(row[component]) + even
        elif mode == "orthorhombic":
            row["pxx"] = float(row["pxx"]) + 0.5 * even
            row["pyy"] = float(row["pyy"]) - 0.5 * even
        else:
            row["pxy"] = float(row["pxy"]) + even

    result = fit_cubic_static_stress_strain(
        records,
        zero_state=zero_state,
        stress_to_gpa=1.0,
    )

    assert result["elasticity"]["independent_targets_gpa"] == pytest.approx(
        {"B": 170.0, "Cprime": 50.0, "C44": 120.0}
    )
    assert result["fit_quality"]["minimum_r2"] < 1.0
    assert result["fit_quality"]["maximum_rmse_gpa"] > 0.0
    assert result["fit_quality"]["maximum_central_difference_std_gpa"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert any(abs(item["intercept_minus_zero_state_gpa"]) > 0.0 for item in result["fits"])


def test_static_three_mode_stress_fit_reports_central_difference_noise() -> None:
    records, zero_state = _static_stress_records(stress_to_gpa=1.0)
    noisy = next(
        row
        for row in records
        if row["mode"] == "shear" and math.isclose(float(row["strain"]), 0.010)
    )
    noisy["pxy"] = float(noisy["pxy"]) + 0.02

    result = fit_cubic_static_stress_strain(
        records,
        zero_state=zero_state,
        stress_to_gpa=1.0,
    )

    shear = next(item for item in result["fits"] if item["mode"] == "shear")
    assert shear["modulus_gpa"] == pytest.approx(120.0)
    assert shear["r2"] < 1.0
    assert shear["rmse_gpa"] > 0.0
    assert shear["central_difference_std_gpa"] > 0.0
    assert result["fit_quality"]["maximum_central_difference_std_gpa"] == pytest.approx(
        shear["central_difference_std_gpa"]
    )
    assert shear["maximum_zero_strain_extrapolation_drift_percent"] > 0.0


def test_static_three_mode_stress_fit_extrapolates_odd_cubic_response_to_zero() -> None:
    records, zero_state = _static_stress_records(stress_to_gpa=1.0)
    cubic_coefficients = {"hydro": 30000.0, "orthorhombic": -15000.0, "shear": 20000.0}
    for row in records:
        strain = float(row["strain"])
        mode = str(row["mode"])
        cubic = cubic_coefficients[mode] * strain**3
        if mode == "hydro":
            for component in ("pxx", "pyy", "pzz"):
                row[component] = float(row[component]) + cubic
        elif mode == "orthorhombic":
            row["pxx"] = float(row["pxx"]) + 0.5 * cubic
            row["pyy"] = float(row["pyy"]) - 0.5 * cubic
        else:
            row["pxy"] = float(row["pxy"]) + cubic

    result = fit_cubic_static_stress_strain(
        records,
        zero_state=zero_state,
        stress_to_gpa=1.0,
    )

    assert result["elasticity"]["independent_targets_gpa"] == pytest.approx(
        {"B": 170.0, "Cprime": 50.0, "C44": 120.0}
    )
    assert any(
        not math.isclose(item["raw_linear_modulus_gpa"], item["modulus_gpa"])
        for item in result["fits"]
    )
    assert result["fit_quality"]["maximum_zero_strain_extrapolation_drift_percent"] == (
        pytest.approx(0.0, abs=1.0e-12)
    )


def test_static_three_mode_stress_fit_rejects_missing_pair_and_nonfinite() -> None:
    records, zero_state = _static_stress_records(stress_to_gpa=1.0)
    missing_pair = [
        row
        for row in records
        if not (row["mode"] == "shear" and math.isclose(float(row["strain"]), -0.010))
    ]
    with pytest.raises(ValueError, match="missing a symmetric pair"):
        fit_cubic_static_stress_strain(
            missing_pair,
            zero_state=zero_state,
            stress_to_gpa=1.0,
        )

    records, zero_state = _static_stress_records(stress_to_gpa=1.0)
    records[0]["pyy"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        fit_cubic_static_stress_strain(
            records,
            zero_state=zero_state,
            stress_to_gpa=1.0,
        )


@pytest.mark.parametrize("bad_conversion", [0.0, -1.0, math.nan, math.inf])
def test_static_three_mode_stress_fit_requires_lammps_convention_and_conversion(
    bad_conversion: float,
) -> None:
    records, zero_state = _static_stress_records(stress_to_gpa=1.0)
    with pytest.raises(ValueError, match="stress_to_gpa"):
        fit_cubic_static_stress_strain(
            records,
            zero_state=zero_state,
            stress_to_gpa=bad_conversion,
        )

    with pytest.raises(ValueError, match="compression_positive"):
        fit_cubic_static_stress_strain(
            records,
            zero_state=zero_state,
            stress_convention="tension_positive",
        )


def test_static_three_mode_stress_fit_requires_explicit_zero_state() -> None:
    records, zero_state = _static_stress_records(stress_to_gpa=1.0)
    bad_zero = dict(zero_state)
    bad_zero["strain"] = 0.001
    with pytest.raises(ValueError, match="zero_state strain must be zero"):
        fit_cubic_static_stress_strain(
            records,
            zero_state=bad_zero,
            stress_to_gpa=1.0,
        )


def test_cubic_minimax_report_uses_independent_targets_and_aliases() -> None:
    elasticity = derive_cubic_properties(240.0, 140.0, 120.0)
    report = cubic_minimax_report(
        elasticity,
        {"bulk": 170.0, "C'": {"value": 55.0}, "C44": 120.0},
    )

    assert report["errors_percent"]["B"] == pytest.approx(100.0 * (173.3333333333 - 170.0) / 170.0)
    assert report["errors_percent"]["Cprime"] == pytest.approx(100.0 * 5.0 / 55.0)
    assert report["maximum_error_percent"] == pytest.approx(100.0 * 5.0 / 55.0)
    assert report["limiting_target"] == "Cprime"
    assert report["signed_errors_percent"]["Cprime"] < 0.0
    assert report["born_stable"] is True
    assert report["eligible"] is True


def test_cubic_minimax_report_fails_closed_for_bad_target_set_or_unstable_model() -> None:
    stable = derive_cubic_properties(240.0, 140.0, 120.0)
    with pytest.raises(ValueError, match="exactly B, Cprime, C44"):
        cubic_minimax_report(stable, {"B": 170.0, "C44": 120.0})
    with pytest.raises(ValueError, match="non-zero"):
        cubic_minimax_report(stable, {"B": 0.0, "Cprime": 50.0, "C44": 120.0})

    unstable = derive_cubic_properties(100.0, 120.0, 50.0)
    report = cubic_minimax_report(
        unstable,
        {"B": 340.0 / 3.0, "Cprime": -10.0, "C44": 50.0},
    )
    assert report["maximum_error_percent"] == pytest.approx(0.0)
    assert report["born_stable"] is False
    assert report["eligible"] is False
