"""Material-independent cubic elasticity calculations.

The functions in this module deliberately stop at pure numerical reduction:
they do not know about a particular element, LAMMPS input, scheduler, or
workflow stage.  Elastic constants and stress/energy densities are expressed
in GPa, while every strain component is dimensionless.  This makes the same
audited implementation usable for static screening and finite-temperature
validation.

The stress convention is always explicit.  ``tension_positive`` is the usual
continuum-mechanics convention; ``compression_positive`` supports pressure
components reported by LAMMPS after applying the caller-supplied conversion
factor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from .constraint_objectives import minimax_relative_errors


_EPS = 1.0e-14
_ENERGY_MODES = ("hydro", "orthorhombic", "shear")
_STRESS_DIRECTIONS = ("x", "y", "z", "xy", "xz", "yz")
_RELEVANT_RESPONSES: dict[str, tuple[tuple[str, str], ...]] = {
    "x": (("sxx", "C11"), ("syy", "C12"), ("szz", "C12")),
    "y": (("syy", "C11"), ("sxx", "C12"), ("szz", "C12")),
    "z": (("szz", "C11"), ("sxx", "C12"), ("syy", "C12")),
    "xy": (("sxy", "C44"),),
    "xz": (("sxz", "C44"),),
    "yz": (("syz", "C44"),),
}


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= _EPS:
        return None
    result = numerator / denominator
    return float(result) if math.isfinite(result) else None


def _r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - observed.mean()) ** 2))
    if total <= _EPS:
        return 1.0 if residual <= _EPS else 0.0
    return float(1.0 - residual / total)


def _unique_magnitudes(values: Sequence[float]) -> list[float]:
    unique: list[float] = []
    for value in sorted(abs(item) for item in values if abs(item) > _EPS):
        if not unique or not math.isclose(value, unique[-1], rel_tol=1.0e-9, abs_tol=1.0e-14):
            unique.append(value)
    return unique


def _mean_at(
    strains: np.ndarray,
    responses: np.ndarray,
    target: float,
) -> float | None:
    mask = np.isclose(strains, target, rtol=1.0e-9, atol=1.0e-14)
    if not np.any(mask):
        return None
    return float(np.mean(responses[mask]))


def derive_cubic_properties(
    c11_gpa: Any,
    c12_gpa: Any,
    c44_gpa: Any,
) -> dict[str, Any]:
    """Derive cubic and isotropic-Hill properties from ``C11/C12/C44``.

    The Born conditions for an unstressed cubic crystal are reported as
    signed margins, rather than collapsed into a boolean alone.  A positive
    margin lies inside the mechanically stable region.  Singular derived
    ratios are returned as ``None`` and make the result ineligible instead of
    leaking a NaN/Inf into a ranking table.
    """
    c11 = _finite_float(c11_gpa, "C11 (GPa)")
    c12 = _finite_float(c12_gpa, "C12 (GPa)")
    c44 = _finite_float(c44_gpa, "C44 (GPa)")

    bulk = (c11 + 2.0 * c12) / 3.0
    cprime = 0.5 * (c11 - c12)
    shear_voigt = (2.0 * cprime + 3.0 * c44) / 5.0
    shear_reuss = _safe_ratio(5.0 * cprime * c44, 2.0 * c44 + 3.0 * cprime)
    shear_hill = None if shear_reuss is None else 0.5 * (shear_voigt + shear_reuss)
    young_hill = (
        None
        if shear_hill is None
        else _safe_ratio(9.0 * bulk * shear_hill, 3.0 * bulk + shear_hill)
    )
    poisson_hill = (
        None
        if shear_hill is None
        else _safe_ratio(3.0 * bulk - 2.0 * shear_hill, 2.0 * (3.0 * bulk + shear_hill))
    )
    zener = _safe_ratio(c44, cprime)

    born_margins = {
        "C11_minus_C12": c11 - c12,
        "C11_plus_2C12": c11 + 2.0 * c12,
        "C44": c44,
    }
    scale = max(abs(c11), abs(c12), abs(c44), _EPS)
    normalized_margins = {name: value / scale for name, value in born_margins.items()}
    born_stable = all(value > 0.0 for value in born_margins.values())
    derived_ratios = (shear_reuss, shear_hill, young_hill, poisson_hill, zener)
    all_derived_finite = all(value is not None and math.isfinite(value) for value in derived_ratios)

    return {
        "units": {"elastic_moduli": "GPa", "strain": "dimensionless"},
        "elastic_constants_gpa": {"C11": c11, "C12": c12, "C44": c44},
        "independent_targets_gpa": {"B": bulk, "Cprime": cprime, "C44": c44},
        "polycrystalline_moduli_gpa": {
            "B": bulk,
            "G_voigt": shear_voigt,
            "G_reuss": shear_reuss,
            "G_hill": shear_hill,
            "E_hill": young_hill,
        },
        "nu_hill": poisson_hill,
        "zener_anisotropy": zener,
        "cauchy_pressure_gpa": c12 - c44,
        "born_stability": {
            "stable": born_stable,
            "margins_gpa": born_margins,
            "minimum_margin_gpa": min(born_margins.values()),
            "normalized_margins": normalized_margins,
            "minimum_normalized_margin": min(normalized_margins.values()),
        },
        "all_derived_finite": all_derived_finite,
        "eligible": bool(born_stable and all_derived_finite),
    }


def fit_cubic_energy_strain(
    records: Sequence[Mapping[str, Any]],
    *,
    reference_energy_density_gpa: Any,
) -> dict[str, Any]:
    """Fit cubic elasticity from symmetric three-mode energy-density data.

    Each record must provide ``mode``, dimensionless ``strain``, and
    ``energy_density_gpa``.  Two or more symmetric non-zero strain magnitudes
    are required for every mode.  Pair averaging cancels the odd response,
    while ``A*strain**2 + Q*strain**4`` separates zero-strain curvature from
    the leading finite-strain contribution.

    The mode normalizations match the conventional three-mode protocol:
    ``A_hydro = 9 B / 2``, ``A_orthorhombic = 2 Cprime``, and
    ``A_shear = C44 / 2``.
    """
    reference = _finite_float(
        reference_energy_density_gpa,
        "reference_energy_density_gpa",
    )
    if not records:
        raise ValueError("Energy-strain records must not be empty")

    parsed: list[tuple[str, float, float]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Energy-strain record {index} must be a mapping")
        mode = str(record.get("mode", ""))
        if mode not in _ENERGY_MODES:
            raise ValueError(f"Energy-strain record {index} has unsupported mode {mode!r}")
        strain = _finite_float(record.get("strain"), f"record {index} strain")
        energy = _finite_float(
            record.get("energy_density_gpa"),
            f"record {index} energy_density_gpa",
        )
        parsed.append((mode, strain, energy))

    fits: list[dict[str, Any]] = []
    quadratic: dict[str, float] = {}
    for mode in _ENERGY_MODES:
        subset = [(strain, energy) for item_mode, strain, energy in parsed if item_mode == mode]
        if not subset:
            raise ValueError(f"Energy-strain dataset is missing mode {mode}")
        strains = np.asarray([item[0] for item in subset], dtype=float)
        energies = np.asarray([item[1] for item in subset], dtype=float)
        magnitudes = _unique_magnitudes(strains.tolist())
        if len(magnitudes) < 2:
            raise ValueError(f"Mode {mode} needs at least two symmetric non-zero strain magnitudes")

        even_values: list[float] = []
        odd_values: list[float] = []
        for magnitude in magnitudes:
            positive = _mean_at(strains, energies, magnitude)
            negative = _mean_at(strains, energies, -magnitude)
            if positive is None or negative is None:
                raise ValueError(f"Mode {mode} is missing a symmetric pair at {magnitude}")
            even_values.append(0.5 * (positive + negative) - reference)
            odd_values.append(0.5 * (positive - negative))

        magnitude_array = np.asarray(magnitudes, dtype=float)
        even_array = np.asarray(even_values, dtype=float)
        strain_scale = float(np.max(magnitude_array))
        scaled_square = (magnitude_array / strain_scale) ** 2
        design = np.column_stack((scaled_square, scaled_square**2))
        if int(np.linalg.matrix_rank(design)) < 2:
            raise ValueError(f"Mode {mode} energy fit is underdetermined")
        coefficients, _, _, singular_values = np.linalg.lstsq(design, even_array, rcond=None)
        predicted = design @ coefficients
        quadratic_value = float(coefficients[0] / strain_scale**2)
        quartic_value = float(coefficients[1] / strain_scale**4)
        if not all(math.isfinite(value) for value in (quadratic_value, quartic_value)):
            raise ValueError(f"Mode {mode} energy fit produced non-finite coefficients")
        quadratic[mode] = quadratic_value
        fits.append(
            {
                "mode": mode,
                "quadratic_energy_density_gpa": quadratic_value,
                "quartic_energy_density_gpa": quartic_value,
                "r2": _r2_score(even_array, predicted),
                "rmse_energy_density_gpa": float(math.sqrt(np.mean((even_array - predicted) ** 2))),
                "maximum_absolute_odd_energy_density_gpa": float(
                    max(abs(value) for value in odd_values)
                ),
                "n_raw_samples": len(subset),
                "n_symmetric_magnitudes": len(magnitudes),
                "degrees_of_freedom": len(magnitudes) - 2,
                "fit_rank": 2,
                "scaled_design_condition_number": float(singular_values[0] / singular_values[-1]),
            }
        )

    bulk = (2.0 / 9.0) * quadratic["hydro"]
    cprime = 0.5 * quadratic["orthorhombic"]
    c44 = 2.0 * quadratic["shear"]
    c11 = bulk + 4.0 * cprime / 3.0
    c12 = bulk - 2.0 * cprime / 3.0
    elasticity = derive_cubic_properties(c11, c12, c44)
    return {
        "method": "symmetric three-mode energy-density curvature",
        "units": {
            "energy_density": "GPa",
            "elastic_moduli": "GPa",
            "strain": "dimensionless",
        },
        "reference_energy_density_gpa": reference,
        "elasticity": elasticity,
        "fits": fits,
        "fit_quality": {
            "minimum_r2": min(item["r2"] for item in fits),
            "maximum_rmse_energy_density_gpa": max(
                item["rmse_energy_density_gpa"] for item in fits
            ),
            "maximum_absolute_odd_energy_density_gpa": max(
                item["maximum_absolute_odd_energy_density_gpa"] for item in fits
            ),
        },
    }


def _linear_fit(strain: np.ndarray, response_gpa: np.ndarray) -> dict[str, Any]:
    unique_strain = np.unique(strain)
    if len(unique_strain) < 3:
        raise ValueError("Stress fit needs at least three distinct strain values")
    scale = float(np.max(np.abs(unique_strain)))
    if scale <= _EPS:
        raise ValueError("Stress fit strain range is zero")
    design = np.column_stack((strain / scale, np.ones_like(strain)))
    if int(np.linalg.matrix_rank(design)) < 2:
        raise ValueError("Stress fit is underdetermined")
    coefficients, _, _, singular_values = np.linalg.lstsq(design, response_gpa, rcond=None)
    predicted = design @ coefficients
    slope = float(coefficients[0] / scale)
    intercept = float(coefficients[1])
    if not all(math.isfinite(value) for value in (slope, intercept)):
        raise ValueError("Stress fit produced non-finite coefficients")
    return {
        "slope_gpa": slope,
        "intercept_gpa": intercept,
        "r2": _r2_score(response_gpa, predicted),
        "rmse_gpa": float(math.sqrt(np.mean((response_gpa - predicted) ** 2))),
        "fit_rank": 2,
        "degrees_of_freedom": int(len(strain) - 2),
        "scaled_design_condition_number": float(singular_values[0] / singular_values[-1]),
    }


def fit_cubic_stress_strain(
    records: Sequence[Mapping[str, Any]],
    *,
    stress_convention: str = "tension_positive",
    stress_to_gpa: Any = 1.0,
) -> dict[str, Any]:
    """Fit ``C11/C12/C44`` from noisy directional stress-strain samples.

    Records use the six directions ``x/y/z/xy/xz/yz`` and stress components
    ``sxx/syy/szz/sxy/sxz/syz``.  A shared record with ``direction='zero'``
    may supply the unstrained baseline, matching finite-temperature sampling
    workflows.  Otherwise every direction must contain its own zero-strain
    record.  At least one symmetric pair and three distinct strain values are
    required per direction.

    ``stress_to_gpa`` converts each input stress number to GPa.  Set
    ``stress_convention='compression_positive'`` for LAMMPS pressure output;
    the sign is then reversed before fitting Hooke's law.
    """
    if stress_convention not in {"tension_positive", "compression_positive"}:
        raise ValueError("stress_convention must be 'tension_positive' or 'compression_positive'")
    conversion = _finite_float(stress_to_gpa, "stress_to_gpa")
    if conversion <= 0.0:
        raise ValueError("stress_to_gpa must be positive")
    if not records:
        raise ValueError("Stress-strain records must not be empty")
    physical_sign = 1.0 if stress_convention == "tension_positive" else -1.0

    parsed: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Stress-strain record {index} must be a mapping")
        direction = str(record.get("direction", ""))
        if direction not in {*_STRESS_DIRECTIONS, "zero"}:
            raise ValueError(
                f"Stress-strain record {index} has unsupported direction {direction!r}"
            )
        strain = _finite_float(record.get("strain", 0.0), f"record {index} strain")
        parsed.append({"direction": direction, "strain": strain, "record": record})

    zero_records = [item for item in parsed if item["direction"] == "zero"]
    shared_zero: dict[str, float] = {}
    if zero_records:
        for component in ("sxx", "syy", "szz", "sxy", "sxz", "syz"):
            shared_zero[component] = float(
                np.mean(
                    [
                        _finite_float(
                            item["record"].get(component),
                            f"shared-zero {component}",
                        )
                        for item in zero_records
                    ]
                )
            )

    fits: list[dict[str, Any]] = []
    constant_samples: dict[str, list[float]] = {"C11": [], "C12": [], "C44": []}
    for direction in _STRESS_DIRECTIONS:
        direction_records = [item for item in parsed if item["direction"] == direction]
        if not direction_records:
            raise ValueError(f"Stress-strain dataset is missing direction {direction}")

        raw_strains = [float(item["strain"]) for item in direction_records]
        if not any(abs(value) <= _EPS for value in raw_strains):
            if not shared_zero:
                raise ValueError(
                    f"Direction {direction} needs a zero-strain record or shared zero state"
                )
            direction_records = [
                *direction_records,
                {"direction": direction, "strain": 0.0, "record": shared_zero},
            ]

        direction_strains = np.asarray(
            [float(item["strain"]) for item in direction_records], dtype=float
        )
        magnitudes = _unique_magnitudes(direction_strains.tolist())
        paired_magnitudes = [
            magnitude
            for magnitude in magnitudes
            if np.any(np.isclose(direction_strains, magnitude, rtol=1.0e-9, atol=1.0e-14))
            and np.any(np.isclose(direction_strains, -magnitude, rtol=1.0e-9, atol=1.0e-14))
        ]
        if not paired_magnitudes:
            raise ValueError(f"Direction {direction} needs at least one symmetric strain pair")

        for component, target_constant in _RELEVANT_RESPONSES[direction]:
            raw_response = np.asarray(
                [
                    _finite_float(
                        item["record"].get(component),
                        f"direction {direction} {component}",
                    )
                    for item in direction_records
                ],
                dtype=float,
            )
            response_gpa = physical_sign * conversion * raw_response
            fit = _linear_fit(direction_strains, response_gpa)
            central_differences: list[float] = []
            for magnitude in paired_magnitudes:
                positive = _mean_at(direction_strains, response_gpa, magnitude)
                negative = _mean_at(direction_strains, response_gpa, -magnitude)
                assert positive is not None and negative is not None
                central_differences.append((positive - negative) / (2.0 * magnitude))
            fit.update(
                {
                    "direction": direction,
                    "stress_component": component,
                    "target_constant": target_constant,
                    "n_samples": len(direction_records),
                    "n_symmetric_magnitudes": len(paired_magnitudes),
                    "central_difference_values_gpa": central_differences,
                    "central_difference_mean_gpa": float(np.mean(central_differences)),
                    "central_difference_std_gpa": (
                        float(np.std(central_differences, ddof=1))
                        if len(central_differences) > 1
                        else 0.0
                    ),
                }
            )
            fits.append(fit)
            constant_samples[target_constant].append(fit["slope_gpa"])

    constant_statistics: dict[str, dict[str, Any]] = {}
    for name, values in constant_samples.items():
        array = np.asarray(values, dtype=float)
        if not len(array) or not np.all(np.isfinite(array)):
            raise ValueError(f"Stress fit produced no finite {name} estimates")
        constant_statistics[name] = {
            "mean_gpa": float(np.mean(array)),
            "std_gpa": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
            "values_gpa": [float(value) for value in array],
        }

    elasticity = derive_cubic_properties(
        constant_statistics["C11"]["mean_gpa"],
        constant_statistics["C12"]["mean_gpa"],
        constant_statistics["C44"]["mean_gpa"],
    )
    return {
        "method": "directional cubic stress-strain least squares",
        "units": {
            "input_stress": "caller-defined",
            "elastic_moduli": "GPa",
            "strain": "dimensionless",
        },
        "stress_convention": stress_convention,
        "stress_to_gpa": conversion,
        "elasticity": elasticity,
        "elastic_constant_statistics_gpa": constant_statistics,
        "fits": fits,
        "fit_quality": {
            "minimum_relevant_r2": min(item["r2"] for item in fits),
            "maximum_rmse_gpa": max(item["rmse_gpa"] for item in fits),
            "maximum_central_difference_std_gpa": max(
                item["central_difference_std_gpa"] for item in fits
            ),
        },
    }


_TARGET_ALIASES = {
    "B": "B",
    "bulk": "B",
    "K": "B",
    "Cprime": "Cprime",
    "cprime": "Cprime",
    "C'": "Cprime",
    "C44": "C44",
    "c44": "C44",
}


def cubic_minimax_report(
    elasticity_or_fit: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an auditable minimax report for independent ``B/Cprime/C44``.

    Target aliases ``bulk``/``K`` and ``C'`` are accepted at the boundary,
    but the report is canonicalized.  All three independent targets are
    required and must be finite and non-zero.  Absolute relative errors are
    delegated to :func:`engine.constraint_objectives.minimax_relative_errors`
    so ranking semantics remain shared across material workflows.
    """
    if not isinstance(elasticity_or_fit, Mapping):
        raise ValueError("elasticity_or_fit must be a mapping")
    elasticity = elasticity_or_fit.get("elasticity", elasticity_or_fit)
    if not isinstance(elasticity, Mapping):
        raise ValueError("elasticity result is missing")
    values = elasticity.get("independent_targets_gpa")
    if not isinstance(values, Mapping):
        raise ValueError("elasticity result is missing independent_targets_gpa")

    canonical_targets: dict[str, dict[str, float]] = {}
    for raw_name, specification in targets.items():
        canonical = _TARGET_ALIASES.get(str(raw_name))
        if canonical is None:
            raise ValueError(f"Unsupported cubic target {raw_name!r}")
        if canonical in canonical_targets:
            raise ValueError(f"Duplicate cubic target alias for {canonical}")
        raw_value = (
            specification.get("value") if isinstance(specification, Mapping) else specification
        )
        value = _finite_float(raw_value, f"target {raw_name}")
        if abs(value) <= _EPS:
            raise ValueError(f"target {raw_name} must be non-zero for relative minimax scoring")
        canonical_targets[canonical] = {"value": value}
    required = {"B", "Cprime", "C44"}
    if set(canonical_targets) != required:
        missing = ", ".join(sorted(required - set(canonical_targets)))
        extra = ", ".join(sorted(set(canonical_targets) - required))
        detail = "; ".join(
            item
            for item in (f"missing {missing}" if missing else "", f"extra {extra}" if extra else "")
            if item
        )
        raise ValueError(f"Cubic minimax targets must be exactly B, Cprime, C44 ({detail})")

    finite_values: dict[str, float] = {}
    for name in ("B", "Cprime", "C44"):
        finite_values[name] = _finite_float(values.get(name), f"calculated {name}")
    report = minimax_relative_errors(finite_values, canonical_targets)
    signed_errors = {
        name: 100.0
        * (finite_values[name] - canonical_targets[name]["value"])
        / abs(canonical_targets[name]["value"])
        for name in ("B", "Cprime", "C44")
    }
    born = elasticity.get("born_stability", {})
    born_stable = bool(born.get("stable", False)) if isinstance(born, Mapping) else False
    finite_score = math.isfinite(float(report["maximum_error_percent"]))
    return {
        **report,
        "units": {"targets": "GPa", "errors": "percent"},
        "values_gpa": finite_values,
        "targets_gpa": {
            name: specification["value"] for name, specification in canonical_targets.items()
        },
        "signed_errors_percent": signed_errors,
        "born_stable": born_stable,
        "eligible": bool(finite_score and born_stable),
    }
