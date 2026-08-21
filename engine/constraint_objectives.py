"""Reusable constraint and minimax scoring primitives.

These functions deliberately keep constraint satisfaction separate from the
objective optimized inside the feasible region.  They contain no material-
specific property names and are shared by elemental, molecular, and future
multicomponent workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence
from typing import Any


_EPS = 1.0e-12


@dataclass(frozen=True)
class ConstraintResult:
    """One observed continuous constraint.

    ``ratio`` is zero at the target and one at the accepted boundary.
    ``margin`` is positive inside the accepted region and negative outside it.
    Missing or non-finite observations fail closed.
    """

    name: str
    observed: float
    target: float
    tolerance: float
    mode: str
    error: float
    ratio: float
    margin: float
    passed: bool


def _finite_float(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return math.nan
    return converted if math.isfinite(converted) else math.nan


def relative_error_percent(observed: Any, target: Any) -> float:
    """Return an absolute relative error in percent, failing closed at zero."""
    value = _finite_float(observed)
    reference = _finite_float(target)
    if not (math.isfinite(value) and math.isfinite(reference)):
        return math.inf
    if abs(reference) <= _EPS:
        return 0.0 if abs(value - reference) <= _EPS else math.inf
    return 100.0 * abs(value - reference) / abs(reference)


def assess_constraint(
    name: str,
    observed: Any,
    target: Any,
    tolerance: Any,
    *,
    mode: str = "relative_percent",
) -> ConstraintResult:
    """Assess a target-centred tolerance without converting it to a penalty.

    Supported modes are ``relative_percent`` and ``absolute``.  The returned
    violation magnitude is represented by ``max(0, ratio - 1)``; all points
    inside the band therefore have zero violation and remain equivalent for
    constraint satisfaction.
    """
    value = _finite_float(observed)
    reference = _finite_float(target)
    limit = _finite_float(tolerance)
    if mode not in {"relative_percent", "absolute"}:
        raise ValueError(f"Unsupported constraint mode: {mode!r}")
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"Constraint {name!r} requires a positive tolerance")

    if mode == "relative_percent":
        error = relative_error_percent(value, reference)
    else:
        error = (
            abs(value - reference)
            if math.isfinite(value) and math.isfinite(reference)
            else math.inf
        )
    ratio = error / limit if math.isfinite(error) else math.inf
    margin = 1.0 - ratio if math.isfinite(ratio) else -math.inf
    return ConstraintResult(
        name=str(name),
        observed=value,
        target=reference,
        tolerance=limit,
        mode=mode,
        error=error,
        ratio=ratio,
        margin=margin,
        passed=ratio <= 1.0 + _EPS,
    )


def summarize_constraints(
    constraints: Sequence[ConstraintResult],
) -> dict[str, Any]:
    """Return a continuous, auditable AND over independent constraints."""
    items = list(constraints)
    if not items:
        raise ValueError("At least one constraint is required")
    finite = all(math.isfinite(item.ratio) for item in items)
    violations = [max(0.0, item.ratio - 1.0) for item in items]
    limiting = max(items, key=lambda item: item.ratio)
    violation = (
        math.sqrt(sum(value * value for value in violations) / len(violations))
        if finite
        else math.inf
    )
    return {
        "feasible": finite and all(item.passed for item in items),
        "constraint_violation": violation,
        "maximum_band_ratio": limiting.ratio if finite else math.inf,
        "minimum_margin": min(item.margin for item in items) if finite else -math.inf,
        "limiting_constraint": limiting.name,
        "failed_constraints": [item.name for item in items if not item.passed],
        "constraints": {
            item.name: {
                "observed": item.observed,
                "target": item.target,
                "tolerance": item.tolerance,
                "mode": item.mode,
                "error": item.error,
                "ratio": item.ratio,
                "margin": item.margin,
                "passed": item.passed,
            }
            for item in items
        },
    }


def minimax_relative_errors(
    values: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    """Score independent objectives by maximum relative error, then RMSE.

    A target may be a number or a mapping containing ``value``.  Missing
    values produce infinite scores and therefore cannot become an incumbent.
    """
    if not targets:
        raise ValueError("At least one minimax target is required")
    errors: dict[str, float] = {}
    for name, specification in targets.items():
        target = (
            specification.get("value")
            if isinstance(specification, Mapping)
            else specification
        )
        errors[str(name)] = relative_error_percent(values.get(name), target)

    finite = all(math.isfinite(value) for value in errors.values())
    if not finite:
        limiting = ""
        maximum = math.inf
        rmse = math.inf
    else:
        limiting = max(errors, key=errors.get)
        maximum = errors[limiting]
        rmse = math.sqrt(sum(value * value for value in errors.values()) / len(errors))
    return {
        "errors_percent": errors,
        "maximum_error_percent": maximum,
        "rmse_percent": rmse,
        "limiting_target": limiting,
    }


def constrained_minimax_rank_key(
    *,
    constraint_summary: Mapping[str, Any],
    objective_summary: Mapping[str, Any],
    born_stable: bool = True,
    fit_r2: Any = 1.0,
    minimum_fit_r2: float = 0.0,
    parameter_contrast: Any = 0.0,
) -> tuple[float, ...]:
    """Return the deterministic final-ranking key used after exact evaluation.

    Feasibility, Born stability, and fit quality are hard ordering layers.
    Only candidates that pass them are scientifically eligible; the remaining
    tuple fields make rejected rows deterministic for diagnostics.
    """
    feasible = bool(constraint_summary.get("feasible", False))
    r2 = _finite_float(fit_r2)
    fit_ok = math.isfinite(r2) and r2 + _EPS >= float(minimum_fit_r2)
    maximum = _finite_float(objective_summary.get("maximum_error_percent"))
    rmse = _finite_float(objective_summary.get("rmse_percent"))
    contrast = _finite_float(parameter_contrast)
    minimum_margin = _finite_float(constraint_summary.get("minimum_margin"))
    return (
        0.0 if feasible else 1.0,
        0.0 if born_stable else 1.0,
        0.0 if fit_ok else 1.0,
        maximum if math.isfinite(maximum) else math.inf,
        rmse if math.isfinite(rmse) else math.inf,
        abs(contrast) if math.isfinite(contrast) else math.inf,
        -minimum_margin if math.isfinite(minimum_margin) else math.inf,
    )
