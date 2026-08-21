"""Material-independent GP acquisition for constrained refinement.

The backend consumes an explicit candidate pool and two distinct evidence
tables.  Structural observations provide continuous gate ratios; mechanical
observations provide a minimax objective only where an *exact* structural
observation passed every gate.  Surrogate predictions are used exclusively
for proposing new coordinates and are never promoted to measured evidence.

The implementation is intentionally conservative.  It fits one low-
dimensional Gaussian process per structural gate and one GP for the measured
mechanical M-infinity score.  If sample support or held-out fit quality is
insufficient, selection falls back to parameter-space coverage and the round
cannot claim scientific convergence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .constraint_objectives import assess_constraint, minimax_relative_errors
from .constrained_refinement import (
    AcquisitionContext,
    AcquisitionResult,
    RefinementError,
    StructuralConstraintSpec,
    _truthy,
)


_EPS = 1.0e-12
_DIAGNOSTIC_SCHEMA = "ffopt-gp-structural-acquisition-v1"


@dataclass(frozen=True)
class _DistributionModel:
    estimator: Any
    residual_floor: float

    def predict(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, standard_deviation = self.estimator.predict(values, return_std=True)
        return (
            np.asarray(mean, dtype=float),
            np.maximum(np.asarray(standard_deviation, dtype=float), self.residual_floor),
        )


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    """Normal CDF without making scipy a core import dependency."""
    try:
        from scipy.special import ndtr
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RefinementError(
            "GP structural acquisition requires the 'bo' optional dependencies"
        ) from exc
    return np.asarray(ndtr(values), dtype=float)


def constrained_expected_improvement(
    *,
    feasible_probability: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    incumbent: float,
    exploration: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return minimisation EI and feasibility-weighted EI.

    Feasibility is a multiplier, not a replacement objective.  Consequently,
    a saturated probability of one leaves genuine mechanical expected
    improvement intact.
    """
    probability = np.clip(np.asarray(feasible_probability, dtype=float), 0.0, 1.0)
    mean = np.asarray(predicted_mean, dtype=float)
    standard_deviation = np.maximum(np.asarray(predicted_std, dtype=float), _EPS)
    improvement = float(incumbent) - mean - float(exploration)
    z_value = improvement / standard_deviation
    cdf = _normal_cdf(z_value)
    density = np.exp(-0.5 * z_value * z_value) / math.sqrt(2.0 * math.pi)
    expected = np.maximum(improvement * cdf + standard_deviation * density, 0.0)
    return expected, probability * expected


def _bernoulli_entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), _EPS, 1.0 - _EPS)
    return -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    ) / math.log(2.0)


def _minimum_distance(points: np.ndarray, references: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=float)
    if len(references) == 0:
        return np.ones(len(points), dtype=float)
    result = np.full(len(points), np.inf, dtype=float)
    for start in range(0, len(references), 256):
        block = references[start : start + 256]
        result = np.minimum(
            result,
            np.linalg.norm(points[:, None, :] - block[None, :, :], axis=2).min(axis=1),
        )
    return result


def _normalise(
    pool: pd.DataFrame,
    observations: pd.DataFrame,
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pool_values = pool[list(names)].to_numpy(dtype=float)
    observation_values = observations[list(names)].to_numpy(dtype=float)
    combined = (
        np.vstack([pool_values, observation_values])
        if len(observation_values)
        else pool_values
    )
    lows = np.min(combined, axis=0)
    spans = np.maximum(np.max(combined, axis=0) - lows, _EPS)
    return (
        (pool_values - lows) / spans,
        (observation_values - lows) / spans,
        lows,
        spans,
    )


def _r2_score(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    if denominator <= _EPS:
        return None
    value = 1.0 - float(np.sum((observed - predicted) ** 2)) / denominator
    return value if math.isfinite(value) else None


def _make_gp(dimensions: int, seed: int) -> Any:
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RefinementError(
            "GP structural acquisition requires the 'bo' optional dependencies"
        ) from exc
    kernel = (
        ConstantKernel(1.0, (1.0e-2, 1.0e2))
        * Matern(
            length_scale=np.full(dimensions, 0.25),
            length_scale_bounds=(0.02, 5.0),
            nu=2.5,
        )
        + WhiteKernel(1.0e-5, (1.0e-8, 1.0e-1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=int(seed),
    )


def _fit_gp(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    name: str,
    seed: int,
    minimum_observations: int,
    minimum_holdout_r2: float,
) -> tuple[_DistributionModel | None, dict[str, Any]]:
    finite = np.isfinite(values).all(axis=1) & np.isfinite(observed)
    x_values = np.asarray(values[finite], dtype=float)
    y_values = np.asarray(observed[finite], dtype=float)
    required = max(int(minimum_observations), 2 * values.shape[1] + 4)
    metrics: dict[str, Any] = {
        "target": str(name),
        "finite_observations": int(len(y_values)),
        "required_observations": int(required),
        "holdout_r2": None,
        "holdout_rmse": None,
        "residual_floor": None,
        "reliable": False,
    }
    if len(y_values) < required:
        metrics["status"] = "insufficient_observations"
        return None, metrics
    if float(np.ptp(y_values)) <= _EPS:
        metrics["status"] = "constant_response_cannot_validate_fit"
        return None, metrics

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(y_values))
    test_count = min(max(3, int(round(0.20 * len(y_values)))), len(y_values) - 5)
    test_indices = order[:test_count]
    train_indices = order[test_count:]
    try:
        provisional = _make_gp(values.shape[1], seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            provisional.fit(x_values[train_indices], y_values[train_indices])
        predicted = np.asarray(provisional.predict(x_values[test_indices]), dtype=float)
        residual = y_values[test_indices] - predicted
        holdout_rmse = float(np.sqrt(np.mean(residual * residual)))
        holdout_r2 = _r2_score(y_values[test_indices], predicted)
        residual_floor = max(
            0.25 * holdout_rmse,
            1.0e-4 * float(np.std(y_values)),
            1.0e-8,
        )
        estimator = _make_gp(values.shape[1], seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(x_values, y_values)
    except (ValueError, np.linalg.LinAlgError, RefinementError) as exc:
        metrics.update({
            "status": "fit_failed",
            "failure": f"{type(exc).__name__}: {exc}",
        })
        return None, metrics

    reliable = holdout_r2 is not None and holdout_r2 >= minimum_holdout_r2
    metrics.update({
        "status": "reliable" if reliable else "poor_holdout_fit",
        "holdout_r2": holdout_r2,
        "holdout_rmse": holdout_rmse,
        "residual_floor": residual_floor,
        "minimum_holdout_r2": float(minimum_holdout_r2),
        "reliable": bool(reliable),
    })
    return _DistributionModel(estimator, residual_floor), metrics


def _exact_structure(
    frame: pd.DataFrame,
    constraints: Sequence[StructuralConstraintSpec],
) -> tuple[np.ndarray, np.ndarray, dict[str, bool], dict[str, float]]:
    ratios = np.full((len(frame), len(constraints)), np.nan, dtype=float)
    feasible = np.zeros(len(frame), dtype=bool)
    by_key: dict[str, bool] = {}
    boundary_distance: dict[str, float] = {}
    for row_index, row in enumerate(frame.to_dict(orient="records")):
        row_ratios: list[float] = []
        passed = True
        for column_index, constraint in enumerate(constraints):
            result = assess_constraint(
                constraint.name,
                row.get(constraint.column),
                constraint.target,
                constraint.tolerance,
                mode=constraint.mode,
            )
            ratios[row_index, column_index] = result.ratio
            row_ratios.append(result.ratio)
            passed = passed and result.passed
        key = str(row["parameter_key"])
        feasible[row_index] = passed
        by_key[key] = bool(passed)
        finite_ratios = [item for item in row_ratios if math.isfinite(item)]
        boundary_distance[key] = (
            abs(max(finite_ratios) - 1.0) if finite_ratios else math.inf
        )
    return ratios, feasible, by_key, boundary_distance


def _eligible_mechanics(
    context: AcquisitionContext,
    structural_feasible: Mapping[str, bool],
) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict[str, Any]] = []
    scores: list[float] = []
    targets = {item.name: item.target for item in context.mechanical_objectives}
    spec = context.refinement_spec
    for row in context.mechanical_observations.to_dict(orient="records"):
        key = str(row["parameter_key"])
        if not structural_feasible.get(key, False):
            continue
        if spec is not None and spec.require_stability:
            if not _truthy(row.get(spec.stability_column, False)):
                continue
        if spec is not None and spec.minimum_fit_quality is not None:
            try:
                fit_quality = float(row.get(spec.fit_quality_column, math.nan))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fit_quality) or fit_quality < spec.minimum_fit_quality:
                continue
        values = {item.name: row.get(item.column) for item in context.mechanical_objectives}
        summary = minimax_relative_errors(values, targets)
        score = float(summary["maximum_error_percent"])
        if not math.isfinite(score):
            continue
        records.append(row)
        scores.append(score)
    if not records:
        return pd.DataFrame(columns=context.mechanical_observations.columns), np.empty(0)
    return pd.DataFrame(records), np.asarray(scores, dtype=float)


def _allocate_counts(total: int, fractions: Sequence[float]) -> tuple[int, ...]:
    if total <= 0:
        return tuple(0 for _ in fractions)
    raw = np.asarray(fractions, dtype=float) * total
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts)):
        if int(counts.sum()) >= total:
            break
        counts[index] += 1
    positive = [index for index, value in enumerate(fractions) if value > 0.0]
    if total >= len(positive):
        for index in positive:
            if counts[index] > 0:
                continue
            donors = [item for item in positive if counts[item] > 1]
            if donors:
                donor = max(donors, key=lambda item: (counts[item], fractions[item]))
                counts[donor] -= 1
                counts[index] += 1
    return tuple(int(item) for item in counts)


def _scaled_score(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    finite = np.isfinite(numeric)
    result = np.zeros(len(numeric), dtype=float)
    if not finite.any():
        return result
    low = float(np.min(numeric[finite]))
    span = max(float(np.max(numeric[finite]) - low), _EPS)
    result[finite] = (numeric[finite] - low) / span
    return result


def _greedy_select(
    *,
    eligible: Sequence[int],
    count: int,
    scores: np.ndarray,
    coordinates: np.ndarray,
    selected: list[int],
    tie_breakers: np.ndarray,
    diversity_weight: float,
) -> list[int]:
    available = [int(item) for item in eligible if int(item) not in selected]
    if count <= 0 or not available:
        return []
    scaled = _scaled_score(scores)
    chosen: list[int] = []
    for _ in range(min(count, len(available))):
        remaining = [item for item in available if item not in chosen]
        references = selected + chosen
        distances = (
            _minimum_distance(coordinates[remaining], coordinates[references])
            if references
            else np.ones(len(remaining), dtype=float)
        )
        distance_scale = max(float(np.max(distances)), _EPS)
        winner = max(
            range(len(remaining)),
            key=lambda position: (
                float(scaled[remaining[position]])
                + float(diversity_weight) * float(distances[position]) / distance_scale
                + 1.0e-9 * float(tie_breakers[remaining[position]]),
                -remaining[position],
            ),
        )
        chosen.append(remaining[winner])
    return chosen


class GaussianProcessStructuralAcquisition:
    """Low-dimensional constrained GP acquisition over an explicit pool."""

    name = "gp_structural_constraints"
    capability = "constrained_gp_pool_acquisition"
    scientific_convergence_capable = True

    def __init__(
        self,
        *,
        improvement_fraction: float = 0.55,
        boundary_fraction: float = 0.30,
        global_fraction: float = 0.15,
        trust_region_radius: float = 0.25,
        minimum_structural_observations: int = 12,
        minimum_mechanical_observations: int = 10,
        minimum_holdout_r2: float = 0.25,
        maximum_dimensions: int = 8,
        diversity_weight: float = 0.20,
        expected_improvement_exploration: float = 0.01,
        maximum_local_anchors: int = 8,
    ) -> None:
        self.improvement_fraction = float(improvement_fraction)
        self.boundary_fraction = float(boundary_fraction)
        self.global_fraction = float(global_fraction)
        fractions = (
            self.improvement_fraction,
            self.boundary_fraction,
            self.global_fraction,
        )
        if any(not 0.0 <= item <= 1.0 for item in fractions):
            raise RefinementError("Acquisition fractions must lie in [0, 1]")
        if not math.isclose(sum(fractions), 1.0, abs_tol=1.0e-12):
            raise RefinementError("Acquisition fractions must sum to one")
        self.trust_region_radius = float(trust_region_radius)
        if not 0.0 < self.trust_region_radius <= 1.0:
            raise RefinementError("trust_region_radius must lie in (0, 1]")
        self.minimum_structural_observations = int(minimum_structural_observations)
        self.minimum_mechanical_observations = int(minimum_mechanical_observations)
        if min(
            self.minimum_structural_observations,
            self.minimum_mechanical_observations,
        ) < 3:
            raise RefinementError("GP acquisition requires at least three observations")
        self.minimum_holdout_r2 = float(minimum_holdout_r2)
        if not math.isfinite(self.minimum_holdout_r2):
            raise RefinementError("minimum_holdout_r2 must be finite")
        self.maximum_dimensions = int(maximum_dimensions)
        if self.maximum_dimensions < 1:
            raise RefinementError("maximum_dimensions must be positive")
        self.diversity_weight = float(diversity_weight)
        if not 0.0 <= self.diversity_weight <= 1.0:
            raise RefinementError("diversity_weight must lie in [0, 1]")
        self.expected_improvement_exploration = float(
            expected_improvement_exploration
        )
        if self.expected_improvement_exploration < 0.0:
            raise RefinementError(
                "expected_improvement_exploration must be non-negative"
            )
        self.maximum_local_anchors = int(maximum_local_anchors)
        if self.maximum_local_anchors < 1:
            raise RefinementError("maximum_local_anchors must be positive")

    def scientific_identity(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "capability": self.capability,
            "scientific_convergence_capable": self.scientific_convergence_capable,
            "improvement_fraction": self.improvement_fraction,
            "boundary_fraction": self.boundary_fraction,
            "global_fraction": self.global_fraction,
            "trust_region_radius": self.trust_region_radius,
            "minimum_structural_observations": self.minimum_structural_observations,
            "minimum_mechanical_observations": self.minimum_mechanical_observations,
            "minimum_holdout_r2": self.minimum_holdout_r2,
            "maximum_dimensions": self.maximum_dimensions,
            "diversity_weight": self.diversity_weight,
            "expected_improvement_exploration": (
                self.expected_improvement_exploration
            ),
            "maximum_local_anchors": self.maximum_local_anchors,
            "joint_feasibility_aggregation": "minimum_marginal_probability",
            "incumbent_semantics": "exact_structural_pass_and_exact_mechanical_label",
        }

    def propose(self, context: AcquisitionContext) -> AcquisitionResult:
        names = tuple(context.parameter_names)
        empty_columns = [
            "candidate_id",
            "selection_role",
            "evidence_kind",
            "parameter_key",
            *names,
        ]
        constraints = tuple(context.structural_constraints)
        objectives = tuple(context.mechanical_objectives)
        diagnostics: dict[str, Any] = {
            "schema": _DIAGNOSTIC_SCHEMA,
            "seed": int(context.seed),
            "round_number": int(context.round_number),
            "parameter_dimensions": int(len(names)),
            "candidate_pool_rows": int(len(context.candidate_pool)),
            "mode": "coverage_fallback",
            "fallback_reasons": [],
            "surrogate_values_are_exact_observations": False,
        }
        if not names:
            raise RefinementError("GP acquisition requires parameter names")
        if not constraints:
            diagnostics["fallback_reasons"].append("no_structural_constraint_specs")
        if not objectives:
            diagnostics["fallback_reasons"].append("no_mechanical_objective_specs")
        exact_value_columns = {
            *(item.column for item in constraints),
            *(item.column for item in objectives),
        }
        ambiguous_pool_columns = sorted(
            exact_value_columns.intersection(context.candidate_pool.columns)
        )
        if ambiguous_pool_columns:
            raise RefinementError(
                "GP candidate pool must contain coordinates, not exact property "
                f"evidence columns: {ambiguous_pool_columns}"
            )
        required_columns = {"parameter_key", *names}
        for label, frame in (
            ("candidate pool", context.candidate_pool),
            ("structural observations", context.structural_observations),
            ("mechanical observations", context.mechanical_observations),
        ):
            missing = sorted(required_columns - set(frame.columns))
            if missing:
                raise RefinementError(f"{label} is missing columns: {missing}")

        observed_keys = set(
            context.structural_observations["parameter_key"].astype(str)
        )
        pool = context.candidate_pool.loc[
            ~context.candidate_pool["parameter_key"].astype(str).isin(observed_keys)
        ].copy()
        pool = pool.drop_duplicates("parameter_key", keep="first").reset_index(drop=True)
        diagnostics["unobserved_unique_candidates"] = int(len(pool))
        if pool.empty or context.proposal_count <= 0:
            diagnostics["fallback_reasons"].append("no_unobserved_candidate_budget")
            return AcquisitionResult(
                pd.DataFrame(columns=empty_columns),
                self.name,
                self.capability,
                False,
                diagnostics,
            )

        structural = context.structural_observations.reset_index(drop=True).copy()
        pool_coordinates, structural_coordinates, lows, spans = _normalise(
            pool, structural, names
        )
        diagnostics["normalization"] = {
            "lower": {name: float(value) for name, value in zip(names, lows)},
            "span": {name: float(value) for name, value in zip(names, spans)},
        }
        if len(names) > self.maximum_dimensions:
            diagnostics["fallback_reasons"].append(
                f"dimension_exceeds_limit:{len(names)}>{self.maximum_dimensions}"
            )

        if constraints:
            missing_constraints = sorted(
                {item.column for item in constraints} - set(structural.columns)
            )
            if missing_constraints and len(structural):
                raise RefinementError(
                    "Structural observations are missing gate columns: "
                    f"{missing_constraints}"
                )
            ratios, exact_feasible, feasible_by_key, boundary_by_key = _exact_structure(
                structural, constraints
            )
        else:
            ratios = np.empty((len(structural), 0))
            exact_feasible = np.zeros(len(structural), dtype=bool)
            feasible_by_key = {
                str(key): False for key in structural["parameter_key"].astype(str)
            }
            boundary_by_key = {
                str(key): math.inf for key in structural["parameter_key"].astype(str)
            }

        eligible_mechanical, mechanical_scores = _eligible_mechanics(
            context, feasible_by_key
        ) if objectives else (
            pd.DataFrame(columns=context.mechanical_observations.columns),
            np.empty(0),
        )
        exact_incumbent: float | None = None
        incumbent_key: str | None = None
        if len(mechanical_scores):
            best_index = int(np.argmin(mechanical_scores))
            exact_incumbent = float(mechanical_scores[best_index])
            incumbent_key = str(
                eligible_mechanical.iloc[best_index]["parameter_key"]
            )
        diagnostics["exact_evidence"] = {
            "structural_rows": int(len(structural)),
            "structural_feasible_rows": int(exact_feasible.sum()),
            "structural_infeasible_rows": int(len(exact_feasible) - exact_feasible.sum()),
            "mechanical_rows": int(len(context.mechanical_observations)),
            "eligible_joint_mechanical_rows": int(len(eligible_mechanical)),
        }
        diagnostics["incumbent"] = {
            "available": exact_incumbent is not None,
            "source": "exact_observed" if exact_incumbent is not None else "none",
            "parameter_key": incumbent_key,
            "mechanical_minimax_error_percent": exact_incumbent,
        }

        structural_models: dict[str, _DistributionModel] = {}
        structural_metrics: dict[str, Any] = {}
        if constraints and len(names) <= self.maximum_dimensions:
            for index, constraint in enumerate(constraints):
                model, metrics = _fit_gp(
                    structural_coordinates,
                    ratios[:, index],
                    name=constraint.name,
                    seed=int(context.seed) + 101 * index,
                    minimum_observations=self.minimum_structural_observations,
                    minimum_holdout_r2=self.minimum_holdout_r2,
                )
                structural_metrics[constraint.name] = metrics
                if model is not None:
                    structural_models[constraint.name] = model
        structural_reliable = bool(constraints) and all(
            structural_metrics.get(item.name, {}).get("reliable", False)
            for item in constraints
        )
        if constraints and not structural_reliable:
            diagnostics["fallback_reasons"].append("structural_gp_quality_insufficient")

        mechanical_model: _DistributionModel | None = None
        mechanical_metrics: dict[str, Any] = {
            "target": "mechanical_minimax_error_percent",
            "finite_observations": int(len(mechanical_scores)),
            "reliable": False,
            "status": "no_eligible_exact_mechanical_labels",
        }
        if len(mechanical_scores) and len(names) <= self.maximum_dimensions:
            mechanical_values = eligible_mechanical[list(names)].to_numpy(dtype=float)
            mechanical_coordinates = (mechanical_values - lows) / spans
            mechanical_model, mechanical_metrics = _fit_gp(
                mechanical_coordinates,
                mechanical_scores,
                name="mechanical_minimax_error_percent",
                seed=int(context.seed) + 10_003,
                minimum_observations=self.minimum_mechanical_observations,
                minimum_holdout_r2=self.minimum_holdout_r2,
            )
        mechanical_reliable = bool(
            exact_incumbent is not None
            and mechanical_model is not None
            and mechanical_metrics.get("reliable", False)
        )
        if not mechanical_reliable:
            diagnostics["fallback_reasons"].append("mechanical_gp_quality_insufficient")
        diagnostics["models"] = {
            "structural_gate_ratio_gps": structural_metrics,
            "mechanical_minimax_gp": mechanical_metrics,
        }

        pool["evidence_kind"] = "surrogate_proposal"
        pool["surrogate_is_exact_observation"] = False
        pool["global_coverage_distance"] = _minimum_distance(
            pool_coordinates, structural_coordinates
        )
        probability = np.full(len(pool), np.nan)
        boundary_entropy = np.full(len(pool), np.nan)
        mechanical_mean = np.full(len(pool), np.nan)
        mechanical_std = np.full(len(pool), np.nan)
        mechanical_ei = np.full(len(pool), np.nan)
        constrained_ei = np.full(len(pool), np.nan)
        model_based = bool(structural_reliable and mechanical_reliable)
        if model_based:
            marginal_probabilities: list[np.ndarray] = []
            marginal_entropies: list[np.ndarray] = []
            for constraint in constraints:
                mean, standard_deviation = structural_models[constraint.name].predict(
                    pool_coordinates
                )
                marginal = np.clip(
                    _normal_cdf((1.0 - mean) / standard_deviation), _EPS, 1.0
                )
                pool[f"surrogate_gate_ratio_mean__{constraint.name}"] = mean
                pool[f"surrogate_gate_ratio_std__{constraint.name}"] = (
                    standard_deviation
                )
                pool[f"surrogate_gate_probability__{constraint.name}"] = marginal
                marginal_probabilities.append(marginal)
                marginal_entropies.append(_bernoulli_entropy(marginal))
            probability = np.min(np.vstack(marginal_probabilities), axis=0)
            boundary_entropy = np.max(np.vstack(marginal_entropies), axis=0)
            mechanical_mean, mechanical_std = mechanical_model.predict(pool_coordinates)
            mechanical_ei, constrained_ei = constrained_expected_improvement(
                feasible_probability=probability,
                predicted_mean=mechanical_mean,
                predicted_std=mechanical_std,
                incumbent=float(exact_incumbent),
                exploration=self.expected_improvement_exploration,
            )
            diagnostics["mode"] = "constrained_gp"
            diagnostics["fallback_reasons"] = []

        pool["surrogate_structural_feasibility_probability"] = probability
        pool["surrogate_structural_boundary_entropy"] = boundary_entropy
        pool["surrogate_mechanical_minimax_mean_percent"] = mechanical_mean
        pool["surrogate_mechanical_minimax_std_percent"] = mechanical_std
        pool["surrogate_mechanical_expected_improvement_percent"] = mechanical_ei
        pool["surrogate_constrained_expected_improvement_percent"] = constrained_ei

        anchor_indices: list[int] = []
        if incumbent_key is not None:
            matches = structural.index[
                structural["parameter_key"].astype(str).eq(incumbent_key)
            ].tolist()
            anchor_indices.extend(int(item) for item in matches[:1])
        if not anchor_indices and len(structural):
            feasible_indices = np.flatnonzero(exact_feasible).tolist()
            anchor_indices.extend(feasible_indices[: self.maximum_local_anchors])
        if not anchor_indices and len(structural):
            ordered_keys = sorted(
                boundary_by_key,
                key=lambda key: (boundary_by_key[key], key),
            )[: self.maximum_local_anchors]
            key_to_index = {
                str(key): index
                for index, key in enumerate(structural["parameter_key"].astype(str))
            }
            anchor_indices.extend(key_to_index[key] for key in ordered_keys)
        anchors = (
            structural_coordinates[anchor_indices]
            if anchor_indices
            else np.empty((0, len(names)))
        )
        if len(anchors):
            trust_distance = np.min(
                np.max(
                    np.abs(
                        pool_coordinates[:, None, :] - anchors[None, :, :]
                    ),
                    axis=2,
                ),
                axis=1,
            )
            inside_trust = trust_distance <= self.trust_region_radius + _EPS
        else:
            trust_distance = np.full(len(pool), np.nan)
            inside_trust = np.ones(len(pool), dtype=bool)
        pool["surrogate_trust_region_distance"] = trust_distance
        pool["surrogate_inside_trust_region"] = inside_trust
        local_indices = np.flatnonzero(inside_trust).tolist()
        global_indices = np.flatnonzero(~inside_trust).tolist()
        if not local_indices:
            local_indices = list(range(len(pool)))
        if not global_indices:
            global_indices = list(range(len(pool)))

        count = min(int(context.proposal_count), len(pool))
        improvement_count, boundary_count, global_count = _allocate_counts(
            count,
            (
                self.improvement_fraction,
                self.boundary_fraction,
                self.global_fraction,
            ),
        )
        coverage_score = pool["global_coverage_distance"].to_numpy(dtype=float)
        if len(structural):
            boundary_anchor_indices = sorted(
                range(len(structural)),
                key=lambda index: (
                    boundary_by_key.get(
                        str(structural.iloc[index]["parameter_key"]), math.inf
                    ),
                    str(structural.iloc[index]["parameter_key"]),
                ),
            )[: self.maximum_local_anchors]
            boundary_anchor_coordinates = structural_coordinates[
                boundary_anchor_indices
            ]
            boundary_coverage_score = 1.0 / np.maximum(
                _minimum_distance(pool_coordinates, boundary_anchor_coordinates),
                1.0e-6,
            )
        else:
            boundary_coverage_score = coverage_score

        rng = np.random.default_rng(int(context.seed))
        tie_breakers = rng.random(len(pool))
        selected: list[int] = []
        roles: dict[int, str] = {}
        first_scores = constrained_ei if model_based else coverage_score
        first_role = (
            "constrained_mechanical_improvement_local"
            if model_based
            else "coverage_local"
        )
        chosen = _greedy_select(
            eligible=local_indices,
            count=improvement_count,
            scores=first_scores,
            coordinates=pool_coordinates,
            selected=selected,
            tie_breakers=tie_breakers,
            diversity_weight=self.diversity_weight,
        )
        selected.extend(chosen)
        roles.update({index: first_role for index in chosen})

        second_scores = boundary_entropy if model_based else boundary_coverage_score
        second_role = (
            "structural_boundary_entropy_local"
            if model_based
            else "coverage_boundary_local"
        )
        chosen = _greedy_select(
            eligible=local_indices,
            count=boundary_count,
            scores=second_scores,
            coordinates=pool_coordinates,
            selected=selected,
            tie_breakers=tie_breakers,
            diversity_weight=self.diversity_weight,
        )
        selected.extend(chosen)
        roles.update({index: second_role for index in chosen})

        chosen = _greedy_select(
            eligible=global_indices,
            count=global_count,
            scores=coverage_score,
            coordinates=pool_coordinates,
            selected=selected,
            tie_breakers=tie_breakers,
            diversity_weight=self.diversity_weight,
        )
        selected.extend(chosen)
        roles.update({index: "global_coverage" for index in chosen})
        if len(selected) < count:
            chosen = _greedy_select(
                eligible=range(len(pool)),
                count=count - len(selected),
                scores=coverage_score,
                coordinates=pool_coordinates,
                selected=selected,
                tie_breakers=tie_breakers,
                diversity_weight=self.diversity_weight,
            )
            selected.extend(chosen)
            roles.update({index: "diverse_batch_fill" for index in chosen})

        proposals = pool.loc[selected].copy().reset_index(drop=True)
        proposals.insert(0, "selection_role", [roles[index] for index in selected])
        proposals.insert(0, "candidate_id", np.arange(1, len(proposals) + 1))
        diagnostics["trust_region"] = {
            "radius_normalized_linf": self.trust_region_radius,
            "anchor_source": (
                "exact_observed_incumbent"
                if incumbent_key is not None
                else "exact_observed_structural"
                if len(anchors)
                else "none_full_pool"
            ),
            "anchor_count": int(len(anchors)),
            "local_candidate_count": int(np.sum(inside_trust)),
            "outside_candidate_count": int(np.sum(~inside_trust)),
        }
        diagnostics["selection_roles"] = {
            role: int(sum(value == role for value in roles.values()))
            for role in sorted(set(roles.values()))
        }
        diagnostics["quality"] = {
            "structural_models_reliable": bool(structural_reliable),
            "mechanical_model_reliable": bool(mechanical_reliable),
            "observed_structural_boundary_bracketed": bool(
                np.any(np.isfinite(ratios).all(axis=1) & exact_feasible)
                and np.any(np.isfinite(ratios).all(axis=1) & ~exact_feasible)
            ),
        }
        boundary_bracketed = diagnostics["quality"][
            "observed_structural_boundary_bracketed"
        ]
        convergence_capable = bool(
            model_based
            and boundary_bracketed
            and exact_incumbent is not None
        )
        diagnostics["scientific_convergence_capable_this_round"] = (
            convergence_capable
        )
        return AcquisitionResult(
            proposals,
            self.name,
            self.capability,
            convergence_capable,
            diagnostics,
        )


__all__ = [
    "GaussianProcessStructuralAcquisition",
    "constrained_expected_improvement",
]
