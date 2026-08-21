"""Stable public API for FFOpt property evaluator plugins."""

from engine.property_contracts import (
    DEFAULT_PROPERTY_COST_CLASS,
    DEFAULT_PROPERTY_FIDELITY,
    DEFAULT_PROPERTY_ROLES,
    PropertyContract,
    PropertyCostClass,
    PropertyRole,
)
from engine.property_evaluators import (
    PropertyEvaluationContext,
    PropertyEvaluator,
    PropertyEvaluatorRegistry,
    PropertyExecutionPlan,
    PropertyStageResult,
    PROPERTY_EVALUATORS,
    register_property_evaluator,
)

__all__ = [
    "DEFAULT_PROPERTY_COST_CLASS",
    "DEFAULT_PROPERTY_FIDELITY",
    "DEFAULT_PROPERTY_ROLES",
    "PropertyContract",
    "PropertyCostClass",
    "PropertyEvaluationContext",
    "PropertyEvaluator",
    "PropertyEvaluatorRegistry",
    "PropertyExecutionPlan",
    "PropertyRole",
    "PropertyStageResult",
    "PROPERTY_EVALUATORS",
    "register_property_evaluator",
]
