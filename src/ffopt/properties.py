"""Stable public API for FFOpt property evaluator plugins."""

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
    "PropertyEvaluationContext",
    "PropertyEvaluator",
    "PropertyEvaluatorRegistry",
    "PropertyExecutionPlan",
    "PropertyStageResult",
    "PROPERTY_EVALUATORS",
    "register_property_evaluator",
]
