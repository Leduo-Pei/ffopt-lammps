"""Validated metadata contracts for composable property evaluators.

The contract is deliberately independent from a particular input schema or
workflow engine.  It describes what an evaluator can be used for and what it
needs/provides; execution remains the responsibility of
``engine.property_evaluators``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Iterable


_EVALUATOR_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
_PROPERTY_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_FIDELITY_RE = re.compile(r"[a-z][a-z0-9_]*")


class PropertyRole(str, Enum):
    """Supported uses of a property evaluator in a material workflow."""

    CONSTRAINT = "constraint"
    OBJECTIVE = "objective"
    PROMOTION = "promotion"
    VALIDATION = "validation"


class PropertyCostClass(str, Enum):
    """Coarse, scheduler-independent evaluator cost classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


DEFAULT_PROPERTY_ROLES = frozenset({
    PropertyRole.OBJECTIVE,
    PropertyRole.VALIDATION,
})
DEFAULT_PROPERTY_FIDELITY = "unspecified"
DEFAULT_PROPERTY_COST_CLASS = PropertyCostClass.MEDIUM


def validate_evaluator_name(value: Any, *, field: str = "name") -> str:
    """Return a validated lower-case evaluator identifier."""

    if not isinstance(value, str) or _EVALUATOR_NAME_RE.fullmatch(value) is None:
        raise ValueError(f"Invalid {field} {value!r}; use lower-case snake_case")
    return value


def validate_fidelity(value: Any) -> str:
    """Return a validated, extensible fidelity identifier."""

    if not isinstance(value, str) or _FIDELITY_RE.fullmatch(value) is None:
        raise ValueError(
            f"Invalid property fidelity {value!r}; use lower-case snake_case "
            "such as 'structural', 'static_0k', or 'dynamic_300k'"
        )
    return value


def coerce_property_role(value: Any) -> PropertyRole:
    try:
        return PropertyRole(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(role.value for role in PropertyRole)
        raise ValueError(
            f"Invalid property role {value!r}; choose one of: {choices}"
        ) from exc


def coerce_cost_class(value: Any) -> PropertyCostClass:
    try:
        return PropertyCostClass(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in PropertyCostClass)
        raise ValueError(
            f"Invalid property cost class {value!r}; choose one of: {choices}"
        ) from exc


def _validated_roles(values: Iterable[Any]) -> frozenset[PropertyRole]:
    if isinstance(values, (str, bytes)):
        raise TypeError("Property contract roles must be an iterable, not a string")
    raw = tuple(values)
    roles = frozenset(coerce_property_role(value) for value in raw)
    if not roles:
        raise ValueError("Property contract roles must not be empty")
    if len(roles) != len(raw):
        raise ValueError("Property contract roles must not contain duplicates")
    return roles


def _validated_dependencies(values: Iterable[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(
            "Property contract dependencies must be an iterable, not a string"
        )
    dependencies = tuple(
        validate_evaluator_name(value, field="dependency") for value in values
    )
    if len(set(dependencies)) != len(dependencies):
        raise ValueError("Property contract dependencies must not contain duplicates")
    return dependencies


def _validated_provides(values: Iterable[Any]) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("Property contract provides must be an iterable, not a string")
    raw = tuple(values)
    provided: set[str] = set()
    for value in raw:
        if not isinstance(value, str) or _PROPERTY_NAME_RE.fullmatch(value) is None:
            raise ValueError(
                f"Invalid provided property {value!r}; use an identifier made of "
                "letters, digits, and underscores"
            )
        provided.add(value)
    if len(provided) != len(raw):
        raise ValueError("Property contract provides must not contain duplicates")
    return frozenset(provided)


@dataclass(frozen=True)
class PropertyContract:
    """Immutable metadata describing one property evaluator.

    ``fidelity`` remains an extensible snake-case identifier so material
    plugins can introduce meaningful levels without changing the core package.
    Roles and cost classes are closed vocabularies because workflow selection
    and budget policies must interpret them consistently.
    """

    roles: frozenset[PropertyRole] = DEFAULT_PROPERTY_ROLES
    fidelity: str = DEFAULT_PROPERTY_FIDELITY
    cost_class: PropertyCostClass = DEFAULT_PROPERTY_COST_CLASS
    dependencies: tuple[str, ...] = ()
    provides: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", _validated_roles(self.roles))
        object.__setattr__(self, "fidelity", validate_fidelity(self.fidelity))
        object.__setattr__(self, "cost_class", coerce_cost_class(self.cost_class))
        object.__setattr__(
            self, "dependencies", _validated_dependencies(self.dependencies)
        )
        object.__setattr__(self, "provides", _validated_provides(self.provides))

    def describe(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible contract metadata."""

        return {
            "roles": sorted(role.value for role in self.roles),
            "fidelity": self.fidelity,
            "cost_class": self.cost_class.value,
            "dependencies": list(self.dependencies),
            "provides": sorted(self.provides),
        }


def contract_from_evaluator(evaluator: Any) -> PropertyContract:
    """Resolve explicit contracts and legacy class attributes uniformly.

    Legacy evaluators require no changes.  New evaluators may either declare
    ``roles``, ``fidelity`` and ``cost_class`` beside the established
    ``dependencies``/``provides`` attributes, or set an explicit
    :class:`PropertyContract`.  Declaring the same field both ways with
    different values is rejected instead of silently choosing one definition.
    """

    explicit = getattr(evaluator, "contract", None)
    if explicit is None:
        return PropertyContract(
            roles=getattr(evaluator, "roles", DEFAULT_PROPERTY_ROLES),
            fidelity=getattr(
                evaluator, "fidelity", DEFAULT_PROPERTY_FIDELITY
            ),
            cost_class=getattr(
                evaluator, "cost_class", DEFAULT_PROPERTY_COST_CLASS
            ),
            dependencies=getattr(evaluator, "dependencies", ()),
            provides=getattr(evaluator, "provides", frozenset()),
        )
    if not isinstance(explicit, PropertyContract):
        raise TypeError(
            "Property evaluator 'contract' must be a PropertyContract instance"
        )

    declared = type(evaluator).__dict__
    comparisons = {
        "roles": explicit.roles,
        "fidelity": explicit.fidelity,
        "cost_class": explicit.cost_class,
        "dependencies": explicit.dependencies,
        "provides": explicit.provides,
    }
    for field, expected in comparisons.items():
        if field not in declared:
            continue
        candidate = PropertyContract(**{
            "roles": explicit.roles,
            "fidelity": explicit.fidelity,
            "cost_class": explicit.cost_class,
            "dependencies": explicit.dependencies,
            "provides": explicit.provides,
            field: declared[field],
        })
        actual = getattr(candidate, field)
        if actual != expected:
            raise ValueError(
                f"Property evaluator declares conflicting {field!r} in its "
                "contract and legacy class attribute"
            )
    return explicit
