"""Declarative workflow-stage definitions and deterministic graph expansion.

The registry deliberately contains no force-field or scheduler logic.  It turns
the stage tokens selected by an input file into a validated, topologically
ordered graph that :mod:`workflow.pipeline` can execute.  Material-specific
packages can build on this API without teaching the core runner every possible
workflow shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class StageDefinition:
    """Declare one executable stage kind or one expansion-only workflow token.

    ``command_token`` is the stable token consumed by the command builder.
    ``expected_artifacts`` are paths relative to the stage output directory.
    Expansion-only definitions use ``expands_to`` and have no command or
    artifacts.  ``repeatable`` gives a single workflow token multiple numbered
    stage instances, which is useful for recoverable active-learning rounds.
    """

    kind: str
    command_token: str | None = None
    expected_artifacts: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    expands_to: tuple[str, ...] = ()
    repeatable: bool = False
    instance_template: str = "{kind}_{index:02d}"

    def __post_init__(self) -> None:
        _validate_token(self.kind, label="stage kind")
        if self.command_token is not None:
            _validate_token(self.command_token, label="command token")
        _validate_unique_tokens(self.requires, label=f"dependencies of {self.kind!r}")
        _validate_unique_tokens(self.expands_to, label=f"expansion of {self.kind!r}")
        if self.expands_to and (
            self.command_token is not None
            or self.expected_artifacts
            or self.repeatable
        ):
            raise ValueError(
                f"Expansion stage {self.kind!r} cannot declare a command, "
                "artifacts, or repetition"
            )
        if not self.expands_to and self.command_token is None:
            raise ValueError(
                f"Executable stage {self.kind!r} must declare command_token"
            )
        if self.expands_to and self.requires:
            raise ValueError(
                f"Expansion stage {self.kind!r} must put dependencies on its "
                "concrete child stages"
            )
        if len(set(self.expected_artifacts)) != len(self.expected_artifacts):
            raise ValueError(f"Duplicate artifacts for stage {self.kind!r}")
        for artifact in self.expected_artifacts:
            normalized = artifact.replace("\\", "/")
            if (
                not normalized
                or normalized.startswith("/")
                or re.match(r"^[A-Za-z]:", normalized)
                or ".." in normalized.split("/")
            ):
                raise ValueError(
                    f"Expected artifact for {self.kind!r} must be a safe relative "
                    f"path: {artifact!r}"
                )
        if self.repeatable:
            try:
                example = self.instance_template.format(kind=self.kind, index=1)
            except (KeyError, ValueError, IndexError) as exc:
                raise ValueError(
                    f"Invalid instance_template for {self.kind!r}: "
                    f"{self.instance_template!r}"
                ) from exc
            _validate_token(example, label="repeated stage name")

    @property
    def is_expansion(self) -> bool:
        return bool(self.expands_to)


@dataclass(frozen=True)
class StageNode:
    """One concrete, executable stage instance in an expanded graph."""

    name: str
    kind: str
    command_token: str
    expected_artifacts: tuple[str, ...]
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class StageGraph:
    """A deterministic topological ordering plus explicit dependency edges."""

    nodes: tuple[StageNode, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)

    def node(self, name: str) -> StageNode:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)


@dataclass
class _DraftNode:
    name: str
    definition: StageDefinition
    ordinal: int
    exact_dependencies: tuple[str, ...] = ()


class StageRegistry:
    """Mutable construction-time registry with immutable expanded graphs."""

    def __init__(self, definitions: Iterable[StageDefinition] = ()) -> None:
        self._definitions: dict[str, StageDefinition] = {}
        for definition in definitions:
            self.register(definition)

    @property
    def definitions(self) -> Mapping[str, StageDefinition]:
        return MappingProxyType(self._definitions)

    def copy(self) -> StageRegistry:
        return StageRegistry(self._definitions.values())

    def register(self, definition: StageDefinition) -> None:
        if definition.kind in self._definitions:
            raise ValueError(f"Stage kind already registered: {definition.kind!r}")
        self._definitions[definition.kind] = definition

    def replace(self, definition: StageDefinition) -> None:
        """Replace one registered contract while preserving registry order."""
        if definition.kind not in self._definitions:
            raise ValueError(f"Cannot replace unknown stage kind: {definition.kind!r}")
        self._definitions[definition.kind] = definition

    def expand(
        self,
        workflow_tokens: Iterable[str],
        *,
        repetitions: Mapping[str, int] | None = None,
        legacy_linear: bool = False,
    ) -> StageGraph:
        """Validate and expand workflow tokens into an executable DAG.

        ``legacy_linear=True`` is the compatibility adapter used by the original
        BTAH pipeline: semantic dependencies are still validated, but each stage
        is chained to the immediately preceding stage so signatures and resume
        invalidation remain byte-for-byte compatible with the historical runner.
        """

        tokens = [str(token) for token in workflow_tokens]
        _validate_unique_tokens(tokens, label="workflow stages")
        for token in tokens:
            if token not in self._definitions:
                raise ValueError(f"Unknown pipeline stage: {token!r}")

        repeat_counts = {str(key): value for key, value in (repetitions or {}).items()}
        for kind, raw_count in repeat_counts.items():
            if kind not in self._definitions:
                raise ValueError(f"Repetition configured for unknown stage: {kind!r}")
            definition = self._definitions[kind]
            if not definition.repeatable:
                raise ValueError(f"Stage {kind!r} is not repeatable")
            if isinstance(raw_count, bool):
                raise ValueError(f"Repetition count for {kind!r} must be a positive integer")
            try:
                count = int(raw_count)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Repetition count for {kind!r} must be a positive integer"
                ) from exc
            if count < 1 or count != raw_count:
                raise ValueError(f"Repetition count for {kind!r} must be a positive integer")
            repeat_counts[kind] = count

        drafts: list[_DraftNode] = []
        origin_nodes: dict[str, list[str]] = {}
        expansion_stack: list[str] = []

        def add_kind(kind: str, *, origin: str) -> None:
            if kind not in self._definitions:
                parent = expansion_stack[-1] if expansion_stack else origin
                raise ValueError(
                    f"Stage {parent!r} expands to unknown stage {kind!r}"
                )
            definition = self._definitions[kind]
            if definition.is_expansion:
                if kind in expansion_stack:
                    cycle = " -> ".join([*expansion_stack, kind])
                    raise ValueError(f"Stage expansion cycle detected: {cycle}")
                expansion_stack.append(kind)
                for child in definition.expands_to:
                    add_kind(child, origin=origin)
                expansion_stack.pop()
                return

            count = int(repeat_counts.get(kind, 1))
            if count != 1 and not definition.repeatable:
                raise ValueError(f"Stage {kind!r} is not repeatable")
            previous: str | None = None
            for index in range(1, count + 1):
                name = (
                    definition.instance_template.format(kind=kind, index=index)
                    if definition.repeatable
                    else kind
                )
                exact = (previous,) if previous is not None else ()
                drafts.append(_DraftNode(name, definition, len(drafts), exact))
                origin_nodes.setdefault(origin, []).append(name)
                previous = name

        for token in tokens:
            add_kind(token, origin=token)

        names = [draft.name for draft in drafts]
        duplicates = _duplicates(names)
        if duplicates:
            raise ValueError(
                "Duplicate concrete pipeline stages after expansion: "
                f"{sorted(duplicates)}"
            )

        by_kind: dict[str, list[str]] = {}
        by_name = {draft.name: draft for draft in drafts}
        for draft in drafts:
            by_kind.setdefault(draft.definition.kind, []).append(draft.name)

        semantic_dependencies: dict[str, list[str]] = {}
        for draft in drafts:
            dependencies = list(draft.exact_dependencies)
            for requirement in draft.definition.requires:
                providers = origin_nodes.get(requirement) or by_kind.get(requirement)
                if not providers:
                    raise ValueError(
                        f"Pipeline stage {draft.name!r} requires missing stage "
                        f"{requirement!r}"
                    )
                provider = providers[-1]
                if provider == draft.name:
                    alternatives = [name for name in providers if name != draft.name]
                    if not alternatives:
                        raise ValueError(
                            f"Pipeline stage {draft.name!r} cannot depend on itself"
                        )
                    provider = alternatives[-1]
                if provider not in dependencies:
                    dependencies.append(provider)
            semantic_dependencies[draft.name] = dependencies

        ordered_names = _stable_topological_order(
            names,
            semantic_dependencies,
            {draft.name: draft.ordinal for draft in drafts},
        )

        nodes: list[StageNode] = []
        previous: str | None = None
        for name in ordered_names:
            draft = by_name[name]
            dependencies = (
                (previous,) if legacy_linear and previous is not None else ()
            ) if legacy_linear else tuple(semantic_dependencies[name])
            nodes.append(StageNode(
                name=name,
                kind=draft.definition.kind,
                command_token=str(draft.definition.command_token),
                expected_artifacts=draft.definition.expected_artifacts,
                dependencies=dependencies,
            ))
            previous = name
        return StageGraph(tuple(nodes))


LEGACY_PIPELINE_STAGES = (
    "bo",
    "sample",
    "nn",
    "al",
    "audit",
    "finalize",
    "validate",
)


def legacy_stage_registry() -> StageRegistry:
    """Return a fresh registry matching the historical BTAH pipeline exactly."""

    return StageRegistry([
        StageDefinition(
            "bo", "bo", ("all_results.csv",),
        ),
        StageDefinition(
            "sample", "sample", ("local_results.csv", "local_replicates.csv"),
            requires=("bo",),
        ),
        StageDefinition(
            "nn", "nn", ("forward_nn.pt", "nn_optimize_result.json"),
            requires=("bo",),
        ),
        StageDefinition(
            "al", "al", ("active_learning_history.json", "final_parameters.json"),
            requires=("bo", "nn"),
        ),
        StageDefinition(
            "audit", "audit", ("stable_results.csv", "stability_replicates.csv"),
            requires=("bo",),
        ),
        StageDefinition(
            "finalize", "finalize",
            ("final_summary.json", "final_parameters.lammps"),
            requires=("audit",),
        ),
        StageDefinition(
            "validate", "validate",
            (
                "validation_summary.json",
                "computed_properties.csv",
                "final_atom_parameters.csv",
                "final_parameters.lammps",
                "final_parameters.json",
            ),
        ),
    ])


def _validate_token(value: str, *, label: str) -> None:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"Invalid {label}: {value!r}")


def _validate_unique_tokens(values: Iterable[str], *, label: str) -> None:
    items = list(values)
    for value in items:
        _validate_token(value, label=label)
    duplicates = _duplicates(items)
    if duplicates:
        raise ValueError(f"Duplicate {label}: {sorted(duplicates)}")


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    return duplicates


def _stable_topological_order(
    names: list[str],
    dependencies: Mapping[str, list[str]],
    ordinal: Mapping[str, int],
) -> list[str]:
    for name, required in dependencies.items():
        unknown = [dependency for dependency in required if dependency not in ordinal]
        if unknown:
            raise ValueError(
                f"Pipeline stage {name!r} has unknown dependencies: {unknown}"
            )
    remaining = {name: set(dependencies[name]) for name in names}
    ready = sorted(
        [name for name, required in remaining.items() if not required],
        key=ordinal.__getitem__,
    )
    ordered: list[str] = []
    while ready:
        name = ready.pop(0)
        ordered.append(name)
        for candidate in names:
            if candidate in ordered or candidate in ready:
                continue
            required = remaining[candidate]
            required.discard(name)
            if not required:
                ready.append(candidate)
                ready.sort(key=ordinal.__getitem__)
    if len(ordered) != len(names):
        cyclic = [name for name in names if name not in ordered]
        raise ValueError(
            "Pipeline stage dependency cycle detected among: "
            f"{cyclic}"
        )
    return ordered


MATERIAL_STAGE_DEFINITIONS = (
    StageDefinition(
        "candidates",
        "candidates",
        (
            "candidates.csv",
            "candidate_pool.csv",
            "static_screen_candidates.csv",
            "candidates_summary.json",
            "stage_manifest.json",
        ),
        requires=("audit",),
    ),
    StageDefinition(
        "static",
        "static",
        (
            "static_results.csv",
            "finalists_selected.csv",
            "best_candidate.json",
            "batch_summary.json",
            "stage_manifest.json",
        ),
        requires=("candidates",),
    ),
    StageDefinition("screen", expands_to=("candidates", "static")),
    StageDefinition(
        "constrained_al",
        "constrained-al",
        (
            "refinement_state.json",
            "exact_structural_mechanical_ranking.csv",
            "structural_proposals.csv",
            "mechanical_proposals.csv",
            "refinement_report.md",
            "stage_disposition.json",
            "round_evidence.json",
            "artifact_manifest.json",
            "material_al_manifest.json",
        ),
        requires=("nn", "static"),
        repeatable=True,
        instance_template="constrained_al_{index:02d}",
    ),
    StageDefinition(
        "finalists",
        "finalists",
        (
            "dynamic_results.csv",
            "dynamic_seed_results.csv",
            "finalists_selected.csv",
            "best_candidate.json",
            "batch_summary.json",
            "stage_manifest.json",
        ),
        requires=("constrained_al",),
    ),
)


def material_stage_registry(base: StageRegistry | None = None) -> StageRegistry:
    """Return material contracts while retaining legacy molecular definitions."""

    registry = base.copy() if base is not None else legacy_stage_registry()
    registry.replace(StageDefinition(
        "bo",
        "bo",
        (
            "all_results.csv",
            "feasible_archive.csv",
            "coverage_anchors.csv",
            "coverage_summary.json",
            "best_parameters.txt",
        ),
    ))
    registry.replace(StageDefinition(
        "nn",
        "material-nn",
        (
            "surrogate_diagnostics.json",
            "initial_proposals.csv",
            "candidate_pool.csv",
            "stage_manifest.json",
        ),
        requires=("static",),
    ))
    registry.replace(StageDefinition(
        "validate",
        "material-validate",
        (
            "validation_summary.json",
            "computed_properties.csv",
            "final_atom_parameters.csv",
            "final_parameters.lammps",
            "final_parameters.json",
            "model_adequacy.json",
            "stage_manifest.json",
        ),
    ))
    for definition in MATERIAL_STAGE_DEFINITIONS:
        registry.register(definition)
    return registry
