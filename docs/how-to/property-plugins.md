# Property evaluator plugins (advanced)

The schema-1 `ffopt.in` compiler recognizes the built-in `bulk`,
`sublimation`, `adsorption`, `surface`, and `elasticity` blocks. Run
`ffopt plugins` and check the `ffopt.in=yes` marker for evaluator-backed
properties. Cubic `elasticity` is instead executed by the native BCC material
stage graph, so it is public input syntax but is not a third-party evaluator
entry point. Installing an external evaluator alone does not add new
`ffopt.in` syntax.

FFOpt discovers external property evaluators from the Python entry-point group
`ffopt.property_evaluators`.

## Minimal evaluator

```python
from ffopt.properties import (
    PropertyEvaluator,
    PropertyStageResult,
)


class DipoleEvaluator(PropertyEvaluator):
    name = "dipole"
    roles = frozenset({"objective", "validation"})
    fidelity = "static_0k"
    cost_class = "low"
    dependencies = ()
    provides = frozenset({"dipole_debye"})

    def evaluate(self, runner, context, accumulated):
        value = run_my_calculation(context.resolved, context.eval_dir)
        return PropertyStageResult(True, {"dipole_debye": value})
```

Register it in the plugin package's `pyproject.toml`:

```toml
[project.entry-points."ffopt.property_evaluators"]
dipole = "my_ffopt_plugin:DipoleEvaluator"
```

The entry point may expose an evaluator instance, a `PropertyEvaluator`
subclass, or a zero-argument factory. Names use lower-case snake_case.

## Property contracts

The five metadata fields above form the evaluator contract. `roles` may contain
`constraint`, `objective`, `promotion`, or `validation`; `fidelity` is an
extensible lower-case identifier such as `structural`, `static_0k`, or
`dynamic_300k`; and `cost_class` is one of `low`, `medium`, `high`, or
`extreme`. Dependencies name other evaluators, while `provides` names numerical
outputs. FFOpt validates all fields at registration time.

For a single explicit object, import and assign `PropertyContract` instead:

```python
from ffopt.properties import PropertyContract, PropertyRole

contract = PropertyContract(
    roles=frozenset({PropertyRole.PROMOTION, PropertyRole.VALIDATION}),
    fidelity="dynamic_300k",
    cost_class="high",
    dependencies=("bulk",),
    provides=frozenset({"elastic_K", "elastic_Cprime", "elastic_C44"}),
)
```

Do not declare a field both on the class and in `contract` with different
values. Legacy plugins that only define `dependencies` and `provides` remain
valid; their compatibility defaults are roles `objective` and `validation`,
fidelity `unspecified`, and cost class `medium`.

Registries can select evaluators and build dependency-complete plans without
hard-coding material names:

```python
selected = registry.select(role="promotion", fidelity="static_0k")
plan = registry.build_plan(role="promotion", fidelity="static_0k")
```

Required dependencies are included even when their own role or fidelity does
not match the filter. `ffopt plugins --json` exposes the normalized contract for
each built-in or third-party evaluator.

## Engine configuration

The following generated/legacy engine configuration is shown for plugin
developers. Ordinary users do not edit YAML, and it is not accepted as a new
schema-1 `property dipole` block:

```yaml
property_evaluators:
  dipole: {enabled: true}

targets:
  dipole_debye:
    value: 4.2
    weight: 1.0
    unit: "D"
```

Dependencies are topologically ordered. Missing dependencies, duplicate names,
unknown evaluators, and active targets with no provider fail during runner
preflight. Inspect the active registry with:

```bash
ffopt plugins
ffopt plugins --json
```

Plugins may call validated `LAMMPSRunner` primitives or implement an independent
calculator. They must write only inside `context.eval_dir` and return numerical
properties through `PropertyStageResult`; objective aggregation remains owned
by FFOpt. A future input-schema extension will let plugins declare their data,
settings, target names, and units to the public compiler.
