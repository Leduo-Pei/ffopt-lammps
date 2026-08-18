# Property evaluator plugins (advanced)

The schema-1 `ffopt.in` compiler currently exposes `bulk`, `sublimation`, and
`adsorption`. Run `ffopt plugins` and check the `ffopt.in=yes` marker before
using a property in the one-file interface. The surface evaluator and external
entry points are engine APIs retained for development and legacy generated
configurations; installing an evaluator alone does not add new `ffopt.in`
syntax.

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
