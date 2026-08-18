# Property evaluator plugins

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

## Project configuration

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
by FFOpt.
