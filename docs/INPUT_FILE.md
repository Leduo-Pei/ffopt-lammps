# FFOpt input reference

## Top level

```text
ffopt 1
project NAME
workflow bo sample nn al validate
```

`ffopt 1` is the required input-schema declaration for version 1. `project
NAME` is only a portable project/result identifier; it does not select
molecule-specific code. Generated results are grouped under `runs/NAME`.

`workflow bo` runs only BO. `workflow bo validate` validates the BO optimum.
The complete common workflow is `bo sample nn al validate`.
`workflow validate` directly evaluates the initial values written on the
`type` lines and saves final trajectories without requiring BO output.

## Parameters

```text
parameters
    range epsilon factor LOW HIGH
    range sigma factor LOW HIGH
    range charge delta SIZE
    charge_limit VALUE
    neutrality derive LABEL
    mixing geometric
    fix epsilon sigma
    type ID LABEL EPSILON SIGMA CHARGE
end
```

- `factor 0.5 2.0` multiplies the initial value by those bounds.
- `delta 0.3` means initial value plus or minus 0.3.
- A trailing `e` is accepted for readability but is optional.
- `absolute LOW HIGH` supplies explicit bounds.
- `charge_limit 1.0` requires every resolved charge to satisfy `|q| <= 1.0`.
- Omitting `fix` optimizes every ranged parameter.
- `fix type LABEL charge` and `range type LABEL charge delta 0.1 e`
  provide per-type overrides.
- The charge named by `neutrality derive` is removed from the independent
  search space and recovered from total charge neutrality.
- `mixing geometric` selects geometric epsilon and sigma mixing.
- `mixing arithmetic` selects geometric epsilon and arithmetic sigma mixing.

## Properties

Bulk uses the fixed minimize + NPT protocol:

```text
property bulk
    data data/crystal.data
    cells_in_data NX NY NZ
    temperature 300 K
    pressure 1 atm
    equilibration 20000
    production 40000
    target density 1.25 g/cm3 weight 1.0 tolerance 0.03
end
```

Sublimation uses bulk NPT mean PE plus minimized single-molecule PE:

```text
property sublimation
    bulk data/crystal.data
    single data/molecule.data
    temperature 298.15 K
    target 98.5 kJ/mol weight 0.3 tolerance 10.0
end
```

Adsorption currently supports deterministic minimization:

```text
property adsorption
    data complex data/complex.data
    data slab data/slab.data
    data molecule data/molecule.data
    protocol minimize
    metal Au
end
```

Without a target, a property is final-validation-only. It does not create BO,
sampling, ANN, or AL labels.

## Methods

Method blocks are optional and override typed package defaults:

```text
bo
    method turbo
    initial_points 48
    max_rounds 300
    early_stop patience 20
end

sample
    points 2000
    centers 16
    radii 0.01 0.025 0.05
    global_fraction 0.0
    seeds 101 202 303
end

nn
    method ann
    ensemble 8
end

al
    acquisition uncertainty
    rounds 2
    candidates 20
end
```

## Commands

```bash
ffopt check ffopt.in
ffopt explain ffopt.in
ffopt run ffopt.in --dry-run
ffopt run ffopt.in
ffopt run ffopt.in --new
ffopt validate --project ffopt.in --initial
```

The expanded engine configuration is recorded as generated JSON under the run
directory for provenance. It is not a user input.
