# Material-workflow unification design

Status: implementation design for the `feature/material-workflows` branch.
The existing molecular-crystal workflow remains the compatibility baseline;
the Fe BCC implementation remains the scientific regression baseline until
both workflows pass the same packaged release.

## Decision

FFOpt-LAMMPS will have one public codebase and one user-facing project layout:

```text
my_project/
|-- ffopt.in
|-- data/
`-- runs/                 # generated and restartable
```

Material-specific behavior is expressed through typed parameter constraints,
property plugins, and a declared workflow graph. It is not selected by copying
different framework trees or by asking users to assemble YAML modules.

The GitHub implementation supplies the execution kernel: strict input parsing,
immutable compiled input, machine profiles, persistent stage state, SLURM
submission, artifact verification, packaging, CI, and documentation. The BCC
implementation supplies the new scientific components: feasible-region
coverage, structural auditing, exact 0 K cubic elasticity, constrained-minimax
acquisition, finite-temperature mechanical screening, and layered final
validation.

## Scientific workflow roles

A property declares both *what is evaluated* and *how its result is used*:

| Role | Meaning | Fe BCC example | Molecular example |
|---|---|---|---|
| `constraint` | A continuously measured feasible-region condition | lattice, angles, density, surface energy | stable NPT cell |
| `objective` | Minimized only inside the feasible region | 0 K `max(error(B), error(Cprime), error(C44))` | weighted target loss |
| `promotion` | Costlier evidence used to select finalists | 300 K cubic elasticity | long production or adsorption |
| `validation` | Independent calculation that never trains the model | all final Fe properties | all final molecular properties |

Pass/fail flags are reporting views of continuous margins. A constraint inside
its tolerance has zero violation; optimization does not chase its experimental
centre at the expense of another accepted property. Raw calculated values,
errors, uncertainty, tolerance, provenance, and pass/fail state are all kept.

For cubic elasticity, the independent target quantities are `B`, `Cprime`, and
`C44` (or equivalently `C11`, `C12`, and `C44`). `G`, `E`, and Poisson's ratio
are derived diagnostics and must not be counted again as independent fitting
dimensions.

## Fe BCC graph

The managed graph is:

```text
structural BO
  -> multi-centre sample
  -> replicated structural audit
  -> candidate merge and exact 0 K labels
  -> low-dimensional surrogate
  -> constrained-minimax AL round
       |-- not stopped: next constrained-minimax AL round
       `-- stopped: 300 K finalist screen
  -> independent validation
  -> ranked result bundle
```

BO identifies and covers the structural feasible region; it does not force an
elastic compromise into the first scalar acquisition. Each AL round evaluates
structural candidates first and promotes eligible points to static elasticity.
Within the observed structural gate, the exact objective is the maximum
relative error of the three independent static elastic targets. RMSE, parameter
contrast, and structural margin are deterministic tie-breakers, not substitutes
for the primary objective.

AL is iterative internally but must not require one command per round. A user
runs one managed command. The scheduler executes each round as a separate job
so that a 24-hour limit does not apply to the whole campaign. The state machine
submits the next round until a scientific stop condition, maximum budget, or a
real failure is recorded.

## Execution state and provenance

Each stage attempt has separate execution and scientific outcomes. For
example, a final validation may execute successfully but classify the best
candidate as `best_effort`; that is neither a scheduler failure nor an accepted
target-tier result.

Every reusable artifact records:

- the exact input and scientific-configuration hashes;
- upstream artifact hashes;
- a canonical parameter key and the resolved type parameters;
- the exact seed set;
- the producing stage, round, attempt, code version, and command;
- expected output hashes and a completion status.

Manifests are written atomically and validated before reuse. A missing,
corrupted, mismatched, or only partially written manifest fails closed. Managed
runs pass explicit artifact identities downstream; modification-time discovery
is not allowed inside a pipeline.

One candidate is the smallest expensive recovery unit. Restarting an audit,
static batch, or finite-temperature batch reuses only candidates whose parameter
key, seed set, configuration, inputs, and output hashes all match. Candidate
rank is display metadata and is never a cache key.

## Stop states

The workflow distinguishes:

- `static_search_converged`: static constraint coverage and constrained improvement
  have stopped changing within the declared patience. This is not whole-workflow
  scientific convergence; dynamic finalist and independent-seed validation remain;
- `target_tier_reached`: the requested reporting band is achieved, but coverage
  and stability conditions still determine whether AL may stop;
- `budget_exhausted`: the configured evaluation or scheduler budget ended;
- `model_inadequate`: the requested physics cannot be represented within the
  model and declared applicability domain;
- `completed_best_effort`: execution completed and the best structurally valid
  candidate was validated, but the requested mechanical tier was not reached;
- `failed`: an unrecoverable execution or data-contract error occurred.

Maximum rounds and wall time are budget stops, not proof of convergence.

## Model-form audit for elemental metals

A conventional BCC lattice has translationally equivalent corner and body
sites. Giving those sites permanent different atom types can provide extra
fitting freedom, but it changes the model into an ordered two-sublattice
surrogate. Different same-element sigma values make energies depend on an
artificial label assignment and can affect surfaces, vacancies, diffusion,
reconstruction, and interfaces.

The current two-type LJ model therefore requires an explicit applicability
contract and preflight tests:

1. swap same-element labels and quantify the energy/property change;
2. compare both surface terminations;
3. test vacancy motion and a short finite-temperature trajectory for label
   sensitivity;
4. report `C11`, `C12`, `C44`, `Cprime`, and the Cauchy difference, not only
   isotropic derived moduli;
5. classify the result as an ordered-sublattice LJ surrogate unless these tests
   justify a broader claim.

The workflow must also support a true single-type elemental model and future
many-body backends. If physical transferability for alpha-Fe is the goal, an
EAM/Finnis-Sinclair/MEAM or modern many-body model is the appropriate comparison
baseline; additional BO cannot repair a missing model degree of freedom.

## Compatibility and release gates

The migration is complete only when all of the following pass from one built
wheel/sdist:

1. every existing molecular/BTAH unit and acceptance test;
2. strict parser tests for molecular, elemental, and multicomponent inputs;
3. an Fe BCC small-budget wiring test with deliberate interruption/resume;
4. candidate-manifest corruption and mismatch tests;
5. an AL test that advances multiple rounds from one public command;
6. finite-temperature batch partial-resume tests keyed by parameters, not rank;
7. a complete Fe scientific campaign and an independent final validation;
8. package, documentation, and Python-version CI.

Until those gates pass, the standalone `01_BCC` tree is retained read-only as
the numerical reference. It is archived only after the unified release can
reproduce the required artifacts and rankings.
