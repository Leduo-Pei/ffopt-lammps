# Changelog

All notable changes are recorded here. The format follows Keep a Changelog;
Python releases use PEP 440 identifiers and SemVer-style compatibility
expectations while the public API remains in alpha.

## Unreleased

## 0.3.0a7 - 2026-08-24

### Changed

- Qualify structural-coverage BO on two independent axes: surrogate-guidance
  integrity and measured feasible-region evidence.  The summary now records
  unique fallback rounds, the model component and actual fallback used,
  normalized feasible-archive affine rank, and genuine outside-boundary anchor
  counts.  Material sampling accepts complete coverage canonically, labels thin
  but non-empty evidence as recovery-only, and stops before compute when no
  strict feasible seed exists.
- Require canonical multi-center evidence to be meaningfully dispersed as well
  as full rank. The first dimension-aware maximin subset must pass normalized
  pair-separation and weakest-direction RMS-span gates; almost coincident point
  clouds remain explicitly recovery-only.
- Expose concise `coverage min_archive`, `min_anchors`, and `max_fallbacks`
  controls in `ffopt.in`, with dimension-aware defaults.  Sampling provenance
  pins the coverage summary alongside the feasible and boundary source files.

### Fixed

- Copy capped coverage-GP objective arrays before replacing non-finite values.
  This restores model-guided acquisition beyond 512 observations under
  Pandas copy-on-write instead of silently degrading later BO rounds to
  novelty selection.
- Preserve the surviving model signal when only one coverage surrogate fails:
  the objective GP can replace a failed feasibility classifier, while a valid
  classifier remains authoritative when only the objective GP fails.
- Parse immutable `.json` runtime configurations as strict JSON, rejecting
  duplicate keys, non-finite or overflowing numbers, and YAML-only syntax.
  Legacy YAML remains supported and cross-format includes use each file's own
  parser, eliminating numeric-type drift between pipeline provenance and BO
  checkpoint identity.
- Bind direct material sampling to the exact coverage summary and hashed
  feasible/boundary CSV artifacts. Boundary quota now uses only measured
  outside anchors: a farther point is labelled recovery-only, while a campaign
  with no outside evidence reallocates that quota to global exploration instead
  of oversampling the feasible interior.

## 0.3.0a6 - 2026-08-22

### Fixed

- Make symmetric three-mode stress--strain slopes the canonical 0 K cubic
  elasticity estimator.  The former energy-curvature estimator is invalid for
  an unshifted hard-cutoff `lj/cut` potential when a neighbour shell crosses
  the cutoff; it is now retained only as a labelled consistency diagnostic and
  can neither train the surrogate nor rank candidates.
- Record the full zero-pressure reference stress tensor and triclinic box
  geometry in every static candidate, and retain per-strain pressures with
  explicit units.  Static fit quality now comes from the same stress response
  that supplies `B`, `Cprime`, and `C44`.
- Split `static_strain` from `dynamic_strain`. Static central differences use
  the two smallest magnitudes to extrapolate `M(h)=M0+q*h^2` to zero strain;
  outer magnitudes audit window drift, while finite-temperature strains remain
  large enough to resolve the signal above trajectory noise. The legacy
  `strain` spelling remains a mutually exclusive compatibility alias.
- Add the public `static_drift VALUE percent` hard eligibility gate (5% by
  default). It rejects a static candidate when any canonical zero-strain
  modulus is too sensitive to the outer-shell/full-window extrapolation audit,
  even when the raw stress fit passes its R2 threshold.
- Close the downstream qualification chain: constrained refinement now
  requires the recorded protocol-specific `fit_quality_pass` flag in addition
  to its scalar R2 check, and Top-N evidence reports retain the measured static
  drift and configured limit.

## 0.3.0a5 - 2026-08-22

### Changed

- Require one explicit global `cutoff VALUE A` in the `parameters` block and
  lock that value across every fitted and validation property. No workflow may
  fall back to a hidden cutoff; elemental BCC inputs additionally require
  `cutoff >= 2.5 * sigma_max`, where `sigma_max` is the largest declared sigma
  optimization bound. Periodic BCC cells also require
  `cutoff <= 0.45 * h_min` after replication, using triclinic-safe face
  heights; generated includes explicitly lock `shift no` and `tail no`.
- Separate the inexpensive multi-seed 300 K finalist promotion protocol from
  the independent long-trajectory validation protocol for the promoted winner,
  and record both protocols in the scientific identity.
- Make the production Fe workflow require a hard floor of 20 structurally and
  mechanically eligible finalists. If fewer than 20 survive, promotion stops
  before launching any finite-temperature LAMMPS work instead of silently
  evaluating a smaller set.
- Evaluate the declared BO centre once as a structural warm-start gate before
  launching LHS candidates, retaining the successful result as exact evidence
  and stopping cheaply when the data/cutoff/protocol no longer reproduces the
  known feasible baseline.

### Fixed

- Select the nested single-node MPI launcher dialect explicitly and fail on
  ambiguous `mpiexec`/`mpirun` paths instead of passing Open MPI-only flags to
  Intel MPI/Hydra. Machine acceptance now exercises every configured worker
  even when the complete worker pool fits on one SLURM node.
- Store newly configured machines in a concise profile schema while expanding
  the existing stage-specific runtime resources internally; legacy expanded
  TOML and YAML profiles remain readable.
- Let `init`, `inspect`, and `data check` consume packaged `builtin:data/...`
  references, document the exact BTAH example paths, and distinguish shell-
  relative paths from paths relative to `ffopt.in`.
- Separate local-workstation and SLURM instructions in generated projects,
  clarify that `local` bypasses the scheduler, and identify `--partition` as a
  site-specific SLURM partition rather than a node name.
- Fingerprint the bytes, size, resolved path, and role of every LAMMPS data
  input. A coordinate-only edit at the same path now changes the pipeline
  scientific identity and cannot reuse old stage evidence.
- Bind every BO checkpoint to the exact pipeline-stage signature, scientific
  configuration, optimization policy, resolved BO method, and ordered parameter
  space. Explicit and automatic resume both reject legacy or incompatible
  `train_X`/`train_Y` before loading them.
- Treat an unsuccessful structural-validation execution as retryable rather
  than publishing an elastic candidate from incomplete properties. Downstream
  artifacts left by affected prereleases are moved into a recoverable attempt
  archive before the retry.
- Verify manifest identity, regular-file type, declared outputs, and SHA-256 for
  every material-stage completion or reuse path, including local execution,
  SLURM completion, resume, and `--from-stage` upstream checks.
- Honor `born off` during elastic ranking and reject a `finalists` workflow at
  input-validation time unless its dynamic elasticity module has the required
  `promotion` role.

## 0.3.0a4 - 2026-08-21

### Added

- Add an experimental, opt-in elemental-BCC workflow to the common one-input
  pipeline: structural feasible-region BO, multi-centre sampling, replicated
  audit, exact 0 K cubic elasticity, constrained-minimax GP active learning,
  finite-temperature finalists, independent holdout validation, and Top-N
  reporting.
- Add typed material/crystal declarations, zero-charge elemental atom types,
  parameter ties and bounded same-element differences, native LAMMPS default
  mixing, and role/fidelity/cost property contracts without changing legacy
  molecular input semantics.
- Add content-addressed candidate/stage manifests and explicit multi-round
  material-stage graphs so one `ffopt run ... --watch` command advances AL and
  the same command safely resumes an interrupted campaign.
- Make `ffopt doctor` validate all active bulk, isolated-molecule, adsorption
  complex, slab, and adsorbate data files as one cross-file physical contract.
- Add `ffopt --debug ...` and concise default CLI errors so input and machine
  mistakes remain readable while full tracebacks are still available.

### Fixed

- Honor the elemental-material sampling split exactly: feasible local centers,
  measured boundary centers, and global coverage now receive their declared
  quotas, and the replicated audit ranks the explicit BO+Sample union rather
  than silently ignoring Sample evidence.
- Report the effective LAMMPS `lj/cut` default as geometric epsilon and sigma
  mixing in `ffopt explain`; execution already delegated this rule correctly.
- Preserve active SLURM stages when `squeue` or `sacct` is temporarily
  unavailable or has not yet indexed a job, preventing an inconclusive query
  from triggering a duplicate expensive submission.
- Bound scheduler status queries to ten seconds and fall back from `squeue` to
  `sacct` without exposing command or timeout tracebacks to users.
- Write `machines.toml` through a flushed temporary file and atomic replace so
  an interrupted configure operation cannot truncate working profiles.
- Report malformed machine TOML, unavailable `sbatch`, and failed `sinfo`
  probes with actionable recovery messages.
- Run installed-wheel smoke tests outside the source checkout so release CI
  cannot accidentally import uninstalled repository modules.
- Include the README workflow SVG and editable Visio source in source
  distributions and enforce their presence during release verification.

## 0.3.0a3 - 2026-08-18

### Added

- Add `ffopt --version` for installation and bug-report diagnostics.
- Add content-addressed input, runtime-configuration, and environment
  snapshots to every pipeline run.
- Package the complete BTAH adsorption data set so every installed example is
  self-contained.
- Fingerprint the packaged scientific-acceptance input and data so a modified,
  incomplete, or stale self-test directory cannot be silently reused.
- Reject self-test directories created by another FFOpt version or acceptance
  definition, even when their visible input files happen to be unchanged.
- Add complete public CLI option descriptions, accept either an atom-type ID
  or label in `ffopt init --derive-charge`, and reject stage jumps whose
  required upstream artifacts are incomplete.
- Enforce portable, path-safe names for reusable machine profiles.
- Expand `ffopt explain` with the effective BO budget, stopping rule,
  sampling design, ANN architecture, AL budget, and final acceptance gates.
- Show target tolerances and the post-AL robust-audit budget in `ffopt
  explain`.
- Dynamically backfill Linux LAMMPS worker slots on Python 3.11+ while still
  recycling each process after one candidate, eliminating wave-level
  straggler stalls without weakening MPI isolation.
- Keep `local` as the only zero-configuration machine; require an explicit
  named profile for every cluster so scheduler resources cannot come from a
  hidden generic template.
- Export a complete resolved per-type epsilon/sigma/charge table from every
  final validation and fail acceptance on missing, non-finite, or unchecked
  fitted properties instead of allowing incomplete metrics to pass.
- Remove the unused 0 K bulk-minimization sublimation template and helper; the
  sole supported definition now unambiguously reuses bulk NPT mean potential
  energy plus isolated-molecule minimization.
- Stop duplicating the immutable pipeline runtime configuration in every
  stage and remove the unconsumed `final_parameters.yaml`; JSON, CSV, and the
  include-ready LAMMPS file remain the supported final exports.
- Make post-AL three-seed auditing and robust finalization part of newly
  generated production workflows, with explicit `audit top_k` and `seeds`
  controls in `ffopt.in`.
- Require an explicit `workflow` declaration so an omitted line can never
  trigger an unintended expensive default pipeline.
- Make `ffopt run --dry-run` strictly read-only; provenance and run
  directories are now created only when execution actually begins.
- Stream Python progress to SLURM logs without block buffering so completed
  candidates and property tables are visible while a stage is running.
- Line-buffer the public CLI as well, keeping redirected `--watch` and
  `self-test --watch` logs current after an SSH terminal disconnects.
- Make `ffopt status` show live SLURM state/reason when available and summarize
  resumable BO checkpoint round, evaluation count, and best objective.
- Test every declared Python version (3.10, 3.11, and 3.12) in CI and run
  `pip check` before regression tests.
- Include the complete user/developer documentation, changelog, citation,
  security policy, and example guides in source distributions.
- Publish a `SHA256SUMS.txt` file beside every GitHub Release wheel and source
  distribution.
- Publish a versioned one/two-node acceptance record with actual SLURM jobs,
  timings, scientific gates, artifact hashes, and final BTAH property values.
- Isolate generated SLURM jobs from user-level Python packages with
  `PYTHONNOUSERSITE=1`.
- Install and execute each built wheel in CI and release jobs so command-line
  entry points and packaged scientific-acceptance resources are tested after
  construction, not only from the source tree.
- Let `ffopt init --target` accept an explicit tolerance, write all effective
  tolerances into generated input, and correct the generated sublimation
  energy-difference formula.
- Report the warm-start centre separately from Latin-hypercube points so the
  initial and total BO evaluation budgets match the actual LAMMPS workload.
- Reject duplicate parameter directives, unknown per-type override labels,
  and fixed/ranged contradictions with source-line diagnostics instead of
  silently overwriting or ignoring them.
- Remove unused adsorption temperature/timestep/seed/run-length fields and
  redundant bulk/sublimation protocol switches from the public input; schema 1
  accepts only property settings that affect the implemented LAMMPS
  calculation. Inert internal defaults remain temporarily to preserve
  prerelease checkpoint hashes.
- Infer molecule size exclusively from the required isolated-molecule data
  file instead of exposing a contradictory `molecule_atoms` override.
- Reject non-finite numbers and invalid physical run controls during
  `ffopt check`, including non-positive cutoffs/timesteps/seeds and bulk
  production windows too short for the fixed averaging interval.
- Stop generating the no-op `validate trajectory final` line; schema 1 final
  validation saves relevant structures and trajectories by default.
- Expose BO stability-audit candidate and seed budgets in `ffopt.in`, and
  report search, audit, and maximum total LAMMPS evaluations separately in
  `ffopt explain` and BO logs.

### Fixed

- Quote generated `ffopt.in` data paths automatically when source filenames
  contain spaces or `#`, so `ffopt init` projects remain directly parseable.
- Stop generating the obsolete no-op `validate trajectory final` setting while
  continuing to read it for checkpoint compatibility; final validation always
  saves the trajectories required by enabled properties.
- Show the effective BO method in `ffopt explain` and `ffopt doctor`, including
  an explicit warning when high-dimensional `auto` mode falls back from SAASBO
  to TuRBO because Pyro is unavailable.
- Remove an unreachable placeholder expression from BO parameter-space plotting.
- Preserve source line numbers for stage settings so invalid BO, sampling, ANN,
  AL, audit, and validation values point to the exact `ffopt.in` line.
- Stream periodic per-candidate progress from LAMMPS batches so a single slow
  tail evaluation no longer makes BO or AL logs appear stalled.
- Write generated inputs as UTF-8 and verify quoted Unicode data paths.
- Replace cross-node SLURM queue publication by atomic rename with direct,
  flushed JSON publication plus partial-read retries. This fixes NFS clients
  that indefinitely retain the pre-rename directory entry on another node.
- Make `ffopt machine test` exercise the real distributed worker count, MPI
  ranks, and node count for multi-node SLURM profiles instead of testing only
  one validation slot.
- Reject SLURM profiles with fewer workers than nodes, which would request
  nodes that can never participate.
- Isolate every distributed machine test by SLURM job ID so stale worker-ready
  files from an earlier test cannot satisfy a later startup check.
- Allocate multiple worker tasks inside the single-node NN job so its
  post-training LAMMPS candidate validation runs concurrently instead of being
  serialized behind `SLURM_NTASKS=1`.
- Show the actual distributed runner in multi-node `machine test --dry-run`
  output instead of the superseded single-slot preview command.
- Reject a missing, charged, or multitype fixed adsorption substrate during
  `ffopt check`, and require the complex molecular labels to match the isolated
  adsorbate instead of failing later inside LAMMPS.
- Make `ffopt status` report the machine profile persisted with an existing
  pipeline when `--machine` is omitted, instead of misleadingly labelling a
  SLURM run as `local`.
- Keep the pure-Python `finalize` stage from starting a SLURM LAMMPS worker
  pool; one-core finalization now writes robust parameter artifacts without a
  contradictory per-evaluation CPU request.
- Exit cleanly when CLI output is piped to a consumer such as `head` that
  closes early, instead of printing a `BrokenPipeError` traceback.

### Changed

- Remove unused epsilon and sigma range declarations from charge-only examples.
- Remove the superseded cluster-smoke input in favor of the gated acceptance
  workflow, and label evaluator APIs that are not yet available in `ffopt.in`.
- Move Matplotlib to the optional `plots` extra because the supported pipeline
  does not import plotting code.
- Preserve the exact command/configuration of an active scheduler stage when
  machine settings are changed during a resume check.
- Keep dimensionality-based `auto` fallback, but reject an unavailable
  explicitly requested SAASBO method and provide a dedicated optional extra.
- Document strict Conda/user-site isolation and scheduler-friendly acceptance
  wall times.
- Distinguish the minute-scale machine smoke test from the hours-scale BTAH
  scientific acceptance instead of presenting both as a five-minute run.
- Scale generated focused-sampling budgets conservatively with parameter
  dimensionality (1500--2500 distinct points) for production-oriented fits.
- Make validate-only inputs genuinely range-free and report zero search
  dimensions when no optimization stage is active.
- Make the built-in serial `local` profile visible to `machine list/show/test`
  instead of requiring a redundant user profile.
- Decouple parameter-space inspection and final LAMMPS validation from the BO
  implementation so validate-only/core installations do not import Torch or
  BoTorch; `doctor` now checks optional runtimes only for active stages.
- Make `bo`, `nn`, `saasbo`, and `xgboost` extras independently complete and
  move SciPy/scikit-learn out of the lightweight validation core.
- Reject SLURM profiles whose per-evaluation timeout is not shorter than the
  scheduler wall time, and use a two-hour timeout in the six-hour acceptance
  profile.
- Enforce portable atom-type labels before they become parameter or CSV column
  names.
- Stop writing the obsolete mutable `runs/<project>/_configs` copy; checks are
  read-only and stages consume only immutable provenance snapshots.
- Hide the redundant public `--resume` spelling while continuing to accept it
  for prerelease compatibility; repeating `ffopt run` always resumes.
- Normalize Python, command input, LAMMPS data/template, documentation,
  citation, and shell-script files to LF for portable SLURM checkouts.
- Remove historical engine labels such as `v8` from logs and generated
  parameter files; the package version reported by `ffopt --version` is the
  only user-facing software version.

## 0.3.0a2 - 2026-08-18

### Changed

- Move XGBoost to the explicit `xgboost`/`all-models` extras so the default
  ANN/BO installation on CPU clusters does not download CUDA/NCCL packages.

## 0.3.0a1 - 2026-08-18

### Added

- Add packaged BTAH machine and scientific acceptance testing with explicit
  objective, relative-error, and absolute-tolerance gates.
- Add a Diataxis-style documentation site, complete Chinese user manual,
  machine-profile reference, output contract, contribution guide, citation
  metadata, and structured issue templates.
- Add strict target-unit validation and richer semantic checks for parameter,
  property, BO, sampling, ANN, AL, and validation settings.
- Add final validation acceptance status to JSON and CSV outputs.

### Changed

- Decouple scientific BO `batch_size` from machine `workers`; one- and
  two-node profiles now evaluate the same candidate budget.
- Reduce the public CLI to one restartable pipeline command plus creation,
  checking, machine, monitoring, and result commands.
- Treat CPU-only PyTorch as valid unless a CUDA device is explicitly required.
- Organize documentation by tutorial, how-to, reference, explanation, and
  development history.

### Removed

- Remove legacy per-stage SLURM submission scripts and obsolete standalone
  sensitivity/replicate utilities from the installed package.

## 0.2.0a3 - 2026-07-30

- Make the SLURM machine test use the same node-local MPI launcher as
  production LAMMPS evaluations.
- Include both scheduler stdout and stderr when a machine test fails.

## 0.2.0a2 - 2026-07-30

- Add molecular LAMMPS data contract and cross-file compatibility checks.
- Add local/SLURM environment probing and executable smoke tests.
- Generate crystal, adsorption, and combined projects from `ffopt init`.
- Support targetless validation-only project generation.
- Add pipeline artifact and SLURM log navigation commands.
- Expand beginner documentation and BTAH regression coverage.

## 0.2.0a1

- Introduce the one-file `ffopt.in` interface, restartable pipeline state, and
  reusable `machines.toml` execution profiles.
