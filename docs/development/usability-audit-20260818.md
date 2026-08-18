# Usability and release audit (2026-08-18)

## Sources used

The redesign follows established patterns from scientific and open-source
software:

- LAMMPS uses a line-oriented command input with comments and explicit
  command categories: <https://docs.lammps.org/Commands_structure.html>.
- GROMACS documents each option with its default, choices, and unit, and ships
  a sample input suitable for editing:
  <https://manual.gromacs.org/current/user-guide/mdp-options.html>.
- ForceBalance separates global options from repeated target blocks and uses
  fixed input, temporary, target, and result locations:
  <https://leeping.github.io/forcebalance/doc/html/usage.html>.
- Diataxis separates tutorials, how-to guides, reference, and explanation:
  <https://www.diataxis.fr/start-here/>.
- Semantic Versioning reserves `0.y.z` for initial development and requires a
  new immutable release for changed content:
  <https://semantic-versioning.org/>.
- Python package versions use PEP 440 syntax, including the `0.3.0a3`
  prerelease form:
  <https://packaging.python.org/en/latest/specifications/version-specifiers/>.
- GitHub recommends a concise root README plus repository health files:
  <https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes>.
- PyPA recommends a `pyproject.toml` build contract and isolated, standards-
  based package builds: <https://packaging.python.org/en/latest/guides/writing-pyproject-toml/>
  and <https://packaging.python.org/en/latest/tutorials/packaging-projects/>.
- Keep a Changelog separates unreleased work from immutable dated releases:
  <https://keepachangelog.com/en/1.1.0/>.

## Findings

1. The tracked repository is small; most visible clutter came from ignored
   local research outputs, build products, and historical runs.
2. The public CLI mixed the restartable pipeline with direct developer-stage
   commands. The supported beginner path is now `check`, `doctor`, `run`,
   `status`, `logs`, and `results`; internal Python modules remain available to
   developers.
3. BO candidate count was coupled to machine worker count. This made one- and
   two-node runs scientifically different. `bo batch_size` is now a scientific
   input and workers control concurrency only.
4. `bulk_protocol`, `single_protocol`, `replicate`, and public sample
   `max_workers` were accepted or exposed despite being unused, ambiguous, or
   machine-specific. They are rejected by schema version 1.
5. Property quantities accepted arbitrary trailing tokens. Units are now
   checked for temperature, pressure, timestep, and cutoff.
6. Target tolerances were recorded but did not determine final acceptance.
   Validation now records absolute error, tolerance status, objective limits,
   and maximum percent error in `validation_summary.json`.
7. CPU-only PyTorch was incorrectly reported as a failed CUDA check. CPU
   fallback is now valid unless a CUDA device is explicitly required.
8. Omitting `workflow` could silently select an expensive default. Schema 1
   now requires the user to state the intended ordered stages explicitly.
9. A dry run created provenance directories. `ffopt run --dry-run` is now
   read-only and prints paths without touching the project.
10. Environment snapshots used the runtime-config hash even though their
    content also includes software and host versions. They are now content-
    addressed, and a started run refuses cross-version checkpoint mixing.
11. Python output was block-buffered in SLURM files. Generated scripts now set
    `PYTHONUNBUFFERED=1`, so candidate and property progress is visible live.
12. AL could hand a single-seed optimum directly to validation. Newly
    generated production inputs now use `audit` and `finalize` to select a
    multi-seed robust result before final validation.
13. The repository carried a superseded cluster-smoke input and an unused 0 K
    bulk sublimation template. Both were removed; BTAH acceptance and the NPT
    potential-energy definition are now the sole supported paths.
14. Public CLI help exposed historical aliases and omitted effective defaults.
    Canonical beginner spellings are now emphasized while old aliases remain
    accepted for prerelease compatibility.
15. Generated inputs failed on Unicode source paths and did not quote spaces or
    `#`. They are now UTF-8 and the initializer verifies its own generated file
    by parsing it before returning.
16. Stage-value errors pointed to line 1 instead of the offending BO, sampling,
    ANN, AL, audit, or validation command. Source line metadata is now retained
    through semantic compilation.
17. `validate trajectory final` was a no-op because final validation always
    saves required trajectories. New inputs no longer generate it, while the
    parser retains the exact old spelling so an existing checkpoint can resume.
18. A live two-node test found that the site NFS client on the second node could
    retain the temporary pre-rename request entry indefinitely. Cross-node JSON
    publication now writes and flushes the final unique name directly, while
    readers retry partial payloads. Multi-node `machine test` now exercises the
    complete worker/MPI/node topology so this class of failure is caught before
    scientific BO.
19. Static dead-code inspection found and removed an unreachable BO plotting
    placeholder. Vulture at 80% confidence and Ruff report no remaining high-
    confidence dead-code finding.
20. Multi-node `machine test --dry-run` initially wrote the correct SLURM
    script but printed the superseded single-slot command. The returned plan
    now exposes the exact distributed runner that will execute.
21. Removing inert adsorption runtime defaults and rejecting the old no-op
    trajectory line stranded checkpoints created by an earlier release
    candidate with the same prerelease version. Public inputs still omit and
    reject unused adsorption controls, while exact inert runtime defaults and
    the legacy `trajectory final` spelling are retained internally until a
    versioned migration can replace them.
22. A live NN job showed that requesting one large SLURM task made the
    post-training LAMMPS worker pool see `SLURM_NTASKS=1`, serializing candidate
    validation. Generated NN profiles now stay on one node but request one
    task per available LAMMPS worker, with MPI ranks assigned per task.
23. The adsorption evaluator assumes one fixed, uncharged substrate type, but
    an unknown metal label or charged/multitype slab previously failed only
    after LAMMPS started. Input compilation now inspects all three adsorption
    files, verifies that contract, and checks complex/isolated molecular labels
    before scheduler submission.
24. `ffopt status` compiled the project with its default `local` profile before
    reading an existing pipeline, so an omitted `--machine` mislabeled SLURM
    results. It now prefers the machine profile persisted in pipeline state.
25. Engine logs and generated parameter files still displayed an unrelated
    historical `v8` development label. Those labels were removed; the PEP 440
    package version from `ffopt --version` is now the sole user-facing version.
26. Live end-to-end acceptance reached `finalize` and exposed a resource-layer
    leak: the one-core bookkeeping job instantiated `LAMMPSRunner`, which tried
    to start a four-core SLURM worker and failed immediately. Finalization now
    uses the runner's parameter/file helpers with scheduler startup explicitly
    disabled; upstream scientific stages remain resumable and untouched.
27. Piping verbose status output through `head` exposed an uncaught broken
    stdout pipe. The CLI now follows normal Unix behavior and exits successfully
    without a traceback when the downstream reader closes early.
28. Final live acceptance completed all seven stages on one- and two-node
    profiles. BO, deterministic sampling, robust audit, final parameters,
    calculated properties, final structures, and the 133 MB NPT trajectory
    matched byte-for-byte across profiles; both final validations passed at
    objective 0.005441760. The evidence is preserved in a versioned reference
    report rather than an uncheckable README claim.

## Repository decision

The tracked repository contains source, tests, examples, three small BTAH
regression data roles, documentation, and GitHub health files. Generated
`build/`, `dist/`, caches, egg metadata, scheduler logs, and runs are not part
of the release. Local historical research directories remain ignored and are
not deleted by package cleanup. A future namespace migration may move the
top-level `engine`, `workflow`, `utils`, and `viz` packages below `ffopt`, but
that import-breaking refactor is deliberately deferred until the one-file
public workflow is stable.

Ignored local `runs/`, `archive/`, and research-analysis directories are not
release files and are intentionally preserved; deleting them would destroy
scientific provenance without making the GitHub package smaller.

## Release decision

These are user-visible additions and behavior corrections during initial
development, so the next prerelease is `0.3.0a3`. The input schema remains
`ffopt 1`; removed keywords were never documented as supported public input.
