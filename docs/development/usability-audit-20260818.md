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
- GitHub recommends a concise root README plus repository health files:
  <https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes>.

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

## Release decision

These are user-visible additions and behavior corrections during initial
development, so the next prerelease is `0.3.0a1`. The input schema remains
`ffopt 1`; removed keywords were never documented as supported public input.
