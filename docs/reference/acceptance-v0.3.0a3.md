# FFOpt-LAMMPS 0.3.0a3 acceptance record

This record captures the release-candidate acceptance run completed on
2026-08-18 at `mag1`. It is evidence for software installation, restart,
single-node, cross-node, and artifact behavior. The packaged BTAH input starts
near a previously validated charge-only solution, so it is not an independent
claim that a small acceptance budget can fit an unknown material from scratch.

## Environment

| Component | Accepted value |
|---|---|
| Python | 3.11.15 |
| FFOpt-LAMMPS | 0.3.0a3 release candidate |
| LAMMPS | 22 Jul 2025, Update 5, OpenMPI build |
| PyTorch | 2.13.0+cpu; CUDA unavailable by design |
| Scheduler | SLURM, `CPU` partition |
| Molecular system | BTAH, charge-only, 13 free charges plus one neutrality-derived charge |

The old `ffopt-cpu` environment and source checkout were removed. The accepted
installation contains one isolated `ffopt` Conda environment; project results
and the pre-existing `vasp_CPU.sh` were retained.

## Machine profiles

| Profile | Nodes | Total cores | Concurrent LAMMPS workers | MPI ranks per worker |
|---|---:|---:|---:|---:|
| `mag1-1node` | 1 | 40 | 10 | 4 |
| `mag1-2node` | 2 | 80 | 20 | 4 |
| `mag1-2node-backfill` | 2 | 32 | 8 | 4 |

The full two-node profile was used for BO. Later two-node stages used the
backfill profile because 40 free cores per node were not immediately
available. This changes throughput only: the candidate budget, seeds, bounds,
targets, and protocols stayed in the same `ffopt.in`.

Final machine tests passed as SLURM jobs `69579` (one node, 3 s) and `69580`
(two nodes, 33 s). The distributed test reported both `ccelab03` and
`ccelab06` and exercised eight workers with four MPI ranks each.

## Scientific budget

| Stage | Acceptance setting |
|---|---|
| BO | TuRBO; 24 LHS points, one warm start, one batch of 24; 49 evaluations total |
| Sampling | 48 deterministic local points around four diverse centres; one acceptance seed |
| ANN | two MLP members per property, hidden layers 64/32, 40 epochs |
| AL | one round, four LAMMPS candidates from a pool of 512 |
| Audit | four distinct candidates, two seeds each |
| Validation | bulk NPT, isolated-molecule minimization, targetless adsorption minimization |

This deliberately reduced budget checks the complete installed workflow. New
materials should use the larger sampling and audit budgets documented in the
user guide.

## Stage record

| Stage | One-node job / elapsed | Two-node job / elapsed |
|---|---|---|
| BO | `69547` / 01:08:16 | `69548` / 01:12:39 |
| Sampling | `69561` / 00:52:00 | `69569` / 01:02:24 |
| ANN | `69568` / 00:48:12 | `69582` / 00:20:36 |
| AL | `69581` / 00:16:28 | `69587` / 00:10:35 |
| Audit | `69585` / 00:17:44 | `69589` / 00:20:17 |
| Finalize | `69591` / 00:00:01 | `69592` / 00:00:01 |
| Validation | `69593` / 00:08:18 | `69594` / 00:08:57 |

The one-node ANN job was retained from before the scheduler-slot correction and
therefore serialized five post-training validations. The corrected NN profile
used by job `69582` supplied four validation slots and reduced that stage from
48:12 to 20:36. Do not interpret node count alone as speed: elapsed time also
depends on worker count, queue placement, node load, NFS latency, and the
slowest candidate.

## Reproducibility

The following one- and two-node artifacts had identical SHA256 content:

| Artifact | SHA256 prefix |
|---|---|
| BO `all_results.csv` | `ef6ba17` |
| Sampling `design.csv` | `3a4a6e5` |
| Sampling `local_results.csv` | `367fbd8` |
| Audit `stable_results.csv` | `b8d39c1` |
| Final atom-parameter table | `7db24c4` |
| Final calculated-property table | `093bb25` |
| Bulk final structure | `d8f0333` |
| Bulk NPT trajectory | `cc2d008` |
| Adsorption-complex final structure | `f41591d` |

The validation JSON files differ only in absolute one-node/two-node provenance
paths. All scientific values and pass/fail decisions are identical.

## Robust selection

The smallest single-seed objective was not accepted blindly. Its two-seed
robust score was poor because of high variability. The finalizer instead chose:

```text
audit mean objective = 0.005663553
audit standard deviation = 0.000221793
robust objective = mean + std = 0.005885345
```

This is the intended BO/ANN/AL contract: candidate generation may use a single
evaluation, while final parameter selection is based on repeated LAMMPS
behavior.

## Final validation

Both profiles produced objective `0.005441760`. Every configured absolute
tolerance passed, and the largest relative error was 1.2224%, below the 3%
acceptance limit.

| Property | Calculated | Target | Relative error | Result |
|---|---:|---:|---:|---|
| a (A) | 4.229526 | 4.242200 | 0.2988% | PASS |
| b (A) | 17.751878 | 17.827000 | 0.4214% | PASS |
| c (A) | 20.705080 | 20.685000 | 0.0971% | PASS |
| alpha (degree) | 73.424296 | 72.630000 | 1.0936% | PASS |
| beta (degree) | 86.819278 | 87.150000 | 0.3795% | PASS |
| gamma (degree) | 87.284061 | 86.230000 | 1.2224% | PASS |
| density (g/cm3) | 1.330545 | 1.328500 | 0.1539% | PASS |
| sublimation estimate (kJ/mol) | 98.539724 | 98.500000 | 0.0403% | PASS |

Targetless adsorption validation reported `-43.552844 kcal/mol`; it is an
output, not a fitted experimental claim. The sublimation quantity is the
documented potential-energy estimate, not a full finite-temperature free-energy
correction.

## Artifacts

Each accepted pipeline contains:

```text
runs/btah_acceptance/pipelines/acceptance/
|-- bo/all_results.csv
|-- sample/local_results.csv
|-- nn/forward_nn.pt
|-- al/active_learning_history.json
|-- audit/stable_results.csv
|-- finalize/final_summary.json
`-- validate/
    |-- validation_summary.json
    |-- computed_properties.csv
    |-- final_atom_parameters.csv
    |-- final_parameters.json
    |-- final_parameters.lammps
    `-- eval_0000/                 # structures and trajectories
```

The complete one-node and two-node acceptance directories were each 207 MB,
primarily because final validation intentionally saves the bulk NPT trajectory.

## Defects found by live acceptance

The test found and verified fixes for four issues that unit tests alone did not
fully expose:

1. NFS clients on the second node could retain a pre-rename request entry;
   direct flushed publication and partial-read retries fixed the worker queue.
2. NN requested one large task, which serialized LAMMPS candidate validation;
   generated profiles now request one task per validation slot.
3. The one-core finalization job unnecessarily started a four-core LAMMPS
   worker; scheduler startup is now disabled for this pure-Python stage.
4. Status without `--machine` mislabeled a recorded SLURM pipeline as `local`;
   it now reads the persisted profile.

Both state databases finish with all seven stages `completed`, all expected
artifacts present, and no FFOpt scheduler job or watcher left running.
