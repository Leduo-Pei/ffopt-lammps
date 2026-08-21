# Run an elemental BCC campaign

The material workflow keeps the same project layout as a molecular fit:

```text
my_project/
|-- ffopt.in
|-- data/
`-- runs/                 # created and resumed by FFOpt
```

Start from `examples/fe_bcc/ffopt.in`, copy the referenced data files, and run:

```bash
ffopt check ffopt.in
ffopt explain ffopt.in
ffopt run ffopt.in --machine cluster --dry-run
ffopt run ffopt.in --machine cluster --watch
```

The same example directory contains `ffopt.canary.in`. It is prominently
marked `NON_SCIENTIFIC_CANARY`: its small replicas, short trajectories,
relaxed gates and narrow ranges test SLURM execution and restart wiring only.
Never publish its numerical values or use them as a fitted Fe potential. Run
the canary with a fixed ID so repeating the identical command addresses the
same saved pipeline:

```bash
ffopt check ffopt.canary.in
ffopt run ffopt.canary.in --machine cluster --run-id fe_bcc_canary_a4 --dry-run
ffopt run ffopt.canary.in --machine cluster --run-id fe_bcc_canary_a4 --watch
```

Interrupting the local `--watch` process does not require another scientific
submission command. Repeat the last command: the controller reconnects to an
active SLURM job or resumes the first incomplete manifested work unit. The
canary must show two concrete executed stages, `constrained_al_01` and
`constrained_al_02`, before `finalists` and `validate` complete.

The last command owns the complete campaign. `al rounds 8` does not mean the
user submits eight commands. It is the maximum number of independently
restartable AL jobs that the stage controller may advance. If the terminal or
controller is interrupted, run the same command again; immutable stage and
candidate manifests determine exactly what can be reused.

To inspect or deliberately bound that same managed run:

```bash
ffopt status ffopt.in --machine cluster
ffopt results ffopt.in
ffopt run ffopt.in --machine cluster --watch --until screen
ffopt run ffopt.in --machine cluster --watch --from-stage al
```

The public bounds use the same concise names as `ffopt.in`. `screen` covers the
candidate assembly and static cubic calculation; `al` covers all configured
restartable constrained-AL rounds. `ffopt status` also shows their concrete
names (`candidates`, `static`, `constrained_al_01`, and so on) when finer-grained
recovery is needed.

For material elasticity, parallelism is exclusively a machine-profile
decision. `parallel.max_workers` caps simultaneous candidate/seed work units,
`parallel.cores_per_worker` is the MPI-rank count of one elastic state, and
`parallel.omp_threads_per_worker` is the thread count of each rank. The runner
derives inner strain-state concurrency from the stage's total allocated CPUs
and rejects any candidate-worker × state-worker × MPI × OMP plan that would
oversubscribe it. These resource fields therefore do not belong in
`ffopt.in` and do not change the scientific candidate budget.

## Evidence flow

```text
structural BO coverage
  -> multi-centre local/global sampling
  -> independent-seed structural audit
  -> exact 0 K cubic screen
  -> constrained-minimax surrogate and AL
  -> diverse 300 K finalist screen
  -> independent final validation
  -> static rank + dynamic rank + final result bundle
```

Structure, density, angles, and surface energy are constraints. Their
continuous violation is zero inside the declared tolerance. The exact static
objective is the maximum relative error among independent `B`, `Cprime`, and
`C44` targets. RMSE and parameter contrast break ties. The requested 20%
mechanical tier labels result quality but never removes the best structurally
valid candidate.

Static and finite-temperature elastic calculations use independent target
triplets and independent protocols. `G`, `E`, and Poisson's ratio are derived
diagnostics, not additional fit dimensions. A finite-temperature ranking can
reverse the static ranking, so every result row keeps both ranks and its
evidence level. In the packaged example, dynamic promotion uses seeds
`101 202 303`, while final validation uses the disjoint holdout set
`404 505 606` declared by `validation_seeds`; deterministic replay of the same
trajectories is not counted as independent validation.

After final validation, the principal user-facing products are under
`runs/<project>/pipelines/<run-id>/validate/`: `validation_summary.json`,
`final_parameters.json`, `computed_properties.csv`, `model_adequacy.json`, and
`TOP_PARAMETERS.csv`/`.json`/`.md`. The Top-N files are generated from the
explicit static and dynamic ranking paths recorded by the pipeline, never from
the newest directory on disk.

## Ordered two-type elemental warning

Two permanent atom types on corner and body sites are an ordered-sublattice LJ
surrogate, not automatically a transferable elemental potential. `tie epsilon
all` and a bounded sigma difference reduce artificial contrast but do not
restore invariance to relabelling identical Fe atoms. Final reporting therefore
includes or requires:

- same-element label-swap sensitivity;
- both surface terminations;
- vacancy/short diffusion sensitivity;
- `C11`, `C12`, `C44`, `Cprime`, Cauchy difference, and Born margins;
- comparison with an EAM/Finnis--Sinclair baseline for transferable Fe use.

`mixing default` means FFOpt does not issue `pair_modify mix`. With `lj/cut`,
LAMMPS resolves the default geometric mixing rule. The rule is fixed and never
optimized.
