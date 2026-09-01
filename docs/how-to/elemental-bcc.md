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

The `parameters` block must contain one explicit global cutoff, for example:

```text
parameters
    range sigma absolute 0.001 5.0
    cutoff 12.5 A
    # remaining constraints and type rows ...
end
```

No property supplies a fallback or override. Bulk, surface, static elasticity,
finite-temperature promotion, and final validation all use this same value.
For elemental BCC, FFOpt checks `cutoff >= 2.5 * sigma_max` against the largest
sigma bound in the complete optimization domain; the example above therefore
requires at least `12.5 A`. Changing it creates a different scientific protocol
and requires a new campaign.

FFOpt additionally requires `cutoff <= 0.45 * h_min` for each periodic bulk,
surface, and elasticity cell after replication. Here `h_min` is computed from
the full (including triclinic) box vectors. This leaves contraction/strain
headroom below the half-box limit; increase `replicate` when needed. Energy
shift and analytical tail correction are explicitly locked off.

The same example directory contains `ffopt.canary.in`. It is prominently
marked `NON_SCIENTIFIC_CANARY`: its small replicas, short trajectories,
relaxed gates and narrow ranges test SLURM execution and restart wiring only.
Never publish its numerical values or use them as a fitted Fe potential. Run
the canary with a fixed ID so repeating the identical command addresses the
same saved pipeline:

```bash
ffopt check ffopt.canary.in
ffopt run ffopt.canary.in --machine cluster --run-id fe_bcc_canary_a6 --dry-run
ffopt run ffopt.canary.in --machine cluster --run-id fe_bcc_canary_a6 --watch
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
  -> 0 K cubic stress-slope screen extrapolated to zero strain
  -> constrained-minimax B/G/E/nu surrogate and AL
  -> 38 cluster-balanced candidates with one quick 300 K seed
  -> exact reranking, then 10 candidates with the two remaining seeds
  -> one promoted winner in an independent long-trajectory validation
  -> static rank + dynamic rank + final result bundle
```

Structure, density, angles, and surface energy are constraints. Their
continuous violation is zero inside the declared tolerance. The static
objective is the maximum relative error among the configured properties. The
packaged Fe input uses `B/G/E/nu`; `Cprime/C44` remain exact Born-stability and
anisotropy diagnostics. It uses symmetric stress slopes; energy curvature is
diagnostic-only because an unshifted LJ cutoff makes energy discontinuous at
neighbour-shell crossings. `r2 0.98` and `static_drift 5 percent` are separate
hard quality gates: the first checks the stress fit, while the second rejects a
zero-strain intercept that changes by more than 5% when the outer strain shell
audits the inner two-shell extrapolation. Use at least three `static_strain`
magnitudes so this audit has independent evidence. RMSE and parameter contrast
break ties. The requested 20% mechanical tier labels result quality but never
removes the best structurally valid candidate.

Static and finite-temperature elastic calculations use independent target
sets and independent protocols. `E` and Poisson's ratio are algebraically
derived from each observed `B/G` pair but remain explicit user-facing ranking
targets when the `B/G/E/nu` basis is selected. A finite-temperature ranking can
reverse the static ranking, so every result row keeps both ranks and its
evidence level. In the packaged example, 38 finalists use seed `101`; exact
300 K evidence then selects 10 candidates for seeds `202 303`. Only its winner
enters the longer
validation protocol, whose disjoint holdout seeds are `404 505 606`;
deterministic replay of a promotion trajectory is not independent validation. The final protocol
has its own strain magnitudes, NPT/NVT equilibration lengths, and production
length, and both protocol fingerprints retain the shared global cutoff.

The example sets adaptive `triage 38`, `confirm 10`, `minimum 10`, and
`require_minimum yes` in its `finalists` block. This is a hard confirmation
floor, not a request to return “up to 10”. `incumbent initial` protects only an
explicit same-material coordinate and never imports its old evidence; new
materials use `incumbent off`.

If one structural gate surrogate has poor held-out quality, it degrades
independently. Its probability is conservatively shrunk toward exact observed
gate prevalence; it no longer disables a reliable mechanical expected-
improvement model. Exact bulk/surface evaluation still applies every hard gate
before a candidate can enter either elastic batch.

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

### Separate publication scopes

FFOpt treats the fitted ordered representation and an elemental
transferability claim separately. The machine-readable publication scope in
this release is one of:

- `ordered_sublattice_bulk` uses the existing lattice, angle, density,
  surface, Born-stability, stress-fit, static-drift, and independent 300 K
  validation gates. It may publish an accepted or explicitly authorized
  best-effort baseline.
- `validated_material_domain` covers a validation that is not the permanent
  multi-type elemental ordered-sublattice case.

The independent evidence object uses
`transferability_evidence.scope=elemental_transferability_only`. A future
trusted runner may establish the conceptual `elemental_transferable`
capability, but that is not a publication-scope value emitted by this release.
Failing the transferability assessment limits the claim and intended use; it
does not silently rerank or invalidate the completed bulk fit.

The machine-readable fields are `formal_invariance_status`,
`empirical_transfer_status`, and `elemental_transferability_claim`. Claim values
are `requires_empirical_validation` when the formal test passes but empirical
evidence is missing, `not_established` when evidence is incomplete or only
attested, `not_supported` after demonstrated sensitivity, `not_applicable` for
a single-type elemental representation, and `unknown` outside the elemental
scope. `supported` is reserved for a future trusted runner that reads and
verifies the empirical manifest, protocol, metrics, and case contents. This
release only validates attestation shape, reports complete positive
attestations as `attested_pass`, and keeps the claim `not_established`. These
claim values are distinct from the broader `physical_transferability.status`
workflow states emitted here: `requires_validation`, `incomplete`,
`ordered_sublattice_only`, `not_transferable`, `requires_verified_runner`, and
`unknown`. A future trusted runner may add a verified positive state.

Formal invariance is evaluated from the final resolved pair coefficients, not
from parameter-range declarations. Any two atom types representing the same
element must have identical interaction rows and columns against every possible
partner type, with equal mass and charge. For a two-type elemental LJ model
this requires `epsilon_11 = epsilon_12 = epsilon_22` and
`sigma_11 = sigma_12 = sigma_22`. A global type-name permutation alone is too
weak because a symmetric lattice can hide assignment sensitivity.

The empirical stage then quantifies the remaining risks with auditable inputs:

1. a harness-control relabelling plus global and spatially separated local
   type swaps, comparing energy and per-atom forces;
2. physically unique BCC(110) cleavages and both artificial label registries,
   without inventing chemical terminations that are only translations;
3. equivalent type-1 and type-2 vacancy formation energies in one fixed box;
4. conjugate seven-image vacancy migration-path sensitivity calculations.

Each case reports `pass`, `fail`, `error`, `not_evaluated`, or
`not_applicable`. A positive attestation requires unique case IDs, the
protocol-specific minimum case count (label swaps 3, surface registries 2,
vacancies 2, migration-path comparisons 2), and SHA-256 identities for its
manifest, protocol, and metrics. Because this pure reporting module does not
read those artifacts, even a well-formed attestation is only `attested_pass`;
the claim stays `not_established`/`requires_verified_runner`. Bare `pass`
strings and zero-case records become `error`. Short 300 K MD
without an observed hop is only a
`diagnostic_only/no_event` stability observation and cannot be reported as a
diffusion pass. An ordered two-sigma Fe fit such as A11 formally fails arbitrary
same-element relabelling, so it is reported as
`elemental_transferability_claim=not_supported` with
`physical_transferability.status=ordered_sublattice_only`, not as “a general
Fe potential”. Its ordered-sublattice bulk validation remains unchanged.
