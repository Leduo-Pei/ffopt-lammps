# Project selection

`project.yaml` is the only selector used by default. Change its single
`include` entry to one of:

- `btah_full.yaml`: 41D epsilon + sigma + charge workflow.
- `btah_fix_sigma.yaml`: 27D epsilon + charge workflow.
- `btah_charge_only.yaml`: 13D charge-only workflow.

Shared properties, methods and machine profiles are defined once in
`btah_common.yaml`. Each regime owns a separate `runs/<project>/` namespace and
an explicit BO/data contract, preventing automatic NN/AL discovery from mixing
different parameter spaces.

Paths are resolved from the repository/project root rather than the installed
Python package. Workstation and SLURM details should be configured once with
`ffopt machine configure`; user profiles override the portable defaults in
`configs/machines/`.
