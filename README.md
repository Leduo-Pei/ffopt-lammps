# FFOpt-LAMMPS

[![tests](https://github.com/Leduo-Pei/ffopt-lammps/actions/workflows/ci.yml/badge.svg)](https://github.com/Leduo-Pei/ffopt-lammps/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/Leduo-Pei/ffopt-lammps?include_prereleases)](https://github.com/Leduo-Pei/ffopt-lammps/releases)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)

FFOpt-LAMMPS is a restartable workflow for fitting molecular force-field
parameters to experimental properties with LAMMPS, Bayesian optimization,
focused sampling, ANN surrogate models, active learning, and final physical
validation.

> **Alpha scope:** the first public release supports molecular crystals,
> isolated molecules, and molecular adsorption models. BTAH is the packaged
> regression and machine-acceptance system. Elemental, alloy, reactive, and
> polymorph workflows are not yet claimed as supported.

## One input, one run

```mermaid
flowchart LR
    I["ffopt.in + LAMMPS data"] --> C["check"]
    C --> B["Bayesian optimization"]
    B --> S["focused sampling"]
    S --> N["ANN surrogate"]
    N --> A["active learning"]
    A --> V["LAMMPS validation"]
    V --> R["parameters + properties + trajectories"]
```

A user project is deliberately small:

```text
my_project/
|-- ffopt.in            # the only scientific control file
|-- data/               # user-supplied LAMMPS data files
`-- runs/               # generated automatically
```

Machine paths, MPI, CPUs, GPUs, SLURM partitions, memory, and wall time are
stored once in `~/.config/ffopt/machines.toml`. They never need to be copied
into a scientific project.

## Install

Create an isolated Python 3.11 environment, install a LAMMPS executable, then
install the current prerelease:

```bash
python -m pip install \
  "ffopt-lammps[full] @ git+https://github.com/Leduo-Pei/ffopt-lammps.git@v0.3.0a2"
```

For development:

```bash
git clone https://github.com/Leduo-Pei/ffopt-lammps.git
cd ffopt-lammps
python -m pip install -e ".[full,dev]"
python -m pytest -q
```

LAMMPS is an external dependency. A CPU-only PyTorch installation is valid;
`torch.cuda.is_available() == False` is not an error on a CPU machine profile.

## Configure and verify a machine

```bash
ffopt machine probe --partition CPU
ffopt machine configure --auto --name cluster-1node \
  --backend slurm --partition CPU --nodes 1 --force
ffopt machine test --name cluster-1node
```

Before fitting a new material, run the packaged scientific acceptance test:

```bash
ffopt self-test --machine cluster-1node --watch
```

The test executes the complete BTAH workflow and fails unless final LAMMPS
validation satisfies the packaged objective and property-error thresholds.
Use the same input with one- and two-node profiles to compare performance; BO
candidate batch size is independent of worker count, so the scientific budget
does not change with the machine.

## Create a project

Validate the data contract first:

```bash
ffopt data check --bulk crystal.data --single molecule.data
```

Then generate a reviewable starting input:

```bash
ffopt init my_crystal \
  --bulk-data crystal.data \
  --single-data molecule.data \
  --cells 2 2 2 \
  --mode charge_only \
  --target a=10.1,1.0,A \
  --target density=1.25,1.0,g/cm3 \
  --target sublimation=80.0,0.3,kJ/mol
```

`ffopt init` reads type labels, LJ values, and charges from the data file and
writes them explicitly. It does not assume that the inferred values or ranges
are chemically correct; review every `type`, `range`, `fix`, and `target` line.

## Check, run, resume

```bash
cd my_crystal
ffopt check ffopt.in
ffopt explain ffopt.in
ffopt doctor ffopt.in --machine cluster-1node
ffopt run ffopt.in --machine cluster-1node --dry-run
ffopt run ffopt.in --machine cluster-1node
```

Repeat the last command after logout, timeout, or node failure. FFOpt verifies
stage artifacts and resumes the first incomplete stage. To stop after BO while
keeping the full workflow in the input, use `--until bo`; later continue with
`--from-stage sample`. Use `--new` only for an independent campaign.

```bash
ffopt status ffopt.in --machine cluster-1node
ffopt logs ffopt.in --stage bo --lines 100
ffopt results ffopt.in
```

## Input at a glance

```text
ffopt 1
project example
workflow bo sample nn al validate

parameters
    range epsilon factor 0.50 2.00
    range sigma   factor 0.85 1.15
    range charge  delta  0.30
    charge_limit 1.0
    neutrality derive N1
    mixing epsilon geometric
    mixing sigma geometric
    fix epsilon sigma
    type 1 N1 0.20 3.30 -0.50
    type 2 H1 0.03 2.42  0.50
end
```

No `fix` line optimizes epsilon, sigma, and charge. `fix epsilon sigma`
produces charge-only optimization. The derived charge is removed from the
independent search space and recovered from exact total neutrality.

## Documentation

- [Complete Chinese user manual](docs/zh_CN/USER_GUIDE.md)
- [Five-minute tutorial](docs/tutorials/quickstart.md)
- [FFOpt input reference](docs/reference/input-file.md)
- [Machine profile reference](docs/reference/machine-profiles.md)
- [Output and naming reference](docs/reference/outputs-and-naming.md)
- [Workflow and accuracy model](docs/explanation/workflow-and-accuracy.md)
- [BTAH examples and acceptance benchmark](examples/btah/README.md)
- [Developer architecture](docs/explanation/architecture.md)

## Scientific definition

The bundled bulk protocol is fixed-box minimization followed by flexible-cell
NPT equilibration and production. The current sublimation observable is a
potential-energy estimate,

```text
E_sub,estimate = E_single,min - <PE_bulk,NPT> / N_molecules
```

fitted to a user-supplied experimental sublimation-enthalpy target. It does
not include translational, rotational, vibrational, or standard-state thermal
corrections and is reported with that provenance in validation outputs.

## Development status

FFOpt-LAMMPS follows semantic prerelease versions while its public input and
supported-material contract are still evolving. See [CHANGELOG.md](CHANGELOG.md),
[CONTRIBUTING.md](CONTRIBUTING.md), and [LICENSE](LICENSE).
