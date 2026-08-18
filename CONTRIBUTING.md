# Contributing

FFOpt-LAMMPS is an alpha scientific package. Contributions should preserve
both software behavior and the physical meaning of saved results.

## Development setup

```bash
git clone https://github.com/Leduo-Pei/ffopt-lammps.git
cd ffopt-lammps
python -m pip install -e ".[full,dev]"
python -m ruff check .
python -m pytest -q
```

Use a short branch from current `main`. Keep changes scoped, add regression
tests, and update the appropriate tutorial/reference/explanation document when
public behavior changes.

## Scientific changes

A pull request that changes a parameter definition, objective, LAMMPS
protocol, unit, property equation, random design, stability criterion, or
default budget must include:

1. The previous and new mathematical or physical definition.
2. A migration note for existing `ffopt.in` files and saved runs.
3. Unit tests for parsing and generated runtime configuration.
4. BTAH acceptance evidence when the executable path is affected.
5. A changelog entry.

Machine concurrency must not silently alter the scientific candidate budget.

## Pull requests

Before opening a pull request, run:

```bash
python -m ruff check .
python -m pytest -q
python -m build --wheel
python tests/check_wheel.py dist/*.whl
```

Do not commit generated `runs/`, model checkpoints, scheduler logs, Conda
environments, credentials, or proprietary simulation data.
