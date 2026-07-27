"""Installed command-line entry point.

The validated workflow CLI remains in :mod:`workflow.cli` during migration.
Keeping this shim tiny makes both ``ffopt`` and ``python -m ffopt`` execute the
same implementation.
"""

from workflow.cli import main

__all__ = ["main"]
