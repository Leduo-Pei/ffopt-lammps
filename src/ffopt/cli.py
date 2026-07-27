"""Installed command-line entry point.

The validated workflow CLI remains in :mod:`workflow.cli` during the first
migration phase. Keeping this shim tiny makes both ``ffopt`` and the historical
``python ffopt.py`` command execute the same implementation.
"""

from workflow.cli import main

__all__ = ["main"]

