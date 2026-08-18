from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_validation_module_does_not_import_bo_runtime():
    code = r"""
import importlib.abc
import sys

class BlockBO(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'torch', 'botorch', 'gpytorch'}:
            raise ModuleNotFoundError(f'blocked optional dependency: {fullname}')
        return None

sys.meta_path.insert(0, BlockBO())
import engine.validate_final_parameters
import engine.local_sampling
print('LIGHTWEIGHT_IMPORT_OK')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "LIGHTWEIGHT_IMPORT_OK" in completed.stdout
