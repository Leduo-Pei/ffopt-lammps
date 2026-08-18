"""Verify that a source distribution is useful and free of local artifacts."""

from __future__ import annotations

import sys
import tarfile
from pathlib import PurePosixPath


archive = sys.argv[1]
with tarfile.open(archive, "r:gz") as handle:
    relative = {
        str(PurePosixPath(*PurePosixPath(name).parts[1:]))
        for name in handle.getnames()
        if len(PurePosixPath(name).parts) > 1
    }

required = {
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/zh_CN/USER_GUIDE.md",
    "docs/reference/input-file.md",
    "examples/btah/README.md",
    "examples/btah/acceptance.in",
}
missing = sorted(required - relative)
if missing:
    raise SystemExit(f"Source distribution is missing: {missing}")

forbidden = ("runs/", "archive/", "analysis/", "analyses/", "dry_run/")
leaked = sorted(path for path in relative if path.startswith(forbidden))
if leaked:
    raise SystemExit(f"Source distribution contains local artifacts: {leaked[:10]}")

print(f"Source distribution OK: {archive} ({len(relative)} entries)")
