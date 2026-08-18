from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")


def test_relative_markdown_links_resolve():
    missing: list[str] = []
    markdown_files = [ROOT / "README.md", ROOT / "CHANGELOG.md"]
    markdown_files.extend(sorted((ROOT / "docs").rglob("*.md")))
    markdown_files.extend(sorted((ROOT / "examples").rglob("*.md")))

    for document in markdown_files:
        text = document.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            raw = match.group(1).strip().strip("<>")
            target = raw.split(maxsplit=1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative = unquote(target.split("#", 1)[0])
            resolved = (document.parent / relative).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")

    assert not missing, "Missing local documentation links:\n" + "\n".join(missing)
