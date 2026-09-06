"""Keep the public diagrams accessible, self-contained and reproducible."""

from pathlib import Path
import runpy
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "utils" / "build_readme_assets.py"))
NS = {"s": "http://www.w3.org/2000/svg"}


@pytest.mark.parametrize("name, factory", [
    ("ffopt-cover.svg", "cover"), ("ffopt-workflow.svg", "workflow"),
])
def test_homepage_svg_is_current_and_self_contained(name, factory):
    actual = ROOT / "docs" / "assets" / name
    root = BUILDER[factory]()
    ET.indent(root, space="  ")
    expected = ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
    assert actual.read_text(encoding="utf-8") == expected
    assert root.attrib["role"] == "img"
    assert root.find("s:title", NS).text
    assert root.find("s:desc", NS).text
    assert not root.findall(".//s:script", NS)
    assert not root.findall(".//s:image", NS)
    for node in root.iter():
        assert not any(key.endswith("href") for key in node.attrib)


def test_cover_has_source_molecule_and_unambiguous_bcc_sites():
    root = BUILDER["cover"]()
    molecule = root.find(".//s:g[@id='btah-molecule']", NS)
    assert len(molecule.findall("s:circle", NS)) == 14
    assert len(molecule.findall("s:line", NS)) == 15
    bcc = root.find(".//s:g[@id='bcc-cell']", NS)
    assert len(bcc.findall("s:line", NS)) == 12
    sites = bcc.findall("s:circle", NS)
    assert len(sites) == 9
    assert len({(site.attrib["cx"], site.attrib["cy"]) for site in sites}) == 9
    assert len({site.attrib["fill"] for site in sites}) == 1


def test_workflow_distinguishes_training_from_physical_evaluation():
    root = BUILDER["workflow"]()
    text = " ".join(node.text or "" for node in root.findall(".//s:text", NS))
    for phrase in ("Molecular crystals", "Elemental BCC", "Offline learning: no MD per epoch",
                   "BO and sampling call LAMMPS", "Test with LAMMPS",
                   "Final LAMMPS validation", "Model limitations"):
        assert phrase in text
