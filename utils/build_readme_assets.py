"""Build dependency-free, text-editable SVG artwork for the GitHub homepage.

Run from any directory: python utils/build_readme_assets.py
Molecular connectivity comes from the bundled BTAH data; BCC is a schematic
conventional unit cell, not a fitted or independently validated structure.
"""

from itertools import combinations, product
from math import sqrt
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)

INK = "#182D35"
MUTED = "#50646C"
LINE = "#CDD9DC"
TEAL = "#087E83"
TEAL_PALE = "#EAF6F5"
RUST = "#B4502B"
RUST_PALE = "#FCF0E9"
WHITE = "#FFFFFF"


def element(parent, tag, **attrs):
    return ET.SubElement(parent, f"{{{NS}}}{tag}", {
        key.replace("_", "-"): str(value) for key, value in attrs.items()
    })


def canvas(width, height, title, description):
    root = ET.Element(f"{{{NS}}}svg", {
        "width": str(width), "height": str(height),
        "viewBox": f"0 0 {width} {height}", "role": "img",
        "aria-labelledby": "title description",
        "font-family": "Arial, Helvetica, sans-serif",
    })
    element(root, "title", id="title").text = title
    element(root, "desc", id="description").text = description
    defs = element(root, "defs")
    for name, color in (("ink", INK), ("teal", TEAL), ("rust", RUST)):
        marker = element(defs, "marker", id=f"arrow-{name}", viewBox="0 0 10 10",
                         refX=9, refY=5, markerWidth=7, markerHeight=7,
                         orient="auto-start-reverse")
        element(marker, "path", d="M 0 0 L 10 5 L 0 10 Z", fill=color)
    element(root, "rect", width=width, height=height, fill=WHITE)
    return root


def text(root, x, y, content, size=24, fill=INK, weight=400, anchor="start"):
    node = element(root, "text", x=x, y=y, font_size=size, fill=fill,
                   font_weight=weight, text_anchor=anchor)
    node.text = content
    return node


def line(root, x1, y1, x2, y2, color=LINE, width=2, arrow=None):
    attrs = {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
             "stroke": color, "stroke_width": width, "fill": "none"}
    if arrow:
        attrs["marker_end"] = f"url(#arrow-{arrow})"
    return element(root, "line", **attrs)


def path(root, d, color=LINE, width=2, arrow=None):
    attrs = {"d": d, "stroke": color, "stroke_width": width,
             "fill": "none", "stroke_linejoin": "round"}
    if arrow:
        attrs["marker_end"] = f"url(#arrow-{arrow})"
    return element(root, "path", **attrs)


def phase(root, y, number, label):
    element(root, "circle", cx=66, cy=y - 8, r=19, fill=INK)
    text(root, 66, y - 1, number, 19, WHITE, 700, "middle")
    text(root, 100, y, label, 28, INK, 700)


def normalized(vector):
    length = sqrt(sum(v * v for v in vector))
    return tuple(v / length for v in vector)


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def molecule(root, x, y, width, height):
    atoms, bonds = {}, []
    section = ""
    source = ROOT / "data" / "molecule" / "BTAH_822_single.data"
    for raw in source.read_text(encoding="utf-8").splitlines():
        content, _, comment = raw.partition("#")
        fields = content.split()
        if not fields:
            continue
        if not fields[0].isdigit():
            section = content.strip()
        elif section == "Atoms":
            atoms[int(fields[0])] = (tuple(map(float, fields[4:7])), comment.strip())
        elif section == "Bonds":
            bonds.append((int(fields[2]), int(fields[3])))
    assert len(atoms) == 14 and len(bonds) == 15
    origin = atoms[13][0]
    horizontal = normalized(sub(atoms[1][0], origin))
    side = sub(atoms[14][0], origin)
    vertical = normalized(tuple(v - dot(side, horizontal) * h
                                for v, h in zip(side, horizontal)))
    projected = {i: (dot(sub(p, origin), horizontal), dot(sub(p, origin), vertical))
                 for i, (p, _) in atoms.items()}
    xmin, xmax = min(p[0] for p in projected.values()), max(p[0] for p in projected.values())
    ymin, ymax = min(p[1] for p in projected.values()), max(p[1] for p in projected.values())
    scale = min((width - 24) / (xmax - xmin), (height - 24) / (ymax - ymin))
    points = {i: (x + width / 2 + (p[0] - (xmin + xmax) / 2) * scale,
                  y + height / 2 + (p[1] - (ymin + ymax) / 2) * scale)
              for i, p in projected.items()}
    group = element(root, "g", id="btah-molecule")
    for a, b in bonds:
        line(group, *points[a], *points[b], color="#82999F", width=4)
    for i, (_, label) in atoms.items():
        hydrogen = label.startswith("bhH")
        color = TEAL if label.startswith("bhN") else INK
        element(group, "circle", cx=points[i][0], cy=points[i][1],
                r=6 if hydrogen else 10, fill=WHITE if hydrogen else color,
                stroke="#82999F" if hydrogen else color, stroke_width=2)


def bcc(root, cx, cy, size):
    group = element(root, "g", id="bcc-cell")
    def point(p):
        a, b, c = p
        return cx + size * (a + 0.48 * b - 0.74), cy + size * (0.32 * b - c + 0.34)
    vertices = list(product((0, 1), repeat=3))
    # Cube edges depict the cell boundary, not chemical bonds.
    for a, b in combinations(vertices, 2):
        if sum(abs(i - j) for i, j in zip(a, b)) == 1:
            line(group, *point(a), *point(b), color="#CFAF9F", width=2.3)
    for p in sorted(vertices + [(0.5, 0.5, 0.5)], key=lambda p: sum(p)):
        px, py = point(p)
        element(group, "circle", cx=px, cy=py, r=10, fill=RUST, stroke=WHITE, stroke_width=2)


def cover():
    root = canvas(1280, 460, "FFOpt-LAMMPS: experiment-guided force-field development",
                  "A general framework for material force-field development. "
                  "Current workflows: molecular crystals and an elemental BCC extension. "
                  "BTAH connectivity and a conventional BCC cell are schematic illustrations.")
    element(root, "rect", width=1280, height=460, fill="#F4F8F8")
    element(root, "rect", width=8, height=460, fill=TEAL)
    text(root, 56, 96, "FFOpt-LAMMPS", 66, INK, 700)
    text(root, 58, 160, "A framework for material", 37, INK)
    text(root, 58, 208, "force-field development", 37, INK)
    text(root, 58, 270, "Guided by experiments. Tested with LAMMPS.", 23, MUTED)
    line(root, 58, 311, 712, 311)
    text(root, 58, 353, "CURRENT MATERIAL WORKFLOWS", 17, MUTED, 700)
    element(root, "circle", cx=66, cy=395, r=6, fill=TEAL)
    text(root, 84, 403, "Molecular crystals", 24, TEAL, 700)
    element(root, "circle", cx=359, cy=395, r=6, fill=RUST)
    text(root, 377, 403, "Elemental BCC extension", 24, RUST, 700)
    line(root, 768, 64, 768, 403, color="#DDE7E8")
    molecule(root, 804, 82, 210, 222)
    bcc(root, 1130, 199, 115)
    text(root, 907, 350, "Molecular", 21, TEAL, 700, "middle")
    text(root, 1130, 350, "BCC", 21, RUST, 700, "middle")
    text(root, 1019, 397, "One framework. Different material models.", 18, MUTED, 400, "middle")
    return root


def workflow():
    root = canvas(1280, 1240, "FFOpt: one framework, material-specific workflows",
                  "Start from data, parameters and targets in ffopt.in. Molecular BO and "
                  "sampling generate property labels; BCC adds structural constraints and "
                  "static elasticity. Train on stored data, select new candidates, calculate "
                  "them with LAMMPS, and return the new results to training. Final selection "
                  "uses repeated calculations and material-specific validation. Surrogate "
                  "training itself does not run molecular dynamics.")
    text(root, 48, 65, "From experimental targets to a tested force field", 39, INK, 700)
    text(root, 48, 106, "Shared execution and data. Material-specific objectives and validation.", 24, MUTED)

    phase(root, 169, "1", "Define the material and the fit")
    element(root, "rect", x=48, y=196, width=1184, height=87, rx=6, fill="#F3F6F7")
    text(root, 72, 231, "One ffopt.in + LAMMPS data", 26, INK, 700)
    text(root, 72, 264, "Initial parameters and bounds   /   Experimental targets   /   Property modules", 23, MUTED)
    line(root, 640, 284, 640, 316, color=INK, arrow="ink")

    phase(root, 357, "2", "Build a material-specific training set")
    for x, color, pale in ((48, TEAL, TEAL_PALE), (658, RUST, RUST_PALE)):
        element(root, "rect", x=x, y=384, width=574, height=226, rx=6, fill=pale)
        element(root, "rect", x=x, y=384, width=5, height=226, fill=color)
    text(root, 72, 425, "Molecular crystals", 29, TEAL, 700)
    text(root, 72, 463, "BO search + focused sampling", 25, INK, 700)
    text(root, 72, 500, "Find promising, repeatable parameter regions.", 22, MUTED)
    text(root, 72, 534, "Calculate the configured molecular properties.", 22, MUTED)
    text(root, 72, 580, "Bulk  /  Sublimation estimate  /  Adsorption", 22, TEAL)
    text(root, 682, 425, "Elemental BCC", 29, RUST, 700)
    text(root, 682, 463, "Feasible-region search + elastic screening", 24, INK, 700)
    text(root, 682, 500, "Map allowed structure and surface properties.", 22, MUTED)
    text(root, 682, 534, "Sample the region; calculate static elasticity.", 22, MUTED)
    text(root, 682, 580, "Structural limits first; then mechanical accuracy.", 22, RUST)
    text(root, 640, 649, "BO and sampling call LAMMPS. Their results become the training data.", 24, INK, 400, "middle")

    phase(root, 715, "3", "Learn the response and test new candidates")
    text(root, 48, 754, "Models in the packaged examples: molecular ANN and BCC Gaussian process (GP).", 23, MUTED)
    columns = [
        (48, "Train the surrogate", ["Stored parameters and properties", "Offline learning: no MD per epoch"], TEAL),
        (467, "Choose candidates", ["Predicted improvement + uncertainty", "Respect the material constraints"], INK),
        (879, "Test with LAMMPS", ["Calculate and compare properties", "Add the new results to the dataset"], RUST),
    ]
    for x, heading, body, color in columns:
        line(root, x, 786, x + 345, 786, color=color, width=4)
        text(root, x, 825, heading, 27, color, 700)
        for i, phrase in enumerate(body):
            text(root, x, 862 + i * 31, phrase, 21, MUTED)
    line(root, 401, 824, 443, 824, color=INK, arrow="ink")
    line(root, 815, 824, 855, 824, color=INK, arrow="ink")
    path(root, "M 1217 909 V 962 H 65 V 918", color=TEAL, width=2.5, arrow="teal")
    element(root, "rect", x=370, y=943, width=548, height=37, fill=WHITE)
    text(root, 644, 969, "New physical evidence updates the model", 21, TEAL, 400, "middle")

    phase(root, 1044, "4", "Select, validate and export")
    text(root, 48, 1085, "Repeated molecular evaluations or BCC finite-temperature screening select the finalists.", 23, MUTED)
    element(root, "rect", x=48, y=1110, width=1184, height=92, rx=6, fill=INK)
    text(root, 72, 1148, "Final LAMMPS validation", 27, WHITE, 700)
    text(root, 72, 1183, "Resolved parameters   /   Target comparison   /   Structures and trajectories   /   Model limitations", 22, WHITE)
    return root


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for filename, root in (("ffopt-cover.svg", cover()), ("ffopt-workflow.svg", workflow())):
        ET.indent(root, space="  ")
        target = ASSETS / filename
        ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
        print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
