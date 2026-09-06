# Homepage artwork

The cover describes FFOpt's general material force-field development goal.
The workflow distinguishes the two currently developed material paths without
claiming that other materials or potential forms have already been validated.

| Asset | Purpose |
|---|---|
| [ffopt-cover.svg](ffopt-cover.svg) | GitHub README cover; molecular and BCC domains |
| [ffopt-workflow.svg](ffopt-workflow.svg) | Shared evidence loop and material-specific calculations |
| [build_readme_assets.py](../../utils/build_readme_assets.py) | Reproducible source for both SVGs; Python standard library only |

Regenerate from a source checkout:

```bash
python utils/build_readme_assets.py
```

Both SVGs contain editable text, vector shapes and accessible descriptions.
They are self-contained: no external fonts, scripts or linked images are
required. A white background keeps them legible in light and dark GitHub themes.

## Scientific meaning

- The molecular illustration uses the atom connectivity and coordinates in
  [the bundled BTAH single-molecule data](../../data/molecule/BTAH_822_single.data).
  It is a projected structural illustration, not evidence from a fitted model.
- The BCC illustration shows the eight boundary corners and one body-centre
  site of a conventional cell. Boundary sites are shared in the periodic
  crystal; the drawing does not mean nine independent atoms per cell. Lines
  mark cell edges, not chemical bonds. All sites represent the same element.
- BO and focused sampling use LAMMPS. Surrogate training consumes stored
  results, while active learning requests selected new physical calculations.
- ANN is not a universal model label. The molecular workflow uses an ANN;
  the packaged BCC example selects a Gaussian process.
- Final validation is distinct from a transferability claim.

The old [ffopt-workflow.vsdx](ffopt-workflow.vsdx) is retained as a historical
Visio source for the earlier molecular-only figure. It does **not** reproduce
the current diagram; use the SVGs or the generator above for current artwork.
