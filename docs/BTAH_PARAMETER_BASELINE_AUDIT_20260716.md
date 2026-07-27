# BTAH parameter-baseline audit (2026-07-16)

## Finding

The BTAH_v2.1 full 41-dimensional project was seeded from the validated
BTAH_v2.0 CHARMM epsilon/sigma values. Its runner generated all same-type LJ
coefficients and charges after `read_data`, so the incorrect Pair Coeffs stored
in the later BTAH_822 data files did not enter those simulations.

The two reduced projects were different: their system YAML files fixed or
bounded epsilon/sigma around the incorrect values stored in the later data
files. Their simulations were internally reproducible but do not represent the
intended CHARMM-based ablation.

## Topology audit

Compared with the BTAH_v2.0 4x2x2 data file, the BTAH_822 8x2x2 data file has:

- Identical masses and per-type initial charges.
- Numerically identical bond, angle, dihedral and improper coefficients.
- Different Pair Coeffs only.

The larger supercell therefore does not require rebuilding after correcting
its Pair Coeffs.

## Recalculation decision

| Project | Decision | Reason |
| --- | --- | --- |
| `btah_full` / BTAH_v2.1 | Keep | 41D config used CHARMM-centred epsilon/sigma ranges and overwrote data Pair Coeffs. |
| `btah_fix_sigma` / BTAH_v2.2_fix_si | Recalculate BO, sampling, NN, AL, validation | Fixed sigma and epsilon ranges came from the incorrect LJ set. |
| `btah_charge_only` / BTAH_v2.2_fix_epi_si | Recalculate BO, sampling, NN, AL, validation | Both fixed epsilon and sigma came from the incorrect LJ set. |

Historical reduced-project outputs are retained only under an archive folder
labelled `invalid_noncharmm_reference`; they must not be used as NN/AL inputs.

## Prevention

`tests/test_btah_charmm_baseline.py` enforces that:

1. Reduced-project fixed parameters/ranges derive from the full project's
   CHARMM initial values.
2. All BTAH Pair Coeffs stored in bulk, molecule and adsorption data files
   match the same reference.
