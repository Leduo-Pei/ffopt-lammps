#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_subspace_config.py -- generate a reduced-dimension config by FIXING a whole
parameter category to known-good values, keeping the rest searched (with init
warm-start). General / material-agnostic.

Examples
--------
  # Fix all sigma, optimize epsilon + charge:
  python make_subspace_config.py --base configs/recipes/cluster_full.yaml \
      --best bo_BTAH_XXXX/best_parameters.txt --fix sigma \
      --out configs/recipes/cluster_fixsigma.yaml

  # Fix epsilon AND sigma, optimize charge only:
  python make_subspace_config.py --base configs/recipes/cluster_full.yaml \
      --best bo_BTAH_XXXX/best_parameters.txt --fix epsilon,sigma \
      --out configs/recipes/cluster_charge_only.yaml

Rules applied to every atom_types param (epsilon/sigma/charge):
  * category in --fix  -> replaced by a fixed SCALAR (= best value)  [not searched]
  * otherwise          -> kept as {min,max} and given init = best value (warm-start)
Sharing / relations (type-i = type-j, opposite charge, formulas) are written by
hand as strings in the config, e.g.  epsilon: "bhC1_epsilon".
"""
import argparse, yaml, sys

from .config_loader import load_config

def parse_best(path):
    vals = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            try:
                vals[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return vals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/recipes/cluster_full.yaml")
    ap.add_argument("--best", required=True, help="best_parameters.txt to take values from")
    ap.add_argument("--fix",  required=True, help="comma list of categories to fix, e.g. sigma or epsilon,sigma")
    ap.add_argument("--out",  required=True)
    a = ap.parse_args()

    fix = {s.strip() for s in a.fix.split(",") if s.strip()}
    cfg = load_config(a.base)
    cfg.pop("_config_path", None)
    cfg.pop("_config_sources", None)
    best = parse_best(a.best)

    n_fixed = n_init = 0
    for at in cfg["atom_types"]:
        label = at["label"]
        for cat in list(at["params"].keys()):
            key = f"{label}_{cat}"
            val = best.get(key)
            if cat in fix:
                if val is None:
                    print(f"  WARN: {key} not in best file; leaving as-is"); continue
                at["params"][cat] = float(val)          # fixed scalar
                n_fixed += 1
            else:
                spec = at["params"][cat]
                if isinstance(spec, dict) and "min" in spec and val is not None:
                    spec["init"] = float(val)            # warm-start
                    n_init += 1

    # Distinct system_name so each experiment writes its own bo_*/nn_output_* dirs
    cfg["manifest"]["system_name"] = (cfg["manifest"]["system_name"]
                                      + "_fix" + "".join(sorted(fix)))

    with open(a.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"Wrote {a.out}: fixed {n_fixed} params ({sorted(fix)}), "
          f"added init to {n_init} searched params.")

if __name__ == "__main__":
    main()
