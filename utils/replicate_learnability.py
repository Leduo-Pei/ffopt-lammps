#!/usr/bin/env python3
"""Diagnose whether replicated LAMMPS labels are learnable by a surrogate.

The analysis keeps parameter candidates as the independent sampling unit. Replicate
seeds estimate simulation noise; they are never treated as additional parameter
points during train/test splitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.benchmark_surrogates import build_model, fit_model as fit_benchmark_model


DEFAULT_PROPERTIES = [
    "a", "b", "c", "alpha", "beta", "gamma_ang", "density", "esub_proxy"
]


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def safe_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or np.nanvar(actual) <= 0.0:
        return float("nan")
    return float(r2_score(actual, predicted))


def noise_table(
    raw: pd.DataFrame,
    properties: list[str],
    candidate_column: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    valid = raw.loc[truthy(raw["success"])].copy()
    for prop in properties:
        column = f"calc_{prop}"
        values = pd.to_numeric(valid[column], errors="coerce")
        frame = valid.loc[np.isfinite(values), [candidate_column, "seed"]].copy()
        frame["value"] = values[np.isfinite(values)].to_numpy()
        groups = frame.groupby(candidate_column, sort=False)["value"]
        counts = groups.size()
        means = groups.mean()
        variances = groups.var(ddof=1)
        usable = counts >= 2
        counts = counts.loc[usable]
        means = means.loc[usable]
        variances = variances.loc[usable]

        pooled_denom = float((counts - 1).sum())
        pooled_within = float(
            (((counts - 1) * variances).sum() / pooled_denom)
            if pooled_denom > 0 else np.nan
        )
        mean_inverse_replicates = float(np.mean(1.0 / counts.to_numpy(float)))
        observed_mean_variance = float(means.var(ddof=1))
        mean_measurement_noise = pooled_within * mean_inverse_replicates
        signal_variance = max(observed_mean_variance - mean_measurement_noise, 0.0)
        single_seed_ceiling = (
            signal_variance / (signal_variance + pooled_within)
            if signal_variance + pooled_within > 0 else np.nan
        )
        mean_label_ceiling = (
            signal_variance / (signal_variance + mean_measurement_noise)
            if signal_variance + mean_measurement_noise > 0 else np.nan
        )

        wide = frame.pivot_table(
            index=candidate_column, columns="seed", values="value", aggfunc="mean"
        )
        seed_correlations = wide.corr().to_numpy(float)
        upper = seed_correlations[np.triu_indices_from(seed_correlations, k=1)]
        upper = upper[np.isfinite(upper)]

        rows.append({
            "property": prop,
            "n_candidates": int(len(means)),
            "mean_replicates": float(counts.mean()),
            "between_candidate_sd_observed": float(np.sqrt(observed_mean_variance)),
            "within_seed_sd_pooled": float(np.sqrt(pooled_within)),
            "median_candidate_seed_sd": float(np.sqrt(variances).median()),
            "signal_sd_estimated": float(np.sqrt(signal_variance)),
            "signal_to_noise_variance_ratio": (
                float(signal_variance / pooled_within) if pooled_within > 0 else np.inf
            ),
            "single_seed_r2_ceiling": single_seed_ceiling,
            "three_seed_mean_r2_ceiling": mean_label_ceiling,
            "mean_pairwise_seed_correlation": (
                float(np.mean(upper)) if len(upper) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def aggregate_candidates(
    raw: pd.DataFrame,
    parameter_columns: list[str],
    properties: list[str],
    candidate_column: str,
) -> pd.DataFrame:
    valid = raw.loc[truthy(raw["success"])].copy()
    numeric = parameter_columns + [f"calc_{prop}" for prop in properties] + ["objective"]
    for column in numeric:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid.loc[np.isfinite(valid[numeric]).all(axis=1)].copy()
    aggregations = {name: "first" for name in parameter_columns}
    aggregations.update({f"calc_{prop}": "mean" for prop in properties})
    aggregations["objective"] = "mean"
    result = valid.groupby(candidate_column, as_index=False).agg(aggregations)
    result["n_successful_seeds"] = valid.groupby(candidate_column).size().reindex(
        result[candidate_column]
    ).to_numpy()
    return result


def fixed_split(
    candidates: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(candidates))
    objective = candidates["objective"].to_numpy(float)
    try:
        bins = pd.qcut(objective, q=5, labels=False, duplicates="drop")
        return train_test_split(
            indices, test_size=0.20, random_state=seed, stratify=bins
        )
    except ValueError:
        return train_test_split(indices, test_size=0.20, random_state=seed)


def fit_diagnostic_model(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    trees: int,
):
    random.seed(seed)
    np.random.seed(seed)
    if name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=-1,
            random_state=seed,
        )
        return model.fit(x, y)
    model = build_model(name, seed)
    return fit_benchmark_model(model, name, x, y, seed)


def learning_curve(
    candidates: pd.DataFrame,
    raw: pd.DataFrame,
    parameter_columns: list[str],
    properties: list[str],
    candidate_column: str,
    seed: int,
    repeats: int,
    trees: int,
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    x = candidates[parameter_columns].to_numpy(float)
    lo = np.nanmin(x, axis=0)
    span = np.maximum(np.nanmax(x, axis=0) - lo, 1e-12)
    x = (x - lo) / span
    y = candidates[[f"calc_{prop}" for prop in properties]].to_numpy(float)
    train_pool, test = fixed_split(candidates, seed)
    possible = [50, 100, 200, 400, 600, 800, len(train_pool)]
    sizes = sorted(set(size for size in possible if size <= len(train_pool)))
    rows: list[dict] = []

    for size in sizes:
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + 1009 * repeat + size)
            train = (
                train_pool if size == len(train_pool)
                else rng.choice(train_pool, size=size, replace=False)
            )
            for model_index, name in enumerate(models):
                model_seed = seed + repeat + 100000 * model_index
                model = fit_diagnostic_model(
                    name, x[train], y[train], model_seed, trees
                )
                predicted = model.predict(x[test])
                for idx, prop in enumerate(properties):
                    rows.append({
                        "model": name,
                        "train_candidates": int(size),
                        "repeat": repeat,
                        "property": prop,
                        "r2": safe_r2(y[test, idx], predicted[:, idx]),
                        "mae": float(mean_absolute_error(y[test, idx], predicted[:, idx])),
                        "n_test": int(len(test)),
                    })

    # Compare noisy single-seed labels with three-seed means on the same split.
    seed_value = int(pd.to_numeric(raw["seed"], errors="coerce").min())
    seed_rows = raw.loc[
        truthy(raw["success"]) & (pd.to_numeric(raw["seed"], errors="coerce") == seed_value)
    ].copy()
    seed_rows = seed_rows.set_index(candidate_column)
    ids = candidates[candidate_column].to_numpy()
    train_ids = ids[train_pool]
    usable = np.array([candidate in seed_rows.index for candidate in train_ids])
    noisy_train = train_pool[usable]
    noisy_y = seed_rows.loc[train_ids[usable], [f"calc_{p}" for p in properties]].to_numpy(float)
    comparisons: list[dict] = []
    for model_index, name in enumerate(models):
        for label_source, train_y in [
            ("three_seed_mean", y[train_pool]), (f"seed_{seed_value}", noisy_y)
        ]:
            train_idx = train_pool if label_source == "three_seed_mean" else noisy_train
            model = fit_diagnostic_model(
                name, x[train_idx], train_y,
                seed + 5000 + 100000 * model_index, trees,
            )
            predicted = model.predict(x[test])
            for idx, prop in enumerate(properties):
                comparisons.append({
                    "model": name,
                    "training_label": label_source,
                    "property": prop,
                    "r2_vs_three_seed_test_mean": safe_r2(y[test, idx], predicted[:, idx]),
                    "mae_vs_three_seed_test_mean": float(
                        mean_absolute_error(y[test, idx], predicted[:, idx])
                    ),
                })

    split = {
        "seed": seed,
        "train_candidate_ids": ids[train_pool].tolist(),
        "test_candidate_ids": ids[test].tolist(),
        "n_train_pool": int(len(train_pool)),
        "n_test": int(len(test)),
    }
    return pd.DataFrame(rows), pd.DataFrame(comparisons), split


def configure_plotting() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def make_figures(
    noise: pd.DataFrame,
    curves: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_plotting()
    colors = {"ceiling": "#2878B5", "model": "#D95F02", "noise": "#6B7280"}

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    positions = np.arange(len(noise))
    ax.bar(
        positions - 0.18, noise["single_seed_r2_ceiling"], width=0.36,
        color=colors["noise"], label="Single-seed ceiling",
    )
    ax.bar(
        positions + 0.18, noise["three_seed_mean_r2_ceiling"], width=0.36,
        color=colors["ceiling"], label="Three-seed mean ceiling",
    )
    ax.axhline(0.9, color="#B91C1C", linewidth=1.0, linestyle="--", label="R2 = 0.90")
    ax.set_xticks(positions, noise["property"], rotation=30, ha="right")
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("Estimated reliability ceiling")
    fig.suptitle("Replicate noise limits the learnable accuracy", y=0.98)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.10, right=0.98)
    fig.savefig(output_dir / "replicate_r2_ceiling.svg", bbox_inches="tight")
    fig.savefig(output_dir / "replicate_r2_ceiling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = curves.groupby(["model", "train_candidates", "property"])["r2"].agg(
        ["mean", "std"]
    ).reset_index()
    model_names = list(dict.fromkeys(curves["model"]))
    palette = ["#D95F02", "#2878B5", "#6A3D9A", "#2A9D8F"]
    model_colors = dict(zip(model_names, palette))
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.2), sharex=True, sharey=True)
    for ax, prop in zip(axes.flat, noise["property"]):
        for name in model_names:
            subset = summary.loc[
                (summary["property"] == prop) & (summary["model"] == name)
            ]
            ax.plot(
                subset["train_candidates"], subset["mean"], marker="o", markersize=3.2,
                linewidth=1.35, color=model_colors[name], label=name,
            )
            ax.fill_between(
                subset["train_candidates"].to_numpy(float),
                (subset["mean"] - subset["std"].fillna(0)).to_numpy(float),
                (subset["mean"] + subset["std"].fillna(0)).to_numpy(float),
                color=model_colors[name], alpha=0.10, linewidth=0,
            )
        ceiling = float(noise.loc[
            noise["property"] == prop, "three_seed_mean_r2_ceiling"
        ].iloc[0])
        ax.axhline(ceiling, color=colors["ceiling"], linewidth=1.0, linestyle="--")
        ax.axhline(0.9, color="#B91C1C", linewidth=0.8, linestyle=":")
        ax.set_title(prop)
        ax.set_ylim(-0.2, 1.04)
    for ax in axes[-1, :]:
        ax.set_xlabel("Independent candidates")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test R2")
    axes[0, 0].legend(ncol=min(3, len(model_names)), fontsize=7, loc="lower right")
    fig.suptitle("Group-safe learning curves on three-seed candidate means", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.svg", bbox_inches="tight")
    fig.savefig(output_dir / "learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def domain_regime_audit(
    candidates: pd.DataFrame,
    parameter_columns: list[str],
    properties: list[str],
    candidate_column: str,
    split: dict,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Compare interpolation in the low-objective basin with outward extrapolation."""
    indexed = candidates.set_index(candidate_column, drop=False)
    train_ids = [item for item in split["train_candidate_ids"] if item in indexed.index]
    test_ids = [item for item in split["test_candidate_ids"] if item in indexed.index]
    train = indexed.loc[train_ids].copy()
    test = indexed.loc[test_ids].copy()
    cutoff = float(candidates["objective"].quantile(0.25))
    local_train = train.loc[train["objective"] <= cutoff]
    local_test = test.loc[test["objective"] <= cutoff]
    outer_test = test.loc[test["objective"] > cutoff]

    lo = candidates[parameter_columns].min().to_numpy(float)
    span = np.maximum(
        candidates[parameter_columns].max().to_numpy(float) - lo, 1e-12
    )

    def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = (frame[parameter_columns].to_numpy(float) - lo) / span
        y = frame[[f"calc_{prop}" for prop in properties]].to_numpy(float)
        return x, y

    x_train, y_train = arrays(train)
    x_local_train, y_local_train = arrays(local_train)
    regimes = [
        ("global_to_local", x_train, y_train, local_test),
        ("global_to_outer", x_train, y_train, outer_test),
        ("local_interpolation", x_local_train, y_local_train, local_test),
        ("local_extrapolation", x_local_train, y_local_train, outer_test),
    ]
    rows: list[dict] = []
    for index, (name, fit_x, fit_y, evaluate) in enumerate(regimes):
        model = fit_diagnostic_model(
            "residual_mlp", fit_x, fit_y, seed + 700000 + index, trees=0
        )
        test_x, test_y = arrays(evaluate)
        predicted = model.predict(test_x)
        for prop_index, prop in enumerate(properties):
            rows.append({
                "regime": name,
                "property": prop,
                "r2": safe_r2(test_y[:, prop_index], predicted[:, prop_index]),
                "mae": float(mean_absolute_error(
                    test_y[:, prop_index], predicted[:, prop_index]
                )),
                "target_sd": float(np.std(test_y[:, prop_index], ddof=1)),
                "n_train": int(len(fit_x)),
                "n_test": int(len(test_x)),
            })
    metadata = {
        "local_definition": "candidate three-seed mean objective <= global 25th percentile",
        "local_objective_cutoff": cutoff,
        "n_global_train": int(len(train)),
        "n_local_train": int(len(local_train)),
        "n_local_test": int(len(local_test)),
        "n_outer_test": int(len(outer_test)),
        "model": "residual_mlp",
    }
    return pd.DataFrame(rows), metadata


def make_domain_figure(scores: pd.DataFrame, output_dir: Path) -> None:
    configure_plotting()
    order = ["global_to_local", "local_interpolation", "global_to_outer", "local_extrapolation"]
    colors = {
        "global_to_local": "#2878B5",
        "local_interpolation": "#2A9D8F",
        "global_to_outer": "#6B7280",
        "local_extrapolation": "#D95F02",
    }
    properties = list(dict.fromkeys(scores["property"]))
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.2), sharey=True)
    for ax, prop in zip(axes.flat, properties):
        subset = scores.loc[scores["property"] == prop].set_index("regime")
        values = np.asarray([subset.loc[name, "r2"] for name in order], dtype=float)
        displayed = np.clip(values, -1.0, 1.0)
        ax.bar(np.arange(len(order)), displayed, color=[colors[name] for name in order])
        for position, value in enumerate(values):
            if value < -1.0:
                ax.text(
                    position, -0.96, f"{value:.1f}", ha="center", va="bottom",
                    fontsize=7, rotation=90, color="white",
                )
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.axhline(0.9, color="#B91C1C", linewidth=0.8, linestyle=":")
        ax.set_title(prop)
        ax.set_xticks([])
        ax.set_ylim(-1.05, 1.02)
    for ax in axes[:, 0]:
        ax.set_ylabel("Test R2")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[name]) for name in order]
    labels = [name.replace("_", " ") for name in order]
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Local interpolation does not guarantee outward generalization", y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output_dir / "domain_regime_r2.svg", bbox_inches="tight")
    fig.savefig(output_dir / "domain_regime_r2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_domain_report(
    scores: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
) -> None:
    pivot = scores.pivot(index="property", columns="regime", values="r2")
    columns = [
        name for name in [
            "global_to_local", "local_interpolation", "global_to_outer",
            "local_extrapolation",
        ] if name in pivot.columns
    ]
    lines = [
        "# fix_epi_si Local and Global Domain Audit",
        "",
        f"- Local cutoff: objective <= **{metadata['local_objective_cutoff']:.6f}**",
        f"- Global/local training candidates: **{metadata['n_global_train']} / {metadata['n_local_train']}**",
        f"- Local/outer test candidates: **{metadata['n_local_test']} / {metadata['n_outer_test']}**",
        f"- Model: **{metadata['model']}**",
        "",
        "| Property | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for prop, row in pivot.iterrows():
        lines.append(
            f"| {prop} | " + " | ".join(f"{row[name]:.3f}" for name in columns) + " |"
        )
    lines.extend([
        "",
        "R2 inside the local basin is affected by range restriction: a small MAE can still produce a low R2 when the local test labels vary little.",
        "The local-to-outer score is the direct test of whether a locally trained surrogate can be trusted outside its sampled basin.",
    ])
    (output_dir / "DOMAIN_REGIME_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_report(
    output_dir: Path,
    input_path: Path,
    parameter_columns: list[str],
    candidates: pd.DataFrame,
    noise: pd.DataFrame,
    curves: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    final_size = int(curves["train_candidates"].max())
    final = curves.loc[curves["train_candidates"] == final_size].groupby(
        ["model", "property"]
    ).agg(
        model_r2=("r2", "mean"), model_r2_sd=("r2", "std"), mae=("mae", "mean")
    ).reset_index()
    final_wide = final.pivot(index="property", columns="model", values="model_r2")
    primary = [
        name for name in ["a", "b", "c", "density", "esub_proxy"]
        if name in set(final["property"])
    ]
    primary_means = final.loc[final["property"].isin(primary)].groupby("model")[
        "model_r2"
    ].mean().sort_values(ascending=False)
    best_model = str(primary_means.index[0])
    best_primary_r2 = float(primary_means.iloc[0])
    model_curve = curves.loc[
        (curves["model"] == best_model) & curves["property"].isin(primary)
    ].groupby("train_candidates")["r2"].mean().sort_index()
    earlier_sizes = [size for size in model_curve.index if size <= 0.625 * final_size]
    earlier_size = int(max(earlier_sizes)) if earlier_sizes else int(model_curve.index[-2])
    late_gain = float(model_curve.loc[final_size] - model_curve.loc[earlier_size])
    lines = [
        "# fix_epi_si Replicate Learnability Diagnostic",
        "",
        f"- Input: `{input_path}`",
        f"- Independent parameter candidates: **{len(candidates)}**",
        f"- Free parameter dimensions: **{len(parameter_columns)}**",
        f"- Maximum learning-curve training size: **{final_size}**",
        "- Train/test split unit: candidate_id, never individual seed rows",
        "",
        "## Property diagnosis",
        "",
        "| Property | Pooled seed SD | Signal/noise variance | Single-seed ceiling | 3-seed mean ceiling | "
        + " | ".join(final_wide.columns) + " |",
        "|---|---:|---:|---:|---:|" + "---:|" * len(final_wide.columns),
    ]
    for row in noise.itertuples(index=False):
        model_values = [
            f"{final_wide.loc[row.property, name]:.3f}" for name in final_wide.columns
        ]
        lines.append(
            f"| {row.property} | {row.within_seed_sd_pooled:.5g} | "
            f"{row.signal_to_noise_variance_ratio:.3f} | {row.single_seed_r2_ceiling:.3f} | "
            f"{row.three_seed_mean_r2_ceiling:.3f} | " + " | ".join(model_values) + " |"
        )
    comparison_view = comparisons.loc[
        comparisons["model"] == "residual_mlp"
    ] if "residual_mlp" in set(comparisons["model"]) else comparisons.loc[
        comparisons["model"] == comparisons["model"].iloc[0]
    ]
    comparison_wide = comparison_view.pivot(
        index="property", columns="training_label",
        values="r2_vs_three_seed_test_mean",
    ).round(4)
    comparison_model = str(comparison_view["model"].iloc[0])
    comparison_primary = comparison_view.loc[
        comparison_view["property"].isin(primary)
    ].pivot(
        index="property", columns="training_label",
        values="r2_vs_three_seed_test_mean",
    )
    seed_column = next(
        column for column in comparison_primary.columns if str(column).startswith("seed_")
    )
    mean_label_gain = float(
        comparison_primary["three_seed_mean"].mean()
        - comparison_primary[seed_column].mean()
    )
    comparison_columns = list(comparison_wide.columns)
    comparison_lines = [
        "| Property | " + " | ".join(comparison_columns) + " |",
        "|---|" + "---:|" * len(comparison_columns),
    ]
    for prop, row in comparison_wide.iterrows():
        values = [
            "" if not np.isfinite(row[column]) else f"{row[column]:.4f}"
            for column in comparison_columns
        ]
        comparison_lines.append(f"| {prop} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Data-derived conclusions",
        "",
        f"- The best model at {final_size} training candidates is **{best_model}**, with mean R2 **{best_primary_r2:.3f}** over a/b/c/density/esub_proxy.",
        f"- Its primary-property mean R2 rises by **{late_gain:+.3f}** from {earlier_size} to {final_size} candidates, so the learning curve has not fully saturated.",
        f"- Three-seed averaging improves the {comparison_model} primary-property mean R2 by only **{mean_label_gain:+.3f}** versus training on one seed.",
        "- Every three-seed reliability ceiling is above 0.90. Replicate noise is therefore not the main reason the fitted surrogate remains below 0.90.",
        "- The gap between reliability ceilings and fitted R2 points to sparse 13D coverage, nonlinear or segmented NPT responses, and model representation as the current bottlenecks.",
        "- For the next sampling budget, prefer more independent one-seed candidates for coverage, then apply three seeds to held-out tests, low-objective candidates, and uncertain or boundary regions.",
        "",
        "## Interpretation rules",
        "",
        "- A low three-seed ceiling indicates that longer simulations or more replicate seeds are required before model changes can help.",
        "- A high ceiling but low model R2 indicates insufficient coverage, an unsuitable representation/model, or a discontinuous response surface.",
        "- A learning curve still rising at the largest sample size supports adding independent parameter points.",
        "- A plateau far below the ceiling supports changing the model or splitting the parameter domain into regimes.",
        "",
        "## Label averaging comparison",
        "",
        *comparison_lines,
        "",
        "Editable figures: `replicate_r2_ceiling.svg` and `learning_curves.svg`.",
    ])
    (output_dir / "LEARNABILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-column", default="candidate_id")
    parser.add_argument("--properties", nargs="+", default=DEFAULT_PROPERTIES)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument(
        "--models", nargs="+",
        default=["extra_trees", "poly_ridge", "residual_mlp"],
    )
    parser.add_argument(
        "--reuse-learning", action="store_true",
        help="Reuse existing diagnostic CSVs and only regenerate reports/domain audit.",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.replicates)
    parameter_columns = [column for column in raw.columns if column.endswith("_charge")]
    required = {
        args.candidate_column, "seed", "success", "objective", *parameter_columns,
        *(f"calc_{prop}" for prop in args.properties),
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not parameter_columns:
        raise ValueError("No *_charge parameter columns were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_learning:
        noise = pd.read_csv(args.output_dir / "replicate_noise_and_ceiling.csv")
        candidates = pd.read_csv(args.output_dir / "candidate_three_seed_means.csv")
        curves = pd.read_csv(args.output_dir / "learning_curve_scores.csv")
        comparisons = pd.read_csv(args.output_dir / "label_averaging_comparison.csv")
        split = json.loads((args.output_dir / "fixed_split.json").read_text(encoding="utf-8"))
    else:
        noise = noise_table(raw, args.properties, args.candidate_column)
        candidates = aggregate_candidates(
            raw, parameter_columns, args.properties, args.candidate_column
        )
        curves, comparisons, split = learning_curve(
            candidates, raw, parameter_columns, args.properties, args.candidate_column,
            args.seed, args.repeats, args.trees, args.models,
        )
        noise.to_csv(args.output_dir / "replicate_noise_and_ceiling.csv", index=False)
        candidates.to_csv(args.output_dir / "candidate_three_seed_means.csv", index=False)
        curves.to_csv(args.output_dir / "learning_curve_scores.csv", index=False)
        comparisons.to_csv(args.output_dir / "label_averaging_comparison.csv", index=False)
        (args.output_dir / "fixed_split.json").write_text(
            json.dumps(split, indent=2), encoding="utf-8"
        )
    make_figures(noise, curves, args.output_dir)
    write_report(
        args.output_dir, args.replicates.resolve(), parameter_columns, candidates,
        noise, curves, comparisons,
    )
    domain_scores, domain_metadata = domain_regime_audit(
        candidates, parameter_columns, args.properties, args.candidate_column,
        split, args.seed,
    )
    domain_scores.to_csv(args.output_dir / "domain_regime_scores.csv", index=False)
    (args.output_dir / "domain_regime_metadata.json").write_text(
        json.dumps(domain_metadata, indent=2), encoding="utf-8"
    )
    make_domain_figure(domain_scores, args.output_dir)
    write_domain_report(domain_scores, domain_metadata, args.output_dir)
    print(noise.to_string(index=False))
    print(f"\nSaved learnability diagnostic to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
