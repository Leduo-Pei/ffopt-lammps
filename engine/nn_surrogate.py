#!/usr/bin/env python3
"""
nn_surrogate.py  |  Independent per-property Ensemble MLP surrogate

Architecture:
  raw BO params -> PhysicsFeatureBuilder -> {prop: EnsembleMLP_prop}
  One EnsembleMLP per target property in the compiled FFOpt configuration
  (a, b, c, alpha, beta,
  gamma_ang, density, surf_energy, ...).
  Each NN predicts the actual property value (not errors).
  Optimization: predict all props -> compute weighted RMSE -> scipy minimize.
  Uncertainty: aggregated normalized std across all property NNs.

Usage:
  python -m engine.nn_surrogate --config runs/.../provenance/runtime_*.json \\
      --bo-dir runs/.../bo

Outputs (pipelines/<run-id>/nn/):
  forward_nn.pt           - all per-property ensemble weights + metadata
  nn_optimize_result.json - best params + LAMMPS validation
  train_history.json      - per-property per-epoch loss curves
  parity_data.csv         - (y_true, y_pred) per property for test set
  feature_names.txt       - ordered feature names after physics embedding
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

from .config_loader import load_config as load_expanded_config, save_config_snapshot
from .lammps_interface import LAMMPSRunner
from .parameter_space import build_parameter_space
from utils.objective_rescoring import (
    active_targets,
    validate_objective_provenance,
)

# `python -m engine.nn_surrogate` executes this file as __main__. Register its
# canonical name so custom sklearn/PyTorch wrappers are pickled with a stable
# import path and can be loaded by later NN/AL processes.
if __name__ == "__main__":
    sys.modules["engine.nn_surrogate"] = sys.modules[__name__]

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# ============================================================================
# Physics feature builder  (unchanged from previous version)
# ============================================================================

class PhysicsFeatureBuilder:
    """
    Pre-compute physically meaningful derived features from raw BO parameters.
    Appended to raw params before the MLP ensemble.

    Features: LB cross terms, LJ well-depth proxy, LJ contact radius,
              charge proxy (when charge enabled), normalised sigma.
    """

    def __init__(self, param_names: List[str], config: dict):
        self.param_names    = param_names
        self.charge_enabled = config["charge"]["enabled"]
        # Saved legacy models do not contain pair_params.  Their feature schema
        # used geometric epsilon plus arithmetic sigma, so keep that default for
        # backward-compatible loading.  New recipes persist the actual rule.
        self.mixing_rule = config.get("pair_params", {}).get("mixing_rule", "arithmetic")
        self._name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(param_names)}
        self._type_eps_idx: List[Optional[int]] = []
        self._type_sig_idx: List[Optional[int]] = []
        self._type_chg_idx: List[Optional[int]] = []
        for at in config["atom_types"]:
            label = at["label"]
            self._type_eps_idx.append(self._name_to_idx.get(f"{label}_epsilon"))
            self._type_sig_idx.append(self._name_to_idx.get(f"{label}_sigma"))
            self._type_chg_idx.append(self._name_to_idx.get(f"{label}_charge"))
        self._derived_charge_pos: Optional[int] = None
        self._derived_charge_feature_name: Optional[str] = None
        self._charge_feature_counts: Optional[np.ndarray] = None
        neutrality = config["charge"].get("neutrality_constraint", {})
        feature_counts = neutrality.get("feature_type_counts")
        if self.charge_enabled and neutrality.get("enabled") and feature_counts:
            derive_type = int(neutrality["derive_from_type"])
            for pos, at in enumerate(config["atom_types"]):
                if int(at["type"]) == derive_type:
                    self._derived_charge_pos = pos
                    self._derived_charge_feature_name = (
                        f"derived_charge_{at['label']}"
                    )
                    break
            counts = np.array(
                [float(feature_counts.get(at["label"], 0.0)) for at in config["atom_types"]],
                dtype=np.float64,
            )
            if self._derived_charge_pos is None or counts[self._derived_charge_pos] <= 0:
                raise ValueError("invalid neutrality feature_type_counts for derived charge")
            self._charge_feature_counts = counts
        self.extra_feature_names = self._build_extra_names(config)

    def _has_charge_feature(self, pos: int) -> bool:
        return (
            self._type_chg_idx[pos] is not None
            or (
                self._derived_charge_pos == pos
                and self._charge_feature_counts is not None
            )
        )

    def _derived_charge(self, X_raw: np.ndarray) -> Optional[np.ndarray]:
        if self._derived_charge_pos is None or self._charge_feature_counts is None:
            return None
        charge_sum = np.zeros(X_raw.shape[0], dtype=X_raw.dtype)
        for pos, idx in enumerate(self._type_chg_idx):
            if pos == self._derived_charge_pos:
                continue
            if idx is None:
                return None
            charge_sum += X_raw[:, idx] * self._charge_feature_counts[pos]
        return -charge_sum / self._charge_feature_counts[self._derived_charge_pos]

    def _build_extra_names(self, config: dict) -> List[str]:
        names: List[str] = []
        labels  = [at["label"] for at in config["atom_types"]]
        n_types = len(labels)
        for i in range(n_types):
            for j in range(i + 1, n_types):
                if self._type_eps_idx[i] is not None and self._type_eps_idx[j] is not None:
                    names.append(f"lb_eps_{labels[i]}_{labels[j]}")
        for i in range(n_types):
            for j in range(i + 1, n_types):
                if self._type_sig_idx[i] is not None and self._type_sig_idx[j] is not None:
                    names.append(f"lb_sig_{labels[i]}_{labels[j]}")
        for i, label in enumerate(labels):
            if self._type_eps_idx[i] is not None and self._type_sig_idx[i] is not None:
                names.append(f"lj_well_{label}")
        for i, label in enumerate(labels):
            if self._type_sig_idx[i] is not None:
                names.append(f"lj_rmin_{label}")
        if self.charge_enabled:
            if self._derived_charge_pos is not None:
                names.append(f"derived_charge_{labels[self._derived_charge_pos]}")
            for i in range(n_types):
                for j in range(i + 1, n_types):
                    if self._has_charge_feature(i) and self._has_charge_feature(j):
                        names.append(f"chg_proxy_{labels[i]}_{labels[j]}")
        for i, label in enumerate(labels):
            if self._type_sig_idx[i] is not None:
                names.append(f"sig_norm_{label}")
        return names

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        N       = X_raw.shape[0]
        n_types = len(self._type_eps_idx)
        extra: List[np.ndarray] = []

        def col(idx): return X_raw[:, idx] if idx is not None else None

        derived_charge = self._derived_charge(X_raw)

        def charge_col(pos):
            if pos == self._derived_charge_pos and derived_charge is not None:
                return derived_charge
            return col(self._type_chg_idx[pos])

        for i in range(n_types):
            for j in range(i + 1, n_types):
                eps_i, eps_j = col(self._type_eps_idx[i]), col(self._type_eps_idx[j])
                if eps_i is not None and eps_j is not None:
                    extra.append(np.sqrt(np.maximum(eps_i * eps_j, 1e-10)))
        for i in range(n_types):
            for j in range(i + 1, n_types):
                sig_i, sig_j = col(self._type_sig_idx[i]), col(self._type_sig_idx[j])
                if sig_i is not None and sig_j is not None:
                    if self.mixing_rule == "geometric":
                        extra.append(np.sqrt(np.maximum(sig_i * sig_j, 1e-10)))
                    elif self.mixing_rule == "arithmetic":
                        extra.append((sig_i + sig_j) / 2.0)
                    else:
                        raise ValueError(
                            "PhysicsFeatureBuilder supports geometric or arithmetic "
                            f"mixing, got {self.mixing_rule!r}"
                        )
        for i in range(n_types):
            eps_i, sig_i = col(self._type_eps_idx[i]), col(self._type_sig_idx[i])
            if eps_i is not None and sig_i is not None:
                extra.append(eps_i * (sig_i ** 6))
        _2_16 = 2.0 ** (1.0 / 6.0)
        for i in range(n_types):
            sig_i = col(self._type_sig_idx[i])
            if sig_i is not None:
                extra.append(sig_i * _2_16)
        if self.charge_enabled:
            if derived_charge is not None:
                extra.append(derived_charge)
            for i in range(n_types):
                for j in range(i + 1, n_types):
                    chg_i, chg_j = charge_col(i), charge_col(j)
                    if chg_i is not None and chg_j is not None:
                        extra.append(chg_i * chg_j)
        all_sigs = [col(self._type_sig_idx[i]) for i in range(n_types)
                    if self._type_sig_idx[i] is not None]
        if all_sigs:
            sig_ref = np.maximum(np.mean(np.stack(all_sigs, axis=1), axis=1), 1e-6)
            for i in range(n_types):
                sig_i = col(self._type_sig_idx[i])
                if sig_i is not None:
                    extra.append(sig_i / sig_ref)

        if extra:
            return np.concatenate([X_raw] + [arr.reshape(N, 1) for arr in extra], axis=1)
        return X_raw.copy()

    def transform_raw_with_derived_charge(self, X_raw: np.ndarray) -> np.ndarray:
        """Append only the neutrality-derived charge, without cross features."""
        derived = self._derived_charge(X_raw)
        if derived is None:
            return X_raw.copy()
        return np.column_stack([X_raw, derived]).astype(X_raw.dtype, copy=False)

    @property
    def raw_with_derived_charge_names(self) -> List[str]:
        names = list(self.param_names)
        if self._derived_charge_feature_name is not None:
            names.append(self._derived_charge_feature_name)
        return names

    @property
    def all_feature_names(self) -> List[str]:
        return list(self.param_names) + self.extra_feature_names


# ============================================================================
# MLP (single member)
# ============================================================================

_ACTIVATIONS: Dict[str, type] = {
    "silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU,
}


class MLP(nn.Module):
    """input_dim -> [hidden_layers] -> 1 with BatchNorm after each hidden layer."""

    def __init__(self, input_dim: int, hidden_layers: List[int], activation: str):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation, nn.SiLU)
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), act_cls()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================================
# EnsembleMLP  (M independent MLPs, predicts mean + std)
# ============================================================================

class EnsembleMLP:
    """
    M independently trained MLPs for one target property.
    Scalers applied internally.  All predict_* accept np.ndarray (physical units).
    """

    def __init__(self, input_dim: int, hidden_layers: List[int],
                 activation: str, ensemble_size: int, device: torch.device):
        self.input_dim     = input_dim
        self.hidden_layers = hidden_layers
        self.activation    = activation
        self.ensemble_size = ensemble_size
        self.device        = device
        self.members: List[MLP] = [
            MLP(input_dim, hidden_layers, activation).to(device)
            for _ in range(ensemble_size)]
        self.X_mean: Optional[np.ndarray] = None
        self.X_std:  Optional[np.ndarray] = None
        self.Y_mean: float = 0.0
        self.Y_std:  float = 1.0

    def _scale_X(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor((X - self.X_mean) / (self.X_std + 1e-8),
                            dtype=torch.float32).to(self.device)

    def _unscale_Y(self, Y: torch.Tensor) -> np.ndarray:
        return Y.detach().cpu().numpy() * self.Y_std + self.Y_mean

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        X_t = self._scale_X(X)
        rows = []
        for m in self.members:
            m.eval()
            with torch.no_grad():
                rows.append(self._unscale_Y(m(X_t)))
        return np.stack(rows, axis=0)   # (M, N)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return self.predict_all(X).mean(axis=0)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        return self.predict_all(X).std(axis=0)

    def state_dicts(self) -> List[dict]:
        return [m.state_dict() for m in self.members]

    def load_state_dicts(self, sds: List[dict]):
        for m, sd in zip(self.members, sds):
            m.load_state_dict(sd)
            m.eval()


class MultiTaskMLP(nn.Module):
    """Shared hidden representation with one output per physical property."""
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_layers: List[int], activation: str,
                 dropout: float = 0.0):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation, nn.SiLU)
        layers: List[nn.Module] = []
        in_dim = input_dim
        for width in hidden_layers:
            layers.extend([nn.Linear(in_dim, width), act_cls()])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = width
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskEnsemble:
    """Ensemble that learns all property responses through shared features."""
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_layers: List[int], activation: str,
                 ensemble_size: int, device: torch.device,
                 dropout: float = 0.0):
        self.device = device
        self.members = [
            MultiTaskMLP(
                input_dim, output_dim, hidden_layers, activation, dropout
            ).to(device)
            for _ in range(ensemble_size)
        ]
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None

    def fit(self, X_train, Y_train, X_val, Y_val, *, seed, learning_rate,
            weight_decay, max_epochs, patience):
        self.X_mean = X_train.mean(axis=0)
        self.X_std = X_train.std(axis=0) + 1.0e-8
        self.Y_mean = Y_train.mean(axis=0)
        self.Y_std = Y_train.std(axis=0) + 1.0e-8

        def scaled(values, mean, std):
            return torch.tensor(
                (values - mean) / std,
                dtype=torch.float32,
                device=self.device,
            )

        Xt = scaled(X_train, self.X_mean, self.X_std)
        Yt = scaled(Y_train, self.Y_mean, self.Y_std)
        Xv = scaled(X_val, self.X_mean, self.X_std)
        Yv = scaled(Y_val, self.Y_mean, self.Y_std)
        for member_index, model in enumerate(self.members):
            torch.manual_seed(seed + member_index * 1000)
            model.apply(NNSurrogate._reinit_weights)
            optimizer = optim.AdamW(
                model.parameters(), lr=learning_rate,
                weight_decay=weight_decay,
            )
            best_loss = float("inf")
            best_state = None
            stale = 0
            for _ in range(max_epochs):
                model.train()
                optimizer.zero_grad()
                loss = nn.functional.smooth_l1_loss(model(Xt), Yt)
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_loss = float(nn.functional.mse_loss(model(Xv), Yv))
                if val_loss < best_loss - 1.0e-7:
                    best_loss = val_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
        return self

    def predict_all(self, X):
        values = torch.tensor(
            (X - self.X_mean) / self.X_std,
            dtype=torch.float32,
            device=self.device,
        )
        predictions = []
        for model in self.members:
            model.eval()
            with torch.no_grad():
                prediction = model(values).cpu().numpy()
            predictions.append(prediction * self.Y_std + self.Y_mean)
        return np.stack(predictions, axis=0)


class MultiTaskPropertyView:
    """Single-property interface over one shared multi-task ensemble."""
    def __init__(self, ensemble: MultiTaskEnsemble, output_index: int):
        self.ensemble = ensemble
        self.output_index = output_index

    def predict_mean(self, X):
        return self.ensemble.predict_all(X)[:, :, self.output_index].mean(axis=0)

    def predict_std(self, X):
        return self.ensemble.predict_all(X)[:, :, self.output_index].std(axis=0)


# ============================================================================
# Pluggable surrogate models - same interface as EnsembleMLP:
#   .fit(X, y)  .predict_mean(X)  .predict_std(X)
# All are picklable (torch.save/load), so forward_nn.pt stays self-contained.
# ============================================================================
class _ScaledModel:
    """Base: standardises X with train-set mean/std."""
    def __init__(self):
        self.X_mean = None
        self.X_std  = None
    def _scale(self, X):
        return (X - self.X_mean) / (self.X_std + 1e-8)


class GPModel(_ScaledModel):
    """Gaussian Process (sklearn). Native predictive std -> ideal for AL."""
    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        Xs = self._scale(X)
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                  + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1)))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=3, alpha=1e-8)
        self.gp.fit(Xs, y)
        return self
    def predict_mean(self, X):
        return self.gp.predict(self._scale(X))
    def predict_std(self, X):
        _, s = self.gp.predict(self._scale(X), return_std=True)
        return s


class RFModel(_ScaledModel):
    """Random forest (sklearn). Std from per-tree spread."""
    def fit(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        self.rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
        self.rf.fit(self._scale(X), y)
        return self
    def predict_mean(self, X):
        return self.rf.predict(self._scale(X))
    def predict_std(self, X):
        Xs = self._scale(X)
        preds = np.stack([t.predict(Xs) for t in self.rf.estimators_], axis=0)
        return preds.std(axis=0)


class XGBModel(_ScaledModel):
    """Gradient boosting (xgboost). Ensemble of seeds -> mean/std.
    Requires `pip install xgboost`."""
    def fit(self, X, y):
        import xgboost as xgb
        Xs = self._scale(X)
        self.models = []
        for seed in range(8):
            m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=seed, n_jobs=-1)
            m.fit(Xs, y)
            self.models.append(m)
        return self
    def predict_mean(self, X):
        Xs = self._scale(X)
        return np.stack([m.predict(Xs) for m in self.models], axis=0).mean(axis=0)
    def predict_std(self, X):
        Xs = self._scale(X)
        return np.stack([m.predict(Xs) for m in self.models], axis=0).std(axis=0)


class ExtraTreesModel(_ScaledModel):
    """Extremely randomized trees; tree spread supplies AL uncertainty."""
    def __init__(self, seed=0):
        super().__init__()
        self.seed = seed

    def fit(self, X, y):
        from sklearn.ensemble import ExtraTreesRegressor
        self.model = ExtraTreesRegressor(
            n_estimators=500,
            max_features=1.0,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=self.seed,
        )
        self.model.fit(self._scale(X), y)
        return self

    def predict_mean(self, X):
        return self.model.predict(self._scale(X))

    def predict_std(self, X):
        Xs = self._scale(X)
        values = np.stack(
            [tree.predict(Xs) for tree in self.model.estimators_], axis=0
        )
        return values.std(axis=0)


class PairProductExtraTreesModel:
    """ExtraTrees over charge columns and all q_i*q_j pair products."""
    def __init__(self, seed=0):
        self.seed = seed

    @staticmethod
    def _augment(X):
        columns = [X]
        for left in range(X.shape[1]):
            for right in range(left + 1, X.shape[1]):
                columns.append((X[:, left] * X[:, right]).reshape(-1, 1))
        return np.concatenate(columns, axis=1)

    def fit(self, X, y):
        from sklearn.ensemble import ExtraTreesRegressor
        augmented = self._augment(X)
        self.X_mean = augmented.mean(axis=0)
        self.X_std = augmented.std(axis=0) + 1.0e-8
        self.model = ExtraTreesRegressor(
            n_estimators=500,
            max_features=1.0,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=self.seed,
        )
        self.model.fit((augmented - self.X_mean) / self.X_std, y)
        return self

    def _scaled(self, X):
        augmented = self._augment(X)
        return (augmented - self.X_mean) / self.X_std

    def predict_mean(self, X):
        return self.model.predict(self._scaled(X))

    def predict_std(self, X):
        scaled = self._scaled(X)
        values = np.stack([
            tree.predict(scaled) for tree in self.model.estimators_
        ], axis=0)
        return values.std(axis=0)


class _BootstrapScaledTargetModel(_ScaledModel):
    """Small bootstrap ensemble with standardised targets."""
    def __init__(self, seed=0, ensemble_size=8):
        super().__init__()
        self.seed = seed
        self.ensemble_size = ensemble_size
        self.models = []
        self.Y_mean = 0.0
        self.Y_std = 1.0

    def _new_model(self):
        raise NotImplementedError

    def fit(self, X, y):
        Xs = self._scale(X)
        self.Y_mean = float(np.mean(y))
        self.Y_std = float(np.std(y)) + 1.0e-8
        ys = (np.asarray(y) - self.Y_mean) / self.Y_std
        rng = np.random.default_rng(self.seed)
        self.models = []
        for member in range(self.ensemble_size):
            if member == 0:
                indices = np.arange(len(Xs))
            else:
                indices = rng.integers(0, len(Xs), size=len(Xs))
            model = self._new_model()
            model.fit(Xs[indices], ys[indices])
            self.models.append(model)
        return self

    def _predict_all(self, X):
        Xs = self._scale(X)
        values = np.stack([model.predict(Xs) for model in self.models], axis=0)
        return values * self.Y_std + self.Y_mean

    def predict_mean(self, X):
        # Member zero is fitted to every stable-core row. Bootstrap members are
        # retained for epistemic spread; averaging them degraded local holdout
        # accuracy in the sparse high-dimensional benchmark used to select this
        # estimator behavior.
        return self._predict_all(X)[0]

    def predict_std(self, X):
        values = self._predict_all(X)
        if len(values) == 1:
            return np.zeros(values.shape[1], dtype=float)
        return np.sqrt(np.mean((values[1:] - values[0]) ** 2, axis=0))


class SVRModel(_BootstrapScaledTargetModel):
    """RBF support-vector regression for smooth local interpolation."""
    def _new_model(self):
        from sklearn.svm import SVR
        return SVR(C=30.0, epsilon=0.01, gamma="scale")


class PolynomialRidgeModel(_BootstrapScaledTargetModel):
    """Second-order local response surface with ridge regularisation."""
    def _new_model(self):
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        return Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", RidgeCV(alphas=np.logspace(-4, 3, 16))),
        ])


# Custom wrappers must keep the same import path whether this file is imported
# or executed with `python -m`; otherwise torch/pickle records `__main__` and AL
# cannot load the saved surrogate in a fresh process.
for _serializable_class in (
    MultiTaskMLP,
    MultiTaskEnsemble,
    MultiTaskPropertyView,
    ExtraTreesModel,
    PairProductExtraTreesModel,
    _BootstrapScaledTargetModel,
    SVRModel,
    PolynomialRidgeModel,
):
    _serializable_class.__module__ = "engine.nn_surrogate"


def make_property_model(kind: str, X_mean, X_std, seed=0, ensemble_size=8):
    """Factory for non-MLP surrogates (mlp_ensemble handled separately)."""
    factories = {
        "gp": lambda: GPModel(),
        "random_forest": lambda: RFModel(),
        "xgboost": lambda: XGBModel(),
        "extra_trees": lambda: ExtraTreesModel(seed=seed),
        "pair_product_extra_trees": lambda: PairProductExtraTreesModel(seed=seed),
        "svr_rbf": lambda: SVRModel(seed=seed, ensemble_size=ensemble_size),
        "polynomial_ridge": lambda: PolynomialRidgeModel(
            seed=seed, ensemble_size=ensemble_size
        ),
    }
    factory = factories.get(kind)
    if factory is None:
        raise ValueError(f"unknown nn.model '{kind}' "
                         "(use mlp_ensemble|gp|random_forest|xgboost|"
                         "extra_trees|pair_product_extra_trees|svr_rbf|"
                         "polynomial_ridge)")
    m = factory()
    m.X_mean = X_mean
    m.X_std  = X_std
    return m


# ============================================================================
# NNSurrogate - per-property independent surrogate training
# ============================================================================

class NNSurrogate:
    """
    Train one EnsembleMLP per target property, then optimize over
    the weighted RMSE of all NN predictions.

    Uncertainty for active learning:
      total_std(x) = sum_prop std_prop(x) / |target_prop|
    """

    def __init__(self, config: dict, bo_dir: str, output_dir: str,
                 no_validate: bool = False, save_traj: bool = False,
                 reuse_models_from: Optional[str] = None,
                 skip_optimize: bool = False):
        self.config      = config
        self.bo_dir      = bo_dir
        self.output_dir  = output_dir
        self.no_validate = no_validate
        self.save_traj   = save_traj
        self.reuse_models_from = reuse_models_from
        self.skip_optimize = skip_optimize

        nn_cfg = config["nn"]
        self.physics_features     = nn_cfg["physics_features"]
        self.derived_charge_feature = bool(
            nn_cfg.get("derived_charge_feature", False)
        )
        self.ensemble_size        = nn_cfg["ensemble_size"]
        self.hidden_layers        = nn_cfg["hidden_layers"]
        self.activation           = nn_cfg["activation"]
        self.lr                   = nn_cfg["learning_rate"]
        self.loss_name            = nn_cfg.get("loss", "huber")
        self.weight_decay         = float(nn_cfg.get("weight_decay", 1.0e-4))
        self.batch_size_nn        = nn_cfg["batch_size"]
        self.max_epochs           = nn_cfg["max_epochs"]
        self.patience             = nn_cfg["early_stopping_patience"]
        self.val_fraction         = nn_cfg["val_fraction"]
        self.test_fraction        = nn_cfg["test_fraction"]
        opt_nn = nn_cfg.get("optimize", {})
        self.n_restarts_opt = opt_nn.get("n_restarts", 100)
        self.opt_method     = opt_nn.get("method",     "L-BFGS-B")
        # Pluggable surrogate model: mlp_ensemble | gp | random_forest | xgboost
        self.model_kind     = nn_cfg.get("model", "mlp_ensemble")
        self.property_model_cfg = dict(nn_cfg.get("property_models", {}) or {})
        self.property_model_kinds: Dict[str, str] = {}
        self.multitask_cfg = dict(nn_cfg.get("multitask", {}) or {})
        self.seed = config["optimization"].get("random_seed", 42)
        training_cfg = nn_cfg.get("training_data", {})
        self.training_data_mode = training_cfg.get("mode", "legacy")
        self.core_objective_max = float(
            training_cfg.get("core_objective_max", 0.03)
        )
        self.buffer_objective_max = float(
            training_cfg.get("buffer_objective_max", 0.075)
        )
        self.core_weight = max(1, int(training_cfg.get("core_weight", 6)))
        self.additional_core_files = list(
            training_cfg.get("additional_core_files", []) or []
        )
        self._data_group_ids: Optional[np.ndarray] = None
        self._core_eval_mask: Optional[np.ndarray] = None

        # Active target properties for the NN surrogate.
        #   - surf_energy dropped when compute_surface=false
        #   - ead dropped when adsorption disabled
        #   - nn.exclude_properties: user-listed props the NN should NOT model
        #     (e.g. cell angles, which are noise-dominated -> R2~0 and only add
        #      noise to NN-based optimisation and AL uncertainty). They are STILL
        #      fitted by BO against ground-truth LAMMPS; this only skips them in
        #      the surrogate.
        compute_surf = config["lammps"]["compute_surface"]
        ads_enabled  = config.get("adsorption", {}).get("enabled", False)
        nn_exclude   = set(config["nn"].get("exclude_properties", []))
        self.targets: Dict = {
            p: info for p, info in config["targets"].items()
            if not (p == "surf_energy" and not compute_surf)
            and not (p == "ead" and not ads_enabled)
            and p not in nn_exclude
            and float(info.get("weight", 1.0)) > 0.0
        }
        self.target_names = list(self.targets.keys())

        # Free BO parameter space
        self.param_space = build_parameter_space(config)
        self.param_names = [p[0] for p in self.param_space]
        self.param_lo    = np.array([p[1] for p in self.param_space])
        self.param_hi    = np.array([p[2] for p in self.param_space])
        self.n_params    = len(self.param_names)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feat_builder = PhysicsFeatureBuilder(self.param_names, config)

        # Trained ensembles: {prop_name: EnsembleMLP}
        self.ensembles: Dict[str, EnsembleMLP] = {}

        self.runner = LAMMPSRunner(config)
        os.makedirs(output_dir, exist_ok=True)
        save_config_snapshot(config, output_dir)

    def _transform_features(self, X_raw: np.ndarray) -> np.ndarray:
        if self.physics_features:
            return self.feat_builder.transform(X_raw)
        if self.derived_charge_feature:
            return self.feat_builder.transform_raw_with_derived_charge(X_raw)
        return X_raw.copy()

    def _feature_names(self) -> List[str]:
        if self.physics_features:
            return self.feat_builder.all_feature_names
        if self.derived_charge_feature:
            return self.feat_builder.raw_with_derived_charge_names
        return list(self.param_names)

    # ====================================================================== #
    # Main entry                                                              #
    # ====================================================================== #

    def run(self):
        t0 = time.time()
        mf = self.config["manifest"]
        print(f"\n{'='*65}")
        print("NN Surrogate - Independent per-property surrogate")
        print(f"  System     : {mf['system_name']}")
        print(f"  Properties : {self.target_names}")
        print(f"  Model      : {self.model_kind}"
              + (f" ({self.ensemble_size} MLPs x {self.hidden_layers})"
                 if self.model_kind == 'mlp_ensemble' else ""))
        if self.property_model_cfg:
            print("  Per-property overrides:")
            for prop, options in self.property_model_cfg.items():
                print(
                    f"    {prop:<12} model={options.get('model', self.model_kind)} "
                    f"source={options.get('training_source', 'core_buffer')}"
                )
        if self.multitask_cfg.get("enabled", False):
            print(
                "  Multi-task  : shared MLP -> "
                f"{self.multitask_cfg.get('use_for_properties', [])}"
            )
        print(f"  Device     : {self.device}")
        print(f"  Physics    : {self.physics_features}")
        print(f"  Derived q  : {self.derived_charge_feature}")
        print(f"{'='*65}")

        # Step 1: load BO data (X_raw + per-property Y columns)
        print("\nStep 1/5 - Loading BO data")
        X_raw, Y_props = self._load_bo_data()
        N = X_raw.shape[0]
        print(f"  Valid evaluations : {N}")
        if N < 10:
            print("  ERROR: fewer than 10 valid points.")
            sys.exit(1)

        # Step 2: physics features
        print("\nStep 2/5 - Building features")
        X_feat = self._transform_features(X_raw)
        feat_names = self._feature_names()
        print(f"  Features: {X_feat.shape[1]}"
              + (f" ({self.n_params} raw + {X_feat.shape[1]-self.n_params} physics)"
                 if self.physics_features else
                 (" (raw + derived neutral charge)"
                  if self.derived_charge_feature else " (raw only)")))

        reusable_models: Dict[str, object] = {}
        reusable_meta: dict = {}
        if self.reuse_models_from:
            reuse_path = Path(self.reuse_models_from).resolve()
            if not reuse_path.exists():
                raise FileNotFoundError(f"Reusable surrogate not found: {reuse_path}")
            reusable_models, _, reused_params, reused_features, reusable_meta = (
                load_ensemble_from_file(str(reuse_path), self.device)
            )
            if reused_params != self.param_names or reused_features != feat_names:
                raise ValueError(
                    "Reusable surrogate has incompatible parameters or features"
                )
            print(f"  Reusing unchanged property models from: {reuse_path}")

        # Step 3: split indices (shared across all properties)
        print("\nStep 3/5 - Splitting data")
        tr_idx, val_idx, test_idx = self._compute_split_indices(N, Y_props)
        print(f"  Train:{len(tr_idx)}  Val:{len(val_idx)}  Test:{len(test_idx)}")

        X_val = X_feat[val_idx]
        X_test = X_feat[test_idx]
        X_test_raw = X_raw[test_idx]
        input_dim = X_feat.shape[1]

        training_indices = {"core_buffer": tr_idx}
        if self._core_eval_mask is not None:
            held_out_groups = set(self._data_group_ids[test_idx])
            core_only_idx = np.flatnonzero(
                self._core_eval_mask
                & np.asarray([
                    group not in held_out_groups
                    for group in self._data_group_ids
                ])
            )
            training_indices["core_only"] = core_only_idx

        multitask_views: Dict[str, MultiTaskPropertyView] = {}
        use_multitask = [
            prop for prop in self.multitask_cfg.get("use_for_properties", [])
            if prop in self.target_names
        ]
        if self.multitask_cfg.get("enabled", False) and use_multitask:
            if self._core_eval_mask is None:
                raise ValueError("nn.multitask requires core_buffer training data")
            held_out_groups = set(
                self._data_group_ids[np.r_[val_idx, test_idx]]
            )
            multitask_train_idx = np.flatnonzero(
                self._core_eval_mask
                & np.asarray([
                    group not in held_out_groups
                    for group in self._data_group_ids
                ])
            )
            Y_matrix = np.column_stack([
                Y_props[prop] for prop in self.target_names
            ])
            shared = MultiTaskEnsemble(
                input_dim,
                len(self.target_names),
                list(self.multitask_cfg.get("hidden_layers", [128, 128, 64])),
                self.multitask_cfg.get("activation", "silu"),
                int(self.multitask_cfg.get("ensemble_size", 5)),
                self.device,
                float(self.multitask_cfg.get("dropout", 0.0)),
            ).fit(
                X_feat[multitask_train_idx],
                Y_matrix[multitask_train_idx],
                X_val,
                Y_matrix[val_idx],
                seed=self.seed,
                learning_rate=float(
                    self.multitask_cfg.get("learning_rate", 1.0e-3)
                ),
                weight_decay=float(
                    self.multitask_cfg.get("weight_decay", 1.0e-4)
                ),
                max_epochs=int(self.multitask_cfg.get("max_epochs", 3000)),
                patience=int(self.multitask_cfg.get("patience", 250)),
            )
            multitask_views = {
                prop: MultiTaskPropertyView(shared, self.target_names.index(prop))
                for prop in use_multitask
            }
            print(
                f"  Multi-task core: train={len(multitask_train_idx)} "
                f"val={len(val_idx)} outputs={len(self.target_names)}"
            )

        # Step 4: train one EnsembleMLP per target property (in parallel)
        print(f"\nStep 4/5 - Training {len(self.target_names)} property models [{self.model_kind}]")
        histories: Dict[str, List] = {}
        test_metrics: Dict[str, dict] = {}
        parity_rows: List[dict] = []

        def train_one_prop(prop: str) -> Tuple[str, EnsembleMLP, List, dict, List]:
            options = self.property_model_cfg.get(prop, {})
            if prop not in self.property_model_cfg and prop in multitask_views:
                ens = multitask_views[prop]
                Y_test_ = Y_props[prop][test_idx]
                Y_pred = ens.predict_mean(X_test)
                Y_std_ = ens.predict_std(X_test)
                r2 = self._r2(Y_test_, Y_pred)
                rmse = float(np.sqrt(np.mean((Y_test_ - Y_pred) ** 2)))
                rows = [
                    {
                        "prop": prop,
                        "y_true": float(yt),
                        "y_pred": float(yp),
                        "y_std": float(ys),
                    }
                    for yt, yp, ys in zip(Y_test_, Y_pred, Y_std_)
                ]
                return prop, ens, [], {
                    "r2": r2,
                    "rmse": rmse,
                    "model": "multitask_mlp",
                    "training_source": "core_only",
                    "n_train": int(len(multitask_train_idx)),
                }, rows
            if prop not in self.property_model_cfg and prop in reusable_models:
                ens = reusable_models[prop]
                prop_kind = reusable_meta.get("property_model_kinds", {}).get(
                    prop, reusable_meta.get("model_kind", "mlp_ensemble")
                )
                Y_test_ = Y_props[prop][test_idx]
                Y_pred = ens.predict_mean(X_test)
                Y_std_ = ens.predict_std(X_test)
                r2 = self._r2(Y_test_, Y_pred)
                rmse = float(np.sqrt(np.mean((Y_test_ - Y_pred) ** 2)))
                rows = [
                    {
                        "prop": prop,
                        "y_true": float(yt),
                        "y_pred": float(yp),
                        "y_std": float(ys),
                    }
                    for yt, yp, ys in zip(Y_test_, Y_pred, Y_std_)
                ]
                metrics = {
                    "r2": r2,
                    "rmse": rmse,
                    "model": prop_kind,
                    "training_source": "reused",
                    "n_train": 0,
                }
                return prop, ens, [], metrics, rows
            prop_kind = options.get("model", self.model_kind)
            source = options.get("training_source", "core_buffer")
            if source not in training_indices:
                raise ValueError(
                    f"nn.property_models.{prop}.training_source={source!r} "
                    "requires core_buffer training data"
                )
            prop_train_idx = training_indices[source]
            Y_all  = Y_props[prop]
            X_prop_tr = X_feat[prop_train_idx]
            Y_tr = Y_all[prop_train_idx]
            Y_val_ = Y_all[val_idx]
            Y_test_ = Y_all[test_idx]
            X_mean = X_prop_tr.mean(axis=0)
            X_std = X_prop_tr.std(axis=0) + 1e-8
            Y_mean = float(Y_tr.mean())
            Y_std = float(Y_tr.std()) + 1e-8
            if prop_kind == "mlp_ensemble":
                ens = EnsembleMLP(input_dim, self.hidden_layers, self.activation,
                                  self.ensemble_size, self.device)
                ens.X_mean = X_mean
                ens.X_std = X_std
                ens.Y_mean = Y_mean
                ens.Y_std = Y_std
                hist = self._train_ensemble(ens, X_prop_tr, Y_tr, X_val, Y_val_,
                                            X_mean, X_std, Y_mean, Y_std,
                                            seed_offset=self.target_names.index(prop)*100)
            else:
                ens = make_property_model(
                    prop_kind,
                    X_mean,
                    X_std,
                    seed=self.seed + self.target_names.index(prop) * 100,
                    ensemble_size=self.ensemble_size,
                )
                ens.fit(X_prop_tr, Y_tr)
                hist = []
            Y_pred = ens.predict_mean(X_test)
            Y_std_ = ens.predict_std(X_test)
            r2     = self._r2(Y_test_, Y_pred)
            rmse   = float(np.sqrt(np.mean((Y_test_ - Y_pred)**2)))
            rows   = [{"prop": prop, "y_true": float(yt), "y_pred": float(yp),
                       "y_std": float(ys)}
                      for yt, yp, ys in zip(Y_test_, Y_pred, Y_std_)]
            metrics = {
                "r2": r2,
                "rmse": rmse,
                "model": prop_kind,
                "training_source": source,
                "n_train": int(len(prop_train_idx)),
            }
            return prop, ens, hist, metrics, rows

        # Train properties sequentially (each uses all CPU/GPU; parallel would contend)
        for prop in self.target_names:
            print(f"  [{prop}]")
            prop, ens, hist, metrics, rows = train_one_prop(prop)
            self.ensembles[prop] = ens
            self.property_model_kinds[prop] = metrics["model"]
            histories[prop]      = hist
            test_metrics[prop]   = metrics
            parity_rows.extend(rows)
            print(
                f"    model={metrics['model']} source={metrics['training_source']} "
                f"R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.6f}"
            )

        # Step 5: optimize
        if self.skip_optimize:
            print("\nStep 5/5 - Candidate optimization skipped")
            candidates = []
        else:
            print(f"\nStep 5/5 - Optimizing over ensemble mean "
                  f"(n_restarts={self.n_restarts_opt})")
            candidates = self._optimize_params()
            print(f"  {len(candidates)} unique candidate(s)")
            for i, c in enumerate(candidates[:3]):
                print(f"    [{i+1}] pred_obj={c['predicted_obj']:.6f}  "
                      + "  ".join(f"{k}={v:.4f}" for k, v in list(c["params"].items())[:3]))

        # LAMMPS validation
        lammps_results: List[dict] = []
        if not self.no_validate:
            k = min(5, len(candidates))
            print(f"\n  LAMMPS validation of top {k} candidates")
            lammps_results = self._validate_candidates(candidates[:k])
        else:
            print("\n  --no-validate: skipping LAMMPS validation")

        self._save_outputs(X_raw, Y_props, X_test_raw, test_idx, feat_names,
                           test_metrics, parity_rows, histories,
                           candidates, lammps_results)
        print(f"\n{'='*65}")
        print(f"Done. Elapsed: {time.time()-t0:.1f}s  |  {self.output_dir}/")

    # ====================================================================== #
    # Data loading                                                            #
    # ====================================================================== #

    def _load_bo_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Load valid BO evaluations.
        Returns:
          X_raw   - (N, D) raw parameter values
          Y_props - {prop_name: (N,) actual property values}
        """
        if self.training_data_mode == "core_buffer":
            return self._load_core_buffer_data()

        stable_path = os.path.join(self.bo_dir, "stable_results.csv")
        prefer_stable = self.config.get("nn", {}).get("prefer_stable_results", True)
        csv_path = os.path.join(self.bo_dir, "all_results.csv")
        if prefer_stable and os.path.exists(stable_path):
            csv_path = stable_path
            print(f"  Using stability-audited data: {csv_path}")
        if not os.path.exists(csv_path):
            import glob as _g
            pattern = "bo_*/stable_results.csv" if prefer_stable else "bo_*/all_results.csv"
            found = sorted(_g.glob(pattern),
                           key=os.path.getmtime, reverse=True)
            if not found and prefer_stable:
                found = sorted(_g.glob("bo_*/all_results.csv"),
                               key=os.path.getmtime, reverse=True)
            if found:
                csv_path = found[0]
                print(f"  Auto-discovered: {csv_path}")
            else:
                print("  ERROR: no stable_results.csv or all_results.csv found.")
                sys.exit(1)

        df = pd.read_csv(csv_path)
        validate_objective_provenance(
            df, active_targets(self.config), str(csv_path)
        )
        if "success" in df.columns:
            success = df["success"].astype(str).str.lower().eq("true")
            df = df.loc[success].copy()
        if "objective" in df.columns:
            df = df[np.isfinite(df["objective"].values)].copy()
        if df.empty:
            print("  ERROR: no valid rows.")
            sys.exit(1)

        # Truncate to the BEST fraction of points (by objective) so the surrogate
        # is accurate in the region that matters, instead of being spread thin
        # over the whole space (incl. the poor initial-LHS points).
        n0 = len(df)
        top_frac = float(self.config["nn"].get("train_top_fraction", 1.0))
        if 0.0 < top_frac < 1.0 and "objective" in df.columns and n0 > 50:
            k = max(50, int(n0 * top_frac))
            df = df.nsmallest(k, "objective").copy()
            print(f"  train_top_fraction={top_frac}: using best {len(df)}/{n0} points "
                  f"(objective <= {df['objective'].max():.4f})")

        for n in self.param_names:
            if n not in df.columns:
                df[n] = np.nan

        X_raw = df[self.param_names].values.astype(np.float64)
        valid  = np.all(np.isfinite(X_raw), axis=1)

        Y_props: Dict[str, np.ndarray] = {}
        for prop in self.target_names:
            col = f"calc_{prop}"
            if col in df.columns:
                vals = df[col].values.astype(np.float64)
                finite = np.isfinite(vals)
                valid &= finite
                Y_props[prop] = vals
            else:
                print(f"  WARNING: column '{col}' not found; skipping property {prop}")

        X_raw = X_raw[valid]
        Y_props = {p: v[valid] for p, v in Y_props.items()}
        return X_raw, Y_props

    def _successful_frame(self, path: str) -> pd.DataFrame:
        frame = pd.read_csv(path)
        validate_objective_provenance(
            frame, active_targets(self.config), str(path)
        )
        if "success" in frame.columns:
            success = frame["success"].astype(str).str.lower().eq("true")
            frame = frame.loc[success].copy()
        if "objective" in frame.columns:
            objective = pd.to_numeric(frame["objective"], errors="coerce")
            frame = frame.loc[np.isfinite(objective)].copy()
            frame["objective"] = objective.loc[frame.index]
        return frame.reset_index(drop=True)

    def _resolve_additional_core_files(self) -> List[str]:
        import glob

        resolved: List[str] = []
        roots = [Path.cwd(), Path(self.bo_dir), Path(self.bo_dir).parent]
        for specification in self.additional_core_files:
            candidate = Path(specification)
            patterns = [str(candidate)] if candidate.is_absolute() else [
                str(root / candidate) for root in roots
            ]
            matches: List[str] = []
            for pattern in patterns:
                matches.extend(glob.glob(pattern))
            for match in matches:
                absolute = str(Path(match).resolve())
                if absolute not in resolved:
                    resolved.append(absolute)
        return resolved

    def _load_core_buffer_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Load a stable core plus a wider raw-BO trend buffer."""
        raw_path = os.path.join(self.bo_dir, "all_results.csv")
        stable_path = os.path.join(self.bo_dir, "stable_results.csv")
        if not os.path.exists(raw_path):
            print(f"  ERROR: core_buffer requires {raw_path}")
            sys.exit(1)

        raw = self._successful_frame(raw_path)
        core_frames: List[pd.DataFrame] = []
        if os.path.exists(stable_path):
            core_frames.append(self._successful_frame(stable_path))
        additional_paths = self._resolve_additional_core_files()
        for path in additional_paths:
            core_frames.append(self._successful_frame(path))
        if not core_frames:
            if "_nn_core_source" in raw.columns:
                marker = raw["_nn_core_source"].astype(str).str.lower().eq("true")
                core_frames.append(raw.loc[marker].copy())
                print("  Reusing core labels embedded in all_results.csv")
            else:
                core_frames.append(raw.copy())
                print("  WARNING: no explicit stable file/core labels; using "
                      "objective-filtered all_results.csv as the core fallback")

        required_columns = self.param_names + [
            f"calc_{prop}" for prop in self.target_names
        ]
        missing_raw = [name for name in required_columns if name not in raw.columns]
        if missing_raw:
            print(f"  ERROR: raw BO data is missing columns: {missing_raw}")
            sys.exit(1)
        core = pd.concat(core_frames, ignore_index=True, sort=False)
        missing_core = [name for name in required_columns if name not in core.columns]
        if missing_core:
            print(f"  ERROR: stable core data is missing columns: {missing_core}")
            print("  Use a recipe whose active properties match the sampling data.")
            sys.exit(1)

        raw_values = raw[required_columns].apply(pd.to_numeric, errors="coerce")
        raw = raw.loc[np.isfinite(raw_values.to_numpy()).all(axis=1)].copy()
        core_values = core[required_columns].apply(pd.to_numeric, errors="coerce")
        core = core.loc[np.isfinite(core_values.to_numpy()).all(axis=1)].copy()
        best_raw_available = float(raw["objective"].min()) if len(raw) else float("nan")
        best_core_available = float(core["objective"].min()) if len(core) else float("nan")
        raw = raw.loc[raw["objective"] <= self.buffer_objective_max].copy()
        core = core.loc[core["objective"] <= self.core_objective_max].copy()
        core = core.drop_duplicates(subset=self.param_names, keep="last")
        if len(core) < 10:
            print(
                f"  ERROR: only {len(core)} stable core rows remain at "
                f"objective<={self.core_objective_max}; best available "
                f"objective={best_core_available:.6f}."
            )
            print(
                "  Use project-specific nn.training_data cutoffs for this "
                "parameter regime."
            )
            sys.exit(1)
        if len(raw) < 10:
            print(
                f"  ERROR: only {len(raw)} raw buffer rows remain at "
                f"objective<={self.buffer_objective_max}; best available "
                f"objective={best_raw_available:.6f}."
            )
            sys.exit(1)

        raw["_nn_is_core"] = False
        raw["_nn_core_copy"] = -1
        weighted_core = []
        for copy_index in range(self.core_weight):
            copy = core.copy()
            copy["_nn_is_core"] = True
            copy["_nn_core_copy"] = copy_index
            weighted_core.append(copy)
        combined = pd.concat([raw] + weighted_core, ignore_index=True, sort=False)

        rounded = np.round(
            combined[self.param_names].to_numpy(dtype=float), decimals=12
        )
        keys = pd.Series(map(tuple, rounded))
        self._data_group_ids = pd.factorize(keys, sort=False)[0]
        self._core_eval_mask = (
            combined["_nn_is_core"].to_numpy(dtype=bool)
            & combined["_nn_core_copy"].eq(0).to_numpy()
        )

        X_raw = combined[self.param_names].to_numpy(dtype=np.float64)
        Y_props = {
            prop: combined[f"calc_{prop}"].to_numpy(dtype=np.float64)
            for prop in self.target_names
        }
        print(
            f"  core_buffer: raw={len(raw)} (objective<={self.buffer_objective_max}) "
            f"core={len(core)} (objective<={self.core_objective_max}) "
            f"core_weight={self.core_weight} rows={len(combined)}"
        )
        if additional_paths:
            print(f"  Additional core files: {len(additional_paths)}")
            for path in additional_paths:
                print(f"    {path}")
        return X_raw, Y_props

    # ====================================================================== #
    # Split                                                                   #
    # ====================================================================== #

    def _compute_split_indices(
            self, N: int, Y_props: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data, keeping repeated parameter groups in only one partition."""
        if self._data_group_ids is not None and self._core_eval_mask is not None:
            return self._compute_core_buffer_split(Y_props)

        # Legacy row-level split retained for old training-data mode.
        np.random.seed(self.seed)
        n_test = max(1, int(N * self.test_fraction))
        n_val  = max(1, int(N * self.val_fraction))
        n_tr   = N - n_val - n_test
        if n_tr < 5:
            n_test = max(1, N // 10)
            n_val = max(1, N // 10)
            n_tr = N - n_val - n_test

        first_prop = list(Y_props.values())[0]
        sorted_idx = np.argsort(first_prop)
        test_idx   = sorted_idx[::max(1, N // n_test)][:n_test]
        remain     = np.setdiff1d(sorted_idx, test_idx, assume_unique=True)
        val_idx    = remain[::max(1, len(remain) // n_val)][:n_val]
        tr_idx     = np.setdiff1d(remain, val_idx, assume_unique=True)
        return tr_idx, val_idx, test_idx

    def _compute_core_buffer_split(
            self, Y_props: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        canonical = np.flatnonzero(self._core_eval_mask)
        rng = np.random.default_rng(self.seed)
        shuffled = rng.permutation(canonical)
        n_test = max(1, int(len(canonical) * self.test_fraction))
        n_val = max(1, int(len(canonical) * self.val_fraction))
        test_idx = shuffled[:n_test]
        val_idx = shuffled[n_test:n_test + n_val]

        held_out_groups = set(self._data_group_ids[np.r_[test_idx, val_idx]])
        train_mask = np.asarray([
            group not in held_out_groups for group in self._data_group_ids
        ])
        tr_idx = np.flatnonzero(train_mask)
        print(
            f"  Group-safe core split: {len(canonical)} unique stable candidates; "
            f"no repeated parameter set crosses train/val/test"
        )
        return tr_idx, val_idx, test_idx

    # ====================================================================== #
    # Training                                                                #
    # ====================================================================== #

    def _train_ensemble(self,
                        ens:      EnsembleMLP,
                        X_tr:     np.ndarray,
                        Y_tr:     np.ndarray,
                        X_val:    np.ndarray,
                        Y_val:    np.ndarray,
                        X_mean:   np.ndarray,
                        X_std:    np.ndarray,
                        Y_mean:   float,
                        Y_std:    float,
                        seed_offset: int = 0) -> List[dict]:
        def to_t(x, y):
            xs = torch.tensor((x - X_mean) / (X_std + 1e-8), dtype=torch.float32).to(self.device)
            ys = torch.tensor((y - Y_mean) / (Y_std + 1e-8), dtype=torch.float32).to(self.device)
            return xs, ys

        Xt_tr, Yt_tr   = to_t(X_tr, Y_tr)
        Xt_val, Yt_val = to_t(X_val, Y_val)
        n_tr   = len(Xt_tr)
        loss_fn = (
            nn.SmoothL1Loss() if self.loss_name == "huber" else nn.MSELoss()
        )
        hists: List[dict] = []

        for m_idx, model in enumerate(ens.members):
            torch.manual_seed(self.seed + seed_offset + m_idx * 1000)
            model.apply(self._reinit_weights)
            model.train()
            opt = optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            sched = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5,
                patience=max(1, self.patience // 3), min_lr=1e-6)

            best_val = float("inf")
            best_sd = None
            no_imp = 0
            tr_losses: List[float] = []
            val_losses: List[float] = []

            for epoch in range(self.max_epochs):
                perm = torch.randperm(n_tr, device=self.device)
                ep_l = 0.0
                nb = 0
                model.train()
                for start in range(0, n_tr, self.batch_size_nn):
                    idx = perm[start:start + self.batch_size_nn]
                    opt.zero_grad()
                    loss = loss_fn(model(Xt_tr[idx]), Yt_tr[idx])
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    ep_l += loss.item()
                    nb += 1
                ep_l /= max(nb, 1)

                model.eval()
                with torch.no_grad():
                    vl = loss_fn(model(Xt_val), Yt_val).item()
                tr_losses.append(ep_l)
                val_losses.append(vl)
                sched.step(vl)
                if vl < best_val - 1e-7:
                    best_val = vl
                    best_sd = {k: v.clone() for k, v in model.state_dict().items()}
                    no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= self.patience:
                    break

            if best_sd:
                model.load_state_dict(best_sd)
            model.eval()
            hists.append({"train": tr_losses, "val": val_losses})

        return hists

    @staticmethod
    def _reinit_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(m.bias, -bound, bound)

    # ====================================================================== #
    # NN-based parameter optimization                                         #
    # ====================================================================== #

    def _objective_from_nn(self, x_raw: np.ndarray) -> float:
        """
        Weighted RMSE of relative errors across all per-property NN predictions.

        Formula mirrors lammps_interface._compute_objective exactly:
            sqrt(sum w_i * relative_error_i^2 / sum w_i)

        Normalising by weight sum makes the value independent of the number of
        active properties; necessary for scipy optimisation across different runs.
        """
        x = np.clip(x_raw, self.param_lo, self.param_hi).reshape(1, -1)
        x = self._transform_features(x)
        sq_sum     = 0.0
        weight_sum = 0.0
        for prop, ens in self.ensembles.items():
            target = float(self.targets[prop]["value"])
            weight = float(self.targets[prop].get("weight", 1.0))
            pred   = float(ens.predict_mean(x)[0])
            rel_err    = abs(pred - target) / (abs(target) + 1e-10)
            sq_sum    += weight * rel_err ** 2
            weight_sum += weight
        return float(np.sqrt(sq_sum / weight_sum)) if weight_sum > 1e-12 else float("inf")

    def _optimize_params(self) -> List[dict]:
        bounds_scipy = list(zip(self.param_lo, self.param_hi))
        n_sobol = max(self.n_restarts_opt * 2, 128)
        sampler = qmc.Sobol(d=self.n_params, scramble=True, seed=self.seed)
        X_start = qmc.scale(sampler.random(n_sobol), self.param_lo, self.param_hi)
        print(f"  Method={self.opt_method}  Restarts={self.n_restarts_opt}")

        raw_results: List[dict] = []

        if self.opt_method == "differential_evolution":
            try:
                res = differential_evolution(
                    self._objective_from_nn, bounds=bounds_scipy,
                    seed=self.seed, maxiter=5000, tol=1e-8, polish=True)
                if np.isfinite(res.fun):
                    raw_results.append({
                        "params": {n: float(np.clip(res.x[i], self.param_lo[i], self.param_hi[i]))
                                   for i, n in enumerate(self.param_names)},
                        "predicted_obj": float(res.fun)})
            except Exception as e:
                print(f"  WARNING: DE failed: {e}")
        else:
            def obj_grad(x):
                f0   = self._objective_from_nn(x)
                eps  = np.maximum(1e-4 * (self.param_hi - self.param_lo), 1e-8)
                grad = np.array([(self._objective_from_nn(x + eps * (np.arange(self.n_params) == i)) -
                                  self._objective_from_nn(x - eps * (np.arange(self.n_params) == i))) /
                                 (2 * eps[i]) for i in range(self.n_params)])
                return f0, grad

            for r_idx in range(min(self.n_restarts_opt, n_sobol)):
                try:
                    if self.opt_method == "L-BFGS-B":
                        res = minimize(obj_grad, X_start[r_idx], method="L-BFGS-B",
                                       jac=True, bounds=bounds_scipy,
                                       options={"maxiter": 1000, "ftol": 1e-10})
                    else:
                        res = minimize(self._objective_from_nn, X_start[r_idx],
                                       method="Nelder-Mead",
                                       options={"maxiter": 5000, "xatol": 1e-8})
                    if np.isfinite(res.fun):
                        x = np.clip(res.x, self.param_lo, self.param_hi)
                        raw_results.append({
                            "params": {n: float(x[i]) for i, n in enumerate(self.param_names)},
                            "predicted_obj": float(res.fun)})
                except Exception:
                    continue

        raw_results.sort(key=lambda r: r["predicted_obj"])
        span = self.param_hi - self.param_lo + 1e-10
        unique: List[dict] = []
        for c in raw_results:
            x = np.array(list(c["params"].values()))
            if not any(np.max(np.abs(x - np.array(list(k["params"].values()))) / span) < 0.01
                       for k in unique):
                unique.append(c)
        return unique or raw_results[:1] if raw_results else []

    # ====================================================================== #
    # LAMMPS validation                                                       #
    # ====================================================================== #

    def _validate_candidates(self, candidates: List[dict]) -> List[dict]:
        if not candidates:
            return []
        val_dir = os.path.join(self.output_dir, "lammps_validation")
        os.makedirs(val_dir, exist_ok=True)
        results = self.runner.evaluate_batch(
            [c["params"] for c in candidates], val_dir, save_traj=self.save_traj)
        enriched = []
        for i, (c, r) in enumerate(zip(candidates, results)):
            e = {"rank": i+1, "params": c["params"], "predicted_obj": c["predicted_obj"],
                 "lammps_success": r.success}
            if r.success:
                e["lammps_obj"] = r.objective
                e["lammps_properties"] = r.properties
                e["per_property_error"] = r.per_property_error
                print(f"    [{i+1}] pred={c['predicted_obj']:.6f}  LAMMPS={r.objective:.6f}")
            else:
                e["lammps_obj"] = None
                e["error_msg"]  = r.error_msg
                print(f"    [{i+1}] pred={c['predicted_obj']:.6f}  FAILED: {r.error_msg}")
            enriched.append(e)
        return enriched

    # ====================================================================== #
    # Save outputs                                                            #
    # ====================================================================== #

    def _save_outputs(self, X_raw, Y_props, X_test_raw, test_idx,
                      feat_names, test_metrics, parity_rows,
                      histories, candidates, lammps_results):
        out = Path(self.output_dir)

        # forward_nn.pt - self-contained, one sub-dict per property
        property_models = {}
        for prop, ens in self.ensembles.items():
            prop_kind = self.property_model_kinds.get(prop, self.model_kind)
            if prop_kind == "mlp_ensemble":
                property_models[prop] = {
                    "model_kind":  prop_kind,
                    "state_dicts": ens.state_dicts(),
                    "X_mean":      ens.X_mean.tolist(),
                    "X_std":       ens.X_std.tolist(),
                    "Y_mean":      ens.Y_mean,
                    "Y_std":       ens.Y_std,
                }
            else:
                property_models[prop] = {
                    "model_kind": prop_kind,
                    "obj": ens,
                }
        save_dict = {
            "property_models": property_models,
            "model_kind":      self.model_kind,
            "property_model_kinds": self.property_model_kinds,
            "target_names":    self.target_names,
            "input_dim":       len(feat_names),
            "hidden_layers":   self.hidden_layers,
            "activation":      self.activation,
            "ensemble_size":   self.ensemble_size,
            "loss":            self.loss_name,
            "weight_decay":    self.weight_decay,
            "physics_features": self.physics_features,
            "derived_charge_feature": self.derived_charge_feature,
            "param_names":     self.param_names,
            "feat_names":      feat_names,
            "n_raw_params":    self.n_params,
            "param_lo":        self.param_lo.tolist(),
            "param_hi":        self.param_hi.tolist(),
            "atom_types":      self.config["atom_types"],
            "charge_cfg":      self.config["charge"],
            "pair_params":     self.config.get("pair_params", {}),
            "targets":         self.config["targets"],
            "compute_surface": self.config["lammps"]["compute_surface"],
        }
        pt_path = out / "forward_nn.pt"
        torch.save(save_dict, pt_path)
        print(f"\n  Saved: {pt_path}")

        # nn_optimize_result.json
        valid_l  = [r for r in lammps_results if r.get("lammps_success")]
        best_lmp = min(valid_l, key=lambda r: r["lammps_obj"]) if valid_l else None
        best_nn  = candidates[0] if candidates else None
        json.dump({
            "best_lammps":      best_lmp,
            "best":             best_lmp or best_nn,
            "best_nn":          best_nn,
            "all_candidates":   candidates,
            "all_lammps":       lammps_results,
            "test_metrics":     test_metrics,
            "target_names":     self.target_names,
            "param_names":      self.param_names,
            "feat_names":       feat_names,
            "ensemble_size":    self.ensemble_size,
            "physics_features": self.physics_features,
            "derived_charge_feature": self.derived_charge_feature,
            "timestamp":        datetime.now().isoformat(),
        }, open(out / "nn_optimize_result.json", "w"), indent=2, default=str)
        print(f"  Saved: {out / 'nn_optimize_result.json'}")

        # train_history.json
        json.dump({"properties": histories},
                  open(out / "train_history.json", "w"), indent=2)

        # parity_data.csv
        pd.DataFrame(parity_rows).to_csv(out / "parity_data.csv", index=False)

        # feature_names.txt
        (out / "feature_names.txt").write_text("\n".join(feat_names))

        # Summary
        print()
        print("Summary:")
        for prop, m in test_metrics.items():
            print(f"  {prop:<14} R2={m['r2']:.4f}  RMSE={m['rmse']:.6f}")
        if best_lmp:
            print(f"\n  Best (LAMMPS-validated) obj={best_lmp['lammps_obj']:.6f}")
            for k, v in best_lmp["params"].items():
                print(f"    {k:<28} = {v:.8f}")

    @staticmethod
    def _r2(y_true, y_pred) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 1.0


# ============================================================================
# Public loader
# ============================================================================

def load_ensemble_from_file(
        pt_path: str,
        device: Optional[torch.device] = None,
) -> Tuple[Dict[str, EnsembleMLP], "PhysicsFeatureBuilder",
           List[str], List[str], dict]:
    """
    Reconstruct per-property EnsembleMLPs from saved forward_nn.pt.

    Returns:
      ensembles    : {prop_name: EnsembleMLP}
      feat_builder : PhysicsFeatureBuilder (for transforming raw params)
      param_names  : list[str]
      feat_names   : list[str]
      meta         : full save_dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        meta = torch.load(pt_path, map_location=device, weights_only=False)
    except TypeError:
        meta = torch.load(pt_path, map_location=device)   # older torch
    model_kind = meta.get("model_kind", "mlp_ensemble")

    ensembles: Dict = {}
    hidden_layers = meta["hidden_layers"]
    activation    = meta["activation"]
    ensemble_size = meta["ensemble_size"]
    input_dim     = meta["input_dim"]
    for prop, pdata in meta["property_models"].items():
        prop_kind = pdata.get("model_kind", model_kind)
        if prop_kind == "mlp_ensemble":
            ens = EnsembleMLP(input_dim, hidden_layers, activation, ensemble_size, device)
            ens.load_state_dicts(pdata["state_dicts"])
            ens.X_mean = np.array(pdata["X_mean"])
            ens.X_std  = np.array(pdata["X_std"])
            ens.Y_mean = float(pdata["Y_mean"])
            ens.Y_std  = float(pdata["Y_std"])
            ensembles[prop] = ens
        else:
            # Picklable sklearn-backed wrapper.
            ensembles[prop] = pdata["obj"]
            if prop_kind == "multitask_mlp":
                shared = ensembles[prop].ensemble
                shared.device = device
                for member in shared.members:
                    member.to(device).eval()

    partial_config = {
        "atom_types": meta["atom_types"],
        "charge": meta["charge_cfg"],
        "pair_params": meta.get("pair_params", {}),
    }
    feat_builder   = PhysicsFeatureBuilder(meta["param_names"], partial_config)

    return ensembles, feat_builder, meta["param_names"], meta["feat_names"], meta


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NN Surrogate - Independent per-property Ensemble MLP")
    parser.add_argument("--config", required=True)
    parser.add_argument("--bo-dir",      default=None,
                        help="BO output dir; stable_results.csv preferred, "
                             "all_results.csv fallback.")
    parser.add_argument("--output-dir",  default=None)
    parser.add_argument("--model",
                        choices=["mlp_ensemble", "gp", "random_forest", "xgboost"],
                        default=None,
                        help="Override nn.model without editing the config.")
    parser.add_argument("--train-top-fraction", type=float, default=None,
                        help="Override nn.train_top_fraction for this run.")
    parser.add_argument(
        "--additional-core-file", action="append", default=None,
        help="Additional stability-mean CSV or glob; repeat for multiple files.",
    )
    parser.add_argument(
        "--core-objective-max", type=float, default=None,
        help="Override nn.training_data.core_objective_max.",
    )
    parser.add_argument(
        "--buffer-objective-max", type=float, default=None,
        help="Override nn.training_data.buffer_objective_max.",
    )
    parser.add_argument(
        "--core-weight", type=int, default=None,
        help="Override the integer weight of stability-audited candidate means.",
    )
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--save-traj",   action="store_true")
    parser.add_argument(
        "--reuse-models-from",
        default=None,
        help="Reuse properties without nn.property_models overrides from an "
             "existing forward_nn.pt.",
    )
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Train, score and save the surrogate without parameter search.",
    )
    args = parser.parse_args()

    config = load_expanded_config(args.config)

    if args.model:
        config.setdefault("nn", {})["model"] = args.model
    if args.train_top_fraction is not None:
        config.setdefault("nn", {})["train_top_fraction"] = args.train_top_fraction
    training_cfg = config.setdefault("nn", {}).setdefault("training_data", {})
    if args.additional_core_file:
        training_cfg["additional_core_files"] = args.additional_core_file
    if args.core_objective_max is not None:
        training_cfg["core_objective_max"] = args.core_objective_max
    if args.buffer_objective_max is not None:
        training_cfg["buffer_objective_max"] = args.buffer_objective_max
    if args.core_weight is not None:
        training_cfg["core_weight"] = args.core_weight

    if not config.get("nn", {}).get("enabled", True):
        print("nn.enabled: false - skipping.")
        sys.exit(0)

    bo_dir = args.bo_dir
    if bo_dir is None:
        import glob as _g
        sysname = config["manifest"]["system_name"]
        # Prefer bo dirs matching THIS config's system_name (so cfg1/cfg2/full
        # experiments don't read each other's data); prefer stability-audited
        # labels when present, then fall back to any bo_*.
        prefer_stable = config.get("nn", {}).get("prefer_stable_results", True)
        found = []
        if prefer_stable:
            found = sorted(_g.glob(f"bo_{sysname}_*/stable_results.csv"),
                           key=os.path.getmtime, reverse=True)
        if not found:
            found = sorted(_g.glob(f"bo_{sysname}_*/all_results.csv"),
                       key=os.path.getmtime, reverse=True)
        if not found and prefer_stable:
            found = sorted(_g.glob("bo_*/stable_results.csv"),
                           key=os.path.getmtime, reverse=True)
        if not found:
            found = sorted(_g.glob("bo_*/all_results.csv"),
                           key=os.path.getmtime, reverse=True)
        bo_dir = os.path.dirname(found[0]) if found else None
        if bo_dir:
            print(f"Auto-selected BO dir: {bo_dir}")
        else:
            print("ERROR: --bo-dir not given and no bo_*/all_results.csv found.")
            sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        sys_name   = config["manifest"]["system_name"]
        output_dir = f"nn_output_{sys_name}_{ts}"

    NNSurrogate(config=config, bo_dir=bo_dir, output_dir=output_dir,
                no_validate=args.no_validate, save_traj=args.save_traj,
                reuse_models_from=args.reuse_models_from,
                skip_optimize=args.skip_optimize).run()


if __name__ == "__main__":
    main()
