#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml/cv_utils.py
--------------
Cross-validation utilities, metrics, and plotting helpers.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Model factory --------------------

def make_model(model_name: str, random_state: int = 42):
    """
    Currently supported:
      - 'logit_enet' (default): StandardScaler + LogisticRegression (saga, elastic net)
    """
    name = (model_name or "logit_enet").strip().lower()
    if name == "logit_enet":
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),  # X already z-scored; keep safe
            ("clf", LogisticRegression(
                solver="saga", penalty="elasticnet", l1_ratio=0.5,
                max_iter=5000, class_weight="balanced", random_state=random_state, n_jobs=1
            ))
        ])
        param_grid = {
            "clf__C": np.logspace(-2, 2, 7),
            "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]
        }
        return pipe, param_grid
    else:
        raise ValueError(f"Unknown model: {model_name}")

# -------------------- Metrics --------------------

def compute_metrics(y_true, y_prob, y_pred) -> Dict[str, float]:
    return dict(
        auc_roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        auc_pr  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        accuracy= accuracy_score(y_true, y_pred),
        bal_acc = balanced_accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, zero_division=0),
        recall    = recall_score(y_true, y_pred, zero_division=0),
        f1        = f1_score(y_true, y_pred, zero_division=0),
    )

# -------------------- CV --------------------

def nested_stratified_cv(
    X: pd.DataFrame,
    y: pd.Series,
    outer_folds: int = 5,
    inner_folds: int = 3,
    model_name: str = "logit_enet",
    random_state: int = 42
) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    """
    Nested CV: outer StratifiedKFold; inner GridSearch for hyperparameters.
    Returns per-fold metrics DataFrame and lists of fold-wise y_true and y_prob.
    """
    pipe, param_grid = make_model(model_name, random_state=random_state)
    outer_cv = StratifiedKFold(n_splits=min(outer_folds, len(y.unique()) and outer_folds), shuffle=True, random_state=random_state)

    rows = []
    y_true_list, y_prob_list = [], []

    for fold, (tr, va) in enumerate(outer_cv.split(X, y), start=1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        inner_cv = StratifiedKFold(n_splits=min(inner_folds, len(ytr.unique()) and inner_folds), shuffle=True, random_state=random_state)
        gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=1, refit=True)
        gs.fit(Xtr, ytr)

        best = gs.best_estimator_
        prob = best.predict_proba(Xva)[:,1]
        pred = (prob >= 0.5).astype(int)

        met = compute_metrics(yva, prob, pred)
        met.update(fold=fold, n_train=len(tr), n_valid=len(va),
                   best_params=str(gs.best_params_))
        rows.append(met)

        y_true_list.append(yva.values)
        y_prob_list.append(prob)

    df = pd.DataFrame(rows)
    return df, y_true_list, y_prob_list

# -------------------- Plotting --------------------

def plot_roc_pr(y_true_list: List[np.ndarray], y_prob_list: List[np.ndarray], out_png: str, title: str):
    # Concatenate all folds for a single curve estimate
    y_true = np.concatenate(y_true_list) if len(y_true_list) else np.array([])
    y_prob = np.concatenate(y_prob_list) if len(y_prob_list) else np.array([])
    if y_true.size == 0:
        _placeholder(out_png, "No test predictions")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    auc_roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else np.nan
    auc_pr  = average_precision_score(y_true, y_prob) if len(np.unique(y_true))>1 else np.nan

    fig, ax = plt.subplots(1,2, figsize=(10,4.5))
    ax[0].plot(fpr, tpr)
    ax[0].plot([0,1],[0,1], ls="--", c="#999999")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[0].set_title(f"ROC (AUC={auc_roc:.3f})" if np.isfinite(auc_roc) else "ROC")

    ax[1].plot(rec, prec)
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title(f"PR (AP={auc_pr:.3f})" if np.isfinite(auc_pr) else "PR")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _placeholder(out_png: str, msg: str):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.axis("off")
    ax.text(0.5,0.5,msg,ha="center",va="center")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
