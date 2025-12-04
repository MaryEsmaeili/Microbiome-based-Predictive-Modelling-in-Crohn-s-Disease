# scripts/ml/cv_utils.py
"""
Cross-validated model utilities for binary microbiome classification.

This module centralizes model construction, nested cross-validation, permutation
testing, and learning-curve generation for the ML part of the project
(e.g. disease vs healthy, PPI use, treatment response, transfer tasks).

Supported models
----------------
get_model_and_grid(model_name) returns a (Pipeline, param_grid) pair for:

  - "logit"  : LogisticRegression (saga), with grids for pure L2 and elastic-net
               penalties (C and l1_ratio).
  - "svm"    : SVC (probability=True), with grids over C, kernel, and gamma.
  - "rf"     : RandomForestClassifier (class_weight="balanced"), with grids over
               n_estimators, max_depth, and min_samples_leaf.
  - "xgb"    : XGBClassifier (if xgboost is installed), with grids over
               max_depth, learning_rate, subsample, colsample_bytree, reg_lambda.

All models are wrapped in a Pipeline with:
  - ColumnTransformer → SimpleImputer(median) + StandardScaler for all columns
  - "clf" final estimator (logit / svm / rf / xgb)

This makes it possible to use a uniform interface for GridSearchCV and
downstream access via best_estimator_.named_steps["clf"].

Nested cross-validation
-----------------------
nested_cv_evaluate(
    X, y, model_name, outdir,
    groups=None,
    strat_labels=None,
    n_splits_outer=5,
    n_splits_inner=4,
    random_state=42,
    min_pos_per_fold=1
) -> Dict[str, Any]

Functionality:
  - Optionally uses StratifiedGroupKFold for the OUTER loop when non-trivial
    groups are provided; otherwise falls back to StratifiedKFold.
  - Uses 'strat_labels' for stratification if given (e.g. combined disease +
    site/batch labels), while 'y' remains the true target.
  - Inner CV: GridSearchCV with StratifiedKFold(n_splits_inner) and ROC-AUC as
    the scoring metric.
  - For each outer fold:
      * Fit the inner grid on training data.
      * Select best estimator, compute fold-level probabilities on test data.
      * Rescale decision_function outputs to [0,1] if needed.
      * Store out-of-fold (OOF) probabilities for all samples.
      * Compute per-fold metrics at a fixed 0.5 threshold:
          - roc_auc, average_precision, accuracy, balanced_accuracy,
            precision, recall, f1.

Outputs (saved to 'outdir'):
  - "cv_metrics.csv"      : one row per outer fold with the metrics above.
  - "cv_predictions.csv"  : per-sample OOF predictions:
                              * sample_id, y_true, y_prob
  - "roc_curve.csv"       : points for ROC curve (fpr, tpr).
  - "pr_curve.csv"        : points for PR curve (recall, precision).
  - "roc.png"             : ROC curve plot (AUC reported in legend).
  - "pr.png"              : Precision–recall curve plot (AP in legend).
  - "calibration.png"     : calibration curve plot (Brier score in legend).
  - "importances.csv"     : (optional) coefficient/feature_importance summary
                            if per-fold coefficients are aggregated.
  - "importances.png"     : (optional) bar plot of top features with mean ± SD
                            across folds.

The function returns a dict of key paths and the final Brier score for the
aggregated OOF predictions.

Plot helpers
------------
The module includes small plotting helpers that produce publication-ready,
compact figures:

  - plot_roc(y_true, y_prob, out_png, title, subtitle)
  - plot_pr(y_true, y_prob, out_png, title, subtitle)
  - plot_calibration(y_true, y_prob, out_png, title, subtitle) -> brier
  - plot_feature_importance_with_errorbars(coef_df, out_png, title, subtitle)

All plots share a consistent layout/use of _decorate() for left-aligned titles
and light grids, and are suitable for direct inclusion in reports.

Permutation testing
-------------------
run_permutation_test(
    X, y, model_name, outdir,
    n_perm=200,
    random_state=7
) -> pd.DataFrame

Implements a label-permutation test for ROC-AUC:

  - For each permutation:
      * Shuffle labels (keeping X fixed).
      * Run a single (non-nested) CV with StratifiedKFold (5 folds).
      * Within each fold, use GridSearchCV(3-fold) with ROC-AUC as scoring.
      * Aggregate OOF predictions and compute ROC-AUC against the TRUE y.
  - Saves "permutation_auc.csv" with columns ['perm', 'auc'] in outdir.
  - Returns the same DataFrame for downstream p-value estimation or plotting.

Learning curves
---------------
save_learning_curve(
    X, y, model_name,
    out_png, out_csv,
    cv_splits=5,
    random_state=42
)

Computes and plots a learning curve (train size vs CV ROC-AUC):

  - Uses StratifiedKFold(cv_splits) as CV.
  - Uses a single "reasonable" base_estimator from get_model_and_grid
    (no inner grid search, to keep runtime manageable).
  - For logistic-type models, sets clf__l1_ratio=0.5 as a default.
  - Saves:
      * out_csv : table with columns:
          - train_size, train_auc_mean, train_auc_std,
            cv_auc_mean, cv_auc_std
      * out_png : line plot of cv_auc_mean vs train_size with ±1 SD band.

Overall, this module provides the core reusable components for robust
cross-validated model evaluation, calibration, and learning-curve diagnostics
for all binary classification tasks in the project.
"""


import os
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, learning_curve
)
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, brier_score_loss, accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
plt.switch_backend("Agg")  # Safe for headless environments

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# -------------------------------
# Model builders + grids
# -------------------------------

def get_model_and_grid(model_name: str):
    """
    Return (pipeline, param_grid) for model_name in {"logit","svm","rf","xgb"}.
    We always wrap the estimator into a Pipeline with a 'clf' step so the rest of the
    code can uniformly access best.named_steps['clf'].
    """
    # simple numeric preprocessor (impute + optional scale). For trees, scaling won't matter.
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # keep StandardScaler so linear / SVM benefit; it won't hurt trees
        ("scale", StandardScaler(with_mean=False))
    ])

    preproc = ColumnTransformer(
        transformers=[("num", numeric, slice(0, None))],
        remainder="drop"
    )

    name = model_name.lower()

    if name == "logit":
        clf = LogisticRegression(
            solver="saga", max_iter=5000, n_jobs=-1
        )
        # Two grids: one for pure L2, one for elastic-net
        param_grid = [
            {"clf__penalty": ["l2"], "clf__C": [0.01, 0.1, 1, 10]},
            {"clf__penalty": ["elasticnet"], "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
             "clf__C": [0.01, 0.1, 1, 10]}
        ]

    elif name == "svm":
        clf = SVC(probability=True)
        param_grid = {
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale", "auto"]
        }

    elif name == "rf":
        clf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
        param_grid = {
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4]
        }

    elif name == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("xgboost is not installed.")
        clf = XGBClassifier(
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_estimators=400,
            n_jobs=-1
        )
        param_grid = {
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.7, 1.0],
            "clf__colsample_bytree": [0.7, 1.0],
            "clf__reg_lambda": [0.0, 1.0]
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipe = Pipeline([("prep", preproc), ("clf", clf)])
    return pipe, param_grid

# -------------------------------
# Plot helpers (publication-ready)
# -------------------------------
def _final_estimator(est):
    # Pipeline? take the 'clf' step (or the last step if unnamed)
    if hasattr(est, "named_steps"):
        return est.named_steps.get("clf", list(est.named_steps.values())[-1])
    return est

def _decorate(ax, title: str, subtitle: str = ""):
    ax.set_title(title + ("\n" + subtitle if subtitle else ""), loc="left", fontsize=11)
    ax.grid(True, alpha=0.25)

def plot_roc(y_true, y_prob, out_png: str, title: str, subtitle: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1], ls="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    _decorate(ax, title, subtitle)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_pr(y_true, y_prob, out_png: str, title: str, subtitle: str):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    _decorate(ax, title, subtitle)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_calibration(y_true, y_prob, out_png: str, title: str, subtitle: str) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, y_prob)
    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    ax.plot(prob_pred, prob_true, marker="o", lw=1.5, label=f"Brier = {brier:.3f}")
    ax.plot([0,1],[0,1], ls="--", lw=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend(loc="upper left")
    _decorate(ax, title, subtitle)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return brier

def plot_feature_importance_with_errorbars(
    coef_df: pd.DataFrame, out_png: str, title: str, subtitle: str, top_k: int = 20
):
    """
    coef_df columns expected: ['feature', 'mean', 'std', 'abs_mean'].
    """
    if coef_df.empty:
        return
    df = coef_df.sort_values("abs_mean", ascending=False).head(top_k)
    fig = plt.figure(figsize=(5.2, max(2.8, 0.25 * len(df))))
    ax = fig.add_subplot(111)
    ax.barh(df["feature"], df["mean"], xerr=df["std"], alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Weight (mean ± SD across folds)")
    _decorate(ax, title, subtitle)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

# -------------------------------
# CV runner
# -------------------------------
def nested_cv_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    outdir: str,
    groups: Optional[pd.Series] = None,
    strat_labels: Optional[pd.Series] = None,   # NEW
    n_splits_outer: int = 5,
    n_splits_inner: int = 4,
    random_state: int = 42,
    min_pos_per_fold: int = 1
) -> Dict[str, Any]:
    """
    Perform nested CV with optional group-aware outer splits. Save figures and CSVs.
    strat_labels (if provided) is used only for stratification of outer folds
    (e.g., combining disease + batch), while y remains the true target.

    NEW:
      - Save per-sample OOF probabilities in 'cv_predictions.csv'
      - Save ROC/PR curve points in 'roc_curve.csv' and 'pr_curve.csv'
    """
    os.makedirs(outdir, exist_ok=True)

    # Decide labels used for stratification in outer CV
    if strat_labels is not None:
        labels_for_outer = strat_labels
    else:
        labels_for_outer = y

    # Align index if it's a Series/DataFrame
    if isinstance(labels_for_outer, (pd.Series, pd.DataFrame)):
        labels_for_outer = labels_for_outer.loc[X.index]

    # Choose splitter
    if groups is not None and len(groups.unique()) > 1 and StratifiedGroupKFold is not None:
        outer = StratifiedGroupKFold(
            n_splits=n_splits_outer,
            shuffle=True,
            random_state=random_state
        )
        use_groups = True
    else:
        outer = StratifiedKFold(
            n_splits=n_splits_outer,
            shuffle=True,
            random_state=random_state
        )
        use_groups = False

    # Model + grid
    model, grid = get_model_and_grid(model_name)

    fold_rows = []
    # OOF predictions for all samples
    oof_prob = pd.Series(index=X.index, dtype=float)

    # (we are not using per_fold_coefs yet)
    per_fold_coefs: List[pd.DataFrame] = []

    for k, (tr, te) in enumerate(
        outer.split(X, labels_for_outer, groups if use_groups else None),
        start=1
    ):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        # Guard: skip degenerate folds
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue

        inner = StratifiedKFold(
            n_splits=n_splits_inner,
            shuffle=True,
            random_state=random_state + 13
        )
        gscv = GridSearchCV(model, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        gscv.fit(Xtr, ytr)

        best = gscv.best_estimator_

        if hasattr(best, "predict_proba"):
            prob = best.predict_proba(Xte)[:, 1]
        else:
            prob = best.decision_function(Xte)

        # If decision_function not in [0,1], rescale to [0,1]
        if prob.min() < 0 or prob.max() > 1:
            pmin, pmax = prob.min(), prob.max()
            if pmax > pmin:
                prob = (prob - pmin) / (pmax - pmin)

        # Store OOF probabilities
        oof_prob.iloc[te] = prob

        # Threshold = 0.5 for metrics
        y_pred = (prob >= 0.5).astype(int)

        row = {
            "fold": k,
            "roc_auc": roc_auc_score(yte, prob),
            "ap": average_precision_score(yte, prob),
            "acc": accuracy_score(yte, y_pred),
            "bal_acc": balanced_accuracy_score(yte, y_pred),
            "precision": precision_score(yte, y_pred, zero_division=0),
            "recall": recall_score(yte, y_pred, zero_division=0),
            "f1": f1_score(yte, y_pred, zero_division=0),
        }
        fold_rows.append(row)

        # Optional: per-fold importances (still not aggregated)
        step = _final_estimator(best)
        imps = None
        if hasattr(step, "coef_"):
            coefs = step.coef_
            if getattr(coefs, "ndim", 1) == 2:
                coefs = coefs[0]
            imps = pd.DataFrame({"feature": X.columns, "weight": coefs})
        elif hasattr(step, "feature_importances_"):
            imps = pd.DataFrame({"feature": X.columns, "weight": step.feature_importances_})

        if imps is not None and outdir is not None:
            imps.sort_values("weight", key=abs, ascending=False)\
                .to_csv(os.path.join(outdir, "importances.csv"), index=False)

    # ---- Fold-level metrics ----
    metrics_df = pd.DataFrame(fold_rows)
    metrics_df.to_csv(os.path.join(outdir, "cv_metrics.csv"), index=False)

    # ---- Build per-sample prediction table (OOF) ----
    valid_mask = oof_prob.notna()
    valid_idx = oof_prob.index[valid_mask]
    y_true = y.loc[valid_idx]
    y_prob = oof_prob.loc[valid_idx]

    pred_df = pd.DataFrame({
        "sample_id": valid_idx.astype(str),
        "y_true": y_true.values,
        "y_prob": y_prob.values,
    })
    pred_df.to_csv(os.path.join(outdir, "cv_predictions.csv"), index=False)

    # ---- ROC / PR curve points as CSV ----
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        os.path.join(outdir, "roc_curve.csv"), index=False
    )
    pd.DataFrame({"recall": rec, "precision": prec}).to_csv(
        os.path.join(outdir, "pr_curve.csv"), index=False
    )

    # ---- Plots (single-model, as before) ----
    title = f"Nested CV — model={model_name}"
    subtitle = (
        f"n={len(y_true)} | pos={int(y_true.sum())} "
        f"({y_true.mean():.2%}) | folds={n_splits_outer}x{n_splits_inner}"
    )

    plot_roc(
        y_true, y_prob,
        os.path.join(outdir, "roc.png"),
        title, subtitle
    )
    plot_pr(
        y_true, y_prob,
        os.path.join(outdir, "pr.png"),
        title, subtitle
    )
    brier = plot_calibration(
        y_true, y_prob,
        os.path.join(outdir, "calibration.png"),
        title, subtitle
    )

    # Optional feature stability across folds (still off by default)
    coef_summary = pd.DataFrame()
    if per_fold_coefs:
        allc = pd.concat(per_fold_coefs, ignore_index=True)
        coef_summary = (
            allc.groupby("feature")["weight"]
                .agg(["mean", "std"])
                .reset_index()
        )
        coef_summary["abs_mean"] = coef_summary["mean"].abs()
        coef_summary.to_csv(os.path.join(outdir, "importances.csv"), index=False)
        plot_feature_importance_with_errorbars(
            coef_summary,
            os.path.join(outdir, "importances.png"),
            title="Top features (mean ± SD across folds)",
            subtitle=subtitle,
        )

    return {
        "metrics_path": os.path.join(outdir, "cv_metrics.csv"),
        "predictions_path": os.path.join(outdir, "cv_predictions.csv"),
        "roc_curve_path": os.path.join(outdir, "roc_curve.csv"),
        "pr_curve_path": os.path.join(outdir, "pr_curve.csv"),
        "roc_path": os.path.join(outdir, "roc.png"),
        "pr_path": os.path.join(outdir, "pr.png"),
        "calibration_path": os.path.join(outdir, "calibration.png"),
        "brier": brier,
        "importances_path": os.path.join(outdir, "importances.csv")
        if not coef_summary.empty
        else "",
    }

def run_permutation_test(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    outdir: str,
    n_perm: int = 200,
    random_state: int = 7
) -> pd.DataFrame:
    """
    Permute labels to get a null distribution of ROC-AUC.
    Saves 'permutation_auc.csv' with columns ['perm','auc'].
    """
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(random_state)
    aucs = []
    for i in range(1, n_perm + 1):
        y_perm = y.sample(frac=1.0, replace=False, random_state=rng).values
        # Single 5-fold CV (non-nested) for speed
        model, grid = get_model_and_grid(model_name)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng.randint(0, 10_000))
        oof = pd.Series(index=X.index, dtype=float)
        for tr, te in cv.split(X, y_perm):
            model_ = GridSearchCV(model, grid, scoring="roc_auc", cv=3, n_jobs=-1)
            model_.fit(X.iloc[tr], y_perm[tr])
            best_est = model_.best_estimator_
            prob = best_est.predict_proba(X.iloc[te])[:, 1] if hasattr(best_est, "predict_proba") else best_est.decision_function(X.iloc[te])
            if prob.min() < 0 or prob.max() > 1:
                pmin, pmax = prob.min(), prob.max()
                if pmax > pmin:
                    prob = (prob - pmin) / (pmax - pmin)
            oof.iloc[te] = prob
        aucs.append(roc_auc_score(y, oof))
    df = pd.DataFrame({"perm": range(1, n_perm + 1), "auc": aucs})
    df.to_csv(os.path.join(outdir, "permutation_auc.csv"), index=False)
    return df


def save_learning_curve(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    out_png: str,
    out_csv: str,
    cv_splits: int = 5,
    random_state: int = 42
):
    """
    Compute and plot a learning curve (train sizes vs. CV score) using ROC-AUC.
    """
    model, grid = get_model_and_grid(model_name)
    # For learning curve we just pick a reasonable default model (no grid for speed):
    base_estimator = model
    if model_name.lower() in ["logit", "logistic", "elasticnet", "logit_en"]:
        base_estimator.set_params(clf__l1_ratio=0.5)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # Compatibility with older sklearn (no random_state / shuffle kwargs)
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            base_estimator, X, y, cv=cv, scoring="roc_auc",
            train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1, shuffle=True, random_state=random_state
        )
    except TypeError:
        # Fallback: no shuffle/random_state args
        train_sizes, train_scores, test_scores = learning_curve(
            base_estimator, X, y, cv=cv, scoring="roc_auc",
            train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1
        )

    df = pd.DataFrame({
        "train_size": train_sizes,
        "train_auc_mean": train_scores.mean(axis=1),
        "train_auc_std": train_scores.std(axis=1),
        "cv_auc_mean": test_scores.mean(axis=1),
        "cv_auc_std": test_scores.std(axis=1)
    })
    df.to_csv(out_csv, index=False)

    fig = plt.figure(figsize=(4.6, 3.6))
    ax = fig.add_subplot(111)
    ax.plot(train_sizes, df["cv_auc_mean"], lw=2, marker="o")
    ax.fill_between(train_sizes,
                    df["cv_auc_mean"] - df["cv_auc_std"],
                    df["cv_auc_mean"] + df["cv_auc_std"], alpha=0.25)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("CV ROC-AUC")
    _decorate(ax, "Learning curve", f"model={model_name} | folds={cv_splits}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
