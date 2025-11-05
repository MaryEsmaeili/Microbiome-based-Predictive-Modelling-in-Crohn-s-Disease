# ml/cv_utils.py
import os
import math
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, GridSearchCV, learning_curve
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, brier_score_loss, accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve

plt.switch_backend("Agg")  # Safe for headless environments (Snakemake)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# ---------------------------------------------------------------------
# Model builders + grids (core entry point: get_model_and_grid)
# ---------------------------------------------------------------------

def get_model_and_grid(model_name: str) -> Tuple[Pipeline, List[Dict[str, List[Any]]]]:
    """
    Return (pipeline, param_grid_list) for model_name in {"logit","svm","rf","xgb"}.

    - All models wrapped in a Pipeline with steps:
          prep -> clf
    - 'prep' does robust numeric impute + scaling on ALL columns.
      (برای میکروبیوم که CLR شده، این فقط نرمال‌سازی واحدهاست)
    """
    # numeric-only preprocessor
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # with_mean=False تا اگر بعداً sparse شد، خطا ندهد
        ("scale", StandardScaler(with_mean=False))
    ])

    preproc = ColumnTransformer(
        transformers=[("num", numeric, slice(0, None))],
        remainder="drop"
    )

    name = model_name.lower()

    if name == "logit":
        # Elastic-net / L2 logistic regression
        clf = LogisticRegression(
            solver="saga",
            max_iter=5000,
            n_jobs=-1
        )
        param_grid = [
            # Pure L2
            {
                "clf__penalty": ["l2"],
                "clf__C": [0.01, 0.1, 1.0, 10.0]
            },
            # Elastic-net
            {
                "clf__penalty": ["elasticnet"],
                "clf__l1_ratio": [0.1, 0.5, 0.9],
                "clf__C": [0.01, 0.1, 1.0, 10.0]
            }
        ]

    elif name == "svm":
        # Kernel SVM با احتمال
        clf = SVC(probability=True)
        param_grid = [
            {
                "clf__kernel": ["linear"],
                "clf__C": [0.01, 0.1, 1.0, 10.0]
            },
            {
                "clf__kernel": ["rbf"],
                "clf__C": [0.1, 1.0, 10.0],
                "clf__gamma": ["scale", "auto"]
            }
        ]

    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
        param_grid = [
            {
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_leaf": [1, 2, 5]
            }
        ]

    elif name == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("xgboost is not installed in this environment.")
        clf = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            max_depth=3,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            n_jobs=-1,
            eval_metric="logloss",
            random_state=42
        )
        param_grid = [
            {
                "clf__max_depth": [2, 3, 4],
                "clf__reg_lambda": [0.1, 1.0, 10.0],
                "clf__reg_alpha": [0.0, 0.5, 1.0]
            }
        ]

    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipe = Pipeline([
        ("prep", preproc),
        ("clf", clf)
    ])
    # همیشه لیست از dict می‌دهیم تا با GridSearchCV سازگار باشد
    if isinstance(param_grid, dict):
        param_grid = [param_grid]
    return pipe, param_grid


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def _final_estimator(est):
    """Extract final estimator from a sklearn Pipeline if present."""
    if hasattr(est, "named_steps"):
        return est.named_steps.get("clf", list(est.named_steps.values())[-1])
    return est


def _decorate(ax, title: str, subtitle: str = ""):
    if subtitle:
        ax.set_title(title + "\n" + subtitle, loc="left", fontsize=11)
    else:
        ax.set_title(title, loc="left", fontsize=11)
    ax.grid(True, alpha=0.25)


def plot_roc(y_true, y_prob, out_png: str, title: str, subtitle: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1)
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
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="quantile"
    )
    brier = brier_score_loss(y_true, y_prob)
    fig = plt.figure(figsize=(4.2, 3.6))
    ax = fig.add_subplot(111)
    ax.plot(prob_pred, prob_true, marker="o", lw=1.5, label=f"Brier = {brier:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1)
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


# ---------------------------------------------------------------------
# Nested CV
# ---------------------------------------------------------------------

def nested_cv_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    outdir: str,
    groups: Optional[pd.Series] = None,
    n_splits_outer: int = 5,
    n_splits_inner: int = 4,
    random_state: int = 42,
    min_pos_per_fold: int = 1
) -> Dict[str, Any]:
    """
    Perform nested CV with optional group-aware outer splits. Save figures and CSVs.

    Parameters
    ----------
    X, y        : features (samples x features), labels (0/1)
    model_name  : "logit" | "svm" | "rf" | "xgb"
    outdir      : output directory
    groups      : optional subject IDs for StratifiedGroupKFold
    min_pos_per_fold : minimum عدد positive در هر فولد (برای small-n)

    Returns
    -------
    dict with paths to metrics and plots.
    """
    os.makedirs(outdir, exist_ok=True)

    # Choose splitter
    if groups is not None and len(pd.Series(groups).unique()) > 1:
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

    # Storage
    fold_rows = []
    oof_prob = pd.Series(index=X.index, dtype=float)
    per_fold_coefs: List[pd.DataFrame] = []

    for k, (tr, te) in enumerate(
        outer.split(X, y, groups if use_groups else None),
        start=1
    ):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        # Skip folds with too few positives/negatives
        pos_tr = int(ytr.sum())
        neg_tr = len(ytr) - pos_tr
        pos_te = int(yte.sum())
        neg_te = len(yte) - pos_te

        if (
            ytr.nunique() < 2 or yte.nunique() < 2 or
            min(pos_tr, neg_tr) < min_pos_per_fold or
            min(pos_te, neg_te) < 1
        ):
            # این فولدها خیلی نامتعادل هستند؛ رد می‌کنیم
            continue

        inner = StratifiedKFold(
            n_splits=n_splits_inner,
            shuffle=True,
            random_state=random_state + 13
        )
        gscv = GridSearchCV(
            model,
            grid,
            scoring="roc_auc",
            cv=inner,
            n_jobs=-1
        )
        gscv.fit(Xtr, ytr)

        best = gscv.best_estimator_

        # Probabilities
        if hasattr(best, "predict_proba"):
            prob = best.predict_proba(Xte)[:, 1]
        else:
            prob = best.decision_function(Xte)
            # Scale decision_function to [0,1]
            pmin, pmax = prob.min(), prob.max()
            if pmax > pmin:
                prob = (prob - pmin) / (pmax - pmin)

        # Store OOF
        oof_prob.iloc[te] = prob

        # Threshold 0.5 predictions
        y_pred = (prob >= 0.5).astype(int)

        # Metrics
        row = {
            "fold": k,
            "roc_auc": roc_auc_score(yte, prob),
            "ap": average_precision_score(yte, prob),
            "acc": accuracy_score(yte, y_pred),
            "bal_acc": balanced_accuracy_score(yte, y_pred),
            "precision": precision_score(yte, y_pred, zero_division=0),
            "recall": recall_score(yte, y_pred, zero_division=0),
            "f1": f1_score(yte, y_pred, zero_division=0),
            "n_test": len(yte),
            "pos_test": int(yte.sum())
        }
        fold_rows.append(row)

        # Coefficients / importances per fold
        step = _final_estimator(best)
        weights = None
        feat_names = X.columns.to_list()
        if hasattr(step, "coef_"):
            coefs = step.coef_
            if getattr(coefs, "ndim", 1) == 2:
                coefs = coefs[0]
            weights = np.asarray(coefs).ravel()
        elif hasattr(step, "feature_importances_"):
            weights = np.asarray(step.feature_importances_).ravel()

        if weights is not None and len(weights) == len(feat_names):
            tmp = pd.DataFrame(
                {"feature": feat_names, "weight": weights, "fold": k}
            )
            per_fold_coefs.append(tmp)

    # Summaries
    metrics_df = pd.DataFrame(fold_rows)
    metrics_df.to_csv(os.path.join(outdir, "cv_metrics.csv"), index=False)

    # OOF plots
    y_prob = oof_prob.dropna()
    y_true = y.loc[y_prob.index]

    title = f"Nested CV — model={model_name}"
    subtitle = (
        f"n={len(y_true)} | pos={int(y_true.sum())} ({y_true.mean():.2%}) "
        f"| folds={n_splits_outer}x{n_splits_inner}"
    )

    plot_roc(y_true, y_prob, os.path.join(outdir, "roc.png"), title, subtitle)
    plot_pr(y_true, y_prob, os.path.join(outdir, "pr.png"), title, subtitle)
    brier = plot_calibration(
        y_true, y_prob,
        os.path.join(outdir, "calibration.png"),
        title, subtitle
    )

    # Feature stability across folds (برای logit / SVM / RF)
    coef_summary = pd.DataFrame()
    if per_fold_coefs:
        allc = pd.concat(per_fold_coefs, ignore_index=True)
        coef_summary = (allc
                        .groupby("feature")["weight"]
                        .agg(["mean", "std"])
                        .reset_index())
        coef_summary["abs_mean"] = coef_summary["mean"].abs()
        coef_summary.to_csv(
            os.path.join(outdir, "importances.csv"), index=False
        )
        plot_feature_importance_with_errorbars(
            coef_summary,
            os.path.join(outdir, "importances.png"),
            title="Top features (mean ± SD across folds)",
            subtitle=subtitle
        )

    return {
        "metrics_path": os.path.join(outdir, "cv_metrics.csv"),
        "roc_path": os.path.join(outdir, "roc.png"),
        "pr_path": os.path.join(outdir, "pr.png"),
        "calibration_path": os.path.join(outdir, "calibration.png"),
        "brier": brier,
        "importances_path": (
            os.path.join(outdir, "importances.csv")
            if not coef_summary.empty else ""
        )
    }


# ---------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------

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
    aucs: List[float] = []

    for i in range(1, n_perm + 1):
        # permute y (but keep X fixed)
        idx_perm = rng.permutation(len(y))
        y_perm = y.values[idx_perm]

        model, grid = get_model_and_grid(model_name)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng.randint(0, 1_000_000))
        oof = pd.Series(index=X.index, dtype=float)

        for tr, te in cv.split(X, y_perm):
            gscv = GridSearchCV(
                model, grid, scoring="roc_auc", cv=3, n_jobs=-1
            )
            gscv.fit(X.iloc[tr], y_perm[tr])
            best = gscv.best_estimator_

            if hasattr(best, "predict_proba"):
                prob = best.predict_proba(X.iloc[te])[:, 1]
            else:
                prob = best.decision_function(X.iloc[te])
                pmin, pmax = prob.min(), prob.max()
                if pmax > pmin:
                    prob = (prob - pmin) / (pmax - pmin)

            oof.iloc[te] = prob

        aucs.append(roc_auc_score(y, oof))

    df = pd.DataFrame({"perm": range(1, n_perm + 1), "auc": aucs})
    df.to_csv(os.path.join(outdir, "permutation_auc.csv"), index=False)
    return df


# ---------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------

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

    # For learning curve, use base estimator (no grid) for speed
    base_estimator = model

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )
    train_sizes, train_scores, test_scores = learning_curve(
        base_estimator, X, y,
        cv=cv,
        scoring="roc_auc",
        train_sizes=np.linspace(0.2, 1.0, 6),
        n_jobs=-1
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
    ax.fill_between(
        train_sizes,
        df["cv_auc_mean"] - df["cv_auc_std"],
        df["cv_auc_mean"] + df["cv_auc_std"],
        alpha=0.25
    )
    ax.set_xlabel("Training samples")
    ax.set_ylabel("CV ROC-AUC")
    _decorate(ax, "Learning curve", f"model={model_name} | folds={cv_splits}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
