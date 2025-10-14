#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train binary classifiers with nested cross-validation on microbiome features.

This script supports three input modes:
  1) Single-table training: --X + --y (optionally --subset-site Oral/Fecal)
  2) Paired delta build:    --make-delta  (needs --X_oral, --X_fecal, --pairs, --y)
  3) Paired fusion build:   --make-fusion (needs --X_oral, --X_fecal, --pairs, --y)

For paired modes we materialize aligned tables and then train on them:
  - Delta:   X_delta = X_fecal - X_oral  (features matched by name)
  - Fusion:  X_fusion = [X_fecal | X_oral] (column-wise concatenation with site prefixes)

Outputs (written to --outdir):
  - cv_metrics.csv           : per-fold metrics (AUC, AUPR, ACC, F1, Brier) + best params
  - oof_predictions.csv      : out-of-fold probabilities per Sample_ID
  - oof_roc.png, oof_pr.png  : ROC and PR curves from OOF probabilities
  - top_features.csv         : top-50 features by model importance/coeffs (if available)
  - shap_summary.png         : SHAP summary (best-effort; skipped if unavailable)
  - model_card.json          : run config + best params from full refit

Notes:
  - We require a column named 'Sample_ID' (string). If absent, we try to infer/rename the
    first column if it looks like an index/ID; otherwise we synthesize one (not ideal).
  - In paired builds we always emit a canonical label column named 'label';
    we then force args.label_col to 'label' for training.
  - If a class has fewer than n_splits samples, nested CV will fail. We check and error early.
"""

from __future__ import annotations

import os
import json
import argparse
import warnings
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    brier_score_loss, roc_curve, precision_recall_curve
)

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

# Optional: SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Optional: XGBoost (falls back to RF if missing)
try:
    from xgboost import XGBClassifier  # type: ignore
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------
# Site name normalization
# ---------------------------------------
SITE_SYNONYMS: Dict[str, str] = {
    "oral": "Oral",
    "mouth": "Oral",
    "fecal": "Fecal",
    "faecal": "Fecal",
    "stool": "Fecal",
}


# ---------------------------------------
# Basic I/O helpers
# ---------------------------------------
def _read_csv_with_id(path: str) -> pd.DataFrame:
    """
    Read CSV and ensure a column named 'Sample_ID' exists (as string, not index).
    If the first column looks like an index/ID ('Unnamed: 0', 'id', etc.), rename it.
    As a last resort, synthesize Sample_IDs from row numbers (string).
    """
    df = pd.read_csv(path)

    # Bring index back if it's named Sample_ID
    if "Sample_ID" not in df.columns and getattr(df.index, "name", None) == "Sample_ID":
        df = df.reset_index()

    if "Sample_ID" not in df.columns:
        first = df.columns[0]
        first_l = str(first).strip().lower()
        if first_l in {"sample_id", "id", "sample", "subject", "unnamed: 0", "index"}:
            df = df.rename(columns={first: "Sample_ID"})
        else:
            # Synthesize IDs (not ideal, but keeps pipeline running)
            df.insert(0, "Sample_ID", np.arange(len(df)).astype(str))

    df["Sample_ID"] = df["Sample_ID"].astype(str)
    return df


def _normalize_site_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a site/body_site/location column exists, normalize to {Oral, Fecal}.
    Non-matching values become NaN.
    """
    candidates = [c for c in df.columns if c.lower() in {"site", "body_site", "location"}]
    if candidates:
        sname = candidates[0]
        site = (
            df[sname].astype(str).str.lower().str.strip()
            .map(lambda x: SITE_SYNONYMS.get(x, x))
        )
        site = site.where(site.isin(["Oral", "Fecal"]), other=np.nan)
        df = df.drop(columns=[sname]).assign(site=site)
    return df


# ---------------------------------------
# Data assembly
# ---------------------------------------
def load_xy(
    X_csv: str,
    y_csv: str,
    label_col: str | None,
    subset_site: str | None
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], pd.DataFrame]:
    """
    Load and align X and y on Sample_ID. Optionally filter y to a specific site.

    Parameters
    ----------
    X_csv : path to feature matrix (must include Sample_ID)
    y_csv : path to design table
    label_col : name of the label column in y (or None to expect canonical 'label')
    subset_site : 'Oral', 'Fecal', or None

    Returns
    -------
    X : (n, p) float array
    y : (n,) int array (0/1)
    featnames : feature column names
    ids : Sample_ID strings (row order)
    df_full : merged DataFrame before extracting arrays (for downstream auditing)
    """
    Xdf = _read_csv_with_id(X_csv)
    ydf = _read_csv_with_id(y_csv)
    ydf = _normalize_site_column(ydf)

    # Locate / normalize label column
    lower_map = {c.lower(): c for c in ydf.columns}
    if label_col is None:
        if "label" not in ydf.columns:
            raise ValueError(f"[load_xy] No label_col specified and 'label' not found in {y_csv}. "
                             f"Got: {list(ydf.columns)}")
    else:
        # If the requested label exists and isn't already named 'label', rename it
        if label_col.lower() in lower_map and lower_map[label_col.lower()] != "label":
            ydf = ydf.rename(columns={lower_map[label_col.lower()]: "label"})
        elif "label" not in ydf.columns:
            raise ValueError(f"[load_xy] '{label_col}' not in {y_csv}. Got: {list(ydf.columns)}")

    # Keep only what's needed from y
    keep_cols = ["Sample_ID", "label"] + (["site"] if "site" in ydf.columns else [])
    ydf = ydf[keep_cols].dropna(subset=["label"])

    # Optional site subset (for single-table training)
    if subset_site:
        want = SITE_SYNONYMS.get(subset_site.lower(), subset_site)
        if "site" not in ydf.columns:
            raise ValueError("[load_xy] --subset-site was set but no site column was found in y.")
        before = len(ydf)
        ydf = ydf[ydf["site"] == want]
        if len(ydf) == 0:
            raise ValueError(f"[load_xy] After site='{want}' filtering, zero rows remain (before={before}).")

    # 1:1 inner join on Sample_ID
    df = Xdf.merge(ydf, on="Sample_ID", how="inner", validate="one_to_one")
    if len(df) == 0:
        raise ValueError("[load_xy] Merge on Sample_ID produced 0 rows. "
                         "Ensure IDs actually match between X and y.")

    # Split to arrays
    meta_cols = {"Sample_ID", "label", "site"}
    featnames = [c for c in df.columns if c not in meta_cols]
    X = df[featnames].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    ids = df["Sample_ID"].astype(str).tolist()

    # Diagnostics
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"[load_xy] n={X.shape[0]} × p={X.shape[1]}; positives={n_pos}, negatives={n_neg}")

    return X, y, featnames, ids, df


# ---------------------------------------
# Paired builders
# ---------------------------------------
def _robust_pair_columns(pairs: pd.DataFrame) -> tuple[str, str]:
    """Resolve oral/fecal column names robustly."""
    lc = {c.lower(): c for c in pairs.columns}
    oral_col  = lc.get("oral_id")  or lc.get("oral")  or "Oral_ID"
    fecal_col = lc.get("fecal_id") or lc.get("fecal") or "Fecal_ID"
    if oral_col not in pairs.columns or fecal_col not in pairs.columns:
        raise ValueError(f"[pairs] Could not find oral/fecal columns in: {list(pairs.columns)}")
    return oral_col, fecal_col


def _align_by_pairs(Xo: pd.DataFrame, Xf: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """
    Align oral and fecal matrices by the pairs table order.
    Returns:
      oral_aligned (rows = pairs order),
      fecal_aligned (rows = pairs order),
      pairs_ok (same rows kept),
      oral_col name,
      fecal_col name
    """
    oral_col, fecal_col = _robust_pair_columns(pairs)
    pairs = pairs[[oral_col, fecal_col]].dropna().astype(str).reset_index(drop=True)

    # Ensure string IDs and set as index
    for df in (Xo, Xf):
        df["Sample_ID"] = df["Sample_ID"].astype(str)
    Xo = Xo.set_index("Sample_ID")
    Xf = Xf.set_index("Sample_ID")

    # Keep only intersecting features in identical order
    common = [c for c in Xf.columns if c in Xo.columns]
    if len(common) == 0:
        raise ValueError("[pairs] No intersecting feature columns between oral and fecal tables.")
    Xo = Xo[common]
    Xf = Xf[common]

    # Reindex by pairs order (indices are oral/fecal IDs)
    idx_oral  = pairs[oral_col].values
    idx_fecal = pairs[fecal_col].values
    oral_aligned  = Xo.reindex(idx_oral)
    fecal_aligned = Xf.reindex(idx_fecal)

    # Build a **positional** mask (NumPy) for rows where both sides are present
    ok_oral  = ~oral_aligned.isna().any(axis=1)
    ok_fecal = ~fecal_aligned.isna().any(axis=1)
    ok = (ok_oral.values) & (ok_fecal.values)  # <- positional mask

    # Apply mask positionally and reset row indexes
    oral_aligned  = oral_aligned.reset_index(drop=True).iloc[ok].reset_index(drop=True)
    fecal_aligned = fecal_aligned.reset_index(drop=True).iloc[ok].reset_index(drop=True)
    pairs_ok      = pairs.iloc[ok].reset_index(drop=True)

    # Helpful debug (counts)
    print(f"[pairs] input pairs={len(pairs)}, kept after alignment={len(pairs_ok)}")

    return oral_aligned, fecal_aligned, pairs_ok, oral_col, fecal_col

def build_delta_matrix(X_oral_csv: str, X_fecal_csv: str, y_csv: str, pairs_csv: str,
                       label_name: str = "disease", crohn_fecal_only: bool = True):
    Xo = _read_csv_with_id(X_oral_csv)
    Xf = _read_csv_with_id(X_fecal_csv)
    y  = _read_csv_with_id(y_csv)
    pairs = pd.read_csv(pairs_csv)

    oral_a, fecal_a, pairs_ok, oral_col, fecal_col = _align_by_pairs(Xo, Xf, pairs)

    # --- Optional Crohn-only restriction (keeps Fecal IDs with disease==1) ---
    if crohn_fecal_only and label_name.lower() == "disease":
        y = _normalize_site_column(y)
        y["Sample_ID"] = y["Sample_ID"].astype(str)
        pos_fecal = set(y[(y.get("site") == "Fecal") & (y["disease"] == 1)]["Sample_ID"].astype(str))
        keep = pairs_ok[fecal_col].isin(pos_fecal).values  # positional mask
        oral_a   = oral_a.iloc[keep].reset_index(drop=True)
        fecal_a  = fecal_a.iloc[keep].reset_index(drop=True)
        pairs_ok = pairs_ok.iloc[keep].reset_index(drop=True)
        print(f"[delta] crohn-only: kept {keep.sum()} / {len(keep)} pairs")

    # === EXACTLY HERE: attach fecal IDs and compute Δ ===
    fecal_ids = pairs_ok[fecal_col].astype(str).values
    Xd_df = pd.DataFrame(fecal_a.values - oral_a.values, columns=fecal_a.columns)  # Δ = fecal - oral
    Xd_df.insert(0, "Sample_ID", fecal_ids)

    # Build y aligned to these fecal Sample_IDs
    y = _normalize_site_column(y)
    lower_map = {c.lower(): c for c in y.columns}
    if label_name.lower() not in lower_map:
        raise ValueError(f"[delta] Label '{label_name}' not found in {y_csv}. Got: {list(y.columns)}")
    y = y.rename(columns={lower_map[label_name.lower()]: "label"})
    yd_df = y[y["Sample_ID"].astype(str).isin(Xd_df["Sample_ID"])][["Sample_ID", "label"] + (["site"] if "site" in y.columns else [])]
    yd_df = yd_df.reset_index(drop=True)

    return Xd_df, yd_df

def build_fusion_matrix(X_oral_csv: str, X_fecal_csv: str, y_csv: str, pairs_csv: str,
                        label_name: str = "ppi_use", crohn_fecal_only: bool = False):
    Xo = _read_csv_with_id(X_oral_csv)
    Xf = _read_csv_with_id(X_fecal_csv)
    y  = _read_csv_with_id(y_csv)
    pairs = pd.read_csv(pairs_csv)

    oral_a, fecal_a, pairs_ok, oral_col, fecal_col = _align_by_pairs(Xo, Xf, pairs)

    if crohn_fecal_only and label_name.lower() == "disease":
        y = _normalize_site_column(y)
        y["Sample_ID"] = y["Sample_ID"].astype(str)
        pos_fecal = set(y[(y.get("site") == "Fecal") & (y["disease"] == 1)]["Sample_ID"].astype(str))
        keep = pairs_ok[fecal_col].isin(pos_fecal).values
        oral_a   = oral_a.iloc[keep].reset_index(drop=True)
        fecal_a  = fecal_a.iloc[keep].reset_index(drop=True)
        pairs_ok = pairs_ok.iloc[keep].reset_index(drop=True)
        print(f"[fusion] crohn-only: kept {keep.sum()} / {len(keep)} pairs")

    # Fuse oral+fecal features with prefixes
    fecal_pref = {c: f"fecal::{c}" for c in fecal_a.columns}
    oral_pref  = {c: f"oral::{c}"  for c in oral_a.columns}
    fused = pd.concat([fecal_a.rename(columns=fecal_pref).reset_index(drop=True),
                       oral_a.rename(columns=oral_pref).reset_index(drop=True)], axis=1)

    # === EXACTLY HERE: attach fecal IDs as Sample_ID for fusion rows ===
    fecal_ids = pairs_ok[fecal_col].astype(str).values
    Xf_df = fused.copy()
    Xf_df.insert(0, "Sample_ID", fecal_ids)

    # Build y aligned to these fecal IDs
    y = _normalize_site_column(y)
    lower_map = {c.lower(): c for c in y.columns}
    if label_name.lower() not in lower_map:
        raise ValueError(f"[fusion] Label '{label_name}' not found in {y_csv}. Got: {list(y.columns)}")
    y = y.rename(columns={lower_map[label_name.lower()]: "label"})
    yf_df = y[y["Sample_ID"].astype(str).isin(Xf_df["Sample_ID"])][["Sample_ID", "label"] + (["site"] if "site" in y.columns else [])]
    yf_df = yf_df.reset_index(drop=True)

    return Xf_df, yf_df

# ---------------------------------------
# Models & utilities
# ---------------------------------------
def model_space(random_state: int = 42):
    """
    Candidate estimators and small grids for inner CV.
    Returns a dict: name -> (estimator_or_pipeline, param_grid)
    """
    space = {
        "logreg_l2": (
            Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(penalty="l2", max_iter=2000, random_state=random_state))
            ]),
            {"clf__C": [0.01, 0.1, 1, 3, 10]},
        ),
        "logreg_l1": (
            Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=random_state))
            ]),
            {"clf__C": [0.01, 0.1, 1, 3, 10]},
        ),
        "linsvm": (
            Pipeline([
                ("sc", StandardScaler()),
                ("clf", LinearSVC(random_state=random_state))
            ]),
            {"clf__C": [0.01, 0.1, 1, 3, 10]},
        ),
        "rf": (
            RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=random_state),
            {"max_depth": [None, 4, 8, 16], "min_samples_leaf": [1, 3, 5]},
        ),
    }

    if XGB_AVAILABLE:
        space["xgb"] = (
            XGBClassifier(
                n_estimators=500,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                learning_rate=0.05,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=random_state,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
            {"max_depth": [3, 4, 6], "reg_lambda": [0.1, 1, 5]},
        )
    return space


def _assert_cv_compat(y: np.ndarray, outer_folds: int):
    """Fail early if any class has fewer than outer_folds samples."""
    uniques, counts = np.unique(y, return_counts=True)
    if len(uniques) < 2:
        raise ValueError(f"[CV] Single-class labels (only {uniques.tolist()}) — cannot run CV.")
    if counts.min() < outer_folds:
        raise ValueError(f"[CV] The minority class has {counts.min()} samples, "
                         f"which is < outer_folds ({outer_folds}). Reduce folds or add data.")


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, out_base: str) -> None:
    """Persist ROC and PR plots."""
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], ls="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC = {auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_base + "_roc.png", dpi=220)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall (AP = {ap:.3f})")
    plt.tight_layout()
    plt.savefig(out_base + "_pr.png", dpi=220)
    plt.close()


def shap_summary(best_model, X: np.ndarray, featnames: List[str], out_png: str) -> None:
    """
    Try to render a SHAP summary. If unavailable/unsupported, skip without failing.
    """
    if not SHAP_AVAILABLE:
        print("[WARN] SHAP not available; skipping summary.")
        return

    try:
        name = type(best_model).__name__.lower()
        if hasattr(best_model, "predict_proba"):
            if ("xgb" in name) or ("forest" in name) or ("randomforest" in name):
                expl = shap.TreeExplainer(best_model)
            elif "logisticregression" in name or "linearsvc" in name:
                expl = shap.LinearExplainer(best_model, X)
            else:
                expl = shap.Explainer(best_model, X)
        else:
            print("[WARN] Estimator has no predict_proba; skipping SHAP.")
            return

        shap_values = expl.shap_values(X)
        plt.figure()
        shap.summary_plot(shap_values, features=X, feature_names=featnames, show=False)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] SHAP failed: {e}")


# ---------------------------------------
# Training orchestration
# ---------------------------------------
def train_and_report(
    X: np.ndarray,
    y: np.ndarray,
    featnames: List[str],
    ids: List[str],
    outdir: str,
    base_model,
    grid: dict,
    outer_folds: int,
    inner_folds: int,
    random_state: int
):
    """Run nested CV, write artifacts, refit on full data, and persist model card."""
    _assert_cv_compat(y, outer_folds)

    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    oof_prob = np.zeros_like(y, dtype=float)
    rows = []

    for k, (tr, te) in enumerate(outer.split(X, y), start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        gs = GridSearchCV(
            base_model, grid, scoring="roc_auc",
            cv=inner_folds, n_jobs=-1, refit=True
        )
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_

        # Probabilities (preferred); else scale decision function to [0,1]
        if hasattr(best, "predict_proba"):
            prob = best.predict_proba(Xte)[:, 1]
        elif hasattr(best, "decision_function"):
            s = best.decision_function(Xte)
            prob = (s - s.min()) / (s.max() - s.min() + 1e-12)
        else:
            prob = best.predict(Xte).astype(float)

        oof_prob[te] = prob

        rows.append({
            "fold": k,
            "auc":   roc_auc_score(yte, prob),
            "aupr":  average_precision_score(yte, prob),
            "acc":   accuracy_score(yte, (prob >= 0.5).astype(int)),
            "f1":    f1_score(yte, (prob >= 0.5).astype(int)),
            "brier": brier_score_loss(yte, prob),
            "best_params": gs.best_params_,
        })

    # Persist metrics
    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(outdir, "cv_metrics.csv"), index=False)
    print("[INFO] CV metrics:\n", mdf)

    # OOF predictions
    pd.DataFrame({"Sample_ID": ids, "y_true": y, "y_prob": oof_prob}).to_csv(
        os.path.join(outdir, "oof_predictions.csv"), index=False
    )

    # Curves
    plot_curves(y, oof_prob, os.path.join(outdir, "oof"))

    # Refit on full data for importances + SHAP
    final = GridSearchCV(base_model, grid, scoring="roc_auc", cv=inner_folds, n_jobs=-1, refit=True)
    final.fit(X, y)
    best_full = final.best_estimator_

    # Importances / coefficients
    try:
        if hasattr(best_full, "coef_"):
            imp = np.abs(best_full.coef_).ravel()
            imp_df = pd.DataFrame({"feature": featnames, "importance": imp}).sort_values("importance", ascending=False)
        elif hasattr(best_full, "feature_importances_"):
            imp = np.asarray(best_full.feature_importances_, dtype=float)
            imp_df = pd.DataFrame({"feature": featnames, "importance": imp}).sort_values("importance", ascending=False)
        else:
            imp_df = pd.DataFrame({"feature": featnames, "importance": np.nan})

        imp_df.head(50).to_csv(os.path.join(outdir, "top_features.csv"), index=False)
    except Exception as e:
        print(f"[WARN] Failed to write feature importances: {e}")

    # SHAP summary (best-effort)
    shap_summary(best_full, X, featnames, os.path.join(outdir, "shap_summary.png"))

    return final.best_params_


# ---------------------------------------
# CLI
# ---------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Nested-CV ML trainer for microbiome features")

    # Single-table mode
    ap.add_argument("--X", help="Feature matrix CSV (rows=Sample_ID)")
    ap.add_argument("--y", required=True, help="Design table CSV")
    ap.add_argument("--label-col", required=True,
                    help="Target column in y (e.g., 'disease' or 'ppi_use'). For paired builds this becomes 'label'.")
    ap.add_argument("--subset-site", default=None, choices=["Oral", "Fecal", "None"],
                    help="For single-table mode only. Use 'None' for no filtering.")

    # Output & CV
    ap.add_argument("--outdir", required=True, help="Output folder")
    ap.add_argument("--model", default="xgb", choices=list(model_space().keys()), help="Model family")
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)

    # Paired delta
    ap.add_argument("--make-delta", action="store_true", help="Build (Fecal−Oral) delta matrix before training")
    ap.add_argument("--X_oral",  help="Oral feature table for paired mode")
    ap.add_argument("--X_fecal", help="Fecal feature table for paired mode")
    ap.add_argument("--pairs",   help="matched_sample_ids.csv")
    ap.add_argument("--delta-label", choices=["disease", "ppi_use"], default="disease",
                    help="Label to use when building delta tables")
    ap.add_argument("--delta-crohn-only", action="store_true",
                    help="Restrict delta to fecal samples with disease==1")

    # Paired fusion
    ap.add_argument("--make-fusion", action="store_true", help="Build early-fusion matrix before training")
    ap.add_argument("--fusion-label", choices=["disease", "ppi_use"], default="disease",
                    help="Label to use when building fusion tables")
    ap.add_argument("--fusion-crohn-only", action="store_true",
                    help="Restrict fusion to fecal samples with disease==1")

    args = ap.parse_args()

    # Normalize subset-site
    if args.subset_site in ("None", "", None):
        args.subset_site = None

    # Ensure outdir
    os.makedirs(args.outdir, exist_ok=True)

    # If XGB not available but requested, fall back to RF
    space = model_space(random_state=args.random_state)
    model_name = args.model
    if model_name == "xgb" and not XGB_AVAILABLE:
        print("[WARN] XGBoost not installed; falling back to RandomForest.")
        model_name = "rf"
    base_model, grid = space[model_name]

    # Build paired tables if requested; force label_col='label' afterward
    if args.make_delta and args.make_fusion:
        raise SystemExit("Choose only one: --make-delta or --make-fusion.")

    if args.make_delta:
        if not (args.X_oral and args.X_fecal and args.pairs):
            raise SystemExit("[args] --make-delta requires --X_oral, --X_fecal, and --pairs.")
        Xd_df, yd_df = build_delta_matrix(
            args.X_oral, args.X_fecal, args.y, args.pairs,
            label_name=args.delta_label,
            crohn_fecal_only=args.delta_crohn_only
        )
        # Write into the model outdir so downstream rules can reuse deterministically
        X_out = os.path.join(args.outdir, "X_delta.csv")
        y_out = os.path.join(args.outdir, "y_delta.csv")
        Xd_df.to_csv(X_out, index=False)
        yd_df.to_csv(y_out, index=False)
        print(f"[delta] wrote {X_out} and {y_out}")

        args.X = X_out
        args.y = y_out
        args.label_col = "label"
        args.subset_site = None  # already fecal-aligned

    elif args.make_fusion:
        if not (args.X_oral and args.X_fecal and args.pairs):
            raise SystemExit("[args] --make-fusion requires --X_oral, --X_fecal, and --pairs.")
        Xf_df, yf_df = build_fusion_matrix(
            args.X_oral, args.X_fecal, args.y, args.pairs,
            label_name=args.fusion_label,
            crohn_fecal_only=args.fusion_crohn_only
        )
        X_out = os.path.join(args.outdir, "X_fusion.csv")
        y_out = os.path.join(args.outdir, "y_fusion.csv")
        Xf_df.to_csv(X_out, index=False)
        yf_df.to_csv(y_out, index=False)
        print(f"[fusion] wrote {X_out} and {y_out}")

        args.X = X_out
        args.y = y_out
        args.label_col = "label"
        args.subset_site = None  # already aligned

    # Load and train
    X, y, featnames, ids, _df_full = load_xy(args.X, args.y, args.label_col, args.subset_site)
    best_params_full = train_and_report(
        X=X, y=y, featnames=featnames, ids=ids,
        outdir=args.outdir,
        base_model=base_model, grid=grid,
        outer_folds=args.outer_folds, inner_folds=args.inner_folds,
        random_state=args.random_state
    )

    # Persist model card
    with open(os.path.join(args.outdir, "model_card.json"), "w") as f:
        json.dump({
            "X": args.X,
            "y": args.y,
            "label_col": args.label_col,
            "subset_site": args.subset_site,
            "model": model_name,
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "random_state": args.random_state,
            "best_params_full_fit": best_params_full
        }, f, indent=2)

    print("[INFO] Saved outputs →", args.outdir)


if __name__ == "__main__":
    main()
