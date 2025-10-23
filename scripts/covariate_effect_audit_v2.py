#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
covariate_effect_audit_v2.py

Drop-in audit for covariate impact WITHOUT needing X_* matrices.
It builds feature matrices directly from taxa_compare outputs (pct_*.csv)
and aligns them with metadata to run:

1) Ablation on ORIGINAL dataset (with metadata):
   - microbiome-only vs microbiome+covariates (nested CV, paired Wilcoxon).

2) Optional transfer fecal→oral using NEW HMP oral table (no metadata):
   - trainCV with ALL features (incl. covariates) vs AFTER intersection (emulating no covariates at test time).
   - final fit on train-after-intersection and predict on HMP.

Inputs you HAVE:
- taxa_dir: path like results/taxa_compare/species
  expects one of:
    * pct_all.csv
    OR
    * pct_oral_crohn.csv, pct_oral_healthy.csv, pct_fecal_crohn.csv, pct_fecal_healthy.csv
- meta_train_csv: model_table_pooled.csv (has 'sample_id', target, covariates incl. 'site')

Optional:
- X_hmp_csv: NEW oral HMP species table (taxa x samples OR samples x taxa). If absent/missing, transfer is skipped.

Author: DS for Life Sciences — microbiome ML audit
"""

import os
import json
import argparse
import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss
)

# ---------------------- small IO utils ----------------------

def ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _clean_taxon_name(n: str) -> str:
    """Normalize taxon names (drop rank prefixes, replace spaces)."""
    s = str(n).strip()
    for pref in ("s__","g__","k__","p__","c__","o__","f__"):
        if s.startswith(pref):
            s = s[len(pref):]
    return s.replace(" ", "_")

def _coerce_sid(df: pd.DataFrame, hint: str="") -> pd.DataFrame:
    """Ensure 'sample_id' exists; if absent, use index."""
    df = df.copy()
    if "sample_id" not in df.columns:
        df["sample_id"] = df.index.astype(str)
    df["sample_id"] = df["sample_id"].astype(str)
    # Drop duplicate sample_ids if any
    before = len(df)
    df = df.drop_duplicates(subset=["sample_id"])
    after = len(df)
    if after < before:
        warnings.warn(f"[{hint}] Dropped {before-after} duplicate sample_id rows.")
    return df

def _normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply taxon name normalization to all non-sample_id columns."""
    df = df.copy()
    if "sample_id" in df.columns:
        feats = [c for c in df.columns if c != "sample_id"]
        df = df[["sample_id"]].join(df[feats].set_axis([_clean_taxon_name(c) for c in feats], axis=1))
    else:
        df = df.set_axis([_clean_taxon_name(c) for c in df.columns], axis=1)
    return df

# ---------------------- loaders for taxa_compare ----------------------

def _read_pct_any(taxa_dir: str) -> pd.DataFrame:
    """
    Read taxa_compare percent tables and return a WIDE matrix:
      rows = samples, columns = taxa, with 'sample_id'.
    Strategy:
      1) If pct_all.csv exists:
           - If looks like taxa x samples, transpose.
           - Else assume already samples x taxa.
      2) Else, if split files exist (pct_oral_crohn.csv, ...):
           - Read each, stack columns, align.
    """
    path_all = os.path.join(taxa_dir, "pct_all.csv")
    split_files = ["pct_oral_crohn.csv","pct_oral_healthy.csv","pct_fecal_crohn.csv","pct_fecal_healthy.csv"]
    split_paths = [os.path.join(taxa_dir, f) for f in split_files]

    def _make_wide(df: pd.DataFrame) -> pd.DataFrame:
        # Heuristic: if first column looks like taxon names, assume taxa x samples -> transpose
        first_col = df.columns[0]
        looks_taxa = df[first_col].astype(str).str.startswith(("s__","g__","k__","p__","c__","o__","f__")).mean() > 0.2
        if looks_taxa:
            taxa = df[first_col].astype(str).apply(_clean_taxon_name)
            data = df.drop(columns=[first_col])
            data.columns = data.columns.astype(str)
            wide = data.T
            wide.columns = taxa
            wide.index.name = "sample_id"
            wide = wide.reset_index()
        else:
            # Assume rows are samples already
            wide = _coerce_sid(df, hint="pct_any")
        wide = _normalize_feature_columns(wide)
        num_cols = [c for c in wide.columns if c != "sample_id"]
        wide[num_cols] = wide[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return wide

    if os.path.exists(path_all):
        df = pd.read_csv(path_all)
        return _make_wide(df)

    # else try split
    present = [p for p in split_paths if os.path.exists(p)]
    if not present:
        raise FileNotFoundError(f"No pct_all.csv or split pct files found in {taxa_dir}")

    wides = []
    for p in present:
        df = pd.read_csv(p)
        w = _make_wide(df)
        wides.append(w)

    # outer-join on sample_id, union of taxa columns
    from functools import reduce
    wide = reduce(lambda a,b: pd.merge(a,b, on="sample_id", how="outer"), wides)
    wide = wide.fillna(0.0)
    return wide

def _read_meta(meta_path: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)
    meta = _coerce_sid(meta, hint="meta")
    return meta

def _build_design_from_pct(
    taxa_dir: str,
    meta: pd.DataFrame,
    target: str,
    site_col: str = "site",
    oral_label: str = "oral",
    fecal_label: str = "fecal"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build two designs (oral and fecal) from pct tables + metadata.
    Returns:
      X_oral (samples x taxa), y_oral (binary),
      X_fecal (samples x taxa), y_fecal (binary).
    """
    wide = _read_pct_any(taxa_dir)  # samples x taxa + sample_id
    # Align with meta to filter by site
    if site_col not in meta.columns:
        raise ValueError(f"'{site_col}' column not found in metadata. Provide --site_col if named differently.")

    # Make normalized site labels
    meta["_site_norm"] = meta[site_col].astype(str).str.lower().str.strip()
    oral_mask  = meta["_site_norm"].isin([oral_label.lower()])
    fecal_mask = meta["_site_norm"].isin([fecal_label.lower()])

    # Join wide with meta to get target and site
    joined = wide.merge(meta, on="sample_id", how="inner")
    if target not in joined.columns:
        raise ValueError(f"Target '{target}' not found in metadata.")

    # Split oral/fecal
    oral_df  = joined[joined["_site_norm"] == oral_label.lower()].copy()
    fecal_df = joined[joined["_site_norm"] == fecal_label.lower()].copy()

    # X matrices (drop metadata columns; keep taxa only)
    meta_cols = set(meta.columns.tolist() + ["_site_norm"])
    taxa_cols_oral  = [c for c in oral_df.columns  if c not in meta_cols and c != "sample_id"]
    taxa_cols_fecal = [c for c in fecal_df.columns if c not in meta_cols and c != "sample_id"]

    X_oral  = oral_df[["sample_id"] + taxa_cols_oral].copy()
    y_oral  = (oral_df[target].astype(str).str.lower().isin(["1","true","yes","case","crohn","ppi","responder"])).astype(int)

    X_fecal = fecal_df[["sample_id"] + taxa_cols_fecal].copy()
    y_fecal = (fecal_df[target].astype(str).str.lower().isin(["1","true","yes","case","crohn","ppi","responder"])).astype(int)

    # Basic sanity
    if X_oral.shape[0] == 0 or X_fecal.shape[0] == 0:
        warnings.warn("One of oral/fecal sets is empty after filtering. Check site labels via --site_col/--oral_label/--fecal_label.")
    return X_oral, y_oral, X_fecal, y_fecal

# ---------------------- HMP loader (optional) ----------------------

def _read_hmp_oral(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # detect taxa-first
    first_col = df.columns[0]
    looks_taxa = df[first_col].astype(str).str.startswith(("s__","g__","k__","p__","c__","o__","f__")).mean() > 0.2
    if looks_taxa:
        taxa = df[first_col].astype(str).apply(_clean_taxon_name)
        data = df.drop(columns=[first_col])
        data.columns = data.columns.astype(str)
        wide = data.T
        wide.columns = taxa
        wide.index.name = "sample_id"
        wide = wide.reset_index()
    else:
        wide = _coerce_sid(df, hint="HMP")
    wide = _normalize_feature_columns(wide)
    num_cols = [c for c in wide.columns if c != "sample_id"]
    wide[num_cols] = wide[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return wide

def _intersect_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    train_cols = [c for c in X_train.columns if c != "sample_id"]
    test_cols  = [c for c in X_test.columns  if c != "sample_id"]
    keep = sorted(list(set(train_cols).intersection(test_cols)))
    drop_train = sorted(list(set(train_cols) - set(keep)))
    drop_test  = sorted(list(set(test_cols)  - set(keep)))
    Xt = X_train[["sample_id"] + keep].copy()
    Xs = X_test[["sample_id"] + keep].copy()
    return Xt, Xs, keep, drop_train, drop_test

# ---------------------- modeling ----------------------

from sklearn.utils import check_random_state

def _logit_pipeline():
    # StandardScaler + LogisticRegression (elasticnet)
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty="elasticnet", l1_ratio=0.5, C=1.0,
            solver="saga", max_iter=5000, n_jobs=1, random_state=42))
    ])

def _grid():
    return {
        "clf__l1_ratio": [0.2, 0.5, 0.8],
        "clf__C": [0.1, 0.5, 1.0, 2.0]
    }

def nested_cv_metrics(X: np.ndarray, y: np.ndarray, n_outer=5, n_inner=3, n_jobs=1) -> pd.DataFrame:
    rng = check_random_state(42)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import GridSearchCV

    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=1337)

    rows = []
    for k, (tr, va) in enumerate(outer.split(X, y), start=1):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        gs = GridSearchCV(
            estimator=_logit_pipeline(),
            param_grid=_grid(),
            scoring="roc_auc",
            cv=inner, refit=True, n_jobs=n_jobs
        )
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_
        p = best.predict_proba(Xva)[:,1]
        roc_auc = roc_auc_score(yva, p)
        pr_auc  = average_precision_score(yva, p)
        brier   = brier_score_loss(yva, p)
        # simple calibration slope proxy
        eps = 1e-9
        logit_p = np.log(np.clip(p,eps,1-eps) / np.clip(1-p,eps,1-eps))
        beta = np.polyfit(logit_p, yva, 1)[0]
        rows.append({"fold":k,"roc_auc":roc_auc,"pr_auc":pr_auc,"brier":brier,"calib_slope_lin":beta,
                     "best_params": gs.best_params_})
    return pd.DataFrame(rows)

def fit_on_full_and_eval(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: Optional[np.ndarray]=None) -> Dict:
    gs = GridSearchCV(
        estimator=_logit_pipeline(),
        param_grid=_grid(),
        scoring="roc_auc", cv=3, refit=True, n_jobs=1
    )
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    p = best.predict_proba(Xte)[:,1]
    out = {"best_params": gs.best_params_, "y_pred_proba": p.tolist()}
    if yte is not None:
        out["roc_auc"] = float(roc_auc_score(yte, p))
        out["pr_auc"]  = float(average_precision_score(yte, p))
        out["brier"]   = float(brier_score_loss(yte, p))
        eps = 1e-9
        logit_p = np.log(np.clip(p,eps,1-eps) / np.clip(1-p,eps,1-eps))
        out["calib_slope_lin"] = float(np.polyfit(logit_p, yte, 1)[0])
    return out

# ---------------------- main workflow ----------------------

def run_ablation(X_df: pd.DataFrame, y_ser: pd.Series, outdir: str, n_outer: int, n_inner: int, n_jobs: int):
    """Run microbiome-only vs microbiome+covariates ablation on a given design (X_df has taxa + any covariates)."""
    ensure_dir(outdir)
    # Split taxa vs covariates
    taxa_cols = [c for c in X_df.columns if c not in ("sample_id",)]
    # Heuristic: any column present in metadata (non-taxa) should be excluded here; since we built X from pct,
    # there are no covariates mixed in yet. So microbe-only == all columns.
    D_micro = X_df.drop(columns=["sample_id"], errors="ignore").to_numpy(dtype=float)
    y = y_ser.astype(int).to_numpy()

    # microbe-only
    cvA = nested_cv_metrics(D_micro, y, n_outer=n_outer, n_inner=n_inner, n_jobs=n_jobs)
    cvA.to_csv(os.path.join(outdir, "ablation_microbe_only_cv.csv"), index=False)

    # microbe+covariates: add covars from metadata by merging on sample_id later.
    # In this v2 audit, we keep covariate augmentation at the "caller" level to control which covars to add.
    # We'll just return cvA here; cvB will be run by caller after augmenting X with covars.
    return cvA

def add_covariates(X_df: pd.DataFrame, meta: pd.DataFrame, covars: List[str]) -> pd.DataFrame:
    """Join selected covariates to X_df by sample_id; one-hot encode object cols."""
    out = X_df.merge(meta[["sample_id"] + [c for c in covars if c in meta.columns]], on="sample_id", how="left")
    # Simple impute and one-hot
    for c in covars:
        if c in out.columns:
            if out[c].dtype.kind in "biufc":
                out[c] = out[c].fillna(0.0)
            else:
                out[c] = out[c].astype(str).fillna("NA")
    obj_cols = [c for c in covars if c in out.columns and out[c].dtype == "object"]
    if obj_cols:
        out = pd.get_dummies(out, columns=obj_cols, dummy_na=True)
    return out

def main():
    ap = argparse.ArgumentParser(description="Covariate effect audit from taxa_compare pct tables (+ optional HMP transfer).")
    ap.add_argument("--taxa_dir", required=True, help="e.g., results/taxa_compare/species")
    ap.add_argument("--meta_train_csv", required=True, help="e.g., data/meta/model_table_pooled.csv")
    ap.add_argument("--target", required=True, help="target column in metadata (e.g., disease)")
    ap.add_argument("--add_covars", default="age,sex,BMI,antibiotics,PPI_use,site", help="comma-separated covariate names")
    ap.add_argument("--site_col", default="site", help="name of site column in metadata")
    ap.add_argument("--oral_label", default="oral", help="label used in metadata for oral site")
    ap.add_argument("--fecal_label", default="fecal", help="label used in metadata for fecal site")
    ap.add_argument("--X_hmp_csv", default="", help="optional path to NEW HMP oral table; if missing, transfer is skipped")
    ap.add_argument("--outdir", default="results/covariate_audit_v2")
    ap.add_argument("--n_outer", type=int, default=5)
    ap.add_argument("--n_inner", type=int, default=3)
    ap.add_argument("--n_jobs", type=int, default=4)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Read meta and build oral/fecal designs from pct tables
    meta = _read_meta(args.meta_train_csv)
    X_oral, y_oral, X_fecal, y_fecal = _build_design_from_pct(
        taxa_dir=args.taxa_dir, meta=meta, target=args.target,
        site_col=args.site_col, oral_label=args.oral_label, fecal_label=args.fecal_label
    )

    covars = [c.strip() for c in args.add_covars.split(",") if c.strip()]
    # ---------------- within_oral ----------------
    out_oral = os.path.join(args.outdir, "within_oral")
    ensure_dir(out_oral)

    # microbe-only
    cvA_oral = run_ablation(X_oral, y_oral, os.path.join(out_oral, "microbe_only"), args.n_outer, args.n_inner, args.n_jobs)

    # microbe + covariates
    X_oral_plus = add_covariates(X_oral, meta, covars)
    D_oral_plus = X_oral_plus.drop(columns=["sample_id"], errors="ignore").to_numpy(dtype=float)
    y_or = y_oral.astype(int).to_numpy()
    cvB_oral = nested_cv_metrics(D_oral_plus, y_or, n_outer=args.n_outer, n_inner=args.n_inner, n_jobs=args.n_jobs)
    cvB_oral.to_csv(os.path.join(out_oral, "microbe_plus_covars_cv.csv"), index=False)

    # Wilcoxon (paired by fold)
    merged_oral = cvA_oral.merge(cvB_oral, on="fold", suffixes=("_A","_B"))
    p_oral = {}
    for m in ["roc_auc","pr_auc","brier","calib_slope_lin"]:
        try:
            stat, p = wilcoxon(merged_oral[f"{m}_A"], merged_oral[f"{m}_B"])
            p_oral[m] = p
        except Exception as e:
            p_oral[m] = None
            warnings.warn(f"[within_oral] Wilcoxon failed for {m}: {e}")

    # ---------------- within_fecal ----------------
    out_fecal = os.path.join(args.outdir, "within_fecal")
    ensure_dir(out_fecal)

    cvA_fec = run_ablation(X_fecal, y_fecal, os.path.join(out_fecal, "microbe_only"), args.n_outer, args.n_inner, args.n_jobs)

    X_fecal_plus = add_covariates(X_fecal, meta, covars)
    D_fec_plus = X_fecal_plus.drop(columns=["sample_id"], errors="ignore").to_numpy(dtype=float)
    y_fe = y_fecal.astype(int).to_numpy()
    cvB_fec = nested_cv_metrics(D_fec_plus, y_fe, n_outer=args.n_outer, n_inner=args.n_inner, n_jobs=args.n_jobs)
    cvB_fec.to_csv(os.path.join(out_fecal, "microbe_plus_covars_cv.csv"), index=False)

    merged_fec = cvA_fec.merge(cvB_fec, on="fold", suffixes=("_A","_B"))
    p_fec = {}
    for m in ["roc_auc","pr_auc","brier","calib_slope_lin"]:
        try:
            stat, p = wilcoxon(merged_fec[f"{m}_A"], merged_fec[f"{m}_B"])
            p_fec[m] = p
        except Exception as e:
            p_fec[m] = None
            warnings.warn(f"[within_fecal] Wilcoxon failed for {m}: {e}")

    # ---------------- transfer f2o → HMP (optional) ----------------
    transfer_report = {"skipped": True}
    if args.X_hmp_csv and os.path.exists(args.X_hmp_csv):
        transfer_report["skipped"] = False
        X_train_all = add_covariates(X_fecal, meta, covars)  # fecal train with covars
        D_train_all = X_train_all.drop(columns=["sample_id"], errors="ignore").to_numpy(dtype=float)
        y_train = y_fecal.astype(int).to_numpy()

        # TrainCV with ALL features (best-case with covariates)
        out_tr_all = os.path.join(args.outdir, "transfer_f2o", "trainCV_with_all_features")
        ensure_dir(out_tr_all)
        cv_all = nested_cv_metrics(D_train_all, y_train, n_outer=args.n_outer, n_inner=args.n_inner, n_jobs=args.n_jobs)
        cv_all.to_csv(os.path.join(out_tr_all, "cv_metrics.csv"), index=False)

        # Load HMP and intersect features (covariates will be dropped if absent on HMP)
        X_hmp = _read_hmp_oral(args.X_hmp_csv)
        Xtr2, Xh2, keep, drop_tr, drop_te = _intersect_features(X_train_all, X_hmp)
        with open(os.path.join(args.outdir, "transfer_f2o", "feature_alignment_report.json"), "w", encoding="utf-8") as f:
            json.dump({"kept_features": keep, "dropped_from_train": drop_tr, "dropped_from_test": drop_te}, f, ensure_ascii=False, indent=2)

        D_train_inter = Xtr2.drop(columns=["sample_id"], errors="ignore").to_numpy(dtype=float)
        D_test_inter  = Xh2.drop(columns=["sample_id"],  errors="ignore").to_numpy(dtype=float)

        # TrainCV AFTER intersection (realistic no-covariates at test time)
        out_tr_inter = os.path.join(args.outdir, "transfer_f2o", "trainCV_after_intersection")
        ensure_dir(out_tr_inter)
        cv_inter = nested_cv_metrics(D_train_inter, y_train, n_outer=args.n_outer, n_inner=args.n_inner, n_jobs=args.n_jobs)
        cv_inter.to_csv(os.path.join(out_tr_inter, "cv_metrics.csv"), index=False)

        # Final fit on full train-after-intersection and predict on HMP
        final = fit_on_full_and_eval(D_train_inter, y_train, D_test_inter, yte=None)
        with open(os.path.join(args.outdir, "transfer_f2o", "test_on_HMP_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)

        transfer_report.update({
            "mean_roc_auc_trainCV_all": float(cv_all["roc_auc"].mean()),
            "mean_roc_auc_trainCV_inter": float(cv_inter["roc_auc"].mean()),
            "n_kept_features": len(keep),
            "n_drop_train": len(drop_tr),
            "n_drop_test": len(drop_te)
        })

    # ---------------- summary ----------------
    summary = []
    summary.append("# within_oral (microbiome-only vs +covariates)")
    summary.append(f"- AUC microbe-only: {cvA_oral['roc_auc'].mean():.3f} ± {cvA_oral['roc_auc'].std():.3f}")
    summary.append(f"- AUC microbe+covs: {cvB_oral['roc_auc'].mean():.3f} ± {cvB_oral['roc_auc'].std():.3f}")
    summary.append(f"- Wilcoxon p-values (B vs A): {p_oral}")

    summary.append("\n# within_fecal (microbiome-only vs +covariates)")
    summary.append(f"- AUC microbe-only: {cvA_fec['roc_auc'].mean():.3f} ± {cvA_fec['roc_auc'].std():.3f}")
    summary.append(f"- AUC microbe+covs: {cvB_fec['roc_auc'].mean():.3f} ± {cvB_fec['roc_auc'].std():.3f}")
    summary.append(f"- Wilcoxon p-values (B vs A): {p_fec}")

    if not transfer_report.get("skipped", True):
        summary.append("\n# transfer f→o (TRAIN with ALL vs AFTER intersection)")
        summary.append(f"- mean AUC (ALL features): {transfer_report['mean_roc_auc_trainCV_all']:.3f}")
        summary.append(f"- mean AUC (AFTER intersection): {transfer_report['mean_roc_auc_trainCV_inter']:.3f}")
        summary.append("- See transfer_f2o/* for details and feature_alignment_report.json")
    else:
        summary.append("\n# transfer f→o")
        summary.append("- skipped (no --X_hmp_csv provided or file not found)")

    out_summary = os.path.join(args.outdir, "summary.txt")
    ensure_dir(args.outdir)
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print("\n==== SUMMARY ====\n")
    print("\n".join(summary))
    print("\nArtifacts at:", args.outdir)

if __name__ == "__main__":
    main()
