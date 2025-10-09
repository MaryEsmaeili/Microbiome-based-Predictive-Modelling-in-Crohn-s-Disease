#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IBD-only ML (Crohn-only within-site) with:
- Clean ID handling and robust abundance reading
- CLR applied ONLY to abundance features (never to covariates)
- Prevalence filtering (global, dataset-level)
- Nested CV (outer RepeatedStratifiedKFold, inner StratifiedKFold grid-search)
- Probability calibration (Platt sigmoid or isotonic)
- Threshold tuning (fixed 0.5 / best F1 / best Youden)
- Confusion/ROC/PR plots, feature importances, differential-abundance (MWU + BH)
- Optional additional covariates (imputed via median) appended to X
- Optional IPTW (propensity) + SMD plot BEFORE/AFTER weighting
- Sensitivity runs over multiple prevalence thresholds (optional)
- Robust guards against NaN and zero-feature edge cases
"""

import os, re, math, argparse, warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, confusion_matrix, precision_recall_curve, roc_curve,
    accuracy_score
)
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(invalid="ignore", divide="ignore")

# --------------------------- FS helpers ---------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def safe_read_csv(p: Optional[str]) -> Optional[pd.DataFrame]:
    if p and os.path.exists(p):
        return pd.read_csv(p)
    return None

def write_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

# -------------------------- ID handling --------------------------

def normalize_id(x: str) -> str:
    """Normalize sample IDs: strip, drop trailing '.N', drop leading zeros if numeric, uppercase."""
    if pd.isna(x):
        return x
    x = str(x).strip()
    x = re.sub(r"\.\d+$", "", x)
    if re.fullmatch(r"\d+", x):
        x = re.sub(r"^0+", "", x) or "0"
    return x.upper()

def norm_index_like(cols: List[str]) -> List[str]:
    return [normalize_id(c) for c in cols]

# ------------------------ Abundance reader -----------------------

def read_abund(path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (M, features, samples) where M is features x samples non-negative matrix."""
    df = safe_read_csv(path)
    if df is None or df.shape[0] == 0:
        return None
    first = df.columns[0]
    if (pd.isna(first) or first == "" or first == "...1" or
        re.fullmatch(r"(?i)(feature|taxon|clade|species|id|name)", str(first) or "")):
        feats = df.iloc[:, 0].astype(str).values
        df = df.iloc[:, 1:].copy()
    else:
        cand = [c for c in df.columns if re.search(r"(?i)feature|taxon|clade|name|id", c)]
        if cand:
            feats = df[cand[0]].astype(str).values
            df = df.drop(columns=[cand[0]])
        else:
            feats = np.arange(1, df.shape[0] + 1).astype(str)

    # numeric coerce; replace negatives/NaNs with 0 (microbiome abundances must be >=0)
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce")
        v = v.fillna(0.0)
        v = v.clip(lower=0.0)
        df[c] = v.values

    M = df.to_numpy(dtype=float)
    keep = (M.sum(axis=1) > 0)
    M = M[keep, :]
    feats = np.array(pd.Series(feats)[keep].astype(str).values, dtype=object)
    samples = np.array(norm_index_like(df.columns.tolist()), dtype=object)
    return (M, feats, samples)

# ------------------------- CLR transform -------------------------

def clr_transform(X: np.ndarray, pseudo: float = 1e-6) -> np.ndarray:
    """
    CLR on samples x features. NaNs -> 0. Negatives -> 0. Add pseudo before log.
    """
    X = np.nan_to_num(X, nan=0.0, posinf=None, neginf=0.0)
    X = np.clip(X, a_min=0.0, a_max=None)
    X2 = X + float(pseudo)
    L = np.log(X2)
    gm = L.mean(axis=1, keepdims=True)
    return L - gm

# --------------------------- Plotting ----------------------------

def plot_roc(y, proba, title, out):
    ensure_dir(os.path.dirname(out))
    if len(np.unique(y)) < 2:
        return
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()

def plot_pr(y, proba, title, out):
    ensure_dir(os.path.dirname(out))
    if len(np.unique(y)) < 2:
        return
    prec, rec, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()

def plot_confusion(cm, title, out):
    ensure_dir(os.path.dirname(out))
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.xticks([0,1], ["Pred 0","Pred 1"])
    plt.yticks([0,1], ["True 0","True 1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.title(title); plt.colorbar()
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()

def plot_top(values, labels, title, out, top=20):
    ensure_dir(os.path.dirname(out))
    order = np.argsort(np.abs(values))[::-1][:top]
    v = np.array(values)[order]
    l = np.array(labels)[order]
    plt.figure(figsize=(6, max(3, 0.3*len(order))))
    y = np.arange(len(order))
    plt.barh(y, v)
    plt.yticks(y, l)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()

# ---------------------- Metrics & thresholds --------------------

def metrics_block(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    out = dict(
        threshold=float(thr),
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        specificity=tn/(tn+fp) if (tn+fp)>0 else np.nan,
        f1=f1_score(y, pred, zero_division=0),
        auroc=roc_auc_score(y, proba) if len(np.unique(y))==2 else np.nan,
        ap=average_precision_score(y, proba) if len(np.unique(y))==2 else np.nan,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )
    return out, pred

def tune_thresholds(y, proba):
    # best F1 on grid
    grid = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y, (proba>=t).astype(int), zero_division=0) for t in grid]
    best_f1_thr = float(grid[int(np.argmax(f1s))])
    # best Youden
    fpr, tpr, thr = roc_curve(y, proba)
    you = tpr - fpr
    best_you_thr = float(thr[int(np.argmax(you))])
    best_you_thr = min(max(best_you_thr, 0.0), 1.0)
    return best_f1_thr, best_you_thr

# ---------------------------- Modeling --------------------------

def make_calibrator(best_estimator, method="sigmoid", cv=3):
    """Version-safe wrapper for CalibratedClassifierCV."""
    try:
        return CalibratedClassifierCV(estimator=best_estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=best_estimator, method=method, cv=cv)

def get_model(kind: str, seed: int, for_importance: bool=False):
    """
    Return Pipeline and param_grid for GridSearchCV.
    - Linear models: Imputer -> Scaler -> LogisticRegression
    - RF: Imputer -> RandomForestClassifier
    """
    if kind == "elasticnet":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", class_weight="balanced",
                max_iter=8000, n_jobs=-1, random_state=seed
            ))
        ])
        grid = {"clf__C":[0.1, 1, 10], "clf__l1_ratio":[0.2, 0.5, 0.8]}
    elif kind == "logreg":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                penalty="l2", solver="liblinear", class_weight="balanced",
                max_iter=5000, random_state=seed
            ))
        ])
        grid = {"clf__C":[0.1, 1, 10]}
    elif kind == "rf":
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500 if for_importance else 400,
                class_weight="balanced_subsample",
                random_state=seed, n_jobs=-1
            ))
        ])
        grid = {"clf__max_depth":[None, 5, 10, 15],
                "clf__max_features":["sqrt", 0.3, 0.5]}
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return pipe, grid

def nested_cv_oof(
    X: np.ndarray, y: np.ndarray, seed: int, kind: str,
    inner_splits=3, outer_splits=5, outer_repeats=1, calib="sigmoid",
    sample_weight: Optional[np.ndarray]=None
):
    """Nested CV with GridSearch + calibration. Returns (oof_proba, mean_auc, splits_info, stability_dict)."""
    proba = np.zeros(len(y), dtype=float)
    aucs = []
    splits_info = []

    rskf = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=outer_repeats, random_state=seed
    )

    for fold_idx, (tr_idx, te_idx) in enumerate(rskf.split(X, y)):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        wtr = sample_weight[tr_idx] if (sample_weight is not None) else None

        pipe, grid = get_model(kind, seed)
        gs = GridSearchCV(
            pipe, grid,
            cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed),
            n_jobs=-1, scoring="roc_auc", refit=True, error_score="raise"
        )
        fit_params = {"clf__sample_weight": wtr} if (wtr is not None and kind in {"elasticnet","logreg"}) else {}
        gs.fit(Xtr, ytr, **fit_params)

        cal = make_calibrator(gs.best_estimator_, method=calib, cv=3)
        cal.fit(Xtr, ytr, **({"sample_weight": wtr} if wtr is not None else {}))

        p = cal.predict_proba(Xte)[:, 1]
        proba[te_idx] = p
        if len(np.unique(yte)) == 2 and np.isfinite(p).all():
            try:
                aucs.append(roc_auc_score(yte, p))
            except Exception:
                pass

        splits_info.append({"fold": fold_idx, "train_n": int(len(tr_idx)), "test_n": int(len(te_idx))})

    return proba, (np.mean(aucs) if aucs else float("nan")), splits_info, {}

# ---------------- Differential Abundance (MWU + BH) --------------

def diff_abundance(X, y, feature_names, pseudo=1e-6, is_clr=False) -> pd.DataFrame:
    rows = []
    ln2 = math.log(2.0)
    for j, f in enumerate(feature_names):
        x0 = X[y==0, j]
        x1 = X[y==1, j]
        # MWU p-value
        if x0.size == 0 or x1.size == 0:
            p = 1.0
        else:
            try:
                p = mannwhitneyu(x0, x1, alternative="two-sided").pvalue
            except Exception:
                p = 1.0
        # effect size
        if is_clr:
            m0 = float(np.nanmean(x0)) if x0.size else 0.0
            m1 = float(np.nanmean(x1)) if x1.size else 0.0
            lf = (m1 - m0) / ln2
        else:
            m0 = float(np.nanmean(x0)) if x0.size else 0.0
            m1 = float(np.nanmean(x1)) if x1.size else 0.0
            num = (m1 + pseudo)
            den = (m0 + pseudo)
            lf = math.log2(num / den) if (num > 0 and den > 0) else 0.0
        rows.append((f, p, lf))

    df = pd.DataFrame(rows, columns=["feature", "p", "log2FC"]).sort_values("p").reset_index(drop=True)
    m = df.shape[0]
    ranks = np.arange(1, m+1, dtype=float)
    q = (df["p"].values * m / ranks)
    q = np.minimum(q, 1.0)
    df["q"] = q
    return df

# --------------------------- IPTW + SMD --------------------------

def compute_iptw(md: pd.DataFrame, target: str, covars: List[str]) -> Optional[np.ndarray]:
    """
    Lightweight IPTW using logistic regression on provided covariates.
    Returns normalized weights aligned to md index; None if covariates missing or degenerate.
    """
    covars = [c for c in covars if c in md.columns]
    if len(covars) < 2:
        return None
    Z = md[covars].apply(pd.to_numeric, errors="coerce")
    Z = Z.fillna(Z.median())  # simple median imputation
    t = pd.to_numeric(md[target], errors="coerce")
    ok = (~t.isna()).values
    if ok.sum() < 10:
        return None
    try:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(Z.values[ok], t.values[ok].astype(int))
        ps = lr.predict_proba(Z.values)[:,1]
        ps = np.clip(ps, 1e-3, 1-1e-3)
        w = t.values/ps + (1-t.values)/(1-ps)
        w = np.where(np.isfinite(w), w, 1.0)
        w = w / np.mean(w)
        return w
    except Exception:
        return None

def _w_mean_var(x: np.ndarray, w: np.ndarray):
    """Weighted mean and variance with numerical guards."""
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    x = np.where(np.isfinite(x), x, np.nan)
    # impute NaN in x by overall median (unweighted) to keep simple
    if np.isnan(x).any():
        med = np.nanmedian(x)
        x = np.where(np.isnan(x), med, x)
    sw = w.sum()
    if sw <= 0:
        return float(np.nan), float(np.nan)
    m = (w * x).sum() / sw
    v = (w * (x - m) ** 2).sum() / sw
    return float(m), float(v)

def _smd_for_cov(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]=None) -> float:
    """
    Standardized Mean Difference between y==1 and y==0.
    If weights given, use weighted means/vars; otherwise unweighted.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    m1 = y == 1
    m0 = y == 0
    if w is None:
        x1, x0 = x[m1], x[m0]
        if x1.size < 2 or x0.size < 2:
            return np.nan
        m1_, v1_ = float(np.nanmean(x1)), float(np.nanvar(x1))
        m0_, v0_ = float(np.nanmean(x0)), float(np.nanvar(x0))
    else:
        w = np.asarray(w, dtype=float)
        m1_, v1_ = _w_mean_var(x[m1], w[m1])
        m0_, v0_ = _w_mean_var(x[m0], w[m0])
    denom = np.sqrt( (v1_ + v0_) / 2.0 ) if np.isfinite(v1_) and np.isfinite(v0_) else np.nan
    if not np.isfinite(denom) or denom == 0:
        return np.nan
    return (m1_ - m0_) / denom

def compute_smd_table(md: pd.DataFrame, target: str, covars: List[str], w: Optional[np.ndarray]) -> pd.DataFrame:
    """Return DataFrame with SMD before and after weighting for numeric covariates."""
    rows = []
    y = pd.to_numeric(md[target], errors="coerce").astype("Int64").astype(float).to_numpy()
    y = np.where(np.isnan(y), np.nan, y).astype(float)
    # keep only samples with non-NaN y
    mask = ~np.isnan(y)
    y = y[mask].astype(int)
    for c in covars:
        if c not in md.columns: 
            continue
        x = pd.to_numeric(md[c], errors="coerce").to_numpy()
        x = x[mask]
        smd_b = _smd_for_cov(x, y, w=None)
        smd_a = _smd_for_cov(x, y, w[mask]) if w is not None else np.nan
        rows.append({"covariate": c, "SMD_before": smd_b, "SMD_after": smd_a})
    df = pd.DataFrame(rows)
    return df

def plot_smd_before_after(df: pd.DataFrame, out_png: str, title: str="Covariate Balance (SMD)"):
    """
    Lollipop-style plot for SMD before/after weighting.
    If 'SMD_after' is NaN (no weights), we plot only 'before'.
    """
    ensure_dir(os.path.dirname(out_png))
    if df.empty:
        # write an empty placeholder figure
        plt.figure()
        plt.text(0.5, 0.5, "No covariates to plot", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
        return

    df = df.copy()
    df = df.sort_values("SMD_before", key=lambda s: np.abs(s.fillna(0.0)), ascending=False)

    y_ticks = np.arange(len(df))
    plt.figure(figsize=(7, max(3, 0.35*len(df))))
    # before
    plt.scatter(df["SMD_before"], y_ticks, label="Before IPTW")
    # after (if present)
    if df["SMD_after"].notna().any():
        plt.scatter(df["SMD_after"], y_ticks, label="After IPTW")
        for i, row in df.iterrows():
            if not np.isnan(row["SMD_after"]):
                plt.plot([row["SMD_before"], row["SMD_after"]], [np.where(df.index==i)[0][0]]*2, alpha=0.3)
    plt.axvline(0.1, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0.2, color="gray", linestyle="--", linewidth=1)
    plt.yticks(y_ticks, df["covariate"].tolist())
    plt.xlabel("Standardized Mean Difference")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220); plt.close()

# ---------------------- Build site dataset ----------------------

def prepare_site(
    site: str, abund_crohn_path: str, covars_path: str, target: str,
    min_prev: float, min_prev_samples: Optional[int], max_features: int,
    use_clr: bool, pseudo: float, outdir: str,
    add_covariates: List[str], propensity_covars: List[str], group_col: Optional[str]
):
    """
    Returns X (n_samples x n_features), y, feature_names, sample_ids, metadata_sub, iptw_weights, used_propensity_covars
    """
    # abundance
    r = read_abund(abund_crohn_path)
    if r is None:
        return None
    M_fxs, feats, samples = r  # features x samples

    # metadata
    md = safe_read_csv(covars_path)
    if md is None or md.shape[0] == 0:
        return None
    md = md.copy()
    if "Sample_ID" not in md.columns or "site" not in md.columns or "disease" not in md.columns:
        raise ValueError("Covariates must include Sample_ID, site, disease.")
    md["Sample_ID"] = md["Sample_ID"].map(normalize_id)
    md["site"] = md["site"].astype(str).str.lower()
    md["disease"] = pd.to_numeric(md["disease"], errors="coerce")

    # site + Crohn only this week
    md = md[(md["disease"] == 1) & (md["site"] == site)].copy()
    if target not in md.columns:
        raise ValueError(f"Target {target} not found in covariates.")
    md[target] = pd.to_numeric(md[target], errors="coerce")

    # intersect IDs
    common = np.intersect1d(samples, md["Sample_ID"].values)
    md = md.set_index("Sample_ID").loc[common]
    y = md[target].astype("Int64").astype(float).to_numpy()
    mask = ~np.isnan(y)
    y = y[mask].astype(int)
    sample_ids = common[mask]

    # align abundance columns to sample order; then transpose to samples x features
    col_idx = [np.where(samples == sid)[0][0] for sid in sample_ids]
    X_abund = M_fxs[:, col_idx].T  # samples x features
    feat_abund = feats.copy()

    # prevalence filter (global, before CV)
    if min_prev_samples is not None:
        keep = (X_abund > 0).sum(axis=0) >= int(min_prev_samples)
    else:
        keep = (X_abund > 0).sum(axis=0) >= int(np.ceil(min_prev * X_abund.shape[0]))
    if keep.sum() == 0:
        site_dir = os.path.join(outdir, site)
        ensure_dir(site_dir)
        write_csv(pd.DataFrame({"note":["no_feature_after_prevalence"]}), os.path.join(site_dir, "sanity_checks.csv"))
        return None
    X_abund = X_abund[:, keep]
    feat_abund = feat_abund[keep]

    # CLR only on abundance
    if use_clr:
        X_abund = clr_transform(X_abund, pseudo=pseudo)

    # Optional covariates → to append to X
    cov_names = []
    X_cov = None
    if add_covariates:
        present = [c for c in add_covariates if c in md.columns]
        missing = [c for c in add_covariates if c not in md.columns]
        if missing:
            print(f"[{site}] WARN: missing covariates skipped: {', '.join(missing)}")
        if present:
            Cov = md[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            col_med = np.nanmedian(Cov, axis=0)
            inds = np.where(np.isnan(Cov))
            Cov[inds] = col_med[inds[1]]
            X_cov = Cov[mask]  # align to y
            cov_names = present

    # combine abundance + covariates
    if X_cov is not None:
        X = np.hstack([X_abund[mask], X_cov])
        features = np.concatenate([feat_abund, np.array(cov_names, dtype=object)])
    else:
        X = X_abund[mask]
        features = feat_abund

    # optional IPTW
    iptw = None
    used_prop_covs = []
    if propensity_covars:
        covs_here = [c for c in propensity_covars if c in md.columns]
        used_prop_covs = covs_here.copy()
        if len(covs_here) < 2:
            print(f"[{site}] WARN: not enough propensity covariates; skip IPTW.")
        else:
            iptw_full = compute_iptw(md, target, covs_here)  # aligned to md index
            if iptw_full is None:
                print(f"[{site}] WARN: IPTW not computed; skip.")
            else:
                iptw = iptw_full[mask]  # align to y
                # normalize again (safety)
                iptw = iptw / np.mean(iptw)

    # grouping hint (not used in CV)
    if group_col and group_col not in md.columns:
        print(f"[{site}] WARN: group column '{group_col}' not found; ignoring grouping.")

    # dumps
    site_dir = os.path.join(outdir, site)
    ensure_dir(site_dir)
    write_csv(pd.DataFrame({"n_samples":[X.shape[0]], "n_features":[X.shape[1]],
                            "min_prev":[min_prev], "min_prev_samples":[min_prev_samples],
                            "use_clr":[use_clr]}),
              os.path.join(site_dir, "matrix_shape.csv"))
    cb = pd.DataFrame({"label":[0,1],
                       "count":[int(np.sum(y==0)), int(np.sum(y==1))]})
    write_csv(cb, os.path.join(site_dir, f"class_balance_{site}.csv"))

    # final NaN guard
    if np.isnan(X).any():
        col_med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = col_med[inds[1]]

    return X, y, features, sample_ids, md, iptw, used_prop_covs

# ------------------------------ Run site -------------------------

def run_site(site: str, abund_path: str, args):
    out_site = os.path.join(args.outdir, site)
    plots_dir = os.path.join(out_site, "plots")
    ensure_dir(out_site); ensure_dir(plots_dir)

    add_covariates = [c.strip() for c in (args.add_covariates or "").split(",") if c.strip()]
    propensity_covars = [c.strip() for c in (args.propensity_covars or "").split(",") if c.strip()]

    prep = prepare_site(
        site=site, abund_crohn_path=abund_path, covars_path=args.covariates, target=args.target,
        min_prev=args.min_prev, min_prev_samples=args.min_prev_samples, max_features=args.max_features,
        use_clr=args.use_clr, pseudo=args.pseudo, outdir=args.outdir,
        add_covariates=add_covariates, propensity_covars=propensity_covars, group_col=args.group_col
    )
    if prep is None:
        write_csv(pd.DataFrame({"note":["no_data_or_no_features"]}), os.path.join(out_site, "summary_onepager.csv"))
        return

    X, y, feat, sids, md, iptw, used_prop_covs = prep

    # --- SMD plot BEFORE/AFTER weighting (if possible) ---
    if used_prop_covs:
        try:
            smd_df = compute_smd_table(md, args.target, used_prop_covs, iptw)
            write_csv(smd_df, os.path.join(out_site, "smd_before_after.csv"))
            plot_smd_before_after(
                smd_df,
                out_png=os.path.join(out_site, "smd_before_after.png"),
                title=f"{site.title()} — Covariate Balance (SMD)"
            )
        except Exception as e:
            print(f"[{site}] WARN: SMD plotting failed: {e}")
    else:
        print(f"[{site}] INFO: No propensity covariates requested → SMD plot skipped.")

    # Models to run
    run_kinds = [k.strip() for k in args.models.split(",") if k.strip()]

    thr_rows = []
    for kind in run_kinds:
        proba, mean_auc, splits, _ = nested_cv_oof(
            X, y, seed=args.seed, kind=kind,
            inner_splits=args.inner_splits, outer_splits=args.outer_splits, outer_repeats=args.outer_repeats,
            calib=args.calibration, sample_weight=(iptw if kind in {"elasticnet","logreg"} else None)
        )

        # ROC/PR
        plot_roc(y, proba, title=f"{site.title()} — ROC ({kind})", out=os.path.join(plots_dir, f"roc_{kind}.png"))
        plot_pr (y, proba, title=f"{site.title()} — PR  ({kind})", out=os.path.join(plots_dir, f"pr_{kind}.png"))

        # thresholds & confusions
        t_f1, t_you = tune_thresholds(y, proba)
        for name, thr in [("fixed", 0.5), ("tunedF1", t_f1), ("tunedYOU", t_you)]:
            m, pred = metrics_block(y, proba, thr)
            write_csv(pd.DataFrame([m]), os.path.join(out_site, f"metrics_{kind}_{name}.csv"))
            cm = confusion_matrix(y, (proba>=thr).astype(int), labels=[0,1])
            suf = "" if name=="fixed" else f"_{name}"
            plot_confusion(cm, title=f"{site.title()} — Confusion ({kind}{' @0.5' if name=='fixed' else ''})",
                           out=os.path.join(plots_dir, f"confusion_{kind}{suf}.png"))

        thr_rows.append({"model":kind, "best_f1":t_f1, "best_youden":t_you})
        # OOF table
        oof = pd.DataFrame({"Sample_ID": sids, "y": y, "proba": proba})
        write_csv(oof, os.path.join(out_site, f"oof_{kind}.csv"))

        # Feature importance by refit on full data
        pipe, grid = get_model(kind, args.seed, for_importance=True)
        gs = GridSearchCV(
            pipe, grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed),
            n_jobs=-1, scoring="roc_auc", refit=True
        )
        gs.fit(X, y)
        best = gs.best_estimator_

        if kind in {"elasticnet", "logreg"}:
            clf = best.named_steps["clf"]
            coef = clf.coef_.ravel()
            imp_df = pd.DataFrame({"feature": feat, "coef": coef, "importance": np.abs(coef)})
            imp_df = imp_df.sort_values("importance", ascending=False)
            write_csv(imp_df, os.path.join(out_site, f"featimp_{kind}.csv"))
            plot_top(coef, feat, title=f"{site.title()} — Feature Importance ({kind} Top-20)",
                     out=os.path.join(plots_dir, f"featimp_{kind}_top20.png"), top=20)
        else:  # rf
            rf = best.named_steps["clf"]
            gini = pd.DataFrame({"feature": feat, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
            write_csv(gini, os.path.join(out_site, f"featimp_{kind}_gini.csv"))
            perm = permutation_importance(best, X, y, scoring="roc_auc", n_repeats=30, random_state=args.seed, n_jobs=-1)
            perm_df = pd.DataFrame({"feature": feat, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
            write_csv(perm_df, os.path.join(out_site, f"featimp_{kind}.csv"))
            plot_top(perm_df["importance"].values, perm_df["feature"].values,
                     title=f"{site.title()} — Feature Importance (RF Top-20)",
                     out=os.path.join(plots_dir, f"featimp_{kind}_top20.png"), top=20)

    write_csv(pd.DataFrame(thr_rows), os.path.join(out_site, "thresholds.csv"))

    # Differential abundance (mostly interpretable for abundance features)
    da = diff_abundance(X, y, feat, pseudo=args.pseudo, is_clr=args.use_clr)
    write_csv(da, os.path.join(out_site, "ppi_diffab.csv"))

# ------------------------------ Main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oral-crohn", required=True)
    ap.add_argument("--fecal-crohn", required=True)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--outdir", default="results/ml-ibd2")
    ap.add_argument("--target", default="PPI_use")

    # prevalence: either min fraction (--min-prev) or absolute count (--min-prev-samples)
    ap.add_argument("--min-prev", type=float, default=0.10)
    ap.add_argument("--min-prev-samples", type=int, default=None)
    ap.add_argument("--max-features", type=int, default=1000)

    ap.add_argument("--pseudo", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--outer-repeats", type=int, default=1)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--calibration", type=str, default="sigmoid", choices=["sigmoid","isotonic"])

    ap.add_argument("--models", type=str, default="elasticnet,rf", help="Comma-separated: elasticnet,logreg,rf")
    ap.add_argument("--use-clr", action="store_true")

    ap.add_argument("--add-covariates", type=str, default="", help="e.g., 'Age,Sex,BMI'")
    ap.add_argument("--propensity-covars", type=str, default="", help="e.g., 'Age,Sex,BMI,Antibiotics_3m'")
    ap.add_argument("--group-col", type=str, default=None)

    # optional sensitivity list for prevalence samples (comma list)
    ap.add_argument("--sens-prev-samples", type=str, default=None)

    args = ap.parse_args()
    ensure_dir(args.outdir)

    # main run
    run_site("oral",  args.oral_crohn,  args)
    run_site("fecal", args.fecal_crohn, args)

    # optional sensitivity runs
    if args.sens_prev_samples:
        vals = [int(v.strip()) for v in args.sens_prev_samples.split(",") if v.strip().isdigit()]
        for v in vals:
            sub_out = args.outdir.rstrip("/") + f"_prev{v}"
            os.makedirs(sub_out, exist_ok=True)
            args2 = argparse.Namespace(**vars(args))
            args2.outdir = sub_out
            args2.min_prev_samples = v
            run_site("oral",  args2.oral_crohn,  args2)
            run_site("fecal", args2.fecal_crohn, args2)

    # sentinel for Snakemake
    open(os.path.join(args.outdir, ".done"), "w").write("ok\n")
    print("[ibd_within_ml] Done")

if __name__ == "__main__":
    main()
