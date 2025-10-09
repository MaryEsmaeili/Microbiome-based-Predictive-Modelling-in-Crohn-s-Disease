#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBD within-site ML — one-shot for BOTH genus & species (oral + fecal)
- Robust ID normalization + ID overlap report
- Collapsing abundances to genus/species (sum across rows)
- CLR only on abundance
- Prevalence filtering (global, dataset-level)
- Nested CV + calibration; threshold tuning; ROC/PR/Confusion
- Feature importances; differential abundance (MWU + BH-like)
- Optional covariates (zero-impute) + IPTW + SMD plot
- Writes per-level/per-site outputs under one outdir
"""

import os, re, math, argparse, warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    if p:
        os.makedirs(p, exist_ok=True)

def safe_read_csv(p: Optional[str]) -> Optional[pd.DataFrame]:
    if p and os.path.exists(p):
        try:
            return pd.read_csv(p, sep=None, engine="python")
        except Exception:
            return pd.read_csv(p)
    return None

def write_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

# -------------------------- ID handling --------------------------

def normalize_id(x: str) -> str:
    """
    Robust normalization:
    - strip
    - drop trailing '.N'
    - remove non-alphanumeric
    - if [Letters][Digits]: drop leading zeros in digits
    - if pure digits: drop leading zeros
    - uppercase
    """
    if pd.isna(x):
        return x
    x = str(x).strip()
    x = re.sub(r"\.\d+$", "", x)
    x = re.sub(r"[^A-Za-z0-9]", "", x)
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", x)
    if m:
        prefix, num = m.groups()
        num = re.sub(r"^0+", "", num) or "0"
        x = f"{prefix}{num}"
    elif re.fullmatch(r"\d+", x):
        x = re.sub(r"^0+", "", x) or "0"
    return x.upper()

def norm_index_like(cols: List[str]) -> List[str]:
    return [normalize_id(c) for c in cols]

# --------------------- Taxonomy parsing & collapse ----------------

def extract_genus(rowname: str):
    parts = str(rowname).split("|")
    g = [p for p in parts if p.startswith("g__")]
    return g[0] if g else None

def extract_species(rowname: str):
    parts = str(rowname).split("|")
    s = [p for p in parts if p.startswith("s__")]
    return s[0] if s else None

def collapse_to_level(M_fxs: np.ndarray, feats: np.ndarray, level: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse rows to genus or species by summing rows with same token.
    Returns (M_collapsed [features x samples], features_collapsed)
    """
    if level not in {"genus","species"}:
        raise ValueError("level must be 'genus' or 'species'")
    if level == "genus":
        tokens = [extract_genus(f) for f in feats]
    else:
        tokens = [extract_species(f) for f in feats]

    df = pd.DataFrame(M_fxs)
    df.insert(0, "token", tokens)
    df = df[~df["token"].isna()].copy()
    if df.empty:
        return np.zeros((0, M_fxs.shape[1])), np.array([], dtype=object)

    grouped = df.groupby("token", as_index=False).sum(numeric_only=True)
    feats_new = grouped["token"].astype(str).values
    M_new = grouped.drop(columns=["token"]).to_numpy(dtype=float)
    return M_new, feats_new

# ------------------------ Abundance reader -----------------------

def read_abund(path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (M, features, samples) where M is features x samples (non-negative)."""
    df = safe_read_csv(path)
    if df is None or df.shape[0] == 0:
        return None

    # detect feature column
    first = df.columns[0]
    if (pd.isna(first) or first == "" or first == "...1" or
        re.fullmatch(r"(?i)(feature|taxon|clade|species|id|name)", str(first) or "")):
        feats = df.iloc[:, 0].astype(str).values
        df = df.iloc[:, 1:].copy()
    else:
        cand = [c for c in df.columns if re.search(r"(?i)feature|taxon|clade|name|id|species", c)]
        if cand:
            feats = df[cand[0]].astype(str).values
            df = df.drop(columns=[cand[0]])
        else:
            feats = np.arange(1, df.shape[0] + 1).astype(str)

    # numeric coerce; negatives/NaNs -> 0
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)
        df[c] = v.values

    M = df.to_numpy(dtype=float)
    keep = (M.sum(axis=1) > 0)
    M = M[keep, :]
    feats = np.array(pd.Series(feats)[keep].astype(str).values, dtype=object)
    samples = np.array(norm_index_like(df.columns.tolist()), dtype=object)
    return (M, feats, samples)

# ------------------------- CLR transform -------------------------

def clr_transform(X: np.ndarray, pseudo: float = 1e-6) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=None, neginf=0.0)
    X = np.clip(X, a_min=0.0, a_max=None)
    L = np.log(X + float(pseudo))
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
    grid = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y, (proba>=t).astype(int), zero_division=0) for t in grid]
    best_f1_thr = float(grid[int(np.argmax(f1s))])
    fpr, tpr, thr = roc_curve(y, proba)
    you = tpr - fpr
    best_you_thr = float(thr[int(np.argmax(you))])
    best_you_thr = min(max(best_you_thr, 0.0), 1.0)
    return best_f1_thr, best_you_thr

# ---------------------------- Modeling --------------------------

def make_calibrator(best_estimator, method="sigmoid", cv=3):
    try:
        return CalibratedClassifierCV(estimator=best_estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=best_estimator, method=method, cv=cv)

def get_model(kind: str, seed: int, for_importance: bool=False):
    if kind == "elasticnet":
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", class_weight="balanced",
                max_iter=8000, n_jobs=-1, random_state=seed
            ))
        ])
        grid = {"clf__C":[0.1, 1, 10], "clf__l1_ratio":[0.2, 0.5, 0.8]}
    elif kind == "logreg":
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                penalty="l2", solver="liblinear", class_weight="balanced",
                max_iter=5000, random_state=seed
            ))
        ])
        grid = {"clf__C":[0.1, 1, 10]}
    elif kind == "rf":
        pipe = Pipeline([
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
    proba = np.zeros(len(y), dtype=float)
    aucs = []
    rskf = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=seed)
    for _, (tr_idx, te_idx) in enumerate(rskf.split(X, y)):
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
    return proba, (np.mean(aucs) if aucs else float("nan")), [], {}

# ---------------- Differential Abundance -------------------------

def diff_abundance(X, y, feature_names, pseudo=1e-6, is_clr=False) -> pd.DataFrame:
    rows = []
    ln2 = math.log(2.0)
    for j, f in enumerate(feature_names):
        x0 = X[y==0, j]; x1 = X[y==1, j]
        if x0.size == 0 or x1.size == 0:
            p = 1.0
        else:
            try:
                p = mannwhitneyu(x0, x1, alternative="two-sided").pvalue
            except Exception:
                p = 1.0
        if is_clr:
            m0 = float(np.nanmean(x0)) if x0.size else 0.0
            m1 = float(np.nanmean(x1)) if x1.size else 0.0
            lf = (m1 - m0) / ln2
        else:
            m0 = float(np.nanmean(x0)) if x0.size else 0.0
            m1 = float(np.nanmean(x1)) if x1.size else 0.0
            lf = math.log2((m1+pseudo)/(m0+pseudo)) if (m1+pseudo)>0 and (m0+pseudo)>0 else 0.0
        rows.append((f, p, lf))
    df = pd.DataFrame(rows, columns=["feature", "p", "log2FC"]).sort_values("p").reset_index(drop=True)
    m = df.shape[0]; ranks = np.arange(1, m+1, dtype=float)
    q = np.minimum(df["p"].values * m / ranks, 1.0)
    df["q"] = q
    return df

# --------------------------- IPTW + SMD --------------------------

def compute_iptw(md: pd.DataFrame, target: str, covars: List[str]) -> Optional[np.ndarray]:
    covars = [c for c in covars if c in md.columns]
    if len(covars) < 2:
        return None
    Z = md[covars].apply(pd.to_numeric, errors="coerce").fillna(0.0)
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
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    x = np.where(np.isfinite(x), x, np.nan)
    if np.isnan(x).any():
        med = np.nanmedian(x); x = np.where(np.isnan(x), med, x)
    sw = w.sum()
    if sw <= 0: return float(np.nan), float(np.nan)
    m = (w * x).sum() / sw
    v = (w * (x - m) ** 2).sum() / sw
    return float(m), float(v)

def _smd_for_cov(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]=None) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=int)
    m1 = y == 1; m0 = y == 0
    if w is None:
        x1, x0 = x[m1], x[m0]
        if x1.size < 2 or x0.size < 2: return np.nan
        m1_, v1_ = float(np.nanmean(x1)), float(np.nanvar(x1))
        m0_, v0_ = float(np.nanmean(x0)), float(np.nanvar(x0))
    else:
        w = np.asarray(w, dtype=float)
        m1_, v1_ = _w_mean_var(x[m1], w[m1]); m0_, v0_ = _w_mean_var(x[m0], w[m0])
    denom = np.sqrt((v1_ + v0_) / 2.0) if np.isfinite(v1_) and np.isfinite(v0_) else np.nan
    if not np.isfinite(denom) or denom == 0: return np.nan
    return (m1_ - m0_) / denom

def compute_smd_table(md: pd.DataFrame, target: str, covars: List[str], w: Optional[np.ndarray]) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(md[target], errors="coerce").astype("Int64").astype(float).to_numpy()
    ymask = ~np.isnan(y); y = y[ymask].astype(int)
    for c in covars:
        if c not in md.columns: continue
        x = pd.to_numeric(md[c], errors="coerce").to_numpy()[ymask]
        rows.append({"covariate": c,
                     "SMD_before": _smd_for_cov(x, y, w=None),
                     "SMD_after":  _smd_for_cov(x, y, w[ymask]) if w is not None else np.nan})
    return pd.DataFrame(rows)

def plot_smd_before_after(df: pd.DataFrame, out_png: str, title: str="Covariate Balance (SMD)"):
    ensure_dir(os.path.dirname(out_png))
    if df.empty:
        plt.figure(); plt.text(0.5, 0.5, "No covariates to plot", ha="center", va="center")
        plt.axis("off"); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close(); return
    df = df.copy().sort_values("SMD_before", key=lambda s: np.abs(s.fillna(0.0)), ascending=False)
    y_ticks = np.arange(len(df))
    plt.figure(figsize=(7, max(3, 0.35*len(df))))
    plt.scatter(df["SMD_before"], y_ticks, label="Before IPTW")
    if df["SMD_after"].notna().any():
        plt.scatter(df["SMD_after"], y_ticks, label="After IPTW")
        for i, row in df.iterrows():
            idx = list(df.index).index(i)
            if not np.isnan(row["SMD_after"]):
                plt.plot([row["SMD_before"], row["SMD_after"]], [idx, idx], alpha=0.3)
    plt.axvline(0.1, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0.2, color="gray", linestyle="--", linewidth=1)
    plt.yticks(y_ticks, df["covariate"].tolist())
    plt.xlabel("Standardized Mean Difference"); plt.title(title)
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

# ---------------------- Build & run site ------------------------

def id_overlap_report(out_site: str, abund_samples: np.ndarray, md_ids_raw: pd.Series, site_label: str):
    samples_n = [normalize_id(s) for s in abund_samples]
    md_ids_n = [normalize_id(s) for s in md_ids_raw.tolist()]
    sa, sm = set(samples_n), set(md_ids_n)
    inter = sorted(sa & sm)
    only_abund = sorted(sa - sm)
    only_md    = sorted(sm - sa)
    df = pd.DataFrame({
        "n_abundance_samples":[len(sa)],
        "n_metadata_samples":[len(sm)],
        "n_intersection":[len(inter)],
        "example_intersection":[ ";".join(inter[:15]) ],
        "example_only_abund":[ ";".join(only_abund[:15]) ],
        "example_only_metadata":[ ";".join(only_md[:15]) ],
        "site":[site_label]
    })
    write_csv(df, os.path.join(out_site, "id_overlap.csv"))
    return len(inter)

def prepare_site_level(
    site: str, level: str, abund_path: str, covars_path: str, target: str,
    min_prev: float, min_prev_samples: Optional[int], max_features: int,
    use_clr: bool, pseudo: float, out_base: str,
    add_covariates: List[str], propensity_covars: List[str],
    group_col: Optional[str]
):
    """
    Build X,y for a specific site & taxonomic level (genus/species).
    Returns X, y, feature_names, sample_ids, md_site, iptw, used_prop_covs.
    """
    out_site = os.path.join(out_base, level, site)
    ensure_dir(out_site)

    # 1) Read abundance
    r = read_abund(abund_path)
    if r is None:
        write_csv(pd.DataFrame({"note":["abundance_read_failed"]}), os.path.join(out_site, "sanity_checks.csv"))
        return None
    M_fxs, feats, samples = r  # features x samples

    # 2) Collapse to level
    M_coll, feats_coll = collapse_to_level(M_fxs, feats, level=level)
    if M_coll.shape[0] == 0:
        write_csv(pd.DataFrame({"note":[f"no_feature_after_tax_collapse:{level}"]}), os.path.join(out_site, "sanity_checks.csv"))
        return None
    # dump cleaned collapsed abundance (deliverable)
    df_clean = pd.DataFrame(M_coll, columns=samples)
    df_clean.insert(0, "feature", feats_coll)
    write_csv(df_clean, os.path.join(out_site, f"abundance_clean_{level}.csv"))

    # 3) Metadata
    md = safe_read_csv(covars_path)
    if md is None or md.shape[0] == 0:
        write_csv(pd.DataFrame({"note":["metadata_read_failed"]}), os.path.join(out_site, "sanity_checks.csv"))
        return None
    for req in ["Sample_ID","site","disease"]:
        if req not in md.columns:
            write_csv(pd.DataFrame({"note":[f"metadata_column_missing:{req}"]}), os.path.join(out_site, "sanity_checks.csv"))
            return None
    md = md.copy()
    md["Sample_ID_raw"] = md["Sample_ID"]
    md["Sample_ID"] = md["Sample_ID"].map(normalize_id)
    md["site"] = md["site"].astype(str).str.lower()
    md["disease"] = pd.to_numeric(md["disease"], errors="coerce")

    # 4) Filter to site + Crohn
    md_site = md[(md["disease"] == 1) & (md["site"] == site)].copy()
    if target not in md_site.columns:
        write_csv(pd.DataFrame({"note":[f"target_not_found:{target}"]}), os.path.join(out_site, "sanity_checks.csv"))
        return None
    md_site[target] = pd.to_numeric(md_site[target], errors="coerce")

    # 5) ID overlap diagnostics
    inter_len = id_overlap_report(out_site, samples, md_site["Sample_ID_raw"], site)
    if inter_len == 0:
        write_csv(pd.DataFrame({"note":[f"no_common_ids_after_normalization:{site}"]}), os.path.join(out_site, "sanity_checks.csv"))
        return None

    # 6) Intersect & align
    common = np.intersect1d(samples, md_site["Sample_ID"].values)
    md_site = md_site.set_index("Sample_ID").loc[common]
    y = md_site[target].astype("Int64").astype(float).to_numpy()
    mask = ~np.isnan(y)
    y = y[mask].astype(int)
    sample_ids = common[mask]

    # align abundance columns to sample order; then transpose (samples x features)
    col_idx = [np.where(samples == sid)[0][0] for sid in sample_ids]
    X_abund = M_coll[:, col_idx].T
    feat_abund = feats_coll.copy()

    # 7) Prevalence filter
    nfeat_before = X_abund.shape[1]
    if min_prev_samples is not None:
        keep = (X_abund > 0).sum(axis=0) >= int(min_prev_samples)
    else:
        keep = (X_abund > 0).sum(axis=0) >= int(np.ceil(min_prev * X_abund.shape[0]))
    if keep.sum() == 0:
        write_csv(pd.DataFrame({
            "note":["no_feature_after_prevalence"],
            "n_features_before":[nfeat_before],
            "n_samples":[X_abund.shape[0]],
            "min_prev":[min_prev],
            "min_prev_samples":[min_prev_samples]}), os.path.join(out_site, "sanity_checks.csv"))
        return None
    X_abund = X_abund[:, keep]; feat_abund = feat_abund[keep]

    # 8) CLR
    if use_clr:
        X_abund = clr_transform(X_abund, pseudo=1e-6)

    # 9) Covariates (zero-impute)
    cov_names, X_cov, impute_rows = [], None, []
    def add_imp_row(col, nmiss, ids):
        impute_rows.append({"scope":"covariate","column":col,"n_missing":int(nmiss),
                            "impute":"zero","impute_value":0.0,
                            "sample_ids":";".join(ids[:200])})

    add_covariates = add_covariates or []
    if add_covariates:
        present = [c for c in add_covariates if c in md_site.columns]
        missing = [c for c in add_covariates if c not in md_site.columns]
        if missing:
            print(f"[{site}/{level}] WARN: missing covariates skipped: {', '.join(missing)}")
        if present:
            Cov_df = md_site[present].apply(pd.to_numeric, errors="coerce")
            na_counts = Cov_df.isna().sum()
            for c in present:
                if na_counts[c] > 0:
                    na_sids = Cov_df.index[Cov_df[c].isna()].tolist()
                    add_imp_row(c, na_counts[c], na_sids)
            Cov = Cov_df.fillna(0.0).to_numpy(dtype=float)
            Cov = Cov[mask]
            X_cov = Cov; cov_names = present

    # 10) Combine; guard NaN->0
    if X_cov is not None:
        X = np.hstack([X_abund[mask], X_cov])
        features = np.concatenate([feat_abund, np.array(cov_names, dtype=object)])
    else:
        X = X_abund[mask]; features = feat_abund

    final_na = int(np.isnan(X).sum())
    if final_na > 0:
        inds = np.where(np.isnan(X)); X[inds] = 0.0
        impute_rows.append({"scope":"X_final","column":"<matrix>","n_missing":final_na,"impute":"zero","impute_value":0.0,"sample_ids":""})

    # 11) Write imputation & summaries
    imp_rep = pd.DataFrame(impute_rows or [], columns=["scope","column","n_missing","impute","impute_value","sample_ids"])
    write_csv(imp_rep, os.path.join(out_site, "imputation_report.csv"))

    write_csv(pd.DataFrame({
        "n_samples":[X.shape[0]], "n_features":[X.shape[1]],
        "n_features_before_prev":[nfeat_before],
        "min_prev":[min_prev], "min_prev_samples":[min_prev_samples],
        "use_clr":[use_clr], "tax_level":[level]
    }), os.path.join(out_site, "matrix_shape.csv"))

    write_csv(pd.DataFrame({"label":[0,1], "count":[int(np.sum(y==0)), int(np.sum(y==1))]}),
              os.path.join(out_site, f"class_balance_{site}.csv"))

    # 12) IPTW
    iptw, used_prop_covs = None, []
    if propensity_covars:
        covs_here = [c for c in propensity_covars if c in md_site.columns]
        used_prop_covs = covs_here.copy()
        if len(covs_here) >= 2:
            try:
                # Note: original md (not subset by mask), weights aligned later
                md_tmp = md_site.copy()
                md_tmp[target] = pd.to_numeric(md_tmp[target], errors="coerce")
                from sklearn.linear_model import LogisticRegression as LR
                Z = md_tmp[covs_here].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
                t = md_tmp[target].to_numpy()
                ok = ~np.isnan(t)
                lr = LR(max_iter=1000).fit(Z[ok], t[ok].astype(int))
                ps = np.clip(lr.predict_proba(Z)[:,1], 1e-3, 1-1e-3)
                w = t/ps + (1-t)/(1-ps)
                w = np.where(np.isfinite(w), w, 1.0); w = w/np.mean(w)
                iptw = w[mask]
            except Exception:
                iptw = None

    return X, y, features, sample_ids, md_site, iptw, used_prop_covs

def run_site_level(site: str, level: str, abund_path: str, args):
    out_site = os.path.join(args.outdir, level, site)
    plots_dir = os.path.join(out_site, "plots")
    ensure_dir(plots_dir)

    add_covariates = [c.strip() for c in (args.add_covariates or "").split(",") if c.strip()]
    propensity_covars = [c.strip() for c in (args.propensity_covars or "").split(",") if c.strip()]

    prep = prepare_site_level(
        site=site, level=level, abund_path=abund_path, covars_path=args.covariates, target=args.target,
        min_prev=args.min_prev, min_prev_samples=args.min_prev_samples, max_features=args.max_features,
        use_clr=args.use_clr, pseudo=args.pseudo, out_base=args.outdir,
        add_covariates=add_covariates, propensity_covars=propensity_covars,
        group_col=args.group_col
    )
    if prep is None:
        write_csv(pd.DataFrame({"note":["no_data_or_no_features"]}), os.path.join(out_site, "summary_onepager.csv"))
        return

    X, y, feat, sids, md, iptw, used_prop_covs = prep

    # SMD
    if used_prop_covs:
        try:
            def _smd_table(md_df, target, covs, w):
                rows=[]
                yv = pd.to_numeric(md_df[target], errors="coerce").astype("Int64").astype(float).to_numpy()
                ymask=~np.isnan(yv); yv=yv[ymask].astype(int)
                for c in covs:
                    if c not in md_df.columns: continue
                    xv = pd.to_numeric(md_df[c], errors="coerce").to_numpy()[ymask]
                    rows.append({"covariate": c,
                                 "SMD_before": _smd_for_cov(xv, yv, None),
                                 "SMD_after": _smd_for_cov(xv, yv, w[ymask]) if w is not None else np.nan})
                return pd.DataFrame(rows)
            smd_df = _smd_table(md, args.target, used_prop_covs, iptw)
            write_csv(smd_df, os.path.join(out_site, "smd_before_after.csv"))
            plot_smd_before_after(smd_df, os.path.join(out_site, "smd_before_after.png"),
                                  f"{site.title()} — Covariate Balance (SMD)")
        except Exception as e:
            print(f"[{site}/{level}] WARN: SMD plotting failed: {e}")

    # models
    run_kinds = [k.strip() for k in args.models.split(",") if k.strip()]
    thr_rows = []
    for kind in run_kinds:
        proba, mean_auc, _, _ = nested_cv_oof(
            X, y, seed=args.seed, kind=kind,
            inner_splits=args.inner_splits, outer_splits=args.outer_splits, outer_repeats=args.outer_repeats,
            calib=args.calibration, sample_weight=(iptw if kind in {"elasticnet","logreg"} else None)
        )
        plot_roc(y, proba, f"{site.title()} — ROC ({kind})", os.path.join(plots_dir, f"roc_{kind}.png"))
        plot_pr (y, proba, f"{site.title()} — PR  ({kind})",  os.path.join(plots_dir, f"pr_{kind}.png"))

        t_f1, t_you = (0.5, 0.5)
        if len(np.unique(y)) == 2:
            t_f1, t_you = tune_thresholds(y, proba)
        for name, thr in [("fixed", 0.5), ("tunedF1", t_f1), ("tunedYOU", t_you)]:
            m, _ = metrics_block(y, proba, thr)
            write_csv(pd.DataFrame([m]), os.path.join(out_site, f"metrics_{kind}_{name}.csv"))
            cm = confusion_matrix(y, (proba>=thr).astype(int), labels=[0,1])
            suf = "" if name=="fixed" else f"_{name}"
            plot_confusion(cm, f"{site.title()} — Confusion ({kind}{' @0.5' if name=='fixed' else ''})",
                           os.path.join(plots_dir, f"confusion_{kind}{suf}.png"))

        thr_rows.append({"model":kind, "best_f1":t_f1, "best_youden":t_you})
        write_csv(pd.DataFrame({"Sample_ID": sids, "y": y, "proba": proba}), os.path.join(out_site, f"oof_{kind}.csv"))

        # refit for importances
        pipe, grid = get_model(kind, args.seed, for_importance=True)
        gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed),
                          n_jobs=-1, scoring="roc_auc", refit=True)
        gs.fit(X, y); best = gs.best_estimator_
        if kind in {"elasticnet","logreg"}:
            coef = best.named_steps["clf"].coef_.ravel()
            imp_df = pd.DataFrame({"feature": feat, "coef": coef, "importance": np.abs(coef)}).sort_values("importance", ascending=False)
            write_csv(imp_df, os.path.join(out_site, f"featimp_{kind}.csv"))
            plot_top(coef, feat, f"{site.title()} — Feature Importance ({kind} Top-20)",
                     os.path.join(plots_dir, f"featimp_{kind}_top20.png"), top=20)
        else:
            rf = best.named_steps["clf"]
            gini = pd.DataFrame({"feature": feat, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
            write_csv(gini, os.path.join(out_site, f"featimp_{kind}_gini.csv"))
            perm = permutation_importance(best, X, y, scoring="roc_auc", n_repeats=30, random_state=args.seed, n_jobs=-1)
            perm_df = pd.DataFrame({"feature": feat, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
            write_csv(perm_df, os.path.join(out_site, f"featimp_{kind}.csv"))
            plot_top(perm_df["importance"].values, perm_df["feature"].values,
                     f"{site.title()} — Feature Importance (RF Top-20)",
                     os.path.join(plots_dir, f"featimp_{kind}_top20.png"), top=20)

    write_csv(pd.DataFrame(thr_rows), os.path.join(out_site, "thresholds.csv"))
    da = diff_abundance(X, y, feat, pseudo=1e-6, is_clr=args.use_clr)
    write_csv(da, os.path.join(out_site, "ppi_diffab.csv"))

# ------------------------------ Main ----------------------------

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oral-crohn", required=True, help="Crohn oral abundance (filtered CSV)")
    ap.add_argument("--fecal-crohn", required=True, help="Crohn fecal abundance (filtered CSV)")
    ap.add_argument("--covariates", required=True, help="Pooled model table (must include Sample_ID, site, disease)")
    ap.add_argument("--outdir", default="results/ml-ibd_levels")
    ap.add_argument("--target", default="PPI_use")

    ap.add_argument("--min-prev", type=float, default=0.20)
    ap.add_argument("--min-prev-samples", type=int, default=10)
    ap.add_argument("--max-features", type=int, default=2000)

    ap.add_argument("--pseudo", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--outer-repeats", type=int, default=10)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--calibration", type=str, default="sigmoid", choices=["sigmoid","isotonic"])
    ap.add_argument("--models", type=str, default="elasticnet,rf", help="Comma-separated: elasticnet,logreg,rf")
    ap.add_argument("--use-clr", action="store_true")

    ap.add_argument("--add-covariates", type=str, default="Age,Sex,BMI")
    ap.add_argument("--propensity-covars", type=str, default="Age,Sex,BMI,Antibiotics_3m")
    ap.add_argument("--group-col", type=str, default=None)
    return ap

def main():
    args = build_argparser().parse_args()
    ensure_dir(args.outdir)

    # One run → both levels, both sites. No "all".
    for level in ["genus", "species"]:
        run_site_level("oral",  level, args.oral_crohn,  args)
        run_site_level("fecal", level, args.fecal_crohn, args)

    # sentinel for Snakemake
    open(os.path.join(args.outdir, ".done"), "w").write("ok\n")
    print("[ibd_within_ml_levels] Done")

if __name__ == "__main__":
    main()
