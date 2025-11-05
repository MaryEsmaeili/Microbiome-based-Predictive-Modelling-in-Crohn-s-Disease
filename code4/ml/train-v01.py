#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — Robust ML runner for the microbiome pipeline

Modes
  - within:       nested CV inside one site (oral|fecal)
  - transfer:     train on fecal, test on oral (cross-site)
  - paired_transfer: subject-level LOSO (train on others' fecal → test on subject's oral)
  - learningcurve: performance vs. train size (default: fecal)

Inputs
  --pct-all : taxa×samples (%) table (rows: taxa, cols: Sample_ID)
  --meta    : pooled metadata
  --pairs   : optional matched oral/fecal pairs (to derive subject_id)

Labels & grouping
  - Binary label 'disease' mapped to {0,1}
  - Group column 'subject_id' to avoid leakage (from meta or pairs; fallback=sample_id)

Guardrails
  - single-class checks (dataset & per-fold)
  - group-aware splitters (StratifiedGroupKFold → fallback StratifiedKFold)
  - feature alignment train↔test (intersection/union)
  - standardization fit on TRAIN only; optional light CORAL
  - safe plotting & placeholder outputs

Outputs
  within:        cv_metrics.csv, roc_pr.png, importances.csv
  transfer:      metrics.csv,    roc_pr.png
  paired_transfer: metrics.csv,  roc_pr.png, per_subject.csv, preds.csv
  learningcurve: learning_curve.csv, learning_curve.png
"""

from __future__ import annotations
import os, argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
try:
    from sklearn.model_selection import StratifiedGroupKFold  # sklearn >=1.1
except Exception:
    StratifiedGroupKFold = None

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)

# ----------------------------- I/O utils -----------------------------

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def read_pct_all(path: str, pseudocount: float = 1e-6) -> pd.DataFrame:
    """Return samples×taxa CLR from taxa×samples % table."""
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.astype(str)
    X = (df.astype(float) / 100.0) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)
    clr = logX.sub(gm, axis=1).T
    clr.index.name = "Sample_ID"
    return clr

def _first_col(df: pd.DataFrame, *cands) -> Optional[str]:
    low = {str(c).lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def norm_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Normalize meta to columns: sample_id, site(Oral/Fecal), disease{0,1}, subject_id?"""
    m = meta.copy()
    m.columns = [str(c).strip() for c in m.columns]

    sid = _first_col(m, "sample_id","id","sample","sampleid")
    site = _first_col(m, "site","body_site","location")
    dis  = _first_col(m, "disease","status","group")
    subc = _first_col(m, "subject_id","subject","patient_id","participant_id","person_id")

    out = pd.DataFrame({
        "sample_id": m[sid].astype(str) if sid else pd.Series(dtype=str),
        "site":      m[site] if site else pd.Series(dtype=object),
        "disease":   m[dis] if dis else pd.Series(dtype=object),
    })
    out["subject_id"] = m[subc].astype(str) if subc else np.nan

    # site to {Oral,Fecal}
    out["site"] = out["site"].astype(str).str.strip().str.capitalize().replace({"Faecal": "Fecal"})
    out.loc[~out["site"].isin(["Oral","Fecal"]), "site"] = np.nan

    # map disease to {0,1}
    def to01(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().lower()
        pos = {"1","yes","true","y","crohn","cd","case","ibd"}
        neg = {"0","no","false","n","healthy","control","hc","non-ibd","nonibd"}
        if s in pos: return 1
        if s in neg: return 0
        try:
            v = int(float(s))
            if v in (0,1): return v
        except Exception:
            pass
        return np.nan
    out["disease"] = out["disease"].map(to01)
    return out

def read_meta(path: str) -> pd.DataFrame:
    return norm_meta(pd.read_csv(path))

def read_pairs_as_subjects(path: Optional[str]) -> Dict[str,str]:
    """Build subject_id map from pairs CSV (oral_id,fecal_id)."""
    if path is None or not os.path.exists(path):
        return {}
    p = pd.read_csv(path)
    low = {c.lower(): c for c in p.columns}
    def pick(*cands):
        for c in cands:
            if c.lower() in low: return low[c.lower()]
        return None
    oc = pick("oral_sample_id","oral_id","oral","oc")
    fc = pick("fecal_sample_id","fecal_id","fecal","fc")
    if oc is None or fc is None: return {}

    submap = {}
    sid = 0
    for _, r in p[[oc,fc]].dropna().astype(str).iterrows():
        sid += 1
        subj = f"S{sid:05d}"
        submap[str(r[oc])] = subj
        submap[str(r[fc])] = subj
    return submap

def attach_subject_id(meta: pd.DataFrame, pairs_map: Dict[str,str]) -> pd.DataFrame:
    """Fill subject_id from pairs_map if missing; fallback = sample_id."""
    out = meta.copy()
    if out["subject_id"].isna().all() and pairs_map:
        out["subject_id"] = out["sample_id"].map(pairs_map)
    if out["subject_id"].isna().any():
        mask = out["subject_id"].isna()
        out.loc[mask, "subject_id"] = out.loc[mask, "sample_id"]
    return out

# ----------------------------- helpers -------------------------------

def class_counts(y: np.ndarray) -> np.ndarray:
    y = pd.Series(y).astype(int).values
    if len(y) == 0: return np.array([0,0])
    return np.bincount(y, minlength=2)[:2]

def has_two_classes(y: np.ndarray) -> bool:
    c = class_counts(y)
    return bool(c[0] > 0 and c[1] > 0)

def decide_n_splits(y: np.ndarray, groups: Optional[np.ndarray], max_splits: int) -> int:
    bc = class_counts(y)
    if not (bc[0] > 0 and bc[1] > 0):
        return 2
    if groups is None:
        return max(2, min(max_splits, int(bc.min())))
    g = pd.DataFrame({"y": pd.Series(y).astype(int), "g": pd.Series(groups).astype(str)})
    grp_min = int(g.groupby("y")["g"].nunique().min())
    return max(2, min(max_splits, grp_min))

def make_splitter(y: np.ndarray, groups: Optional[np.ndarray], max_splits: int):
    k = decide_n_splits(y, groups, max_splits)
    if groups is not None and StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

def write_skip_placeholder(outdir: str, fname: str, reason: str):
    ensure_dir(outdir)
    pd.DataFrame([{"status":"skipped","reason":reason}]).to_csv(os.path.join(outdir, fname), index=False)
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    for i, ttl in enumerate(["ROC (n/a)","PR (n/a)"]):
        ax[i].axis("off"); ax[i].set_title(ttl)
        ax[i].text(0.5,0.5,f"Skipped: {reason}",ha="center",va="center")
    fig.tight_layout(); fig.savefig(os.path.join(outdir,"roc_pr.png"),dpi=170); plt.close(fig)

# ----------------------------- modeling ------------------------------
def coef_importances(fitted: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """
    استخراج ضرایب مدل لجستیک (پس از GridSearchCV.best_estimator_)
    و برگرداندن جدول feature, coef, abs(|coef|) به‌صورت مرتب‌شده.
    """
    try:
        coef = np.ravel(fitted.named_steps["clf"].coef_)
        df = pd.DataFrame({
            "feature": feature_names,
            "coef": coef
        })
        df["abs"] = np.abs(df["coef"].astype(float))
        return df.sort_values("abs", ascending=False)
    except Exception:
        # اگر به هر دلیلی نتونست ضرایب رو بخونه، خروجی خالی/NaN بده
        return pd.DataFrame({"feature": feature_names, "coef": np.nan, "abs": np.nan})

def build_model(name: str, random_state: int=42) -> Tuple[Pipeline, Dict[str, List]]:
    name = (name or "logit_enet").lower()
    if name != "logit_enet":
        raise ValueError(f"Unknown model: {name}")
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            solver="saga", penalty="elasticnet",
            l1_ratio=0.5, C=1.0, max_iter=20000,
            class_weight="balanced", random_state=random_state, n_jobs=1
        ))
    ])
    grid = {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1], "clf__l1_ratio":[0.0,0.2,0.5,0.8,1.0]}
    return pipe, grid

def safe_roc_pr(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    roc_res = {"fpr":None,"tpr":None,"auc":np.nan}
    pr_res  = {"precision":None,"recall":None,"ap":np.nan}
    if has_two_classes(y_true):
        fpr,tpr,_ = roc_curve(y_true,y_score)
        roc_res.update({"fpr":fpr,"tpr":tpr,"auc":float(roc_auc_score(y_true,y_score))})
    try:
        prec,rec,_ = precision_recall_curve(y_true,y_score)
        pr_res.update({"precision":prec,"recall":rec,"ap":float(average_precision_score(y_true,y_score))})
    except Exception:
        pass
    return roc_res, pr_res

# ---------- plotting helpers ----------
def plot_importances_lollipop(imp_df: pd.DataFrame, out_png: str,
                              top: int = 20, title: str = ""):
    """
    لولی‌پاپ افقی از مهم‌ترین فیچرها بر اساس |coef|.
    رنگ آبی برای ضرایب مثبت و نارنجی برای ضرایب منفی.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from textwrap import shorten

    # آماده‌سازی
    df = imp_df.copy()
    if "abs" not in df.columns:
        df["abs"] = np.abs(df["coef"].astype(float))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["coef", "abs"])
    if df.empty:
        # خروج امن
        fig, ax = plt.subplots(figsize=(7.2, 4))
        ax.axis("off"); ax.text(0.5, 0.5, "No importances", ha="center", va="center")
        fig.savefig(out_png, dpi=180); plt.close(fig); return

    topk = df.sort_values("abs", ascending=False).head(top).iloc[::-1]  # برعکس برای بالا→پایین
    y = np.arange(len(topk))

    colors = np.where(topk["coef"] >= 0, "#2a6fdb", "#f08a24")  # + / –
    x0 = np.zeros_like(y, dtype=float)
    x1 = topk["coef"].values.astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    # stem (بدون مارکر پیش‌فرض)
    for yi, xi, ci in zip(y, x1, colors):
        ax.plot([0, xi], [yi, yi], lw=2.5, color=ci, alpha=0.9)
        ax.scatter(xi, yi, s=40, color=ci, zorder=3)

    # محور و برچسب‌ها
    ax.set_yticks(y)
    ax.set_yticklabels([shorten(str(f), width=45, placeholder="…") for f in topk["feature"]])
    ax.axvline(0, lw=1, color="#999", ls="--", alpha=0.7)
    ax.set_xlabel("Coefficient (elastic-net logistic)")
    if title: ax.set_title(title)
    # مارجین کمی برای برچسب‌ها
    lo, hi = np.nanmin([0, x1.min()]), np.nanmax([0, x1.max()])
    span = hi - lo if hi > lo else 1.0
    ax.set_xlim(lo - 0.08*span, hi + 0.12*span)
    # تم تمیز
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_roc_pr(roc_res, pr_res, out_png: str, title: str):
    """Draw side-by-side ROC and PR; robust to single-class test."""
    ensure_dir(os.path.dirname(out_png))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))

    # ROC
    if roc_res.get("fpr") is None or roc_res.get("tpr") is None:
        ax[0].axis("off")
        ax[0].text(0.5, 0.5, "ROC n/a (single-class)", ha="center", va="center")
    else:
        ax[0].plot(roc_res["fpr"], roc_res["tpr"], lw=2)
        ax[0].plot([0, 1], [0, 1], "--", lw=1)
        auc_txt = "n/a" if not np.isfinite(roc_res.get("auc", np.nan)) else f"{roc_res['auc']:.3f}"
        ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title(f"ROC (AUC={auc_txt})")

    # PR
    prec, rec, ap = pr_res.get("precision"), pr_res.get("recall"), pr_res.get("ap", np.nan)
    if prec is None or rec is None:
        ax[1].axis("off")
        ax[1].text(0.5, 0.5, "PR n/a", ha="center", va="center")
    else:
        ax[1].plot(rec, prec, lw=2)
        ap_txt = "n/a" if not np.isfinite(ap) else f"{ap:.3f}"
        ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title(f"PR (AP={ap_txt})")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_cv_folds(df_cv: pd.DataFrame, out_png: str, title: str = ""):
    """Bar chart of per-fold ROC-AUC and PR-AP from cv_metrics.csv."""
    try:
        dfp = df_cv.copy()
        dfp = dfp[dfp["fold"] != "mean"].copy()
        if dfp.empty:
            return
        dfp["fold"] = dfp["fold"].astype(int)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        ax[0].bar(dfp["fold"], dfp["auc"]); ax[0].set_ylim(0, 1)
        ax[0].set_xlabel("Fold"); ax[0].set_ylabel("ROC-AUC"); ax[0].set_title("Per-fold ROC-AUC")
        ax[1].bar(dfp["fold"], dfp["ap"]); ax[1].set_ylim(0, 1)
        ax[1].set_xlabel("Fold"); ax[1].set_ylabel("PR-AP"); ax[1].set_title("Per-fold PR-AP")
        if title: fig.suptitle(title)
        fig.tight_layout()
        ensure_dir(os.path.dirname(out_png))
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
    except Exception:
        pass


def plot_importances_bar(imp_df: pd.DataFrame, out_png: str, top: int = 20, title: str = "Top features"):
    """Horizontal bar plot of |coef| (top-k)."""
    try:
        if imp_df is None or imp_df.empty:
            return
        dfp = imp_df.copy()
        if "abs" not in dfp.columns:
            dfp["abs"] = np.abs(dfp["coef"].astype(float))
        dfp = dfp.sort_values("abs", ascending=False).head(top).iloc[::-1]  # bottom-up
        fig, ax = plt.subplots(figsize=(6.8, max(3.0, 0.25 * len(dfp))))
        ax.barh(dfp["feature"], dfp["abs"])
        ax.set_xlabel("|coef|"); ax.set_ylabel("feature"); ax.set_title(title)
        fig.tight_layout()
        ensure_dir(os.path.dirname(out_png))
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
    except Exception:
        pass

# ----------------------- feature alignment / CORAL -------------------

def _feature_align(Xtr, Xte, mode="intersection", min_freq=0):
    if mode == "union_zero":
        cols = list(Xtr.columns); return Xtr, Xte.reindex(columns=cols).fillna(0.0), cols
    inter = Xtr.columns.intersection(Xte.columns)
    if mode == "intersection_minfreq" and min_freq > 0:
        tr_nz = (Xtr[inter]!=0).sum(0); te_nz = (Xte[inter]!=0).sum(0)
        inter = inter[(tr_nz>=min_freq) & (te_nz>=min_freq)]
    return Xtr[inter].copy(), Xte[inter].copy(), list(inter)

def _fit_standardizer(X): m=X.mean(0); s=X.std(0,ddof=0).replace(0,np.nan); return m,s
def _apply_standardizer(X,m,s): return ((X-m)/s).fillna(0.0)

def _coral_align(Xtr, Xte, alpha=0.8):
    if Xtr.shape[1] < 2: return Xtr, Xte
    eps=1e-6
    Ctr = np.cov(Xtr.values,rowvar=False)+eps*np.eye(Xtr.shape[1])
    Cte = np.cov(Xte.values,rowvar=False)+eps*np.eye(Xte.shape[1])
    wtr,Vtr = np.linalg.eigh(Ctr); wte,Vte = np.linalg.eigh(Cte)
    Ctr_m12 = Vtr @ np.diag(wtr**-0.5) @ Vtr.T
    Cte_p12 = Vte @ np.diag(wte**0.5) @ Vte.T
    A = alpha*(Ctr_m12 @ Cte_p12) + (1-alpha)*np.eye(Xtr.shape[1])
    Xtr_c = Xtr.values @ A
    return pd.DataFrame(Xtr_c,index=Xtr.index,columns=Xtr.columns), Xte

# ----------------------- data assembly functions --------------------

def assemble_within(pct, meta, site: str):
    site = site.capitalize()
    M = meta[meta["site"]==site].copy()
    ids = set(M["sample_id"].astype(str))
    X = pct.loc[pct.index.isin(ids)].copy()
    M = M.set_index("sample_id").loc[X.index]
    y = M["disease"].astype(float).values
    g = M["subject_id"].astype(str).values
    keep = np.isfinite(y); X=X.loc[keep]; y=y[keep].astype(int); g=g[keep]
    return X.values, y, g, list(X.columns)

def assemble_transfer(
    pct_all, meta, train_site: str, test_site: str,
    disjoint_drop: str = "test",   # "train" or "test"
    feature_mode: str="intersection",
    min_freq: int=0,
    site_standardize: bool=False,
    coral: bool=False
):
    tr = train_site.capitalize(); te = test_site.capitalize()
    Mtr = meta[meta["site"] == tr].copy()
    Mte = meta[meta["site"] == te].copy()

    # decide where to drop shared subjects (to keep splits clean)
    inter = set(Mtr["subject_id"]).intersection(set(Mte["subject_id"]))
    if disjoint_drop not in ("train", "test"):
        disjoint_drop = "test"
    if len(inter) > 0:
        if disjoint_drop == "train":
            Mtr = Mtr[~Mtr["subject_id"].isin(inter)].copy()
        else:
            Mte = Mte[~Mte["subject_id"].isin(inter)].copy()

    # ── here was the bug: use pct_all, not pct ──
    Xtr = pct_all.loc[pct_all.index.isin(set(Mtr["sample_id"]))].copy()
    Xte = pct_all.loc[pct_all.index.isin(set(Mte["sample_id"]))].copy()

    Mtr = Mtr.set_index("sample_id").loc[Xtr.index]
    Mte = Mte.set_index("sample_id").loc[Xte.index]

    ytr = Mtr["disease"].astype(float).values
    yte = Mte["disease"].astype(float).values

    Xtr, Xte, feats = _feature_align(Xtr, Xte, mode=feature_mode, min_freq=min_freq)

    if site_standardize:
        m,s = _fit_standardizer(Xtr); Xtr=_apply_standardizer(Xtr,m,s); Xte=_apply_standardizer(Xte,m,s)
    if coral:
        Xtr,Xte = _coral_align(Xtr,Xte,alpha=0.8)

    keep_tr = np.isfinite(ytr); Xtr=Xtr.loc[keep_tr]; ytr=ytr[keep_tr].astype(int)
    keep_te = np.isfinite(yte); Xte=Xte.loc[keep_te]; yte=yte[keep_te].astype(int)

    gtr = Mtr.loc[Xtr.index, "subject_id"].astype(str).values
    return Xtr.values, ytr, gtr, Xte.values, yte, feats

# ------------------------------ runners -----------------------------
def run_within(args):
    ensure_dir(args.outdir)
    pct = read_pct_all(args.pct_all)
    meta = attach_subject_id(read_meta(args.meta), read_pairs_as_subjects(args.pairs))
    meta = meta[meta["sample_id"].isin(pct.index)]

    X, y, g, feat = assemble_within(pct, meta, args.site)
    if len(y) < 3 or not has_two_classes(y):
        write_skip_placeholder(args.outdir, "cv_metrics.csv", "single_class_or_too_few_samples")
        return

    base, grid = build_model(args.model, args.random_state)
    groups_used = g if args.group_col else None
    outer = make_splitter(y, groups_used, args.folds)

    oof = np.zeros(len(y), dtype=float)
    rows = []

    for i, (tr, te) in enumerate(outer.split(X, y, groups_used), start=1):
        ytr, yte = y[tr], y[te]
        gtr = g[tr] if args.group_col else None
        inner = make_splitter(ytr, gtr, args.inner_folds)

        gcv = GridSearchCV(base, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True, error_score=np.nan)
        gcv.fit(X[tr], ytr, **({"groups": gtr} if gtr is not None else {}))

        try:
            p = gcv.predict_proba(X[te])[:, 1]
        except Exception:
            s = gcv.decision_function(X[te])
            p = (s - s.min()) / (s.max() - s.min() + 1e-9)

        roc_res, pr_res = safe_roc_pr(yte, p)
        rows.append({"fold": i, "auc": roc_res["auc"], "ap": pr_res["ap"]})
        oof[te] = p

    if len(rows) == 0:
        write_skip_placeholder(args.outdir, "cv_metrics.csv", "no_cv_rows")
        return

    df_cv = pd.DataFrame(rows)
    df_cv.loc[len(df_cv)] = {"fold": "mean",
                             "auc": np.nanmean(df_cv["auc"]),
                             "ap":  np.nanmean(df_cv["ap"])}
    df_cv.to_csv(os.path.join(args.outdir, "cv_metrics.csv"), index=False)

    # Overall ROC/PR (OOF)
    roc_res, pr_res = safe_roc_pr(y, oof)
    plot_roc_pr(roc_res, pr_res,
                os.path.join(args.outdir, "roc_pr.png"),
                f"Within-site ({args.site}, {args.rank})")

    # Per-fold bars
    plot_cv_folds(df_cv,
                  os.path.join(args.outdir, "cv_folds.png"),
                  title=f"CV folds — {args.site} ({args.rank})")

    # Refit on all to export importances + bar plot
    inner_all = make_splitter(y, groups_used, args.inner_folds)
    gcv_all = GridSearchCV(base, grid, scoring="roc_auc", cv=inner_all, n_jobs=-1, refit=True, error_score=np.nan)
    gcv_all.fit(X, y, **({"groups": groups_used} if groups_used is not None else {}))
    imp_df = coef_importances(gcv_all.best_estimator_, feat)
    imp_df.to_csv(os.path.join(args.outdir, "importances.csv"), index=False)

    plot_importances_lollipop(
        imp_df,
        os.path.join(args.outdir, "importances_top20.png"),
        top=20,
        title=f"Top features — {args.site} ({args.rank})"
    )

def run_transfer(args):
    ensure_dir(args.outdir)
    pct = read_pct_all(args.pct_all)
    meta = attach_subject_id(read_meta(args.meta), read_pairs_as_subjects(args.pairs))
    meta = meta[meta["sample_id"].isin(pct.index)]

    Xtr, ytr, gtr, Xte, yte, feat_names = assemble_transfer(
        pct, meta, args.train_site, args.test_site,
        disjoint_drop=args.disjoint_drop,
        feature_mode=args.feature_mode,
        min_freq=args.min_freq,
        site_standardize=args.site_standardize,
        coral=args.coral
    )


    if len(np.unique(ytr))<2 or len(np.unique(yte))<2:
        write_skip_placeholder(args.outdir, "metrics.csv", "single_class_train_or_test"); return

    base,grid = build_model(args.model, args.random_state)
    inner = make_splitter(ytr, gtr if args.group_col else None, args.inner_folds)
    gcv = GridSearchCV(base, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True, error_score=np.nan)
    gcv.fit(Xtr, ytr, **({"groups":gtr} if args.group_col else {}))

    try: p = gcv.predict_proba(Xte)[:,1]
    except Exception:
        s = gcv.decision_function(Xte); p=(s-s.min())/(s.max()-s.min()+1e-9)

    roc_res,pr_res = safe_roc_pr(yte,p)
    pd.DataFrame([{
        "auc": roc_res["auc"], "ap": pr_res["ap"],
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "feature_mode": args.feature_mode,
        "site_standardize": bool(args.site_standardize),
        "coral": bool(args.coral),
        "disjoint_drop": args.disjoint_drop
    }]).to_csv(os.path.join(args.outdir,"metrics.csv"), index=False)
    plot_roc_pr(roc_res,pr_res, os.path.join(args.outdir,"roc_pr.png"),
                f"Transfer {args.train_site.capitalize()}→{args.test_site.capitalize()} ({args.rank})")

def run_paired_transfer(args):
    """LOSO on paired subjects: train on others' fecal → test on subject's oral."""
    ensure_dir(args.outdir)
    pct = read_pct_all(args.pct_all)
    meta = attach_subject_id(read_meta(args.meta), read_pairs_as_subjects(args.pairs))
    meta = meta[meta["sample_id"].isin(pct.index)]
    meta["site"] = meta["site"].astype(str)

    grp = meta.groupby("subject_id")["site"].apply(lambda s: set(s))
    paired = grp[grp.apply(lambda s: {"Oral","Fecal"}.issubset(s))].index.tolist()
    if not paired:
        write_skip_placeholder(args.outdir, "metrics.csv", "no_paired_subjects"); return

    base,grid = build_model(args.model, args.random_state)
    all_y, all_p = [], []; per_rows=[]; pred_rows=[]; used=0

    for sid in paired:
        trm = meta[(meta.site=="Fecal") & (meta.subject_id!=sid) & (meta.disease.isin([0,1]))].copy()
        if trm.empty: continue
        Xtr = pct.loc[trm.sample_id]; ytr = trm.disease.astype(int).values; gtr = trm.subject_id.astype(str).values
        if len(np.unique(ytr))<2:
            per_rows.append({"subject_id":sid,"n_train":len(ytr),"reason":"single_class_train"}); continue

        tem = meta[(meta.site=="Oral") & (meta.subject_id==sid) & (meta.disease.isin([0,1]))].copy()
        if tem.empty:
            per_rows.append({"subject_id":sid,"n_train":len(ytr),"reason":"no_oral_for_subject"}); continue

        Xte = pct.loc[tem.sample_id].reindex(columns=Xtr.columns).fillna(0.0); yte=tem.disease.astype(int).values
        inner = make_splitter(ytr, gtr if args.group_col else None, args.inner_folds)
        gcv = GridSearchCV(base, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True, error_score=np.nan)
        gcv.fit(Xtr.values, ytr, **({"groups":gtr} if args.group_col else {}))

        try: p = gcv.predict_proba(Xte.values)[:,1]
        except Exception:
            s = gcv.decision_function(Xte.values); p=(s-s.min())/(s.max()-s.min()+1e-9)

        used += 1; all_y.append(yte); all_p.append(p)
        yhat=(p>=0.5).astype(int); acc=float((yhat==yte).mean()) if len(yte)>0 else np.nan
        per_rows.append({"subject_id":sid,"n_train":int(len(ytr)),"n_test":int(len(yte)),
                         "y_pos_test":int(yte.sum()),"acc":acc,"best_params":str(gcv.best_params_)})
        for samp,yy,pp in zip(tem.sample_id.tolist(), yte, p):
            pred_rows.append({"subject_id":sid,"sample_id":samp,"y_true":int(yy),"y_prob":float(pp)})
        
        # after the loop:
        if used == 0:
            write_skip_placeholder(args.outdir, "metrics.csv", "no_usable_paired_subjects")
            return

        per_subj_df = pd.DataFrame(per_rows)     # <- NOT per_subj_rows
        per_subj_df.to_csv(os.path.join(args.outdir, "per_subject.csv"), index=False)
        pd.DataFrame(pred_rows).to_csv(os.path.join(args.outdir, "preds.csv"), index=False)

        y_all = np.concatenate(all_y) if all_y else np.array([])
        p_all = np.concatenate(all_p) if all_p else np.array([])
        if y_all.size == 0:
            write_skip_placeholder(args.outdir, "metrics.csv", "no_test_samples")
            return

        roc_res, pr_res = safe_roc_pr(y_all, p_all)
        two_class = (len(np.unique(y_all)) >= 2)
        auc_val = roc_res["auc"] if two_class else np.nan

        pd.DataFrame([{
            "auc": auc_val,
            "ap":  pr_res["ap"],
            "n_subjects": int(used),
            "n_test_samples": int(len(y_all)),
            "n_test_pos": int((y_all==1).sum()),
            "n_test_neg": int((y_all==0).sum()),
            "note": "single_class_test" if not two_class else ""
        }]).to_csv(os.path.join(args.outdir, "metrics.csv"), index=False)

        # richer panel (handles single-class ROC gracefully)
        plot_paired_panel(
            y_all, p_all, per_subj_df,
            os.path.join(args.outdir, "roc_pr.png"),
            title=f"Paired Transfer F→O ({args.rank})"
        )

def plot_paired_panel(y, p, per_subj_df, out_png, title=""):
    """
    پانل 2×2:
      1) ROC (اگر دوکلاسه نبود، پیام نشان می‌دهد)
      2) PR (همیشه تلاش می‌کنیم)
      3) هیستوگرام امتیازها جدا برای کلاس‌ها
      4) دات‌پلات per-subject (میانگین احتمال + رنگ با y_true)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    y = np.asarray(y).astype(int)
    two_class = (len(np.unique(y)) >= 2)

    roc_res, pr_res = safe_roc_pr(y, p)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (1) ROC
    ax = axes[0,0]
    if two_class and roc_res["fpr"] is not None:
        ax.plot(roc_res["fpr"], roc_res["tpr"], lw=2)
        ax.plot([0,1],[0,1],'--',lw=1,color="#888")
        ax.set_title(f"ROC (AUC={roc_res['auc']:.3f})")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    else:
        ax.axis("off"); ax.text(0.5,0.5,"ROC n/a (single-class test)",ha="center",va="center")

    # (2) PR
    ax = axes[0,1]
    if pr_res["precision"] is not None:
        ax.plot(pr_res["recall"], pr_res["precision"], lw=2)
        ap = pr_res["ap"]; ap_txt = "n/a" if not np.isfinite(ap) else f"{ap:.3f}"
        ax.set_title(f"PR (AP={ap_txt})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    else:
        ax.axis("off"); ax.text(0.5,0.5,"PR n/a",ha="center",va="center")

    # (3) توزیع احتمال‌ها
    ax = axes[1,0]
    if np.any(y==0):
        ax.hist(np.asarray(p)[y==0], bins=20, alpha=0.6, label="True 0")
    if np.any(y==1):
        ax.hist(np.asarray(p)[y==1], bins=20, alpha=0.6, label="True 1")
    ax.set_xlabel("Predicted probability (class=1)")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution by true label")
    ax.legend(frameon=False)

    # (4) دات‌پلات per-subject
    ax = axes[1,1]
    try:
        tmp = per_subj_df.copy()
        # اگر per_subject.csv فقط acc/params دارد، از preds.csv بهتره؛
        # ولی اینجا از per_subj_df استفاده می‌کنیم: acc را به‌صورت نقطه نشان بده
        ax.scatter(range(len(tmp)), tmp["acc"], c=np.where(tmp["y_pos_test"]>0,"tab:red","tab:blue"))
        ax.set_xticks([])
        ax.set_ylim(-0.05,1.05)
        ax.set_ylabel("Per-subject accuracy")
        ax.set_title("Per-subject accuracy (red: has positives)")
    except Exception:
        ax.axis("off"); ax.text(0.5,0.5,"per-subject view n/a",ha="center",va="center")

    if title: fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def run_learningcurve(args):
    ensure_dir(args.outdir)
    pct = read_pct_all(args.pct_all)
    meta = attach_subject_id(read_meta(args.meta), read_pairs_as_subjects(args.pairs))
    meta = meta[meta["sample_id"].isin(pct.index)]
    site = getattr(args,"site","fecal")
    X,y,g,_ = assemble_within(pct, meta, site)
    if len(y)<5 or not has_two_classes(y):
        write_skip_placeholder(args.outdir,"learning_curve.csv","single_class_or_too_few_samples"); return

    base,_ = build_model(args.model, args.random_state)
    splitter = make_splitter(y, g if args.group_col else None, args.folds)

    from sklearn.model_selection import learning_curve as sk_lc
    sizes, tr_sc, te_sc = sk_lc(estimator=base, X=X, y=y,
                                train_sizes=np.linspace(0.3,1.0,5),
                                cv=splitter, scoring="roc_auc",
                                n_jobs=-1, groups=g if args.group_col else None)
    df = pd.DataFrame({
        "train_size": sizes,
        "test_auc_mean": te_sc.mean(axis=1), "test_auc_sd": te_sc.std(axis=1),
        "train_auc_mean": tr_sc.mean(axis=1), "train_auc_sd": tr_sc.std(axis=1)
    })
    df.to_csv(os.path.join(args.outdir,"learning_curve.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.8,4.4))
    ax.plot(df["train_size"], df["train_auc_mean"], marker="o", label="Train AUC")
    ax.fill_between(df["train_size"], df["train_auc_mean"]-df["train_auc_sd"],
                    df["train_auc_mean"]+df["train_auc_sd"], alpha=0.2)
    ax.plot(df["train_size"], df["test_auc_mean"], marker="s", label="CV AUC")
    ax.fill_between(df["train_size"], df["test_auc_mean"]-df["test_auc_sd"],
                    df["test_auc_mean"]+df["test_auc_sd"], alpha=0.2)
    ax.set_xlabel("Train size"); ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Learning curve — {site.capitalize()} ({args.rank})")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(args.outdir,"learning_curve.png"), dpi=180); plt.close(fig)

# ------------------------------- CLI --------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="ML runner for microbiome pipeline")
    sub = ap.add_subparsers(dest="mode", required=True)

    def add_shared(a):
        a.add_argument("--pct-all", required=True)
        a.add_argument("--meta", required=True)
        a.add_argument("--rank", required=True, choices=["genus", "species"])
        a.add_argument("--model", default="logit_enet")
        a.add_argument("--random-state", type=int, default=42)
        a.add_argument("--outdir", required=True)
        a.add_argument("--pairs", default=None)
        a.add_argument("--group-col", default="subject_id")
        a.add_argument("--folds", type=int, default=5)
        a.add_argument("--inner-folds", type=int, default=3)
        return a

    # within
    w = add_shared(sub.add_parser("within"))
    w.add_argument("--site", required=True, choices=["oral", "fecal"])

    # paired transfer
    add_shared(sub.add_parser("paired_transfer"))

    # transfer
    t = add_shared(sub.add_parser("transfer"))
    t.add_argument("--train-site", required=True, choices=["oral", "fecal"])
    t.add_argument("--test-site", required=True, choices=["oral", "fecal"])
    t.add_argument("--allow-shared-subjects", action="store_true")
    t.add_argument(
        "--disjoint-drop",
        choices=["train", "test"],
        default="train",
        help="If subjects are shared across sites, drop them from this side (default: train).",
    )
    t.add_argument(
        "--feature-mode",
        choices=["union_zero", "intersection", "intersection_minfreq"],
        default="intersection",
    )
    t.add_argument("--min-freq", type=int, default=0)
    t.add_argument("--site-standardize", action="store_true")
    t.add_argument("--coral", action="store_true")

    # learning curve
    lc = add_shared(sub.add_parser("learningcurve"))
    lc.add_argument("--site", default="fecal", choices=["oral", "fecal"])

    return ap.parse_args()


def main():
    args = parse_args()
    if hasattr(args,"site"): args.site = args.site.lower()
    if hasattr(args,"train_site"): args.train_site = args.train_site.lower()
    if hasattr(args,"test_site"): args.test_site = args.test_site.lower()

    if args.mode == "within":            run_within(args)
    elif args.mode == "transfer":        run_transfer(args)
    elif args.mode == "paired_transfer": run_paired_transfer(args)
    elif args.mode == "learningcurve":   run_learningcurve(args)
    else: raise SystemExit(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
