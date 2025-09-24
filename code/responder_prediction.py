#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified responder prediction
- Levels: genus, species
- Modes: oral, fecal, combined
- Models: ElasticNet (logistic, saga) + RandomForest
- CV: StratifiedKFold, OOF probabilities
- Transforms: optional CLR (+ pseudocount), then StandardScaler
- Cohort: intersection (same STUDY_ID set for all modes) or union
- Outputs per (level/mode):
    * pr.png, roc.png (both models on one plot, + baseline on PR)
    * elasticnet_scores.csv, rf_scores.csv
    * oof_elasticnet.csv, oof_rf.csv
    * elasticnet_top_features.csv (coef & abs_coef, signed), rf_top_features.csv
    * elasticnet_decision_card.png, rf_decision_card.png
    * elasticnet_features_tornado.png, rf_features_panel.png
"""

import argparse
from pathlib import Path
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix
)

# ------------------------------- IO helpers -------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_csv(p: str) -> pd.DataFrame:
    return pd.read_csv(p, low_memory=False)

def detect_tax_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in ("clade_name","taxon","taxonomy","name","feature","#clade_name"):
            return c
    return df.columns[0]

def load_raw_matrix(path_csv: str) -> pd.DataFrame:
    df = read_csv(path_csv)
    tax = detect_tax_col(df)
    df = df.rename(columns={tax:"taxon"})
    val = [c for c in df.columns if c!="taxon"]
    # coerce numeric, NA->0
    for c in val:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # drop all-zero taxa
    df = df.loc[(df[val]!=0).any(axis=1)].reset_index(drop=True)
    return df

def prep_level(df_raw: pd.DataFrame, level: str) -> pd.DataFrame:
    tax = df_raw["taxon"].astype(str)
    if level == "species":
        m = tax.str.contains("s__")
        dd = df_raw.loc[m].copy()
        dd["taxon"] = dd["taxon"].str.extract(r"(s__[^|;]+)$", expand=False)
        return dd.set_index("taxon")
    # genus
    has_g = tax.str.contains("g__")
    # prefer pure genus rows (no s__)
    if (has_g & ~tax.str.contains("s__")).any():
        dd = df_raw.loc[has_g & ~tax.str.contains("s__")].copy()
        dd["taxon"] = dd["taxon"].str.extract(r"(g__[^|;]+)", expand=False)
        return dd.set_index("taxon")
    # aggregate to genus
    agg = df_raw.copy()
    agg["gen"] = agg["taxon"].str.extract(r"(g__[^|;]+)", expand=False)
    agg = agg.dropna(subset=["gen"])
    val = [c for c in agg.columns if c not in ("taxon","gen")]
    mat = agg.groupby("gen")[val].sum()
    mat.index.name = "taxon"
    return mat

# ------------------------------- Mapping / designs -------------------------------

def build_mapping(meta_fp: str, oral_cols, fecal_cols) -> pd.DataFrame:
    meta = read_csv(meta_fp)
    need = ["STUDY_ID","Responder","Oral_sample_ID","Fecal_sample_ID"]
    for c in need:
        if c not in meta.columns:
            raise RuntimeError(f"Metadata missing column: {c}")
    m = meta[need].copy()
    m["Responder"] = pd.to_numeric(m["Responder"], errors="coerce").fillna(0).astype(int)
    m["Oral_col_ok"]  = m["Oral_sample_ID"].astype(str).isin(list(oral_cols))
    m["Fecal_col_ok"] = m["Fecal_sample_ID"].astype(str).isin(list(fecal_cols))
    return m

def sample_set_for_cohort(mapping: pd.DataFrame, cohort: str) -> list[str]:
    oral_ids  = mapping.loc[mapping["Oral_col_ok"],  "STUDY_ID"].astype(str)
    fecal_ids = mapping.loc[mapping["Fecal_col_ok"], "STUDY_ID"].astype(str)
    if cohort.lower().startswith("inter"):
        ids = sorted(set(oral_ids) & set(fecal_ids))
    else:  # union/any
        ids = sorted(set(oral_ids) | set(fecal_ids))
    return ids

def design_for_mode(mapping: pd.DataFrame,
                    oral_mat: pd.DataFrame,
                    fecal_mat: pd.DataFrame,
                    mode: str,
                    cohort_ids: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns X (samples × features) and y aligned to STUDY_ID index.
    """
    m = mapping.copy()
    m = m[m["STUDY_ID"].astype(str).isin(cohort_ids)].copy()

    def site_block(mat, colname, prefix):
        ok = m[colname].astype(str).isin(mat.columns)
        sub = m.loc[ok]
        if not ok.any():
            return pd.DataFrame(index=m["STUDY_ID"].astype(str).values)
        X = mat[sub[colname].astype(str).tolist()].T
        X.index = sub["STUDY_ID"].astype(str).values
        X = X.rename(columns=lambda t: f"{prefix}|{t}")
        return X

    if mode=="oral":
        X = site_block(oral_mat, "Oral_sample_ID", "oral")
    elif mode=="fecal":
        X = site_block(fecal_mat, "Fecal_sample_ID", "fecal")
    elif mode=="combined":
        Xo = site_block(oral_mat, "Oral_sample_ID", "oral")
        Xf = site_block(fecal_mat, "Fecal_sample_ID", "fecal")
        # union of indices within cohort_ids
        all_ids = pd.Index(cohort_ids, dtype=str)
        X = pd.DataFrame(index=all_ids)
        X = X.join(Xo, how="left").join(Xf, how="left").fillna(0.0)
    else:
        raise ValueError(mode)

    # y on STUDY_ID index
    y = m.set_index("STUDY_ID")["Responder"].reindex(X.index).astype(int)
    # drop all-zero columns
    X = X.loc[:, (X != 0).any(axis=0)]
    return X.astype(float), y

# ------------------------------- Transforms & filters -------------------------------

def filter_prevalence(X: pd.DataFrame, min_prev: float) -> pd.DataFrame:
    X = X.loc[:, (X != 0).any(axis=0)]
    if min_prev <= 0:
        return X
    prev = (X > 0).mean(axis=0)
    keep = prev >= min_prev
    return X.loc[:, keep]

def clr_transform(X: pd.DataFrame, pseudo: float) -> pd.DataFrame:
    if X.shape[1] == 0:
        return X.copy()
    V = X.clip(lower=0).astype(float)
    rsum = V.sum(axis=1).replace(0, np.nan)
    V = V.div(rsum, axis=0)
    V = np.log(V + pseudo)
    gm = V.mean(axis=1)
    V = V.sub(gm, axis=0)
    return V.fillna(0.0)

# ------------------------------- Plot helpers -------------------------------

def plot_roc_pr(y, probs_dict, title, outdir, baseline=None):
    # ROC
    plt.figure(figsize=(6,5))
    for name, p in probs_dict.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"--",alpha=.5)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "roc.png", dpi=130); plt.close()

    # PR
    plt.figure(figsize=(6,5))
    for name, p in probs_dict.items():
        rc, pr, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        plt.plot(rc, pr, label=f"{name} (AP={ap:.3f})")
    base = y.mean() if baseline is None else baseline
    plt.hlines(base, xmin=0, xmax=1, linestyles="--", label=f"baseline={base:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "pr.png", dpi=130); plt.close()

def decision_card(y, p, model_name, title, out_png):
    rc, pr, thr = precision_recall_curve(y, p)
    rc_thr, pr_thr = rc[:-1], pr[:-1]
    f1 = (2*pr_thr*rc_thr) / (pr_thr + rc_thr + 1e-12)
    k = int(np.nanargmax(f1))
    t_star = thr[k]; rc_star = rc_thr[k]; pr_star = pr_thr[k]; f1_star = f1[k]
    auc = np.nan
    try: auc = roc_auc_score(y, p)
    except Exception: pass
    ap = average_precision_score(y, p)
    yhat = (p >= t_star).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    prec_star = tp / (tp+fp+1e-12); rec_star = tp / (tp+fn+1e-12)

    fig = plt.figure(figsize=(9,4.8))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.25)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(rc, pr, lw=2, label=f"PR (AP={ap:.3f})")
    ax1.hlines(y.mean(), 0, 1, linestyles="--", label=f"baseline={y.mean():.2f}")
    ax1.plot([rc_star], [pr_star], marker="o", ms=7, label=f"F1-max @ t={t_star:.3f}")
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
    ax1.set_title(f"{title} — {model_name} | AUC={auc:.3f}")
    ax1.legend(loc="lower left")

    ax2 = fig.add_subplot(gs[0,1])
    cm = np.array([[tn, fp],[fn, tp]])
    im = ax2.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, int(cm[i,j]), ha="center", va="center")
    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    ax2.set_xticklabels(["Pred 0","Pred 1"]); ax2.set_yticklabels(["True 0","True 1"])
    ax2.set_title(f"@ t={t_star:.3f} | Prec={prec_star:.2f}, Rec={rec_star:.2f}, F1={f1_star:.2f}")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)

def tornado_signed(en_signed: pd.Series, top: int, title: str, out_png: Path):
    s = en_signed.sort_values()
    # pick symmetric top
    left = s.head(top//2)
    right = s.tail(top - len(left))
    s = pd.concat([left, right])
    fig, ax = plt.subplots(figsize=(6, max(4, 0.3*len(s))))
    y = np.arange(len(s))
    ax.barh(y, s.values)
    ax.set_yticks(y); ax.set_yticklabels(s.index, fontsize=8)
    ax.axvline(0, ls="--", lw=1)
    ax.set_xlabel("ElasticNet coefficient (signed)")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)

def bar_importance(imp: pd.Series, top: int, title: str, out_png: Path):
    s = imp.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.3*len(s))))
    y = np.arange(len(s))
    ax.barh(y, s.values)
    ax.set_yticks(y); ax.set_yticklabels(s.index, fontsize=8)
    ax.set_xlabel("RF importance")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)

# ------------------------------- ML core -------------------------------

def run_block(X_raw: pd.DataFrame,
              y: pd.Series,
              level: str, mode: str,
              outdir: Path,
              use_clr: bool, pseudo: float,
              splits: int, seed: int,
              rf_trees: int, enet_C: float, l1_ratio: float):
    """
    Fit EN & RF with CV, write all outputs for this (level, mode).
    """
    outdir = ensure_dir(outdir)

    # guard
    n_pos = int(y.sum()); n_neg = int((1-y).sum())
    if X_raw.shape[1] == 0 or len(np.unique(y)) < 2:
        (outdir / "EMPTY_DESIGN.txt").write_text(
            f"Empty or single-class design: X={X_raw.shape}, y={y.unique()}\n"
        )
        return

    # transform
    X = X_raw.copy()
    if use_clr:
        X = clr_transform(X, pseudo=pseudo)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X.values)

    # models
    en = LogisticRegression(
        penalty="elasticnet", solver="saga",
        l1_ratio=l1_ratio, C=enet_C,
        class_weight="balanced", max_iter=5000
    )
    rf = RandomForestClassifier(
        n_estimators=rf_trees, random_state=seed,
        class_weight="balanced", n_jobs=1
    )
    models = [("elasticnet", en), ("rf", rf)]

    # folds
    n_splits = min(splits, int(y.value_counts().min()))
    if n_splits < 2: n_splits = 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    probs = {}
    for name, model in models:
        oof = np.zeros(len(y), dtype=float)
        idx = np.arange(len(y))
        for tr, te in skf.split(Xs, y.values):
            model.fit(Xs[tr], y.values[tr])
            oof[te] = model.predict_proba(Xs[te])[:, 1]
        probs[name] = oof

        # scores
        auc = roc_auc_score(y, oof)
        ap  = average_precision_score(y, oof)
        pd.DataFrame({"ROC_AUC":[auc], "PR_AUC":[ap]}).to_csv(outdir / f"{name}_scores.csv", index=False)

        # save OOF with labels (and STUDY_ID if index name looks like it)
        oof_df = pd.DataFrame({
            "STUDY_ID": y.index,
            "y": y.values,
            "prob": oof
        })
        oof_df.to_csv(outdir / f"oof_{name}.csv", index=False)

    # combined plots
    title = f"{level} — {mode} (n={len(y)}, pos={int(y.sum())})"
    plot_roc_pr(y.values, probs, title, outdir)

    # decision cards
    for name, p in probs.items():
        decision_card(y.values, p, name, title, outdir / f"{name}_decision_card.png")

    # top features (refit on full transformed data)
    # EN (signed)
    en.fit(Xs, y.values)
    coef = pd.Series(en.coef_.ravel(), index=X.columns, name="coef")
    top_en = coef.abs().sort_values(ascending=False)
    pd.DataFrame({
        "coef": coef.loc[top_en.index],
        "abs_coef": top_en
    }).head(200).to_csv(outdir / "elasticnet_top_features.csv")

    tornado_signed(coef, top=20,
                   title=f"{level} — {mode} — ElasticNet signed coefficients",
                   out_png=outdir / "elasticnet_features_tornado.png")

    # RF
    rf.fit(Xs, y.values)
    imp = pd.Series(rf.feature_importances_, index=X.columns, name="importance").sort_values(ascending=False)
    imp.head(200).to_csv(outdir / "rf_top_features.csv")
    bar_importance(imp, top=20,
                   title=f"{level} — {mode} — RF top features",
                   out_png=outdir / "rf_features_panel.png")

    # debug
    dbg = [
        f"n={len(y)}, pos={int(y.sum())}, neg={int((1-y).sum())}",
        f"prevalence={y.mean():.3f}",
        f"X shape: {X_raw.shape}",
        f"use_clr={use_clr}, pseudo={pseudo}",
    ]
    (outdir / "debug.txt").write_text("\n".join(dbg) + "\n")

# ------------------------------- Runner -------------------------------

def run_level(level: str,
              meta_fp: str, oral_fp: str, fecal_fp: str,
              outroot: Path,
              min_prev: float, use_clr: bool, pseudo: float,
              cohort: str, seed: int, splits: int,
              rf_trees: int, enet_C: float, l1_ratio: float):
    outroot = ensure_dir(outroot / level)

    oral_raw  = load_raw_matrix(oral_fp)
    fecal_raw = load_raw_matrix(fecal_fp)
    oral_mat  = prep_level(oral_raw, level)     # taxa × Sxxxx
    fecal_mat = prep_level(fecal_raw, level)    # taxa × Sxxxx

    mapping = build_mapping(meta_fp, oral_mat.columns, fecal_mat.columns)
    cohort_ids = sample_set_for_cohort(mapping, cohort)
    if len(cohort_ids) == 0:
        (outroot / "EMPTY_LEVEL.txt").write_text(f"No patients for cohort={cohort}\n"); return

    modes = ["combined", "fecal", "oral"]  # order for plots
    for mode in modes:
        X, y = design_for_mode(mapping, oral_mat, fecal_mat, mode, cohort_ids)
        X = filter_prevalence(X, min_prev)
        modir = ensure_dir(outroot / mode)
        if X.shape[0]==0 or X.shape[1]==0 or len(np.unique(y))<2:
            (modir / "EMPTY_DESIGN.txt").write_text(
                f"Empty or single-class design: X={X.shape}, unique(y)={np.unique(y)}\n"
            )
            continue

        run_block(X, y, level, mode, modir,
                  use_clr, pseudo, splits, seed,
                  rf_trees, enet_C, l1_ratio)

def main():
    ap = argparse.ArgumentParser(description="Unified responder prediction (genus/species × oral/fecal/combined)")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--oral",      required=True)
    ap.add_argument("--fecal",     required=True)
    ap.add_argument("--outdir",    required=True)

    ap.add_argument("--min-prevalence", type=float, default=0.00)
    ap.add_argument("--use-clr",        type=int,   default=1)
    ap.add_argument("--pseudocount",    type=float, default=1e-6)
    ap.add_argument("--cohort",         type=str,   default="intersection", choices=["intersection","union"])

    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--splits",  type=int,   default=5)
    ap.add_argument("--rf-trees",type=int,   default=1000)
    ap.add_argument("--enet-C", "--enet_C", dest="enet_C", type=float, default=1.0)
    ap.add_argument("--l1-ratio",type=float, default=0.5)

    args = ap.parse_args()

    outroot = ensure_dir(Path(args.outdir))

    for level in ["genus","species"]:
        run_level(level,
                  args.metadata, args.oral, args.fecal, outroot,
                  min_prev=args.min_prevalence,
                  use_clr=bool(args.use_clr),
                  pseudo=args.pseudocount,
                  cohort=args.cohort,
                  seed=args.seed,
                  splits=args.splits,
                  rf_trees=args.rf_trees,
                  enet_C=args.enet_C,
                  l1_ratio=args.l1_ratio)

if __name__ == "__main__":
    main()

