#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Microbial Ratio Methods: Best Single Ratio (BSR) + Pairwise Log-Ratio L1 (PLR-L1)
- Sites: oral, fecal
- Levels: genus, species (collapse from rownames like "g__...|s__...")
- Baselines: Elastic-Net, RF (abundance-only inside this script for fair ratio comparison)
- Outputs:
  - AUROC/AP for baselines and ratio methods
  - ROC/PR plots per method
  - Ratio coefficients (PLR-L1) + aggregated taxa importance
  - Selected best single ratios per outer fold
  - summary comparison table at results/ml-ratios/comparison.csv
"""

import os, re, math, argparse, warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(invalid="ignore", divide="ignore")

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- I/O helpers ----------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def safe_read_csv(p: Optional[str]) -> Optional[pd.DataFrame]:
    if p and os.path.exists(p):
        try: return pd.read_csv(p, sep=None, engine="python")
        except Exception: return pd.read_csv(p)
    return None

def write_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path)); df.to_csv(path, index=False)

# ---------- IDs & taxonomy ----------
def normalize_id(x: str) -> str:
    if pd.isna(x): return x
    s = str(x).strip()
    s = re.sub(r"\.\d+$", "", s)
    s = re.sub(r"[^A-Za-z0-9]", "", s)
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", s)
    if m:
        pref, num = m.groups()
        num = re.sub(r"^0+", "", num) or "0"
        s = f"{pref}{num}"
    elif re.fullmatch(r"\d+", s):
        s = re.sub(r"^0+", "", s) or "0"
    return s.upper()

def extract_genus(rowname: str):
    parts = str(rowname).split("|")
    g = [p for p in parts if p.startswith("g__")]
    return g[0] if g else None

def extract_species(rowname: str):
    parts = str(rowname).split("|")
    s = [p for p in parts if p.startswith("s__")]
    return s[0] if s else None

def collapse_to_level(M_fxs: np.ndarray, feats: np.ndarray, level: str) -> Tuple[np.ndarray, np.ndarray]:
    if level not in {"genus","species"}: raise ValueError("level must be 'genus' or 'species'")
    tokens = [extract_genus(f) if level=="genus" else extract_species(f) for f in feats]
    df = pd.DataFrame(M_fxs); df.insert(0, "token", tokens)
    df = df[~df["token"].isna()].copy()
    if df.empty: return np.zeros((0, M_fxs.shape[1])), np.array([], dtype=object)
    g = df.groupby("token", as_index=False).sum(numeric_only=True)
    feats_new = g["token"].astype(str).values
    M_new = g.drop(columns=["token"]).to_numpy(dtype=float)
    return M_new, feats_new

# ---------- abundance reader + CLR/log ----------
def read_abund(path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    df = safe_read_csv(path)
    if df is None or df.shape[0]==0: return None
    first = df.columns[0]
    if (pd.isna(first) or first=="" or first=="...1" or re.fullmatch(r"(?i)(feature|taxon|clade|species|id|name)", str(first) or "")):
        feats = df.iloc[:,0].astype(str).values
        df = df.iloc[:,1:].copy()
    else:
        cand = [c for c in df.columns if re.search(r"(?i)feature|taxon|clade|name|id|species", c)]
        if cand:
            feats = df[cand[0]].astype(str).values; df = df.drop(columns=[cand[0]])
        else:
            feats = np.arange(1, df.shape[0]+1).astype(str)
    # numeric, NA->0, negatives->0
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)
        df[c] = v.values
    M = df.to_numpy(dtype=float)
    keep = (M.sum(axis=1) > 0)
    M = M[keep,:]; feats = np.array(pd.Series(feats)[keep].astype(str).values, dtype=object)
    samples = np.array([normalize_id(c) for c in df.columns.tolist()], dtype=object)
    return (M, feats, samples)

def clr_transform(X: np.ndarray, pseudo: float=1e-6) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0); X = np.clip(X, 0.0, None)
    L = np.log(X + float(pseudo))
    gm = L.mean(axis=1, keepdims=True)
    return L - gm

# ---------- plotting ----------
def plot_roc(y, proba, title, out):
    if len(np.unique(y))<2: return
    fpr, tpr, _ = roc_curve(y, proba); auc = roc_auc_score(y, proba)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    plt.legend(loc="lower right"); plt.tight_layout(); ensure_dir(os.path.dirname(out)); plt.savefig(out, dpi=200); plt.close()

def plot_pr(y, proba, title, out):
    if len(np.unique(y))<2: return
    prec, rec, _ = precision_recall_curve(y, proba); ap = average_precision_score(y, proba)
    plt.figure(); plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.legend(loc="lower left"); plt.tight_layout(); ensure_dir(os.path.dirname(out)); plt.savefig(out, dpi=200); plt.close()

# ---------- CV helpers ----------
def calibrate_and_oof(pipe, X, y, seed=13, inner_splits=3, outer_splits=5, outer_repeats=5, calibration="sigmoid", param_grid=None, fit_params=None):
    proba = np.zeros(len(y), dtype=float); aucs=[]
    rskf = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=seed)
    for tr, te in rskf.split(X, y):
        Xtr, Xte = X[tr], X[te]; ytr, yte = y[tr], y[te]
        gs = GridSearchCV(pipe, param_grid or {}, cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed),
                          n_jobs=-1, scoring="roc_auc", refit=True)
        gs.fit(Xtr, ytr, **(fit_params or {}))
        best = gs.best_estimator_
        cal = CalibratedClassifierCV(best, method=calibration, cv=3)
        cal.fit(Xtr, ytr)
        p = cal.predict_proba(Xte)[:,1]; proba[te] = p
        if len(np.unique(yte))==2: aucs.append(roc_auc_score(yte, p))
    return proba, (np.mean(aucs) if aucs else float("nan"))

# ---------- build design ----------
def build_site_level(oral_or_fecal: str, level: str, abund_path: str, covars_path: str, target: str,
                     min_prev_samples: int=10, use_clr_for_baseline=True, pseudo=1e-6):
    """Return X_abund (samples x feats), feats, y, sample_ids, logs for ratios"""
    out_logs = {}
    # read abundance
    r = read_abund(abund_path)
    if r is None: return None
    M_fxs, feats, samples = r  # feats x samples
    # collapse
    M_coll, feats_coll = collapse_to_level(M_fxs, feats, level)
    if M_coll.shape[0]==0: return None
    # metadata
    md = safe_read_csv(covars_path)
    if md is None or md.shape[0]==0: return None
    for req in ["Sample_ID","site","disease"]:
        if req not in md.columns: return None
    md = md.copy()
    md["Sample_ID"] = md["Sample_ID"].map(normalize_id)
    md["site"] = md["site"].astype(str).str.lower()
    md["disease"] = pd.to_numeric(md["disease"], errors="coerce")
    md_site = md[(md["disease"]==1) & (md["site"]==oral_or_fecal)].copy()
    if target not in md_site.columns: return None
    md_site[target] = pd.to_numeric(md_site[target], errors="coerce")
    # intersect
    common = np.intersect1d(samples, md_site["Sample_ID"].values)
    if common.size==0: return None
    md_site = md_site.set_index("Sample_ID").loc[common]
    y = md_site[target].astype("Int64").astype(float).to_numpy()
    mask = ~np.isnan(y); y = y[mask].astype(int); sample_ids = common[mask]
    # align abundance to sample order
    col_idx = [np.where(samples==sid)[0][0] for sid in sample_ids]
    X_ab = M_coll[:, col_idx].T    # samples x feats
    feats2 = feats_coll.copy()

    # prevalence filter (global)
    keep = (X_ab > 0).sum(axis=0) >= int(min_prev_samples)
    if keep.sum()==0: return None
    X_ab = X_ab[:, keep]; feats2 = feats2[keep]
    # CLR for baseline (optional)
    X_baseline = clr_transform(X_ab, pseudo=pseudo) if use_clr_for_baseline else X_ab.copy()
    # LOG for ratios
    X_log = np.log(X_ab + float(pseudo))

    out_logs["n_samples"] = X_ab.shape[0]
    out_logs["n_features_after_prev"] = X_ab.shape[1]
    return X_baseline, feats2, y, sample_ids, X_log

# ---------- candidate selection for ratios ----------
def select_top_taxa(X_log: np.ndarray, y: np.ndarray, feats: np.ndarray, top_m: int=30) -> np.ndarray:
    """Pick taxa with highest variance on log scale among those with reasonable prevalence (already filtered)."""
    if X_log.shape[1] <= top_m: return np.arange(X_log.shape[1])
    var = np.var(X_log, axis=0)
    order = np.argsort(var)[::-1][:top_m]
    return order

def build_all_pairwise_ratios(X_log: np.ndarray, feats: np.ndarray, idxs: np.ndarray):
    """Return R (samples x P), names, where R[:,k]=log(A_i)-log(A_j) for i<j in idxs."""
    sel_feats = feats[idxs]
    L = X_log[:, idxs]
    pairs = []
    mats = []
    n = len(idxs)
    for i in range(n):
        Li = L[:, i][:,None]
        for j in range(i+1, n):
            r = Li - L[:, j][:,None]   # (samples x 1)
            mats.append(r)
            pairs.append((sel_feats[i], sel_feats[j]))
    if not mats:
        return np.zeros((X_log.shape[0], 0)), []
    R = np.hstack(mats)  # samples x num_pairs
    names = [f"{a}/{b}" for (a,b) in pairs]
    return R, names

# ---------- methods ----------
def baseline_models(X_baseline: np.ndarray, y: np.ndarray, seed=13):
    results = []
    # Elastic-Net
    pipe_en = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                        ("clf", LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced",
                                                   max_iter=8000, n_jobs=-1, random_state=seed))])
    grid_en = {"clf__C":[0.1,1,10], "clf__l1_ratio":[0.2,0.5,0.8]}
    p_en, auc_en = calibrate_and_oof(pipe_en, X_baseline, y, seed=seed, param_grid=grid_en)
    results.append(("ElasticNet", p_en, auc_en))

    # RandomForest
    pipe_rf = Pipeline([("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                       random_state=seed, n_jobs=-1))])
    grid_rf = {"clf__max_depth":[None,5,10,15], "clf__max_features":["sqrt", 0.3, 0.5]}
    p_rf, auc_rf = calibrate_and_oof(pipe_rf, X_baseline, y, seed=seed, param_grid=grid_rf)
    results.append(("RF", p_rf, auc_rf))
    return results

def method_plr_l1(R: np.ndarray, y: np.ndarray, seed=13):
    """L1 logistic over all pairwise ratios."""
    if R.shape[1]==0:
        return np.zeros(len(y)), float("nan"), None
    pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                     ("clf", LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced",
                                                max_iter=5000, random_state=seed))])
    grid = {"clf__C":[0.1, 1, 10]}
    p, auc = calibrate_and_oof(pipe, R, y, seed=seed, param_grid=grid)
    # refit on full data to get coefficients
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
                      n_jobs=-1, scoring="roc_auc", refit=True)
    gs.fit(R, y)
    best = gs.best_estimator_
    coef = best.named_steps["clf"].coef_.ravel()
    return p, auc, coef

def method_best_single_ratio(X_log: np.ndarray, y: np.ndarray, pairs_matrix: np.ndarray, pair_names: List[str], seed=13):
    """Inner-CV selects best single ratio per outer fold; returns OOF probs and fold-wise chosen names."""
    if pairs_matrix.shape[1]==0:
        return np.zeros(len(y)), float("nan"), []
    oof = np.zeros(len(y), dtype=float); aucs = []; chosen = []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    for tr, te in rskf.split(X_log, y):
        Xtr = pairs_matrix[tr]; ytr = y[tr]
        Xte = pairs_matrix[te]; yte = y[te]
        # inner: pick best single ratio by AUROC
        best_auc, best_j = -1, None
        # quick scan on a subset if too many pairs
        cols = range(Xtr.shape[1])
        for j in cols:
            rtr = Xtr[:, j].reshape(-1,1)
            # logistic on single feature
            clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000, random_state=seed)
            try:
                clf.fit(rtr, ytr)
                pval = clf.predict_proba(rtr)[:,1]
                aucj = roc_auc_score(ytr, pval)
                if aucj > best_auc:
                    best_auc, best_j = aucj, j
            except Exception:
                continue
        if best_j is None:
            # fallback: mean zero
            oof[te] = 0.5; chosen.append("NA"); continue
        chosen.append(pair_names[best_j])
        # train on train fold with best single ratio, predict test fold
        clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(Xtr[:, best_j].reshape(-1,1), ytr)
        p = clf.predict_proba(Xte[:, best_j].reshape(-1,1))[:,1]
        oof[te] = p
        if len(np.unique(yte))==2:
            try: aucs.append(roc_auc_score(yte, p))
            except Exception: pass
    return oof, (np.mean(aucs) if aucs else float("nan")), chosen

# ---------- metrics ----------
def ap_score(y, p):
    try: return average_precision_score(y, p)
    except Exception: return float("nan")

def metrics_block(y, p):
    try:
        auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
    except Exception:
        auc, ap = float("nan"), float("nan")
    return auc, ap

# ---------- main orchestration ----------
def run_one(oral_or_fecal, level, abund_path, covars_path, args, comp_rows):
    out_dir = os.path.join(args.outdir, level, oral_or_fecal); ensure_dir(out_dir); ensure_dir(os.path.join(out_dir,"plots"))
    built = build_site_level(oral_or_fecal, level, abund_path, covars_path, args.target,
                             min_prev_samples=args.min_prev_samples, use_clr_for_baseline=True, pseudo=args.pseudo)
    if built is None:
        write_csv(pd.DataFrame({"note":[f"build_failed_{oral_or_fecal}_{level}"]}), os.path.join(out_dir, "sanity_checks.csv"))
        return
    X_base, feats, y, sids, X_log = built
    # save clean collapsed tables for deliverable
    df_clean = pd.DataFrame(X_base if not args.save_log else X_log, columns=feats)
    df_clean.insert(0, "Sample_ID", sids)
    write_csv(df_clean, os.path.join(out_dir, f"table_{'log' if args.save_log else 'clr'}_{level}.csv"))

    # ---------- baselines ----------
    base_results = baseline_models(X_base, y, seed=args.seed)
    for name, p, auc in base_results:
        auc_b, ap_b = metrics_block(y, p)
        comp_rows.append({"site":oral_or_fecal, "level":level, "method":name, "auroc":auc_b, "ap":ap_b})
        plot_roc(y, p, f"{oral_or_fecal.title()} — ROC ({level} {name})", os.path.join(out_dir, "plots", f"roc_{name}.png"))
        plot_pr (y, p, f"{oral_or_fecal.title()} — PR  ({level} {name})", os.path.join(out_dir, "plots", f"pr_{name}.png"))

    # ---------- candidate ratios ----------
    idxs = select_top_taxa(X_log, y, feats, top_m=args.top_m)
    R, pair_names = build_all_pairwise_ratios(X_log, feats, idxs)
    write_csv(pd.DataFrame({"pair":pair_names}), os.path.join(out_dir, "ratio_pairs_catalog.csv"))

    # ---------- PLR-L1 ----------
    p_plr, auc_plr, coef_plr = method_plr_l1(R, y, seed=args.seed)
    auc_b, ap_b = metrics_block(y, p_plr)
    comp_rows.append({"site":oral_or_fecal, "level":level, "method":"PLR-L1", "auroc":auc_b, "ap":ap_b})
    plot_roc(y, p_plr, f"{oral_or_fecal.title()} — ROC ({level} PLR-L1)", os.path.join(out_dir, "plots", f"roc_PLR_L1.png"))
    plot_pr (y, p_plr, f"{oral_or_fecal.title()} — PR  ({level} PLR-L1)",  os.path.join(out_dir, "plots", f"pr_PLR_L1.png"))

    # save coefficients & taxa importance
    if coef_plr is not None and len(pair_names)==len(coef_plr):
        dfc = pd.DataFrame({"pair":pair_names, "coef":coef_plr, "abscoef":np.abs(coef_plr)})
        dfc = dfc[dfc["coef"]!=0].sort_values("abscoef", ascending=False)
        write_csv(dfc, os.path.join(out_dir, "ratio_coefs_PLR_L1.csv"))
        # taxa importance (sum |coef| over pairs containing the taxon)
        taxa_imp = {}
        for pr, c in zip(pair_names, coef_plr):
            if c==0: continue
            a,b = pr.split("/",1)
            taxa_imp[a] = taxa_imp.get(a,0.0) + abs(c)
            taxa_imp[b] = taxa_imp.get(b,0.0) + abs(c)
        imp_df = pd.DataFrame({"taxon":list(taxa_imp.keys()), "importance":list(taxa_imp.values())}).sort_values("importance", ascending=False)
        write_csv(imp_df, os.path.join(out_dir, "taxa_importance_PLR_L1.csv"))

    # ---------- Best Single Ratio ----------
    p_bsr, auc_bsr, chosen = method_best_single_ratio(X_log, y, R, pair_names, seed=args.seed)
    auc_b, ap_b = metrics_block(y, p_bsr)
    comp_rows.append({"site":oral_or_fecal, "level":level, "method":"BestSingleRatio", "auroc":auc_b, "ap":ap_b})
    plot_roc(y, p_bsr, f"{oral_or_fecal.title()} — ROC ({level} BestSingleRatio)", os.path.join(out_dir, "plots", f"roc_BSR.png"))
    plot_pr (y, p_bsr, f"{oral_or_fecal.title()} — PR  ({level} BestSingleRatio)",  os.path.join(out_dir, "plots", f"pr_BSR.png"))
    # save chosen ratios per fold (frequency summary)
    if chosen:
        freq = pd.Series(chosen).value_counts().reset_index()
        freq.columns = ["pair","count"]
        write_csv(freq, os.path.join(out_dir, "best_single_ratio_frequencies.csv"))

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oral-crohn", required=True)
    ap.add_argument("--fecal-crohn", required=True)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--outdir", default="results/ml-ratios")
    ap.add_argument("--target", default="PPI_use")
    ap.add_argument("--min-prev-samples", type=int, default=10)
    ap.add_argument("--pseudo", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--top-m", type=int, default=30, help="num taxa for pairwise ratios (creates ~M*(M-1)/2 pairs)")
    ap.add_argument("--save-log", action="store_true", help="save per-sample log table instead of CLR (debug)")
    return ap

def main():
    args = build_argparser().parse_args()
    ensure_dir(args.outdir)
    comp_rows = []
    for level in ["genus","species"]:
        run_one("oral",  level, args.oral_crohn,  args.covariates, args, comp_rows)
        run_one("fecal", level, args.fecal_crohn, args.covariates, args, comp_rows)
    if comp_rows:
        comp = pd.DataFrame(comp_rows, columns=["site","level","method","auroc","ap"])
        write_csv(comp, os.path.join(args.outdir, "comparison.csv"))
    open(os.path.join(args.outdir, ".done"), "w").write("ok\n")
    print("[ratio_methods] Done")

if __name__ == "__main__":
    main()
