#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-taxon PPI & interactions (CLR; OLS or MixedLM) — clean
----------------------------------------------------------
- Volcano plots: ALL (PPI), Oral (PPI), Fecal (PPI), disease:PPI, disease:site
- Delta analysis: ONLY writes ppi_effects_delta.csv (no delta volcano)
"""

from __future__ import annotations
import os, argparse, warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

# ------------------------- small utils -------------------------

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def bh_qvalues(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    # NaNs -> +inf (بی‌اثر در رتبه‌بندی)
    p = np.where(np.isnan(p), np.inf, p)
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i, idx in enumerate(order[::-1], start=1):
        rank = n - i + 1
        val = min(prev, (p[idx] * n) / rank)
        q[idx] = val
        prev = val
    q = np.clip(q, 0, 1)
    q[np.isinf(p)] = np.nan
    return q

def clr_transform(pct_taxa_by_samples: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    X = (pct_taxa_by_samples.astype(float) / 100.0) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)
    return logX.sub(gm, axis=1)

def prettify_taxon(label: str, rank: str) -> str:
    if label is None:
        return None
    r = (rank or "").lower()
    if r == "genus" and label.startswith("g__"):
        name = label.replace("g__", "").replace("_", " ")
        return " ".join(w.capitalize() for w in name.split())
    if r == "species" and label.startswith("s__"):
        name = label.replace("s__", "").replace("_", " ")
        toks = name.split()
        if toks:
            toks[0] = toks[0].capitalize()
            toks[1:] = [t.lower() for t in toks[1:]]
            return " ".join(toks)
    return label

def save_placeholder_png(out_png: str, title: str) -> None:
    ensure_dir(os.path.dirname(out_png))
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.42, "No data or no testable features.", ha="center", va="center", fontsize=10)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

def volcano(df, x, sig_col, name_col, out_png, title,
            alpha=0.1, k_onplot_each=15, metric_label="q", near_min=5):
    ensure_dir(os.path.dirname(out_png))
    if df is None or df.empty or x not in df or sig_col not in df:
        save_placeholder_png(out_png, title); return

    X = pd.to_numeric(df[x], errors="coerce").values
    S = pd.to_numeric(df[sig_col], errors="coerce").clip(lower=1e-300).values
    Y = -np.log10(S)
    names = df[name_col].astype(str).values if name_col in df else df.index.astype(str).values

    is_sig = S <= alpha
    score = np.abs(X) * (Y + 1)

    up_idx = np.where(is_sig & (X > 0))[0]
    dn_idx = np.where(is_sig & (X < 0))[0]
    up_sel = up_idx[np.argsort(-score[up_idx])[:k_onplot_each]]
    dn_sel = dn_idx[np.argsort(-score[dn_idx])[:k_onplot_each]]
    label_idxs = list(np.concatenate([up_sel, dn_sel]))

    if len(label_idxs) < near_min:
        target = -np.log10(alpha)
        dist = np.abs(Y - target)
        candidates = np.setdiff1d(np.arange(len(X)), label_idxs, assume_unique=False)
        order = np.lexsort((-np.abs(X[candidates]), dist[candidates]))
        extra = candidates[order][: (near_min - len(label_idxs))]
        label_idxs = list(label_idxs) + list(extra)

    fig = plt.figure(figsize=(10.5, 6.2)); ax = fig.add_subplot(111)
    ax.scatter(X[~is_sig], Y[~is_sig], s=22, alpha=0.35, color="#999999", zorder=1)
    ax.scatter(X[is_sig & (X > 0)], Y[is_sig & (X > 0)], s=24, alpha=0.9, color="#1B4F72", zorder=2)
    ax.scatter(X[is_sig & (X < 0)], Y[is_sig & (X < 0)], s=24, alpha=0.9, color="#7FB3D5", zorder=2)
    ax.axhline(-np.log10(alpha), ls="--", lw=1, color="gray", alpha=0.8)
    ax.axvline(0, ls="--", lw=1, color="gray", alpha=0.8)
    ax.grid(True, ls=":", lw=0.6, alpha=0.4)
    ax.set_title(title); ax.set_ylabel(f"-log10({metric_label})")
    for i in label_idxs:
        ax.text(X[i], Y[i], names[i], fontsize=9, ha="left", va="bottom",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=3)
    plt.tight_layout(); fig.savefig(out_png, dpi=320, bbox_inches="tight"); plt.close(fig)

# ------------------------- metadata helpers -------------------------

def norm_cols_meta(meta: pd.DataFrame) -> pd.DataFrame:
    m = meta.copy()
    m.columns = [c.strip() for c in m.columns]
    lower_map = {c.lower(): c for c in m.columns}
    aliases = {
        "sample_id": ["sample_id","id","sid","sampleid","sample"],
        "subject_id": ["subject_id","subject","participant","patient","pair_id"],
        "site": ["site","body_site","location"],
        "disease": ["disease","status","group","label"],
        "age": ["age"], "sex": ["sex","gender"], "bmi": ["bmi"],
        "ppi_use": ["ppi_use","ppi","ppi3m","ppi_current"],
        "antibiotics_3m": ["antibiotics_3m","abx_3m","antibiotics_last3m","abx3m"],
        "smoking": ["smoking","smoker"]
    }
    out = {}
    for canon, keys in aliases.items():
        found = next((lower_map[k] for k in keys if k in lower_map), None)
        if found is not None:
            out[canon] = m[found]
    cols = [k for k in ["sample_id","subject_id","site","disease","age","sex","bmi",
                        "ppi_use","antibiotics_3m","smoking"] if k in out]
    return pd.DataFrame({k: out[k] for k in cols})

def to_int01(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"1","true","yes","y","crohn","cd","case","ibd"}: return 1
    if s in {"0","false","no","n","healthy","control","hc","non-ibd","nonibd"}: return 0
    try:
        v = int(float(s))
        if v in (0,1): return v
    except Exception:
        pass
    return np.nan

def coerce_covariates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "disease" in out: out["disease"] = out["disease"].map(to_int01)
    if "ppi_use" in out: out["ppi_use"] = out["ppi_use"].map(to_int01)
    if "antibiotics_3m" in out: out["antibiotics_3m"] = out["antibiotics_3m"].map(to_int01)
    if "smoking" in out: out["smoking"] = out["smoking"].map(to_int01)

    if "site" in out:
        out["site"] = (
            out["site"].astype(str).str.strip().str.capitalize().replace({"Faecal":"Fecal"})
        )
        out.loc[~out["site"].isin(["Oral","Fecal"]), "site"] = np.nan

    for c in ["age","bmi"]:
        if c in out: out[c] = pd.to_numeric(out[c], errors="coerce")

    if "sex" in out:
        out["sex"] = out["sex"].astype(str).str.strip().str.capitalize()

    return out

# ------------------------- modeling core -------------------------

def _fit_model(formula: str, df: pd.DataFrame, use_mixed: bool, group_col: str | None):
    if use_mixed and group_col and group_col in df and df[group_col].notna().any():
        try:
            md = MixedLM.from_formula(formula, groups=df[group_col], data=df)
            return md.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        except Exception as e:
            warnings.warn(f"MixedLM failed with {e}; falling back to OLS.")
    return smf.ols(formula=formula, data=df).fit()

def fit_per_taxon(long_df: pd.DataFrame, formula: str, rank: str, model_label: str,
                  min_n: int = 8, use_mixed: bool = False, group_col: str | None = None) -> pd.DataFrame:
    rows = []
    for tax, d in long_df.groupby("taxon", sort=False):
        dd = d.dropna(subset=["CLR"])
        must_have = [c for c in ["disease","ppi_use","site"] if (c in dd.columns) and (c in formula)]
        dd = dd.dropna(subset=must_have) if must_have else dd
        if dd.shape[0] < min_n:
            continue
        try:
            m = _fit_model(formula, dd, use_mixed=use_mixed, group_col=group_col)
            for term, beta in m.params.items():
                if term == "Intercept":
                    continue
                rows.append({
                    "taxon": tax,
                    "pretty_taxon": prettify_taxon(tax, rank),
                    "model": model_label,
                    "term": term,
                    "beta": float(beta),
                    "p": float(m.pvalues.get(term, np.nan)),
                    "n": int(dd.shape[0]),
                })
        except Exception:
            rows.append({
                "taxon": tax, "pretty_taxon": prettify_taxon(tax, rank),
                "model": model_label, "term": "__fit_failed__", "beta": np.nan,
                "p": np.nan, "n": int(dd.shape[0]),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = out.groupby("model")["p"].transform(lambda s: bh_qvalues(s.values))
    return out

# ------------------------- delta utils (NO volcano) -------------------------

def load_or_build_deltas(deltas_csv: str | None,
                         pct_all: pd.DataFrame | None,
                         pairs_csv: str | None,
                         pseudocount: float = 1e-6) -> pd.DataFrame | None:
    # 1) provided deltas
    if deltas_csv and os.path.isfile(deltas_csv):
        D = pd.read_csv(deltas_csv)
        cols = {c.lower(): c for c in D.columns}
        subj = next((cols.get(k) for k in ["subject_id","pair_id","id","subject"] if k in cols), None)
        tax  = next((cols.get(k) for k in ["taxon","feature","name"] if k in cols), None)
        delt = next((cols.get(k) for k in ["delta_clr","delta","fc_minus_oc","fecal_minus_oral"] if k in cols), None)
        if subj and tax and delt:
            out = D[[subj, tax, delt]].rename(columns={subj:"subject_id", tax:"taxon", delt:"delta_clr"})
            out["subject_id"] = out["subject_id"].astype(str)
            out["delta_clr"] = pd.to_numeric(out["delta_clr"], errors="coerce")
            return out.dropna(subset=["subject_id","taxon","delta_clr"])
        pair_col = cols.get("pair_id")
        if pair_col:
            D2 = D.copy()
            D2.columns = [str(c) for c in D2.columns]
            long = D2.melt(id_vars=[pair_col], var_name="taxon", value_name="delta_clr") \
                     .rename(columns={pair_col:"subject_id"})
            long["subject_id"] = long["subject_id"].astype(str)
            long["delta_clr"] = pd.to_numeric(long["delta_clr"], errors="coerce")
            return long.dropna(subset=["subject_id","taxon","delta_clr"])
        warnings.warn("Unrecognized deltas schema; attempting to compute from pct_all+pairs.")

    # 2) compute from pairs
    if (pct_all is not None) and pairs_csv and os.path.isfile(pairs_csv):
        pairs = pd.read_csv(pairs_csv)
        c = {x.lower(): x for x in pairs.columns}
        sid  = next((c.get(k) for k in ["subject_id","pair_id","id","subject"] if k in c), None)
        oral = next((c.get(k) for k in ["oral","oral_id","oc","sample_oral"] if k in c), None)
        fec  = next((c.get(k) for k in ["fecal","faecal","fecal_id","fc","sample_fecal"] if k in c), None)
        if not (sid and oral and fec):
            warnings.warn("Pairs CSV lacks required columns to compute deltas.")
            return None
        clr = clr_transform(pct_all, pseudocount=pseudocount)
        rows = []
        for _, r in pairs[[sid, oral, fec]].dropna().iterrows():
            subj_id = str(r[sid]); oc = str(r[oral]); fc = str(r[fec])
            if oc not in clr.columns or fc not in clr.columns: continue
            delta = clr[fc] - clr[oc]
            for taxon, dval in delta.items():
                rows.append({"subject_id": subj_id, "taxon": taxon, "delta_clr": float(dval)})
        if not rows: return None
        return pd.DataFrame(rows)

    return None

def fit_deltas_per_taxon(delta_long: pd.DataFrame,
                         meta: pd.DataFrame,
                         rank: str) -> pd.DataFrame:
    # فقط جدول ضرایب؛ بدون هیچ پلاطی
    m = meta.copy()
    if "subject_id" not in m and "sample_id" in m:
        m["subject_id"] = m["sample_id"]
    keep = ["subject_id","disease","ppi_use","age","sex","bmi","antibiotics_3m","smoking"]
    m = m[[c for c in keep if c in m]].drop_duplicates()
    m["subject_id"] = m["subject_id"].astype(str)

    long = delta_long.merge(m, on="subject_id", how="left")
    covs = [c for c in ["ppi_use","disease","age","sex","bmi","antibiotics_3m","smoking"] if c in long.columns]
    formula = "delta_clr ~ " + (" + ".join(covs) if covs else "1")

    rows = []
    for tax, d in long.groupby("taxon", sort=False):
        dd = d.dropna(subset=["delta_clr"])
        need = [c for c in ["ppi_use","disease"] if c in dd.columns and (c in formula)]
        dd = dd.dropna(subset=need) if need else dd
        if dd.shape[0] < 8:
            continue
        try:
            mod = smf.ols(formula=formula, data=dd).fit()
            for term, beta in mod.params.items():
                if term == "Intercept": 
                    continue
                rows.append({
                    "taxon": tax, "pretty_taxon": prettify_taxon(tax, rank),
                    "model": "DELTA", "term": term, "beta": float(beta),
                    "p": float(mod.pvalues.get(term, np.nan)), "n": int(dd.shape[0]),
                })
        except Exception:
            rows.append({
                "taxon": tax, "pretty_taxon": prettify_taxon(tax, rank),
                "model": "DELTA", "term": "__fit_failed__", "beta": np.nan,
                "p": np.nan, "n": int(dd.shape[0]),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["taxon","pretty_taxon","model","term","beta","p","q","n"])
    out["q"] = out.groupby("model")["p"].transform(lambda s: bh_qvalues(s.values))
    return out

# ------------------------- CLI -------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Per-taxon PPI & interactions (volcano except delta).")
    ap.add_argument("--pct-all", required=True)
    ap.add_argument("--meta",     required=True)
    ap.add_argument("--rank",     required=True, choices=["genus","species"])
    ap.add_argument("--outdir",   required=True)
    ap.add_argument("--pairs",    required=False, default=None)
    ap.add_argument("--deltas",   required=False, default=None)
    ap.add_argument("--heat-topk", type=int, default=15)
    ap.add_argument("--volcano-metric", choices=["q","p"], default="q")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--use-mixed", action="store_true")
    ap.add_argument("--near-min", type=int, default=5)
    return ap.parse_args()

# ------------------------- main -------------------------

def main():
    args = parse_args()
    out_rank = os.path.join(args.outdir, args.rank); ensure_dir(out_rank)

    # 1) abundance → CLR
    pct = pd.read_csv(args.pct_all, index_col=0)
    pct.columns = pct.columns.astype(str)
    clr = clr_transform(pct, pseudocount=1e-6)
    clr_T = clr.T; clr_T.index.name = "Sample_ID"

    # 2) metadata
    meta_raw = pd.read_csv(args.meta)
    meta0 = norm_cols_meta(meta_raw)
    req = {"sample_id","site","disease"}
    miss = req - set(meta0.columns)
    if miss:
        raise ValueError(f"[meta] Missing required columns: {sorted(list(miss))}")
    meta = coerce_covariates(meta0)
    meta = meta.loc[meta["sample_id"].isin(clr.columns)].copy()
    if meta.empty:
        raise ValueError("[meta] No overlapping Sample_ID with pct_all columns.")

    # 3) long join
    long = clr_T.stack().reset_index()
    long.columns = ["Sample_ID","taxon","CLR"]
    long["Sample_ID"] = long["Sample_ID"].astype(str)
    mm = meta.rename(columns={"sample_id":"Sample_ID"})
    long = long.merge(mm, on="Sample_ID", how="inner")
    if long.empty:
        raise ValueError("No overlap between abundance and meta after join.")

    # 4) design counts
    design_rows = []
    for label, sub in [("ALL", long), ("Oral", long[long["site"]=="Oral"]), ("Fecal", long[long["site"]=="Fecal"])]:
        m = sub.dropna(subset=["disease","ppi_use"]) if "ppi_use" in sub.columns else sub
        design_rows.append({
            "scope": label,
            "n_rows": len(sub),
            "n_with_disease_ppi": len(m),
            "n_unique_samples": sub["Sample_ID"].nunique(),
            "n_crohn": int((sub.get("disease", pd.Series(dtype=float))==1).sum()),
            "n_healthy": int((sub.get("disease", pd.Series(dtype=float))==0).sum()),
            "n_ppi_yes": int((sub.get("ppi_use", pd.Series(dtype=float))==1).sum()),
            "n_ppi_no": int((sub.get("ppi_use", pd.Series(dtype=float))==0).sum()),
        })
    pd.DataFrame(design_rows).to_csv(os.path.join(out_rank, "ppi_design_counts.csv"), index=False)

    # 5) ALL model + volcanoes (غیر دلتا)
    rhs_parts = []
    if "disease" in long.columns: rhs_parts.append("disease")
    if "site"    in long.columns: rhs_parts.append("site")
    if "ppi_use" in long.columns: rhs_parts.append("ppi_use")
    if set(["disease","site"]).issubset(long.columns):    rhs_parts.append("disease:site")
    if set(["disease","ppi_use"]).issubset(long.columns): rhs_parts.append("disease:ppi_use")
    for c in ["age","sex","bmi","antibiotics_3m","smoking"]:
        if c in long.columns: rhs_parts.append(c)
    formula_all = "CLR ~ " + (" + ".join(rhs_parts) if rhs_parts else "1")

    eff_all = fit_per_taxon(long, formula=formula_all, rank=args.rank, model_label="ALL",
                            min_n=12, use_mixed=args.use_mixed, group_col="subject_id")
    eff_all.to_csv(os.path.join(out_rank, "ppi_effects_all.csv"), index=False)

    metric, alpha = args.volcano_metric, args.alpha
    sub = eff_all[eff_all["term"].astype(str).str.startswith("ppi_use", na=False)].copy()
    if not sub.empty: sub["q"] = bh_qvalues(sub["p"].values)
    volcano(sub, x="beta", sig_col=metric, name_col="pretty_taxon",
            out_png=os.path.join(out_rank, "volcano_ppi_all.png"),
            title="Volcano (ALL) — PPI effect",
            alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric, near_min=args.near_min)

    sub = eff_all[eff_all["term"].astype(str).str.startswith("disease:ppi_use", na=False)].copy()
    if not sub.empty: sub["q"] = bh_qvalues(sub["p"].values)
    volcano(sub, x="beta", sig_col=metric, name_col="pretty_taxon",
            out_png=os.path.join(out_rank, "volcano_int_disease_ppi.png"),
            title="Volcano (ALL) — disease:PPI interaction",
            alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric, near_min=args.near_min)

    sub = eff_all[eff_all["term"].astype(str).str.startswith("disease:site", na=False)].copy()
    if not sub.empty: sub["q"] = bh_qvalues(sub["p"].values)
    volcano(sub, x="beta", sig_col=metric, name_col="pretty_taxon",
            out_png=os.path.join(out_rank, "volcano_int_disease_site.png"),
            title="Volcano (ALL) — disease:site interaction",
            alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric, near_min=args.near_min)

    # 6) Site-specific volcanoes (Oral/Fecal)
    for site in ["Oral","Fecal"]:
        subdf = long[long["site"] == site].copy()
        out_csv = os.path.join(out_rank, f"ppi_effects_{site.lower()}.csv")
        out_png = os.path.join(out_rank, f"volcano_ppi_{site.lower()}.png")
        if subdf.empty or subdf["disease"].dropna().nunique() < 2:
            pd.DataFrame(columns=["taxon","pretty_taxon","model","term","beta","p","q","n"]).to_csv(out_csv, index=False)
            save_placeholder_png(out_png, f"Volcano ({site}) — PPI effect")
            continue
        covs = [t for t in ["disease","ppi_use","age","sex","bmi","antibiotics_3m","smoking"] if t in subdf.columns]
        formula_site = "CLR ~ " + (" + ".join(covs) if covs else "1")
        eff_site = fit_per_taxon(subdf, formula=formula_site, rank=args.rank, model_label=site,
                                 min_n=10, use_mixed=args.use_mixed, group_col="subject_id")
        eff_site.to_csv(out_csv, index=False)
        sub_v = eff_site[eff_site["term"].astype(str).str.startswith("ppi_use", na=False)].copy()
        if not sub_v.empty: sub_v["q"] = bh_qvalues(sub_v["p"].values)
        volcano(sub_v, x="beta", sig_col=metric, name_col="pretty_taxon",
                out_png=out_png, title=f"Volcano ({site}) — PPI effect",
                alpha=alpha, k_onplot_each=args.heat_topk, metric_label=metric, near_min=args.near_min)

    # 7) Delta (بدون ولکانو؛ فقط جدول ضرایب اگر ورودی/جفت‌ها موجود بود)
    delta_long = load_or_build_deltas(args.deltas, pct, args.pairs, pseudocount=1e-6)
    if (delta_long is not None) and (not delta_long.empty):
        eff_delta = fit_deltas_per_taxon(delta_long, meta, rank=args.rank)
        eff_delta.to_csv(os.path.join(out_rank, "ppi_effects_delta.csv"), index=False)
    else:
        pd.DataFrame(columns=["taxon","pretty_taxon","model","term","beta","p","q","n"]).to_csv(
            os.path.join(out_rank, "ppi_effects_delta.csv"), index=False
        )

    print("[INFO] PPI/interaction analysis done →", out_rank)

if __name__ == "__main__":
    main()
