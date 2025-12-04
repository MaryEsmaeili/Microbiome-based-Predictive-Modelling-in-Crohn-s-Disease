# scripts/da_unified_module.py

"""
Unified differential-abundance module for the Crohn microbiome project.

This script runs all DA analyses on the species/genus-level MetaPhlAn tables
used in the thesis. It supports two main modes:

- UNADJUSTED tests:
  * Uses only the source-map (pct_all_sources.csv) for labels.
  * Performs site-specific Mann–Whitney U tests for Crohn vs Healthy in
    oral and fecal samples.
  * Writes per-taxon CSVs and volcano plots for unadjusted CH effects.

- ADJUSTED models:
  * Reads full metadata, normalizes column names and encodings, and
    merges with CLR-transformed abundance tables.
  * Fits per-taxon OLS models (via statsmodels) with disease, site,
    PPI_use and other covariates, plus interaction terms where supported.
  * Produces CSV files and volcano plots for:
      - disease effects (ALL, oral-only, fecal-only)
      - paired oral↔fecal Crohn deltas (Wilcoxon on FC–OC)
      - PPI main effects and disease×PPI interactions.

All outputs are written under --outdir/<rank>/ as:
  - da_*_unadj_<rank>.csv / volcano_*_unadj_<rank>.png   (unadjusted CH)
  - da_*_adj_<rank>.csv   / volcano_*_adj_<rank>.png     (adjusted disease)
  - da_paired_<rank>.csv  / volcano_paired_<rank>.png    (paired OC↔FC)
  - ppi_effects*.csv      / volcano_ppi*.png             (PPI-related effects)

This module is designed as the single source of truth for all DA and PPI
results reported in the thesis.
"""

from __future__ import annotations
import os, argparse, warnings
import numpy as np
import pandas as pd

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from scipy.stats import mannwhitneyu, wilcoxon
import statsmodels.formula.api as smf
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests

# ========================== Small utilities ==========================

def ensure_dir(d: str) -> None:
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_placeholder(out_png: str, title: str) -> None:
    """Render a neutral placeholder figure."""
    ensure_dir(os.path.dirname(out_png))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.42, "No data or no testable features.", ha="center", va="center", fontsize=10)
    fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig)

def bh_q(p):
    """Benjamini–Hochberg FDR."""
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n, float); prev = 1.0
    for i, idx in enumerate(order[::-1], start=1):
        rank = n - i + 1
        val = min(prev, p[idx] * n / rank)
        q[idx] = val; prev = val
    return np.clip(q, 0, 1)

def clr_table(pct: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """Convert % table (rows=taxa, cols=samples) to CLR."""
    X = (pct.astype(float) / 100.0) + pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)
    return logX.sub(gm, axis=1)

def pretty_rank_name(taxon: str, rank: str) -> str:
    """Prettify MetaPhlAn-style clade names for labels."""
    s = str(taxon)
    if rank == "species":
        nm = s.split("|s__")[-1] if "|s__" in s else s.split("|")[-1]
        return nm.replace("_", " ")
    if rank == "genus":
        nm = s.split("|g__")[-1] if "|g__" in s else s.split("|")[-1]
        return nm.replace("_", " ")
    return s.split("|")[-1].replace("_", " ")

def volcano(df, x, sig, name, out_png, title,
            alpha=0.05, k=10, ylab="q", jitter=0.0):
    """
    Generic volcano plot helper.

    - x: effect size (e.g. delta CLR or beta)
    - sig: p or q (used for y-axis and significance threshold)
    - name: taxon label column
    """
    ensure_dir(os.path.dirname(out_png))

    if df is None or df.empty or x not in df or sig not in df:
        save_placeholder(out_png, title)
        return

    X = pd.to_numeric(df[x], errors="coerce").values
    S = pd.to_numeric(df[sig], errors="coerce").clip(lower=1e-300).values
    Y = -np.log10(S)
    names = df[name].astype(str).values if name in df else df.index.astype(str).values

    is_sig = S <= alpha
    score = np.abs(X) * (Y + 1.0)

    up = np.where(is_sig & (X > 0))[0]
    dn = np.where(is_sig & (X < 0))[0]

    lab_idx = np.concatenate([
        up[np.argsort(-score[up])[: (k // 2 + k % 2)]],
        dn[np.argsort(-score[dn])[: (k // 2)]]
    ]) if (len(up) + len(dn)) > 0 else np.array([], dtype=int)

    if jitter:
        rng = np.random.RandomState(0)
        Xp = X + rng.normal(0, jitter, size=X.size)
    else:
        Xp = X

    if np.all(~np.isfinite(Xp)):
        save_placeholder(out_png, title)
        return

    q99 = float(np.nanpercentile(np.abs(Xp[np.isfinite(Xp)]), 99.5))
    xmax = max(0.5, q99)
    xmin = -xmax

    fig = plt.figure(figsize=(12, 6.5))
    ax = fig.add_subplot(111)

    ax.scatter(Xp[~is_sig], Y[~is_sig], s=22, alpha=0.35, color="#999999")

    ax.scatter(Xp[is_sig & (X > 0)], Y[is_sig & (X > 0)],
               s=26, alpha=0.9, color="#1B4F72")
    ax.scatter(Xp[is_sig & (X < 0)], Y[is_sig & (X < 0)],
               s=26, alpha=0.9, color="#7FB3D5")

    ax.axhline(-np.log10(alpha), ls="--", lw=1, color="gray", alpha=0.8)
    ax.axvline(0, ls="--", lw=1, color="gray", alpha=0.8)

    ax.set_xlim(xmin, xmax)
    ax.grid(True, ls=":", lw=0.6, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(f"-log10({ylab})")

    for i in lab_idx:
        if not np.isfinite(Xp[i]) or not np.isfinite(Y[i]):
            continue
        label_txt = f"{names[i]} ({ylab}={S[i]:.2g})"
        ax.text(
            Xp[i], Y[i], label_txt,
            fontsize=9, ha="left", va="bottom",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )

    n_up = int((is_sig & (X > 0)).sum())
    n_dn = int((is_sig & (X < 0)).sum())
    summary_txt = f"{ylab}\u2264{alpha:g}: up={n_up}, down={n_dn}"

    ax.text(
        0.98, 0.02, summary_txt,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

# ========================== UNADJUSTED (NO METADATA) ==========================

def load_labels_from_sources(source_map_csv: str, clr_cols) -> pd.DataFrame:
    """
    Build labels for UNADJUSTED tests *only* from pct_all_sources.csv, independent of metadata.

    Expected columns in source_map_csv:
      - Sample_ID (string)
      - site_fallback in {"Oral","Fecal"} (case-insensitive; 'Faecal' normalized to 'Fecal')
      - disease_fallback in {0,1}  (0=Healthy, 1=Crohn)

    Returns a DataFrame with columns: sample_id, site, disease
    Only samples present in clr_cols are kept.
    """
    if not (source_map_csv and os.path.exists(source_map_csv)):
        raise ValueError("[UNADJUSTED] --source-map is required and must exist.")

    s = pd.read_csv(source_map_csv)
    s = s.rename(columns={c: c.strip() for c in s.columns})
    need = {"Sample_ID", "site_fallback", "disease_fallback"}
    if not need.issubset(set(s.columns)):
        raise ValueError("[UNADJUSTED] source-map must contain columns: Sample_ID, site_fallback, disease_fallback")

    s["Sample_ID"] = s["Sample_ID"].astype(str)
    s["site"] = s["site_fallback"].astype(str).str.strip().str.capitalize().replace({"Faecal": "Fecal"})
    s["disease"] = pd.to_numeric(s["disease_fallback"], errors="coerce").astype("Int64")

    # Keep only valid labels and only samples we actually have in CLR
    s = s[s["site"].isin(["Oral", "Fecal"])]
    s = s[s["disease"].isin([0, 1])]
    s = s[s["Sample_ID"].isin(list(map(str, clr_cols)))]

    out = s.drop_duplicates(subset=["Sample_ID"])[["Sample_ID", "site", "disease"]]
    out = out.rename(columns={"Sample_ID": "sample_id"})
    if out.empty:
        raise ValueError("[UNADJUSTED] No labeled samples found in source-map that overlap with CLR columns.")
    return out

def mwu_unadjusted_with_labels(clr: pd.DataFrame, labels: pd.DataFrame, site: str, rank: str) -> pd.DataFrame:
    """
    Mann–Whitney U for CH within a site using explicit labels (no metadata).
    labels: DataFrame with columns [sample_id, site, disease]
    """
    df = clr.T
    df.index.name = "Sample_ID"
    L = labels[(labels["site"] == site) & (labels["disease"].isin([0, 1]))]
    if L.empty:
        return pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","delta_clr","n_1","n_0"])

    inter = df.index.intersection(L["sample_id"].astype(str))
    if len(inter) == 0:
        return pd.DataFrame(columns=["taxon","pretty_taxon","test","stat","p","q","delta_clr","n_1","n_0"])

    df = df.loc[inter]
    lab = L.set_index("sample_id").loc[inter, "disease"].astype(int).values

    rows = []
    for tax in clr.index:
        x = df[tax].astype(float).values
        a = x[lab == 1]; b = x[lab == 0]
        if a.size == 0 or b.size == 0:
            U, p, delta = np.nan, 1.0, np.nan
        else:
            try:
                U, p = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                U, p = np.nan, 1.0
            delta = float(np.nanmean(a) - np.nanmean(b))  # Crohn minus Healthy
        rows.append({
            "taxon": tax,
            "pretty_taxon": pretty_rank_name(tax, rank),
            "test": f"MWU_{site}",
            "stat": float(U) if np.isfinite(U) else np.nan,
            "p": float(p),
            "delta_clr": delta,
            "n_1": int((lab == 1).sum()),
            "n_0": int((lab == 0).sum()),
        })
    out = pd.DataFrame(rows)
    if len(out): out["q"] = bh_q(out["p"].values)
    return out

# ========================== ADJUSTED (WITH METADATA) ==========================

def norm_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Normalize likely column names in metadata to a common schema."""
    m = meta.copy(); m.columns = [c.strip() for c in m.columns]
    lower = {c.lower(): c for c in m.columns}

    def pick(keys):
        for k in keys:
            if k in lower: return lower[k]
        return None

    out = {}
    out["sample_id"] = m[pick(["sample_id","id","sid","sampleid","sample"])] if pick(["sample_id","id","sid","sampleid","sample"]) else None
    out["site"]      = m[pick(["site","body_site","location"])]              if pick(["site","body_site","location"]) else None
    out["disease"]   = m[pick(["disease","status","group","label"])]         if pick(["disease","status","group","label"]) else None

    for k, keys in {
        "subject_id":["subject_id","pair_id","subject","patient"],
        "age":["age"], "sex":["sex","gender"], "bmi":["bmi"],
        "ppi_use":["ppi_use","ppi","ppi3m","ppi_current"],
        "antibiotics_3m":["antibiotics_3m","abx_3m","antibiotics_last3m","abx3m"],
        "smoking":["smoking","smoker"]
    }.items():
        col = pick(keys)
        if col: out[k] = m[col]

    out = {k:v for k,v in out.items() if v is not None}
    return pd.DataFrame(out)

def to_01(x):
    """Map various encodings to {0,1} where possible."""
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in {"1","true","yes","y","crohn","cd","case","ibd"}: return 1
    if s in {"0","false","no","n","healthy","control","hc","non-ibd","nonibd"}: return 0
    try:
        v = int(float(s))
        if v in (0, 1): return v
    except Exception:
        pass
    return np.nan

def coerce_covs(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast/normalize covariates for adjusted models."""
    d = df.copy()
    if "disease" in d: d["disease"] = d["disease"].map(to_01)
    if "ppi_use" in d: d["ppi_use"] = d["ppi_use"].map(to_01)
    if "antibiotics_3m" in d: d["antibiotics_3m"] = d["antibiotics_3m"].map(to_01)
    if "smoking" in d: d["smoking"] = d["smoking"].map(to_01)
    if "site" in d:
        d["site"] = d["site"].astype(str).str.strip().str.capitalize().replace({"Faecal":"Fecal"})
        d.loc[~d["site"].isin(["Oral","Fecal"]), "site"] = np.nan
    for c in ["age","bmi"]:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    if "sex" in d: d["sex"] = d["sex"].astype(str).str.strip().str.capitalize()
    return d

def merge_meta_with_fallback(meta, srcmap, clr_cols):
    """
    For adjusted models: merge metadata and fill missing site/disease
    from source-map when available. Restricted to samples present in CLR.
    """
    m = meta.copy()
    m["sample_id"] = m["sample_id"].astype(str)

    if srcmap and os.path.exists(srcmap):
        s = pd.read_csv(srcmap, comment="#"); s = s.rename(columns={c: c.strip() for c in s.columns})
        if {"Sample_ID","site_fallback","disease_fallback"}.issubset(s.columns):
            s["Sample_ID"] = s["Sample_ID"].astype(str)
            mm = m.merge(s, left_on="sample_id", right_on="Sample_ID", how="left")
            if "site_fallback" in mm:
                mm["site"] = mm["site"].where(mm["site"].notna(),
                                              mm["site_fallback"].astype(str).str.strip().str.capitalize().replace({"Faecal":"Fecal"}))
            if "disease_fallback" in mm:
                mm["disease"] = mm["disease"].where(mm["disease"].notna(),
                                                    pd.to_numeric(mm["disease_fallback"], errors="coerce").astype("Int64"))
            keep = ["sample_id","site","disease","ppi_use","age","sex","bmi","antibiotics_3m","smoking","subject_id"]
            keep = [c for c in keep if c in mm.columns]
            m = mm[keep]

    m = m[m["sample_id"].isin(list(map(str, clr_cols)))]
    return m

# ---------- ancillary adjusted helpers ----------

def fit_ols_per_taxon(long_df: pd.DataFrame, formula: str, rank: str, label: str, min_n: int = 8) -> pd.DataFrame:
    """Fit OLS per taxon and return tidy results."""
    res = []
    for tax, d in long_df.groupby("taxon"):
        dd = d.dropna(subset=["CLR","disease"])
        if dd.shape[0] < min_n: 
            continue
        try:
            m = smf.ols(formula=formula, data=dd).fit()
            for term in m.params.index:
                if term == "Intercept": 
                    continue
                res.append({
                    "taxon": tax,
                    "pretty_taxon": pretty_rank_name(tax, rank),
                    "model": label,
                    "term": term,
                    "beta": float(m.params[term]),
                    "p": float(m.pvalues.get(term, np.nan)),
                    "n": int(dd.shape[0]),
                })
        except Exception:
            res.append({"taxon":tax,"pretty_taxon":pretty_rank_name(tax, rank),"model":label,
                        "term":"__fit_failed__","beta":np.nan,"p":np.nan,"n":int(dd.shape[0])})
    out = pd.DataFrame(res)
    if not out.empty:
        out["q"] = out.groupby("model")["p"].transform(lambda s: bh_q(s.values))
    return out

def simple_ols_site(clr: pd.DataFrame, meta: pd.DataFrame, site: str, rank: str) -> pd.DataFrame:
    """Simple per-feature OLS (label ~ feature) inside a site; for QC/ancillary only."""
    df = clr.T.reset_index().rename(columns={"index":"Sample_ID"})
    m = meta.loc[meta["site"]==site, ["sample_id","disease"]] \
            .rename(columns={"sample_id":"Sample_ID","disease":"label"})
    mm = df.merge(m, on="Sample_ID", how="inner")
    feats = [c for c in mm.columns if c not in {"Sample_ID","label"}]
    rows = []; Y = mm["label"].astype(float).values
    for f in feats:
        x = mm[f].astype(float).values
        try:
            mdl = OLS(Y, add_constant(x), hasconst=True).fit()
            rows.append({"feature":f,"beta":float(mdl.params[1]),"pval":float(mdl.pvalues[1]),"n":len(mm)})
        except Exception:
            rows.append({"feature":f,"beta":np.nan,"pval":1.0,"n":len(mm)})
    out = pd.DataFrame(rows)
    if len(out):
        ok = np.isfinite(out["pval"].values); q = np.full(len(out), np.nan)
        if ok.sum() > 0: q[ok] = multipletests(out.loc[ok,"pval"].values, method="fdr_bh")[1]
        out["qval"] = q
    out["site"] = site
    return out[["feature","beta","pval","qval","n","site"]]

# ========================== Pairs (paired OC↔FC) ==========================

def load_or_build_deltas(deltas_csv, pct_all, pairs_csv):
    """
    Load precomputed FC-OC deltas or construct them from pairs list.
    Returns (delta_wide, debug_info).
    """
    info = {"mode": None, "n_pairs": 0, "missing_oc": [], "missing_fc": [], "cols": {}}
    if deltas_csv and os.path.exists(deltas_csv):
        D = pd.read_csv(deltas_csv)
        if "pair_id" not in D.columns:
            D = D.rename(columns={D.columns[0]: "pair_id"})
        D = D.set_index("pair_id")
        info["mode"] = "precomputed"; info["n_pairs"] = D.shape[0]
        return D, info

    if not (pairs_csv and os.path.exists(pairs_csv)):
        return None, info

    pairs = pd.read_csv(pairs_csv)
    pairs = pairs.rename(columns={c: c.strip().lower().replace(" ", "_") for c in pairs.columns})
    info["cols"]["all"] = list(pairs.columns)

    def pick(cands):
        for c in cands:
            if c in pairs.columns: return c
        return None

    sid  = pick(["subject_id","pair_id","subject","id","study_id"])
    oral = pick(["oral","oral_id","oc","sample_oral","oral_sample","oral_sid","oral_sample_id","oral_original_kept"])
    fec  = pick(["fecal","faecal","fecal_id","fc","sample_fecal","fecal_sample","fecal_sid","fecal_sample_id","fecal_original_kept"])
    info["cols"]["sid"]=sid; info["cols"]["oral"]=oral; info["cols"]["fecal"]=fec
    if not (oral and fec): 
        return None, info
    if not sid:
        sid = "__row_id__"; pairs[sid] = np.arange(1, len(pairs) + 1).astype(str)

    clr = clr_table(pct_all)
    cols = set(clr.columns.astype(str))
    rows = []
    for _, r in pairs[[sid, oral, fec]].dropna().iterrows():
        pid = str(r[sid]).strip(); oc = str(r[oral]).strip(); fc = str(r[fec]).strip()
        if oc not in cols: info["missing_oc"].append(oc); continue
        if fc not in cols: info["missing_fc"].append(fc); continue
        delta = clr[fc] - clr[oc]; delta.name = pid; rows.append(delta)

    if rows:
        D = pd.DataFrame(rows).groupby(lambda x: x).mean()
        info["mode"] = "built_from_pairs"; info["n_pairs"] = D.shape[0]
        return D, info
    return None, info

def wilcoxon_from_deltas(deltas_wide: pd.DataFrame, rank: str) -> pd.DataFrame:
    """Paired Wilcoxon test per taxon from FC-OC deltas."""
    rows = []
    for tax in deltas_wide.columns:
        dif = deltas_wide[tax].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if dif.size < 5 or np.allclose(dif, 0.0):
            W, p, dm = np.nan, 1.0, np.nan
        else:
            try:
                W, p = wilcoxon(dif, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                W, p = np.nan, 1.0
            dm = float(np.nanmedian(dif))
        rows.append({"taxon": tax, "pretty_taxon": pretty_rank_name(tax, rank),
                     "test":"Wilcoxon_OC_vs_FC_Crohn", "stat": float(W) if np.isfinite(W) else np.nan,
                     "p": float(p), "delta_median": dm, "n_pairs": int(len(dif))})
    out = pd.DataFrame(rows)
    if len(out): out["q"] = bh_q(out["p"].values)
    return out

# ========================== CLI ==========================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Unified DA + PPI; UNADJUSTED uses source-map only, ADJUSTED uses metadata."
    )
    ap.add_argument("--pct-all", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--deltas", default=None)

    # Required for UNADJUSTED
    ap.add_argument("--source-map", required=True,
                    help="pct_all_sources.csv with columns: Sample_ID, site_fallback, disease_fallback")

    ap.add_argument("--rank", required=True, choices=["genus","species"])
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--heat-topk", type=int, default=10)
    ap.add_argument("--volcano-metric", choices=["q","p"], default="q")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--label-topk", type=int, default=10)
    ap.add_argument("--paired-min-pairs", type=int, default=5)
    ap.add_argument("--plot-jitter", type=float, default=0.0)
    return ap.parse_args()

# ========================== Main ==========================

def main():
    args = parse_args()
    out_rank = os.path.join(args.outdir, args.rank); ensure_dir(out_rank)

    # Load abundance & CLR
    pct_all = pd.read_csv(args.pct_all, index_col=0); pct_all.columns = pct_all.columns.astype(str)
    clr = clr_table(pct_all)

    # ---------- UNADJUSTED (NO METADATA): build labels from source-map ----------
    labels = load_labels_from_sources(args.source_map, clr.columns)
    # Oral
    da_oral = mwu_unadjusted_with_labels(clr, labels, site="Oral", rank=args.rank)
    da_oral.to_csv(os.path.join(out_rank, f"da_oral_unadj_{args.rank}.csv"), index=False)
    # audit
    lab_oral = labels[(labels["site"]=="Oral") & (labels["sample_id"].isin(clr.columns.astype(str)))]
    with open(os.path.join(out_rank, "audit_unadj_oral.txt"), "w") as fh:
        ids1 = sorted(lab_oral.loc[lab_oral["disease"]==1, "sample_id"].astype(str))
        ids0 = sorted(lab_oral.loc[lab_oral["disease"]==0, "sample_id"].astype(str))
        fh.write("Oral UNADJUSTED using source-map (no metadata)\n")
        fh.write(f"n_Crohn (1) = {len(ids1)}\n")
        fh.write(f"n_Healthy (0) = {len(ids0)}\n")
        fh.write("Crohn_IDs:\n" + ",".join(ids1) + "\n")
        fh.write("Healthy_IDs:\n" + ",".join(ids0) + "\n")
    metric = args.volcano_metric; alpha = args.alpha
    if not da_oral.empty:
        if metric not in da_oral and "p" in da_oral: da_oral[metric] = bh_q(da_oral["p"]) if metric=="q" else da_oral["p"]
        volcano(da_oral.rename(columns={"delta_clr":"x"}), x="x", sig=metric, name="pretty_taxon",
                out_png=os.path.join(out_rank, f"volcano_oral_unadj_{args.rank}.png"),
                title=f"Volcano (Oral CH unadjusted) — {args.rank}",
                alpha=alpha, k=args.heat_topk, ylab=metric, jitter=args.plot_jitter)
    else:
        save_placeholder(os.path.join(out_rank, f"volcano_oral_unadj_{args.rank}.png"),
                         f"Volcano (Oral CH unadjusted) — {args.rank}")

    # Fecal
    da_fecal = mwu_unadjusted_with_labels(clr, labels, site="Fecal", rank=args.rank)
    da_fecal.to_csv(os.path.join(out_rank, f"da_fecal_unadj_{args.rank}.csv"), index=False)
    lab_fec = labels[(labels["site"]=="Fecal") & (labels["sample_id"].isin(clr.columns.astype(str)))]
    with open(os.path.join(out_rank, "audit_unadj_fecal.txt"), "w") as fh:
        ids1 = sorted(lab_fec.loc[lab_fec["disease"]==1, "sample_id"].astype(str))
        ids0 = sorted(lab_fec.loc[lab_fec["disease"]==0, "sample_id"].astype(str))
        fh.write("Fecal UNADJUSTED using source-map (no metadata)\n")
        fh.write(f"n_Crohn (1) = {len(ids1)}\n")
        fh.write(f"n_Healthy (0) = {len(ids0)}\n")
        fh.write("Crohn_IDs:\n" + ",".join(ids1) + "\n")
        fh.write("Healthy_IDs:\n" + ",".join(ids0) + "\n")
    if not da_fecal.empty:
        if metric not in da_fecal and "p" in da_fecal: da_fecal[metric] = bh_q(da_fecal["p"]) if metric=="q" else da_fecal["p"]
        volcano(da_fecal.rename(columns={"delta_clr":"x"}), x="x", sig=metric, name="pretty_taxon",
                out_png=os.path.join(out_rank, f"volcano_fecal_unadj_{args.rank}.png"),
                title=f"Volcano (Fecal CH unadjusted) — {args.rank}",
                alpha=alpha, k=args.heat_topk, ylab=metric, jitter=args.plot_jitter)
    else:
        save_placeholder(os.path.join(out_rank, f"volcano_fecal_unadj_{args.rank}.png"),
                         f"Volcano (Fecal CH unadjusted) — {args.rank}")

    # ---------- ADJUSTED (WITH METADATA) ----------
    meta_raw = pd.read_csv(args.meta)
    meta0 = norm_meta(meta_raw)
    if "sample_id" not in meta0.columns:
        raise ValueError("[ADJUSTED] metadata must contain a sample_id column.")
    meta = coerce_covs(meta0)
    meta = merge_meta_with_fallback(meta, args.source_map, clr.columns)

    # Long-format table for OLS
    clr_T = clr.T; clr_T.index.name = "Sample_ID"
    long = clr_T.stack().reset_index(); long.columns = ["Sample_ID","taxon","CLR"]
    mm = meta.rename(columns={"sample_id":"Sample_ID"})
    long = long.merge(mm, on="Sample_ID", how="inner")

    base = [c for c in ["disease","site","ppi_use","age","sex","bmi","antibiotics_3m","smoking"] if c in long.columns]
    formula_all = "CLR ~ " + (" + ".join(base) if base else "1")
    da_all = fit_ols_per_taxon(long, formula_all, args.rank, "ALL")
    da_all.to_csv(os.path.join(out_rank, f"da_all_adj_{args.rank}.csv"), index=False)

    ds = da_all[da_all["term"].str.startswith("disease", na=False)].copy()
    if not ds.empty:
        if metric not in ds and "p" in ds: ds[metric] = bh_q(ds["p"]) if metric=="q" else ds["p"]
        volcano(ds.rename(columns={"beta":"x"}), x="x", sig=metric, name="pretty_taxon",
                out_png=os.path.join(out_rank, f"volcano_all_adj_{args.rank}.png"),
                title=f"Volcano (Adjusted ALL; term=disease) — {args.rank}",
                alpha=args.alpha, k=args.heat_topk, ylab=metric, jitter=args.plot_jitter)

    for site in ["Oral","Fecal"]:
        sub = long[long["site"] == site].copy()
        out_csv = os.path.join(out_rank, f"da_{site.lower()}_adj_{args.rank}.csv")
        if sub.empty or sub["disease"].dropna().nunique() < 2:
            pd.DataFrame(columns=["taxon","model","term","beta","p","q","n"]).to_csv(out_csv, index=False)
            save_placeholder(os.path.join(out_rank, f"volcano_{site.lower()}_adj_{args.rank}.png"),
                             f"Volcano (Adjusted {site}; term=disease) — {args.rank}")
        else:
            terms = [t for t in base if t != "site"]
            formula_site = "CLR ~ " + (" + ".join(terms) if terms else "1")
            da_site = fit_ols_per_taxon(sub, formula_site, args.rank, site)
            da_site.to_csv(out_csv, index=False)
            ds = da_site[da_site["term"].str.startswith("disease", na=False)].copy()
            if not ds.empty:
                if metric not in ds and "p" in ds: ds[metric] = bh_q(ds["p"]) if metric=="q" else ds["p"]
                volcano(ds.rename(columns={"beta":"x"}), x="x", sig=metric, name="pretty_taxon",
                        out_png=os.path.join(out_rank, f"volcano_{site.lower()}_adj_{args.rank}.png"),
                        title=f"Volcano (Adjusted {site}; term=disease) — {args.rank}",
                        alpha=args.alpha, k=args.heat_topk, ylab=metric, jitter=args.plot_jitter)

    # ---------- Paired OC↔FC ----------
    paired_csv = os.path.join(out_rank, f"da_paired_{args.rank}.csv")
    D, dbg = load_or_build_deltas(args.deltas, pct_all, args.pairs)
    with open(os.path.join(out_rank, "paired_debug.txt"), "w") as fh:
        fh.write(f"mode={dbg.get('mode')}\n")
        fh.write(f"n_pairs={dbg.get('n_pairs')}\n")
        fh.write(f"cols_detected={dbg.get('cols')}\n")
        if dbg.get("missing_oc"): fh.write("missing_oral_ids=" + ",".join(dbg["missing_oc"][:50]) + "\n")
        if dbg.get("missing_fc"): fh.write("missing_fecal_ids=" + ",".join(dbg["missing_fc"][:50]) + "\n")

    if D is None or D.empty or D.shape[0] < args.paired_min_pairs:
        pd.DataFrame(columns=["taxon","test","stat","p","q","delta_median","n_pairs"]).to_csv(paired_csv, index=False)
        save_placeholder(os.path.join(out_rank, f"volcano_paired_{args.rank}.png"),
                        f"Volcano (Paired OC↔FC) — {args.rank}")
    else:
        paired = wilcoxon_from_deltas(D, args.rank)
        paired["n_pairs"] = D.shape[0]
        paired.to_csv(paired_csv, index=False)
        sig = args.volcano_metric
        if sig not in paired and "p" in paired:
            paired[sig] = bh_q(paired["p"]) if sig == "q" else paired["p"]
        volcano(paired.rename(columns={"delta_median":"x"}), x="x", sig=sig, name="pretty_taxon",
                out_png=os.path.join(out_rank, f"volcano_paired_{args.rank}.png"),
                title=f"Volcano (Paired OC↔FC; x=median FC−OC) — {args.rank}",
                alpha=args.alpha, k=args.label_topk, ylab=sig)

    # ---------- PPI (adjusted; needs metadata) ----------
    def n_nonmiss(col): return int(long[col].notna().sum()) if col in long else 0
    def n_unique(col):  return int(long[col].dropna().nunique()) if col in long else 0

    rhs = []
    if n_nonmiss("disease")>=12 and n_unique("disease")>1: rhs.append("disease")
    if n_nonmiss("site")>=12 and n_unique("site")>1: rhs.append("site")
    if n_nonmiss("ppi_use")>=12 and n_unique("ppi_use")>1: rhs.append("ppi_use")
    if {"disease","site"}.issubset(rhs): rhs.append("disease:site")
    if {"disease","ppi_use"}.issubset(rhs): rhs.append("disease:ppi_use")
    for c in ["age","sex","bmi","antibiotics_3m","smoking"]:
        if n_nonmiss(c)>=12 and n_unique(c)>1: rhs.append(c)

    form_all = "CLR ~ " + (" + ".join(rhs) if rhs else "1")
    eff_all = fit_ols_per_taxon(long, form_all, args.rank, "ALL", min_n=12)
    eff_all.to_csv(os.path.join(out_rank, "ppi_effects_all.csv"), index=False)

    metric = args.volcano_metric; alpha = args.alpha
    sub = eff_all[eff_all["term"].astype(str).str.startswith("ppi_use", na=False)].copy()
    if not sub.empty and metric not in sub and "p" in sub: sub[metric] = bh_q(sub["p"]) if metric=="q" else sub["p"]
    volcano(sub.rename(columns={"beta":"x"}), x="x", sig=metric, name="pretty_taxon",
            out_png=os.path.join(out_rank, "volcano_ppi_all.png"),
            title="Volcano (ALL) — PPI effect", alpha=alpha, k=args.heat_topk, ylab=metric)

    sub = eff_all[eff_all["term"].astype(str).str.startswith("disease:ppi_use", na=False)].copy()
    if not sub.empty and metric not in sub and "p" in sub: sub[metric] = bh_q(sub["p"]) if metric=="q" else sub["p"]
    volcano(sub.rename(columns={"beta":"x"}), x="x", sig=metric, name="pretty_taxon",
            out_png=os.path.join(out_rank, "volcano_int_disease_ppi.png"),
            title="Volcano (ALL) — disease:PPI", alpha=alpha, k=args.heat_topk, ylab=metric)

    sub = eff_all[eff_all["term"].astype(str).str.startswith("disease:site", na=False)].copy()
    if not sub.empty and metric not in sub and "p" in sub: sub[metric] = bh_q(sub["p"]) if metric=="q" else sub["p"]
    volcano(sub.rename(columns={"beta":"x"}), x="x", sig=metric, name="pretty_taxon",
            out_png=os.path.join(out_rank, "volcano_int_disease_site.png"),
            title="Volcano (ALL) — disease:site", alpha=alpha, k=args.label_topk, ylab=metric)

    for site in ["Oral","Fecal"]:
        subdf = long[long["site"] == site].copy()
        out_csv = os.path.join(out_rank, f"ppi_effects_{site.lower()}.csv")
        out_png = os.path.join(out_rank, f"volcano_ppi_{site.lower()}.png")
        if subdf.empty or subdf["disease"].dropna().nunique()<2 or subdf["ppi_use"].dropna().nunique()<2:
            pd.DataFrame(columns=["taxon","model","term","beta","p","q","n"]).to_csv(out_csv, index=False)
            save_placeholder(out_png, f"Volcano ({site}) — PPI effect")
        else:
            covs = [t for t in ["disease","ppi_use","age","sex","bmi","antibiotics_3m","smoking"]
                    if t in subdf.columns and subdf[t].dropna().nunique()>1]
            form_site = "CLR ~ " + (" + ".join(covs) if covs else "1")
            eff_site = fit_ols_per_taxon(subdf, form_site, args.rank, site, min_n=10)
            eff_site.to_csv(out_csv, index=False)
            vv = eff_site[eff_site["term"].astype(str).str.startswith("ppi_use", na=False)].copy()
            if not vv.empty and metric not in vv and "p" in vv: vv[metric] = bh_q(vv["p"]) if metric=="q" else vv["p"]
            volcano(vv.rename(columns={"beta":"x"}), x="x", sig=metric, name="pretty_taxon",
                    out_png=out_png, title=f"Volcano ({site}) — PPI effect",
                    alpha=alpha, k=args.heat_topk, ylab=metric)

    print("[INFO] DA unified done →", out_rank)

if __name__ == "__main__":
    main()
