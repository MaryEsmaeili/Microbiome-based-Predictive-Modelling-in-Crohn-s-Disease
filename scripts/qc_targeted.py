#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, warnings
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from numpy.random import default_rng

# ------------------------ utils ------------------------
def ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def save_empty_png(path: str, msg: str) -> None:
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)

def _pick(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None

def to01(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    pos = {"1","yes","true","y","crohn","cd","case","ibd"}
    neg = {"0","no","false","n","healthy","control","hc","non-ibd","nonibd"}
    if s in pos: return 1
    if s in neg: return 0
    try:
        v = int(float(s)); return v if v in (0,1) else np.nan
    except Exception:
        return np.nan

def normalize_meta(meta_csv: str) -> pd.DataFrame:
    m = pd.read_csv(meta_csv)
    m.columns = [str(c).strip() for c in m.columns]
    out = pd.DataFrame({
        "sample_id": m[_pick(m, ["sample_id","id","sample","sampleid"])].astype(str),
        "site":      m[_pick(m, ["site","body_site","location"])],
        "disease":   m[_pick(m, ["disease","status","group"])],
        "ppi_use":   m[_pick(m, ["ppi_use","ppi","ppi_current","ppi3m"])]
    })
    out["site"] = out["site"].astype(str).str.capitalize().replace({"Faecal":"Fecal"})
    bad = ~out["site"].isin(["Oral","Fecal"]); out.loc[bad,"site"] = np.nan
    out["disease"] = out["disease"].map(to01)
    out["ppi_use"] = out["ppi_use"].map(to01)
    return out

def clr_on_percent(pct_df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    X = (pct_df.astype(float)/100.0) + pseudocount
    L = np.log(X)
    gm = L.mean(axis=0)
    return L.sub(gm, axis=1)

def prevalence_mean_filter(pct_df: pd.DataFrame, prevalence: float, mean_pct: float) -> pd.Index:
    nz_frac = (pct_df > 0).sum(axis=1) / pct_df.shape[1]
    mean_vals = pct_df.mean(axis=1)
    keep = (nz_frac >= prevalence) & (mean_vals >= mean_pct)
    return pct_df.index[keep]

# ------------------------ plotting helpers ------------------------
def add_n_to_title(ax, title, n_text):
    ax.set_title(f"{title}\n{n_text}", fontsize=12)

def bar_counts(df: pd.DataFrame, cols: List[str], out_png: str, title: str):
    ensure_dir(os.path.dirname(out_png))
    counts = df.value_counts(cols).reset_index(name="n")
    if counts.empty:
        save_empty_png(out_png, "No data for counts"); return counts
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = counts[cols].astype(str).agg(" × ".join, axis=1)
    ax.bar(labels, counts["n"])
    ax.set_ylabel("n")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    return counts

def pca_scatter(M: pd.DataFrame, meta: pd.DataFrame, out_png: str):
    ensure_dir(os.path.dirname(out_png))
    if M.shape[0] < 3 or M.shape[1] < 2:
        save_empty_png(out_png, "Not enough data for PCA"); return
    X = M.values.astype(float)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pcs = U[:, :2]*S[:2]
    var = (S**2)/(X.shape[0]-1); var = var/var.sum()
    fig, ax = plt.subplots(figsize=(7,6))
    color = {"Oral":"#00798c","Fecal":"#d1495b"}
    marker = {0:"o",1:"s"}
    meta = meta.reindex(M.index)
    for (site, dis), idx in meta.reset_index().groupby(["site","disease"]).groups.items():
        pts = pcs[list(idx), :]
        ax.scatter(pts[:,0], pts[:,1], c=color.get(site,"#777"), marker=marker.get(dis,"o"),
                   s=60, label=f"{site}, {'Healthy' if dis==0 else 'Crohn'} (n={len(idx)})")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)"); ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.legend(frameon=False, ncol=2); ax.set_title("PCA (CLR, filtered taxa)")
    ax.axhline(0,color="#bbb",lw=0.8); ax.axvline(0,color="#bbb",lw=0.8)
    fig.tight_layout(); fig.savefig(out_png,dpi=300,bbox_inches="tight"); plt.close(fig)

def permanova_one_factor(M: pd.DataFrame, meta: pd.DataFrame, factor: str,
                         pairs: Optional[pd.DataFrame]=None, n_perm: int=999, seed: int=1):
    rng = default_rng(seed)
    meta = meta.reindex(M.index)
    y = meta[factor].values
    mask = ~pd.isna(y)
    X = M.values[mask, :].astype(float)
    y = y[mask].astype(str)
    if X.shape[0] < 4 or len(pd.unique(y))<2:
        return (np.nan, np.nan, np.nan, 0)
    X = X - X.mean(axis=0, keepdims=True)
    D = squareform(pdist(X, metric="euclidean"))
    n = X.shape[0]
    levels = pd.unique(y)

    def anova_terms(labels):
        SSW, dfw = 0.0, 0
        for lv in levels:
            idx = np.where(labels==lv)[0]
            if len(idx)<=1: continue
            Dg = D[np.ix_(idx, idx)]
            SSW += (Dg**2).sum()/len(idx)
            dfw += (len(idx)-1)
        TSS = (D**2).sum()/n
        SSB = TSS - SSW
        dfb = len(levels)-1
        MSB = SSB/dfb if dfb>0 else np.nan
        MSW = SSW/dfw if dfw>0 else np.nan
        F = MSB/MSW if (MSW>0 and np.isfinite(MSB)) else np.nan
        R2 = SSB/TSS if TSS>0 else np.nan
        return F, R2

    Fobs, R2obs = anova_terms(y)

    blocks = None
    if pairs is not None and {"oral","fecal"}.issubset({c.lower() for c in pairs.columns}):
        p = pairs.copy(); p.columns = [c.lower() for c in p.columns]
        long = pd.DataFrame({"sid": list(p["oral"].astype(str))+list(p["fecal"].astype(str)),
                             "block": np.repeat(np.arange(len(p)), 2)})
        block_map = long.set_index("sid")["block"]
        idx = [block_map.get(i, np.nan) for i in M.index[mask].astype(str)]
        if pd.Series(idx).notna().sum()>0:
            blocks = np.array(idx)

    def perm_within(labels):
        if blocks is None: return rng.permutation(labels)
        out = labels.copy()
        b = pd.Series(blocks)
        for blk, ids in b.groupby(b).groups.items():
            ids = np.array(list(ids))
            if len(ids)>1:
                out[ids] = rng.permutation(out[ids])
        return out

    ge, eff = [], 0
    for _ in range(n_perm):
        yp = perm_within(y)
        if np.array_equal(yp, y):  # skip identity
            continue
        Fp, _ = anova_terms(yp)
        if np.isfinite(Fp):
            ge.append(Fp); eff += 1
    if eff==0 or not np.isfinite(Fobs):
        return (np.nan, R2obs, np.nan, eff)
    ge = np.array(ge)
    p = (1 + (ge >= Fobs).sum()) / (1 + eff)
    return (float(Fobs), float(R2obs), float(p), eff)

# ------------------------ PPI helpers ------------------------
def load_ppi_effects(ppi_csv: Optional[str]) -> pd.DataFrame:
    if not ppi_csv or not os.path.exists(ppi_csv):
        return pd.DataFrame()
    ppi = pd.read_csv(ppi_csv)
    ppi.columns = [str(c).strip() for c in ppi.columns]
    return ppi

def build_ppi_maps(ppi: pd.DataFrame) -> Dict[str,str]:
    """برچسب ترم PPI برای هر تاکسون (اگر ستون‌ها موجود باشند)."""
    if ppi.empty: return {}
    cols = {c.lower(): c for c in ppi.columns}
    term_col = cols.get("term")
    tax_col  = cols.get("taxon") or cols.get("feature") or cols.get("name")
    if not (term_col and tax_col): return {}
    # ترجیحاً فقط ردیف‌های مرتبط با PPI (اگر قابل تشخیص بود)
    sub = ppi
    if ppi[term_col].astype(str).str.contains("ppi", case=False, na=False).any():
        sub = ppi[ppi[term_col].astype(str).str.contains("ppi", case=False, na=False)]
    return dict(zip(sub[tax_col].astype(str), sub[term_col].astype(str)))

def ppi_filter_taxa(ppi: pd.DataFrame, metric: str, thr: float) -> Optional[pd.Index]:
    if metric == "none" or ppi.empty:
        return None
    cols = {c.lower(): c for c in ppi.columns}
    tax_col = cols.get("taxon") or cols.get("feature") or cols.get("name")
    if not tax_col: return None

    if metric == "q":
        mcol = cols.get("q") or cols.get("qvalue") or cols.get("q_value")
        if not mcol: return None
        keep = ppi[mcol] <= thr
        return ppi.loc[keep, tax_col].astype(str).unique()

    if metric == "p":
        mcol = cols.get("p") or cols.get("pvalue") or cols.get("p_value")
        if not mcol: return None
        keep = ppi[mcol] <= thr
        return ppi.loc[keep, tax_col].astype(str).unique()

    if metric == "absbeta":
        mcol = cols.get("beta") or cols.get("effect") or cols.get("coef") or cols.get("estimate")
        if not mcol: return None
        keep = ppi[mcol].abs() >= thr
        return ppi.loc[keep, tax_col].astype(str).unique()

    return None

# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct-all", required=True, help="taxa×samples percent table (0..100), index=taxa")
    ap.add_argument("--meta", required=True, help="pooled metadata CSV")
    ap.add_argument("--ppi-effects-all", required=True, help="CSV from PPI interactions")
    ap.add_argument("--pairs", default=None, help="optional pairs CSV with Oral/Fecal IDs")
    ap.add_argument("--rank", required=True, choices=["genus","species"])
    ap.add_argument("--outdir", required=True)
    # thresholds (آستانه‌محور، نه Top-K)
    ap.add_argument("--prevalence", type=float, default=0.20, help="keep taxa present in ≥ this fraction")
    ap.add_argument("--mean-pct",  type=float, default=1.0,  help="keep taxa with mean %% ≥ this threshold")
    ap.add_argument("--cap", type=int, default=30, help="soft cap (max taxa plotted); selection by CLR variance")
    # PPI filtering / annotation
    ap.add_argument("--ppi-filter-metric", choices=["none","q","p","absbeta"], default="none")
    ap.add_argument("--ppi-filter-thresh", type=float, default=0.10)
    ap.add_argument("--annotate-ppi", dest="annotate_ppi", action="store_true", default=False)
    # PCA/PERMANOVA
    ap.add_argument("--with-pca-permanova", action="store_true")
    args = ap.parse_args()

    base = os.path.join(args.outdir, args.rank)
    ensure_dir(base)

    # ---- load inputs
    pct = pd.read_csv(args.pct_all, index_col=0)
    pct.columns = pct.columns.astype(str)
    meta = normalize_meta(args.meta)
    meta = meta[meta["sample_id"].isin(pct.columns)]
    meta = meta.dropna(subset=["site","disease"])  # essential
    pct = pct.loc[:, meta["sample_id"]]           # align

    ppi_df = load_ppi_effects(args.ppi_effects_all)
    ppi_terms_map = build_ppi_maps(ppi_df)

    # ---- threshold filters (presence & mean%)
    keep_taxa = set(prevalence_mean_filter(pct, prevalence=args.prevalence, mean_pct=args.mean_pct))

    # ---- optional PPI-evidence filter (افزوده بر آستانه‌ها)
    ppi_keep = ppi_filter_taxa(ppi_df, args.ppi_filter_metric, args.ppi_filter_thresh)
    if ppi_keep is not None:
        keep_taxa = keep_taxa.intersection(set(map(str, ppi_keep)))

    if len(keep_taxa) == 0:
        # خروجی‌های خالی اما معتبر
        pd.DataFrame(columns=["taxon"]).to_csv(os.path.join(base,"top_taxa_selected.csv"), index=False)
        for f in ["box_by_ppi_oral.png","box_by_ppi_fecal.png","interaction_means.png",
                  "paired_spaghetti_crohn.png","density_top_taxa_oral.png","density_top_taxa_fecal.png",
                  "pca_permanova/pca_scatter.png"]:
            save_empty_png(os.path.join(base,f), "No taxa after filtering")
        pd.DataFrame(columns=["factor","F","R2","p","n_perm"]).to_csv(
            os.path.join(base,"pca_permanova","permanova.tsv"), sep="\t", index=False)
        print("[QC-targeted] No taxa after filters."); return

    pct_f = pct.loc[list(keep_taxa)]
    kept = pct_f.shape[0]

    # ---- CLR & variance-based capping for visuals
    clr = clr_on_percent(pct_f, pseudocount=1e-6)   # taxa×samples (CLR)
    clr_T = clr.T                                   # samples×taxa
    if kept > args.cap:
        var = clr_T.var(axis=0).sort_values(ascending=False)
        taxa_vis = var.index[:args.cap]
    else:
        taxa_vis = clr_T.columns

    # ذخیرهٔ انتخاب نهایی تاکسا برای ویژوال‌ها
    pd.DataFrame({"taxon": taxa_vis}).to_csv(os.path.join(base,"top_taxa_selected.csv"), index=False)

    # ---- کلاس‌کانت‌ها
    class_counts = meta.value_counts(["site","disease"]).reset_index(name="n")
    class_counts.to_csv(os.path.join(base, "class_counts_site_disease.tsv"), sep="\t", index=False)
    ensure_dir(base)
    # تصویر شمارش‌ها در Snakefile جدید لازم نیست، ولی می‌سازیم اگر خواستی اضافه‌اش کنی
    # (می‌تونی نادیده بگیری)
    # bar_counts(meta, ["site","disease"], os.path.join(base,"counts_site_disease.png"),
    #            "Counts by site × disease")

    # ---- long table
    long = clr_T.stack().reset_index()
    long.columns = ["Sample_ID","taxon","CLR"]
    long = long.merge(meta.rename(columns={"sample_id":"Sample_ID"}), on="Sample_ID", how="left")

    # ---- helpers
    def label_with_ppi(t: str) -> str:
        if (not args.annotate_ppi) or (t not in ppi_terms_map):
            return t
        return f"{t} [{ppi_terms_map[t]}]"

    # ---- PCA & PERMANOVA
    if args.with_pca_permanova:
        M = long[long["taxon"].isin(taxa_vis)].pivot_table(index="Sample_ID", columns="taxon", values="CLR")
        pca_dir = os.path.join(base, "pca_permanova"); ensure_dir(pca_dir)
        pca_scatter(M, meta.set_index("sample_id"), os.path.join(pca_dir, "pca_scatter.png"))
        pairs_df = pd.read_csv(args.pairs) if args.pairs and os.path.exists(args.pairs) else None
        F1,R21,p1,eff1 = permanova_one_factor(M, meta.set_index("sample_id"), "site",    pairs_df)
        F2,R22,p2,eff2 = permanova_one_factor(M, meta.set_index("sample_id"), "disease", pairs_df)
        pd.DataFrame([{"factor":"site","F":F1,"R2":R21,"p":p1,"n_perm":eff1},
                      {"factor":"disease","F":F2,"R2":R22,"p":p2,"n_perm":eff2}]\
                    ).to_csv(os.path.join(pca_dir,"permanova.tsv"), sep="\t", index=False)
    else:
        # اگر Snakemake انتظار فایل‌ها را داشت، آن‌ها را حداقلی بسازیم
        pca_dir = os.path.join(base, "pca_permanova"); ensure_dir(pca_dir)
        save_empty_png(os.path.join(pca_dir,"pca_scatter.png"), "PCA disabled")
        pd.DataFrame([{"factor":"site","F":np.nan,"R2":np.nan,"p":np.nan,"n_perm":0},
                      {"factor":"disease","F":np.nan,"R2":np.nan,"p":np.nan,"n_perm":0}]\
                    ).to_csv(os.path.join(pca_dir,"permanova.tsv"), sep="\t", index=False)

    # ---- Box-by-PPI (per site)
    def box_by_ppi(site: str, out_png: str):
        d = long[(long["site"]==site) & (long["taxon"].isin(taxa_vis)) & long["ppi_use"].notna()]
        if d.empty:
            save_empty_png(out_png, f"No PPI data for {site}"); return
        taxa = d["taxon"].unique().tolist()
        n0_samp = d.loc[d["ppi_use"]==0, "Sample_ID"].nunique()
        n1_samp = d.loc[d["ppi_use"]==1, "Sample_ID"].nunique()
        pos = np.arange(len(taxa))
        fig, ax = plt.subplots(figsize=(min(14, 1.2*len(taxa)+4), 5))
        for i,t in enumerate(taxa):
            x0 = d[(d["taxon"]==t) & (d["ppi_use"]==0)]["CLR"].values
            x1 = d[(d["taxon"]==t) & (d["ppi_use"]==1)]["CLR"].values
            bp0 = ax.boxplot([x0], positions=[i-0.18], widths=0.32, patch_artist=True)
            bp1 = ax.boxplot([x1], positions=[i+0.18], widths=0.32, patch_artist=True)
            for b in bp0['boxes']: b.set(facecolor="#9ad1d4")
            for b in bp1['boxes']: b.set(facecolor="#2b8a9a")
        ax.set_xticks(pos); ax.set_xticklabels([label_with_ppi(t) for t in taxa], rotation=60, ha="right")
        ax.set_ylabel("CLR")
        add_n_to_title(ax, f"Box by PPI — {site}",
                       f"samples: PPI=0 (n={n0_samp}), PPI=1 (n={n1_samp}); "
                       f"rows: PPI=0 (n={(d['ppi_use']==0).sum()}), PPI=1 (n={(d['ppi_use']==1).sum()})")
        fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)

    box_by_ppi("Oral",  os.path.join(base, "box_by_ppi_oral.png"))
    box_by_ppi("Fecal", os.path.join(base, "box_by_ppi_fecal.png"))

    # ---- KDE by site
    def kde_site(site: str, out_png: str):
        d = long[(long["site"]==site) & (long["taxon"].isin(taxa_vis))]
        if d.empty: save_empty_png(out_png, f"No data for {site}"); return
        fig, ax = plt.subplots(figsize=(10,6))
        for t in d["taxon"].unique():
            v = d.loc[d["taxon"]==t,"CLR"].astype(float).values
            v = v[np.isfinite(v)]
            if len(v)>1:
                xs = np.linspace(np.min(v)-1, np.max(v)+1, 200)
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(v)
                ax.plot(xs, kde(xs), lw=2, label=label_with_ppi(t), alpha=0.9)
        ax.set_xlabel("CLR"); ax.set_ylabel("Density")
        add_n_to_title(ax, f"CLR density — {site}", f"taxa shown: {len(d['taxon'].unique())}")
        ax.legend(fontsize=8, frameon=False, ncol=2)
        fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)

    kde_site("Oral",  os.path.join(base, "density_top_taxa_oral.png"))
    kde_site("Fecal", os.path.join(base, "density_top_taxa_fecal.png"))

    # ---- Interaction means (disease × site)
    sel = long[long["taxon"].isin(taxa_vis)]
    if not sel.empty:
        fig, ax = plt.subplots(figsize=(max(12, 1.0*len(taxa_vis)+6), 6))
        xbase = {t:i for i,t in enumerate(taxa_vis)}
        for t in taxa_vis:
            for dis, ls, mk in [(0,"-","o"), (1,"--","s")]:
                vo = sel[(sel["taxon"]==t)&(sel["site"]=="Oral")&(sel["disease"]==dis)]["CLR"].values
                vf = sel[(sel["taxon"]==t)&(sel["site"]=="Fecal")&(sel["disease"]==dis)]["CLR"].values
                if len(vo)==0 or len(vf)==0: continue
                mO, mF = np.mean(vo), np.mean(vf)
                ax.plot([xbase[t]+0.0, xbase[t]+0.6], [mO, mF], ls=ls, marker=mk, lw=2)
        ax.set_xticks([xbase[t]+0.3 for t in taxa_vis])
        ax.set_xticklabels([label_with_ppi(t) for t in taxa_vis], rotation=55, ha="right")
        ax.set_ylabel("Mean CLR"); ax.axhline(0,ls=":",color="#aaa")
        ax.legend([plt.Line2D([0],[0], ls="-", marker="o"),
                   plt.Line2D([0],[0], ls="--", marker="s")],
                  ["Healthy", "Crohn"], frameon=False, loc="upper left")
        add_n_to_title(ax, "Interaction means (disease × site)",
                       f"filters: prevalence≥{args.prevalence:.0%}, mean%≥{args.mean_pct:.2f}; cap={args.cap}; "
                       f"PPI filter={args.ppi_filter_metric} (thr={args.ppi_filter_thresh})")
        fig.tight_layout(); fig.savefig(os.path.join(base,"interaction_means.png"),
                                        dpi=300, bbox_inches="tight"); plt.close(fig)

    # ---- Spaghetti برای جفت‌های Crohn
    if args.pairs and os.path.exists(args.pairs):
        pairs = pd.read_csv(args.pairs)
        cols = {c.lower(): c for c in pairs.columns}
        ocol = cols.get("oral") or cols.get("oral_id") or cols.get("oc")
        fcol = cols.get("fecal") or cols.get("fecal_id") or cols.get("fc")
        if ocol and fcol:
            crohn = long[long["disease"]==1]
            usable = []
            sids = set(crohn["Sample_ID"].astype(str))
            for _,r in pairs.iterrows():
                o, f = str(r[ocol]), str(r[fcol])
                if o in sids and f in sids:
                    usable.append((o,f))
            fig, axes = plt.subplots(nrows=min(len(taxa_vis), 10), ncols=1,
                                     figsize=(8.2, 2.1*min(len(taxa_vis),10)), sharex=True)
            if not isinstance(axes, np.ndarray): axes = np.array([axes])
            for ax, t in zip(axes, taxa_vis[:len(axes)]):
                dd = crohn[crohn["taxon"]==t].set_index(["Sample_ID","site"])["CLR"].unstack("site")
                if dd is None or "Oral" not in dd or "Fecal" not in dd:
                    ax.axis("off"); continue
                inc = []
                for (o,f) in usable:
                    if o in dd.index and f in dd.index:
                        xo, xf = dd.at[o,"Oral"], dd.at[f,"Fecal"]
                        if pd.notna(xo) and pd.notna(xf):
                            ax.plot([0,1],[xo,xf], color="#2ca02c" if xf>xo else "#d62728",
                                    alpha=0.35, lw=1.2, marker="o")
                            inc.append(xf>xo)
                ax.axhline(0,ls=":",color="#aaa")
                lab = label_with_ppi(t)
                ax.set_title(f"{lab} — % increasing: {np.mean(inc)*100:.1f}%" if inc else lab)
            axes[-1].set_xticks([0,1]); axes[-1].set_xticklabels(["Oral","Fecal"])
            fig.tight_layout(); fig.savefig(os.path.join(base,"paired_spaghetti_crohn.png"),
                                            dpi=300, bbox_inches="tight"); plt.close(fig)
        else:
            save_empty_png(os.path.join(base,"paired_spaghetti_crohn.png"), "Pairs file missing Oral/Fecal cols")
    else:
        save_empty_png(os.path.join(base,"paired_spaghetti_crohn.png"), "No pairs file provided")

    print("[QC-targeted] Done →", base)

if __name__ == "__main__":
    main()
