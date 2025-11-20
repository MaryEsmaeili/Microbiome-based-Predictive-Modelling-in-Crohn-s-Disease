#!/usr/bin/env python3
import os, argparse, re
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from numpy.random import default_rng
import statsmodels.api as sm

# ================================================================
# Small utilities
# ================================================================

def ensure_dir(p: str):
    """
    Create directory `p` if it does not exist.
    Does nothing if `p` is empty or already exists.
    """
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def save_empty_png(path: str, msg: str):
    """
    Save a placeholder PNG with a central message.
    Used when there is not enough data to make a real plot.
    """
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def _pick(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    """
    Pick the first column in `df` matching any of the candidate names
    (case-insensitive). Returns the actual column name or None.
    """
    low = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None

def to01(x):
    """
    Map various encodings to {0,1} for binary variables:
    - Positive (1): yes, true, crohn, cd, case, ibd, etc.
    - Negative (0): no, false, healthy, control, non-ibd, etc.
    Returns np.nan when mapping is not clear.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    pos = {"1","yes","true","y","crohn","cd","case","ibd"}
    neg = {"0","no","false","n","healthy","control","hc","non-ibd","nonibd"}
    if s in pos:
        return 1
    if s in neg:
        return 0
    try:
        v = int(float(s))
        return v if v in (0, 1) else np.nan
    except Exception:
        return np.nan

def normalize_meta(meta_csv: str) -> pd.DataFrame:
    """
    Load and normalize metadata into a common schema.

    Returned columns (if available):
      - sample_id
      - site         (Oral/Fecal; others set to NaN)
      - disease      (0/1 via `to01`)
      - ppi_use      (0/1)
      - age          (numeric)
      - sex          (capitalized string)
      - bmi          (numeric)
      - antibiotics_3m (0/1)
      - smoking      (0/1)
    """
    m = pd.read_csv(meta_csv)
    m.columns = [str(c).strip() for c in m.columns]

    out = pd.DataFrame({
        "sample_id":       m[_pick(m, ["sample_id","id","sample","sampleid"])].astype(str),
        "site":            m[_pick(m, ["site","body_site","location"])],
        "disease":         m[_pick(m, ["disease","status","group","label"])],
        "ppi_use":         m[_pick(m, ["ppi_use","ppi","ppi_current","ppi3m"])],
        "age":             m[_pick(m, ["age"])],
        "sex":             m[_pick(m, ["sex","gender"])],
        "bmi":             m[_pick(m, ["bmi"])],
        "antibiotics_3m":  m[_pick(m, ["antibiotics_3m","abx_3m","antibiotics_last3m","abx3m"])],
        "smoking":         m[_pick(m, ["smoking","smoker"])],
    })

    # Normalize site to 'Oral' / 'Fecal'
    out["site"] = out["site"].astype(str).str.strip().str.capitalize().replace({"Faecal":"Fecal"})
    out.loc[~out["site"].isin(["Oral","Fecal"]), "site"] = np.nan

    # Map binary covariates to 0/1
    out["disease"]         = out["disease"].map(to01)
    out["ppi_use"]         = out["ppi_use"].map(to01)
    out["antibiotics_3m"]  = out["antibiotics_3m"].map(to01)
    out["smoking"]         = out["smoking"].map(to01)

    # Numeric covariates
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["bmi"] = pd.to_numeric(out["bmi"], errors="coerce")

    # Sex as a clean string (no numeric encoding here)
    if "sex" in out:
        out["sex"] = out["sex"].astype(str).str.strip().str.capitalize()

    return out

def clr_on_percent(pct_df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """
    Compute CLR transformation from a percent-abundance table.

    Input:
      - pct_df: taxa x samples, entries in [0,100]
    Output:
      - same shape, centered log-ratio values
    """
    X = (pct_df.astype(float) / 100.0) + pseudocount
    L = np.log(X)
    gm = L.mean(axis=0)  # geometric mean (in log space)
    return L.sub(gm, axis=1)

def prevalence_mean_filter(pct_df: pd.DataFrame, prevalence: float, mean_pct: float) -> pd.Index:
    """
    Select taxa that pass a global filter:
      - prevalence: fraction of samples with >0
      - mean_pct : mean percentage abundance across all samples

    Returns an index of taxa (row names) to keep.
    """
    nz_frac = (pct_df > 0).sum(axis=1) / pct_df.shape[1]
    mean_vals = pct_df.mean(axis=1)
    keep = (nz_frac >= prevalence) & (mean_vals >= mean_pct)
    return pct_df.index[keep]

def pretty_label(t: str) -> str:
    """
    Prettify a clade name for plotting (MetaPhlAn-like):
      - prefer species label after '|s__'
      - otherwise genus after '|g__'
      - otherwise last pipe-separated token
      - underscores replaced by spaces
    """
    s = str(t)
    if "|s__" in s:
        s = s.split("|s__")[-1]
    elif "|g__" in s:
        s = s.split("|g__")[-1]
    else:
        s = s.split("|")[-1]
    return s.replace("_", " ")

# ================================================================
# PCA & PERMANOVA
# ================================================================

def pca_scatter(M: pd.DataFrame, meta_idx: pd.DataFrame, out_png: str,
                title: str = "PCA (CLR)"):
    """
    Simple PCA (via SVD) on CLR matrix and scatter by site/disease.

    Parameters
    ----------
    M : DataFrame
        Samples x taxa CLR matrix (rows: Sample_ID).
    meta_idx : DataFrame
        Metadata indexed by sample_id, with columns 'site' and 'disease'.
    out_png : str
        Path to save the PNG.
    title : str
        Plot title.
    """
    ensure_dir(os.path.dirname(out_png))
    if M.shape[0] < 3 or M.shape[1] < 2:
        save_empty_png(out_png, "Not enough data for PCA")
        return

    X = M.values.astype(float)
    # Center features
    X = X - X.mean(axis=0, keepdims=True)

    # SVD-based PCA
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pcs = U[:, :2] * S[:2]

    # Explained variance
    var = (S**2) / (X.shape[0] - 1)
    var = var / var.sum()

    fig, ax = plt.subplots(figsize=(7.3, 6.1))
    color = {"Oral":"#00798c", "Fecal":"#d1495b"}   # site colors
    marker = {0:"o", 1:"s"}                        # disease shapes

    # Align meta to M rows
    meta_idx = meta_idx.reindex(M.index)

    # Group by (site, disease) for plotting
    for (site, dis), idx in meta_idx.reset_index().groupby(["site","disease"]).groups.items():
        pts = pcs[list(idx), :]
        lab = f"{site}, {'Healthy' if dis == 0 else 'Crohn'} (n={len(idx)})"
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=color.get(site, "#777"),
            marker=marker.get(dis, "o"),
            s=60,
            alpha=0.9,
            label=lab
        )

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.legend(frameon=False, ncol=2)
    ax.set_title(title)
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.axvline(0, color="#bbb", lw=0.8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def permanova_one_factor(M: pd.DataFrame, meta_idx: pd.DataFrame, factor: str,
                         n_perm: int = 999, seed: int = 1) -> Tuple[float,float,float,int]:
    """
    One-factor PERMANOVA using Euclidean distance on CLR matrix.

    Parameters
    ----------
    M : DataFrame
        Samples x taxa CLR matrix (rows: Sample_ID).
    meta_idx : DataFrame
        Metadata indexed by sample_id, must contain column `factor`.
    factor : str
        Column name in meta_idx to test (e.g. 'site', 'disease', 'ppi_use').
    n_perm : int
        Number of permutations.
    seed : int
        Random seed for permutation.

    Returns
    -------
    Fobs : float
        Observed F-statistic.
    R2 : float
        R-squared (effect size).
    p : float
        Permutation p-value (upper tail).
    eff : int
        Effective number of permutations used.
    """
    rng = default_rng(seed)
    meta_idx = meta_idx.reindex(M.index)
    y = meta_idx[factor].values

    # Restrict to non-missing labels
    mask = ~pd.isna(y)
    X = M.values[mask, :].astype(float)
    y = y[mask].astype(str)

    if X.shape[0] < 4 or len(pd.unique(y)) < 2:
        return (np.nan, np.nan, np.nan, 0)

    # Center
    X = X - X.mean(axis=0, keepdims=True)

    # Distance matrix
    D = squareform(pdist(X, metric="euclidean"))
    n = X.shape[0]
    levels = pd.unique(y)

    def anova_terms(labels):
        """
        Compute SSB/SSW-based F and R2 for PERMANOVA given group labels.
        """
        SSW, dfw = 0.0, 0
        for lv in levels:
            idx = np.where(labels == lv)[0]
            if len(idx) <= 1:
                continue
            Dg = D[np.ix_(idx, idx)]
            # Within-group sum of squares following Anderson's PERMANOVA logic
            SSW += (Dg**2).sum() / len(idx)
            dfw += (len(idx) - 1)
        TSS = (D**2).sum() / n
        SSB = TSS - SSW
        dfb = len(levels) - 1
        MSB = SSB / dfb if dfb > 0 else np.nan
        MSW = SSW / dfw if dfw > 0 else np.nan
        F = MSB / MSW if (MSW > 0 and np.isfinite(MSB)) else np.nan
        R2 = SSB / TSS if TSS > 0 else np.nan
        return F, R2

    # Observed statistic
    Fobs, R2obs = anova_terms(y)

    # Permutations
    ge, eff = [], 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if np.array_equal(yp, y):
            # Skip identity permutation
            continue
        Fp, _ = anova_terms(yp)
        if np.isfinite(Fp):
            ge.append(Fp)
            eff += 1

    if eff == 0 or not np.isfinite(Fobs):
        return (np.nan, float(R2obs), np.nan, eff)

    ge = np.array(ge)
    p = (1 + (ge >= Fobs).sum()) / (1 + eff)
    return (float(Fobs), float(R2obs), float(p), eff)

# ================================================================
# Covariate residualization
# ================================================================

def residualize(clr: pd.DataFrame, meta: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    """
    Residualize CLR values w.r.t. selected covariates using OLS.

    For each taxon:
      - Fit OLS: CLR ~ covariates_to_remove
      - Return residuals of that model (so variation due to these covariates is removed).
      - Covariates listed in `keep_cols` are *not* regressed out.

    Parameters
    ----------
    clr : DataFrame
        Taxa x samples CLR matrix.
    meta : DataFrame
        Metadata with a 'sample_id' column.
    keep_cols : list
        Covariate names that should NOT be removed (kept in the residual).

    Returns
    -------
    DataFrame with same shape as `clr`, containing residual CLR values.
    """
    if "sample_id" not in meta.columns:
        # Make sure `sample_id` is a column, not an index
        if meta.index.name == "sample_id":
            meta = meta.reset_index()
        else:
            meta = meta.reset_index().rename(columns={"index": "sample_id"})

    meta_idx = meta.set_index("sample_id").reindex(clr.columns)

    # Candidate covariates we might remove
    cand = ["age","sex","bmi","antibiotics_3m","smoking","ppi_use","disease","site"]
    covs = [c for c in cand if c in meta_idx.columns and c not in keep_cols]
    if not covs:
        # Nothing to residualize out
        return clr.copy()

    # One-hot encode categorical covariates (e.g. sex, site)
    X = pd.get_dummies(meta_idx[covs], drop_first=True)
    X = sm.add_constant(X, has_constant="add").apply(pd.to_numeric, errors="coerce")

    try:
        Xv = X.to_numpy(dtype=float)
    except (TypeError, ValueError):
        Xv = np.asarray(X.values, dtype=float)

    rows = []
    for tax in clr.index:
        y = pd.to_numeric(clr.loc[tax], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y) & np.all(np.isfinite(Xv), axis=1)
        if ok.sum() < 8:
            # Too few usable samples, keep original values
            rows.append(pd.Series(y, index=clr.columns))
            continue
        m = sm.OLS(y[ok], Xv[ok], hasconst=True).fit()
        r = np.full_like(y, np.nan, dtype=float)
        r[ok] = y[ok] - m.predict(Xv[ok])
        rows.append(pd.Series(r, index=clr.columns))

    return pd.DataFrame(rows, index=clr.index, columns=clr.columns)

# ================================================================
# Plotting helpers
# ================================================================

def kde_site(long_df: pd.DataFrame, taxa: List[str], site: str, out_png: str):
    """
    Plot CLR density (Gaussian KDE) per taxon within a given site.
    """
    d = long_df[(long_df["site"] == site) & (long_df["taxon"].isin(taxa))]
    if d.empty:
        save_empty_png(out_png, f"No data for {site}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for t in d["taxon"].unique():
        v = d.loc[d["taxon"] == t, "CLR"].astype(float).values
        v = v[np.isfinite(v)]
        if len(v) > 1 and np.nanstd(v) > 1e-12:
            try:
                xs = np.linspace(np.min(v) - 1, np.max(v) + 1, 200)
                kde = gaussian_kde(v)
                ax.plot(xs, kde(xs), lw=2, label=pretty_label(t), alpha=0.9)
            except Exception:
                # Silently skip degenerate cases
                pass

    ax.set_xlabel("CLR")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.set_title(f"CLR density — {site}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def box_by_group(long_df: pd.DataFrame, taxa: List[str], site: str, out_png: str,
                 group_col: str, groups=(0,1), labels=("0","1")):
    """
    Boxplots of CLR for selected taxa, split by group_col (e.g. disease or ppi_use),
    within a given site.

    Each taxon is plotted as side-by-side boxplots for the two groups.
    """
    d = long_df[(long_df["site"] == site) & (long_df["taxon"].isin(taxa))]
    if d.empty:
        save_empty_png(out_png, f"No data for {site}")
        return

    taxa_list = list(taxa)
    pos = np.arange(len(taxa_list))
    fig, ax = plt.subplots(figsize=(min(16, 1.4*len(taxa_list)+6), 5.5))

    for i, t in enumerate(taxa_list):
        vals = []
        for g in groups:
            vals.append(d[(d["taxon"] == t) & (d[group_col] == g)]["CLR"].values)
        offs = np.linspace(-0.18, 0.18, len(groups))
        for v, off in zip(vals, offs):
            bp = ax.boxplot([v], positions=[i + off], widths=0.32, patch_artist=True)
            for b in bp['boxes']:
                b.set(facecolor="#9ad1d4" if off < 0 else "#2b8a9a")

    ax.set_xticks(pos)
    ax.set_xticklabels([pretty_label(t) for t in taxa_list], rotation=60, ha="right")
    ax.set_ylabel("CLR")
    ax.set_title(f"CLR boxplots — {site} ({group_col})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def interaction_means(long_df: pd.DataFrame, taxa: List[str], out_png: str):
    """
    Line plot of mean CLR by site (Oral vs Fecal) for each taxon,
    stratified by disease (Healthy vs Crohn).

    This visually highlights disease × site interactions for the selected taxa.
    """
    sel = long_df[long_df["taxon"].isin(taxa)]
    if sel.empty:
        save_empty_png(out_png, "No data")
        return

    fig, ax = plt.subplots(figsize=(max(12, 1.0*len(taxa)+6), 6))
    xbase = {t: i for i, t in enumerate(taxa)}

    for t in taxa:
        for dis, ls, mk in [(0, "-", "o"), (1, "--", "s")]:
            vo = sel[(sel["taxon"] == t) & (sel["site"] == "Oral")  & (sel["disease"] == dis)]["CLR"].values
            vf = sel[(sel["taxon"] == t) & (sel["site"] == "Fecal") & (sel["disease"] == dis)]["CLR"].values
            if len(vo) == 0 or len(vf) == 0:
                continue
            ax.plot(
                [xbase[t] + 0.0, xbase[t] + 0.6],
                [np.mean(vo), np.mean(vf)],
                ls=ls, marker=mk, lw=2
            )

    ax.set_xticks([xbase[t] + 0.3 for t in taxa])
    ax.set_xticklabels([pretty_label(t) for t in taxa], rotation=55, ha="right")
    ax.set_ylabel("Mean CLR")
    ax.axhline(0, ls=":", color="#aaa")
    ax.set_title("Interaction means (disease × site)")
    ax.legend(
        [plt.Line2D([0],[0], ls="-",  marker="o"),
         plt.Line2D([0],[0], ls="--", marker="s")],
        ["Healthy", "Crohn"],
        frameon=False,
        loc="upper left"
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def spaghetti_pairs(long_df: pd.DataFrame, pairs_csv: Optional[str], taxa: List[str], out_png: str):
    """
    Spaghetti plots for Crohn paired Oral–Fecal samples.

    Each panel shows one taxon; each line is a pair (Oral→Fecal) for a Crohn subject.
    Colors indicate direction (increase vs decrease from Oral to Fecal).
    """
    if not pairs_csv or not os.path.exists(pairs_csv):
        save_empty_png(out_png, "No pairs file")
        return

    pairs = pd.read_csv(pairs_csv)
    cols = {c.lower(): c for c in pairs.columns}
    ocol = cols.get("oral") or cols.get("oral_id") or cols.get("oc")
    fcol = cols.get("fecal") or cols.get("fecal_id") or cols.get("fc")
    if not (ocol and fcol):
        save_empty_png(out_png, "Pairs missing Oral/Fecal")
        return

    crohn = long_df[long_df["disease"] == 1]
    sids = set(crohn["Sample_ID"].astype(str))

    # Build list of usable (oral_id, fecal_id) pairs
    usable = [
        (str(r[ocol]), str(r[fcol]))
        for _, r in pairs.iterrows()
        if str(r[ocol]) in sids and str(r[fcol]) in sids
    ]
    if not usable:
        save_empty_png(out_png, "No Crohn pairs found")
        return

    rows = min(len(taxa), 10)
    fig, axes = plt.subplots(rows, 1, figsize=(8.4, 2.0*rows), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, t in zip(axes, taxa[:rows]):
        dd = crohn[crohn["taxon"] == t].set_index(["Sample_ID","site"])["CLR"].unstack("site")
        if dd is None or "Oral" not in dd or "Fecal" not in dd:
            ax.axis("off")
            continue
        for o, f in usable:
            if (
                o in dd.index and f in dd.index and
                pd.notna(dd.at[o,"Oral"]) and pd.notna(dd.at[f,"Fecal"])
            ):
                ax.plot(
                    [0, 1],
                    [dd.at[o,"Oral"], dd.at[f,"Fecal"]],
                    color="#2ca02c" if dd.at[f,"Fecal"] > dd.at[o,"Oral"] else "#d62728",
                    alpha=0.35, lw=1.2, marker="o"
                )
        ax.axhline(0, ls=":", color="#aaa")
        ax.set_title(pretty_label(t))

    axes[-1].set_xticks([0, 1])
    axes[-1].set_xticklabels(["Oral", "Fecal"])

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ================================================================
# Core helpers for long-format tables and feature selection
# ================================================================

def make_long(matrix: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Convert taxa x samples CLR matrix to a long-form table and merge metadata.

    Output columns:
      - Sample_ID
      - taxon
      - CLR
      - all metadata columns (merged by Sample_ID)
    """
    M_T = matrix.T
    long = M_T.stack().reset_index()
    long.columns = ["Sample_ID","taxon","CLR"]
    return long.merge(
        meta.rename(columns={"sample_id":"Sample_ID"}),
        on="Sample_ID",
        how="left"
    )

def select_top_by_site(clr: pd.DataFrame, site_vec: pd.Series, cap: int) -> Dict[str, List[str]]:
    """
    For each site (Oral/Fecal), select up to `cap` taxa with highest variance
    within that site. Used to choose features for visualization.

    Returns a dict: { "Oral": [taxa...], "Fecal": [taxa...] }.
    """
    top = {}
    for site in ["Oral","Fecal"]:
        mask = (site_vec == site).reindex(clr.columns).fillna(False).values
        if mask.sum() < 3:
            top[site] = []
            continue
        var = pd.Series(
            np.var(clr.loc[:, mask], axis=1),
            index=clr.index
        ).sort_values(ascending=False)
        top[site] = list(var.index[:min(cap, len(var))])
    return top

# ================================================================
# Main CLI
# ================================================================

def main():
    """
    Entry point for the QC / PCA / PERMANOVA visualization module.

    High-level steps:
      1) Load percent abundance table and metadata.
      2) Normalize metadata, restrict to overlapping samples.
      3) Global taxa filter by prevalence and mean percentage.
      4) Build CLR matrix and then:
         - NOCOV: no residualization.
         - WITHCOV: residualize against covariates, keep site+disease.
         - PPI_WITHCOV: focus on PPI users (Crohn ± healthy) and residualize.
      5) For each mode:
         - Select top-variance taxa per site for plotting.
         - Run PCA and PERMANOVA.
         - Generate density plots, boxplots, interaction plots, and paired spaghetti plots.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct-all", required=True, help="Percent table for all samples (taxa x samples).")
    ap.add_argument("--meta", required=True, help="Metadata CSV.")
    ap.add_argument("--pairs", default=None, help="Pairs CSV for Oral/Fecal Crohn subjects (for spaghetti plots).")
    ap.add_argument("--rank", required=True, choices=["genus","species"], help="Taxonomic rank label used only for naming subfolder.")
    ap.add_argument("--outdir", required=True, help="Base output directory.")
    ap.add_argument("--prevalence", type=float, default=0.20, help="Global prevalence filter (fraction of samples>0).")
    ap.add_argument("--mean-pct",  type=float, default=0.10, help="Global mean abundance filter in percent.")
    ap.add_argument("--cap", type=int, default=30, help="Maximum number of top-variance taxa per site.")
    ap.add_argument("--modes", nargs="+", choices=["NOCOV","WITHCOV","PPI_WITHCOV"],
                    default=["NOCOV","WITHCOV","PPI_WITHCOV"],
                    help="Which analysis modes to run.")
    ap.add_argument("--ppi-include-healthy", action="store_true",
                    help="For PPI_WITHCOV mode, include healthy with PPI usage as well.")
    args = ap.parse_args()

    base = os.path.join(args.outdir, args.rank)
    ensure_dir(base)

    # ------------------------------------------------------------
    # Load percent table and convert from proportions if needed
    # ------------------------------------------------------------
    pct = pd.read_csv(args.pct_all, index_col=0)
    pct.columns = pct.columns.astype(str)

    nz = pct.values[pct.values > 0]
    # Heuristic: if median non-zero value <=1.0, assume [0,1] and convert to percent
    if nz.size and np.nanmedian(nz) <= 1.0:
        pct = pct * 100.0

    # ------------------------------------------------------------
    # Load / normalize metadata and align with pct columns
    # ------------------------------------------------------------
    meta = normalize_meta(args.meta).dropna(subset=["site","disease"])
    pct = pct.loc[:, pct.columns.isin(meta["sample_id"].astype(str))]
    meta = meta[meta["sample_id"].isin(pct.columns)]

    if pct.empty:
        # No overlapping samples between abundance and metadata
        for suf in ["_NOCOV","_WITHCOV","_PPI_WITHCOV"]:
            save_empty_png(os.path.join(base, f"pca_scatter{suf}.png"), "No samples overlap")
        print("[QC] No overlap between pct and meta.")
        return

    # ------------------------------------------------------------
    # Global taxa filtering by prevalence & mean abundance
    # ------------------------------------------------------------
    keep = prevalence_mean_filter(pct, args.prevalence, args.mean_pct)
    if len(keep) == 0:
        # Nothing passes global filter → create placeholders and a dummy PERMANOVA file
        for suf in ["_NOCOV","_WITHCOV","_PPI_WITHCOV"]:
            for f in ["pca_scatter","density_top_taxa_oral","density_top_taxa_fecal",
                      "interaction_means","box_oral","box_fecal","paired_spaghetti_crohn"]:
                save_empty_png(os.path.join(base, f"{f}{suf}.png"), "No taxa after filtering")
            pd.DataFrame([{"factor":"site","F":np.nan,"R2":np.nan,"p":np.nan,"n_perm":0}])\
              .to_csv(os.path.join(base, f"permanova{suf}.tsv"), sep="\t", index=False)
        print("[QC] No taxa after filters.")
        return

    # CLR on filtered taxa
    clr_all = clr_on_percent(pct.loc[keep])

    # ------------------------------------------------------------
    # Build analysis tracks (NOCOV, WITHCOV, PPI_WITHCOV)
    # ------------------------------------------------------------
    tracks = []

    # disease/site tracks (with/without covariate adjustment)
    if "NOCOV" in args.modes or "WITHCOV" in args.modes:
        # For NOCOV we do not residualize, but we treat it as a track with these "kept" columns
        tracks.append(("NOCOV",   "site_disease", ["site","disease"], ["age","sex","bmi","antibiotics_3m","smoking","ppi_use"]))
        # For WITHCOV we will residualize and keep site+disease
        tracks.append(("WITHCOV", "site_disease", ["site","disease"], []))

    # PPI track: subset on PPI usage
    if "PPI_WITHCOV" in args.modes:
        mm = meta.copy()
        if args.ppi_include_healthy:
            # Include any sample with PPI usage
            mm = mm[mm["ppi_use"].notna()]
        else:
            # Restrict to Crohn with PPI usage
            mm = mm[(mm["ppi_use"].notna()) & (mm["disease"] == 1)]

        if "sample_id" not in mm.columns:
            if mm.index.name == "sample_id":
                mm = mm.reset_index()
            else:
                mm = mm.reset_index().rename(columns={"index": "sample_id"})

        common = clr_all.columns.intersection(mm["sample_id"])
        tracks.append((
            "PPI_WITHCOV",
            "ppi",
            ["site","ppi_use"],
            [],
            clr_all.loc[:, common],
            mm.set_index("sample_id").loc[common].reset_index()
        ))

    # ------------------------------------------------------------
    # Run each track
    # ------------------------------------------------------------
    site_vec = meta.set_index("sample_id")["site"]

    for mode, kind, keep_cols, _skip_cov, *opt in tracks:
        suffix = f"_{mode}"

        # Decide which CLR matrix & metadata to use in this track
        if kind == "ppi":
            clr = opt[0]
            meta_use = opt[1]
            if clr.shape[1] < 6:
                # Too few PPI samples – just placeholders and dummy PERMANOVA
                for f in ["pca_scatter","density_top_taxa_oral","density_top_taxa_fecal",
                          "interaction_means","box_oral","box_fecal","paired_spaghetti_crohn"]:
                    save_empty_png(os.path.join(base, f"{f}{suffix}.png"), "Too few PPI samples")
                pd.DataFrame([
                    {"factor":"ppi_use","F":np.nan,"R2":np.nan,"p":np.nan,"n_perm":0},
                    {"factor":"site","F":np.nan,"R2":np.nan,"p":np.nan,"n_perm":0}
                ]).to_csv(os.path.join(base, f"permanova{suffix}.tsv"), sep="\t", index=False)
                continue
        else:
            # Use all CLR and all metadata
            clr = clr_all.copy()
            meta_use = meta.copy()
            if "sample_id" not in meta_use.columns:
                if meta_use.index.name == "sample_id":
                    meta_use = meta_use.reset_index()
                else:
                    meta_use = meta_use.reset_index().rename(columns={"index": "sample_id"})

        # --------------------------------------------------------
        # Residualization mode
        # --------------------------------------------------------
        if mode == "NOCOV":
            # No covariates removed
            adj = clr
        elif mode == "WITHCOV":
            # Remove all candidate covariates except site + disease
            adj = residualize(clr, meta_use, keep_cols=["site","disease"])
        elif mode == "PPI_WITHCOV":
            # For PPI, keep site + ppi_use, and optionally disease (if healthy excluded)
            keep_cols_ppi = ["site","ppi_use"] + ([] if args.ppi_include_healthy else ["disease"])
            adj = residualize(clr, meta_use, keep_cols=keep_cols_ppi)
        else:
            adj = clr

        # Make sure we have a 'sample_id' column
        if "sample_id" not in meta_use.columns:
            if meta_use.index.name == "sample_id":
                meta_use = meta_use.reset_index()
            else:
                meta_use = meta_use.reset_index().rename(columns={"index": "sample_id"})

        # --------------------------------------------------------
        # Site-specific top taxa selection by variance
        # --------------------------------------------------------
        site_series = meta_use.set_index("sample_id")["site"]
        top = select_top_by_site(adj, site_series, args.cap)

        # Save which taxa were used for plots in each site
        for site in ["Oral","Fecal"]:
            taxa_vis = top.get(site, [])
            pd.DataFrame({"taxon": taxa_vis})\
              .to_csv(os.path.join(base, f"top_taxa_{site.lower()}{suffix}.csv"), index=False)

        # Build union of taxa used in either site
        taxa_union = list(dict.fromkeys((top["Oral"] or []) + (top["Fecal"] or [])))
        if not taxa_union:
            # No taxa selected for plotting
            for f in ["pca_scatter","density_top_taxa_oral","density_top_taxa_fecal",
                      "interaction_means","box_oral","box_fecal","paired_spaghetti_crohn"]:
                save_empty_png(os.path.join(base, f"{f}{suffix}.png"), "No taxa selected")
            continue

        adj_use = adj.loc[taxa_union]
        long = make_long(adj_use, meta_use)

        # --------------------------------------------------------
        # Sample counts: who actually went into this mode
        # --------------------------------------------------------
        counts = (
            long.assign(
                disease=long["disease"].astype("Int64"),
                ppi_use=long["ppi_use"].astype("Int64")
            )
            .drop_duplicates(["Sample_ID"])
        )

        # site × disease counts
        counts_sd = counts.value_counts(["site","disease"], dropna=False).reset_index(name="n")
        counts_sd.to_csv(os.path.join(base, f"sample_counts_site_disease{suffix}.csv"), index=False)

        # site × ppi_use counts (only for PPI track)
        if kind == "ppi":
            counts_sp = counts.value_counts(["site","ppi_use"], dropna=False).reset_index(name="n")
            counts_sp.to_csv(os.path.join(base, f"sample_counts_site_ppi{suffix}.csv"), index=False)

        # --------------------------------------------------------
        # Prepare wide matrix (samples x taxa) for PCA / PERMANOVA
        # --------------------------------------------------------
        M = long.pivot_table(index="Sample_ID", columns="taxon", values="CLR")

        # PCA
        title = "PCA (CLR)" if kind != "ppi" else "PCA (CLR residuals) — PPI"
        pca_scatter(M, meta_use.set_index("sample_id"),
                    os.path.join(base, f"pca_scatter{suffix}.png"),
                    title=title)

        # PERMANOVA
        if kind == "ppi":
            # Test PPI effect and site effect
            F1, R21, p1, eff1 = permanova_one_factor(M, meta_use.set_index("sample_id"), "ppi_use")
            F2, R22, p2, eff2 = permanova_one_factor(M, meta_use.set_index("sample_id"), "site")
            pd.DataFrame([
                {"factor":"ppi_use","F":F1,"R2":R21,"p":p1,"n_perm":eff1},
                {"factor":"site","F":F2,"R2":R22,"p":p2,"n_perm":eff2}
            ]).to_csv(os.path.join(base, f"permanova{suffix}.tsv"), sep="\t", index=False)

            # PPI boxplots by site (group = ppi_use)
            box_by_group(long, top.get("Oral", []),  "Oral",
                         os.path.join(base, f"box_oral{suffix}.png"),
                         group_col="ppi_use", groups=(0,1), labels=("No","Yes"))
            box_by_group(long, top.get("Fecal", []), "Fecal",
                         os.path.join(base, f"box_fecal{suffix}.png"),
                         group_col="ppi_use", groups=(0,1), labels=("No","Yes"))
        else:
            # Test site effect and disease effect
            F1, R21, p1, eff1 = permanova_one_factor(M, meta_use.set_index("sample_id"), "site")
            F2, R22, p2, eff2 = permanova_one_factor(M, meta_use.set_index("sample_id"), "disease")
            pd.DataFrame([
                {"factor":"site","F":F1,"R2":R21,"p":p1,"n_perm":eff1},
                {"factor":"disease","F":F2,"R2":R22,"p":p2,"n_perm":eff2}
            ]).to_csv(os.path.join(base, f"permanova{suffix}.tsv"), sep="\t", index=False)

            # Disease boxplots by site
            box_by_group(long, top.get("Oral", []),  "Oral",
                         os.path.join(base, f"box_oral{suffix}.png"),
                         group_col="disease")
            box_by_group(long, top.get("Fecal", []), "Fecal",
                         os.path.join(base, f"box_fecal{suffix}.png"),
                         group_col="disease")

        # Density plots, interaction plots, and paired spaghetti
        kde_site(long, top.get("Oral", []),  "Oral",
                 os.path.join(base, f"density_top_taxa_oral{suffix}.png"))
        kde_site(long, top.get("Fecal", []), "Fecal",
                 os.path.join(base, f"density_top_taxa_fecal{suffix}.png"))
        interaction_means(long, taxa_union,
                          os.path.join(base, f"interaction_means{suffix}.png"))
        spaghetti_pairs(long, args.pairs, taxa_union,
                        os.path.join(base, f"paired_spaghetti_crohn{suffix}.png"))

    print("[QC] Done ->", base)

if __name__ == "__main__":
    main()
