#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC / Outliers

- Alpha outliers (Shannon, IQR) from existing alpha CSVs (oral / fecal / healthy)
- Beta outliers from PCoA (classical MDS) on a precomputed Bray–Curtis distance matrix:
    * Global: top-pct farthest from overall centroid
    * Within-group (optional): top-pct farthest from each group's centroid (>=1 per group)
- Missing IDs (metadata vs. matrices) and duplicates
- PCoA figures with outliers highlighted
- Colors are read from config.yaml

Outputs (under --outdir, e.g. results/qc/):
  outliers_alpha.csv
  outliers_beta.csv
  outliers_beta_within.csv
  missing_ids.csv
  duplicates.csv
  pcoa_bray_allgroups_outliers_marked.png
  pcoa_bray_within_groups_outliers_marked.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Set, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml  # for reading colors from config.yaml

# -------------------------- small utils --------------------------

def _normalize_ids(s: pd.Series) -> pd.Series:
    """Trim + uppercase IDs for robust matching across files."""
    return s.astype(str).str.strip().str.upper()

def _pick_first(cols, candidates, must=True, what=""):
    """Pick first matching column name (case/space/underscore tolerant)."""
    cols = [str(c) for c in cols]
    lut_exact = {c.lower(): c for c in cols}
    lut_compact = {"".join(c.lower().split()).replace("_", ""): c for c in cols}
    for cand in candidates:
        key = str(cand).lower()
        if key in lut_exact:
            return lut_exact[key]
        key2 = "".join(key.split()).replace("_", "")
        if key2 in lut_compact:
            return lut_compact[key2]
    if must:
        raise KeyError(f"Missing column for {what}; tried: {candidates}\nAvailable: {list(cols)}")
    return None

def _load_colors_from_config(config_path: str = "config.yaml") -> Dict[str, str]:
    """Read a colors dict from config.yaml. Fallback to sensible defaults."""
    cfg_p = Path(config_path)
    if not cfg_p.exists():
        return {"Oral": "#516D99", "Fecal": "#213547", "Healthy": "#59A14F", "Unknown": "#999999", "Other": "#999999"}
    with open(cfg_p, "r") as f:
        cfg = yaml.safe_load(f) or {}
    colors = cfg.get("colors", {}) or {}
    # add safe fallbacks if not provided
    colors.setdefault("Oral", "#516D99")
    colors.setdefault("Fecal", "#213547")
    colors.setdefault("Healthy", colors.get("Healthy-Oral", "#59A14F"))
    colors.setdefault("Unknown", colors.get("Other", "#999999"))
    colors.setdefault("Other", "#999999")
    return colors

def _color_for_group(g: str, colors: Dict[str, str]) -> str:
    """Resolve color for a group name using several possible keys."""
    for k in (g, f"Crohn-{g}", f"{g}-Oral", f"{g}-Fecal", "Other", "Unknown"):
        if k in colors:
            return colors[k]
    return "#999999"

# -------------------------- read alpha --------------------------

def read_alpha_csv(path: str) -> pd.DataFrame:
    """
    Read alpha CSV (results/alpha_diversity/alpha_*.csv).
    Normalizes to columns: sample_id, Shannon, [Simpson], [Richness], group
    Group is inferred from filename: contains 'oral' -> Oral, 'fecal' -> Fecal, 'healthy' -> Healthy
    """
    df = pd.read_csv(path)
    print(f"[alpha] reading {path} -> cols: {list(df.columns)}")

    sample_col  = _pick_first(df.columns,
                              ["sample_id","SampleID","Sample","ID","Oral_sample_ID","Fecal_sample_ID","Healthy_sample_ID"],
                              True, "sample_id")
    shannon_col = _pick_first(df.columns,
                              ["Shannon","shannon","Shannon_index","Shannon.index","H","H_"],
                              True, "Shannon")
    simpson_col = _pick_first(df.columns,
                              ["Simpson","simpson","D","Simpson_index","Simpson.index"],
                              must=False, what="Simpson")
    rich_col    = _pick_first(df.columns,
                              ["Richness","richness","Observed","Observed_ASVs","S","n_features"],
                              must=False, what="Richness")

    out = pd.DataFrame({
        "sample_id": _normalize_ids(df[sample_col]),
        "Shannon":   pd.to_numeric(df[shannon_col], errors="coerce"),
    })
    if simpson_col: out["Simpson"] = pd.to_numeric(df[simpson_col], errors="coerce")
    if rich_col:    out["Richness"] = pd.to_numeric(df[rich_col], errors="coerce")

    # infer group from filename
    name = Path(path).name.lower()
    if "healthy" in name:
        out["group"] = "Healthy"
    elif "fecal" in name or "faecal" in name:
        out["group"] = "Fecal"
    elif "oral" in name:
        out["group"] = "Oral"
    else:
        out["group"] = "Unknown"

    return out.dropna(subset=["sample_id"])

def alpha_outliers_iqr(alpha_df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR rule on Shannon per group.
    Returns: sample_id, group, metric=Shannon, q1,q3,iqr,lower_bound,upper_bound
    """
    rows = []
    for g, d in alpha_df.groupby("group"):
        x = pd.to_numeric(d["Shannon"], errors="coerce").dropna()
        if len(x) == 0:
            continue
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (d["Shannon"] < low) | (d["Shannon"] > high)
        sel = d.loc[mask, ["sample_id","group"]].copy()
        sel["metric"] = "Shannon"
        sel["q1"] = q1; sel["q3"] = q3; sel["iqr"] = iqr
        sel["lower_bound"] = low; sel["upper_bound"] = high
        rows.append(sel)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["sample_id","group","metric","q1","q3","iqr","lower_bound","upper_bound"])

# -------------------------- beta: read + PCoA --------------------------

def read_beta_dist_csv(path: str) -> pd.DataFrame:
    """
    Read a square distance matrix CSV where one column is the ID column
    (e.g., 'Unnamed: 0', 'Sample', 'ID', ...), and other columns are the same IDs.
    Returns a numeric DataFrame with normalized uppercase IDs, trimmed to rows/cols intersection.
    """
    df = pd.read_csv(path)
    id_col = _pick_first(df.columns, ["sample_id","SampleID","Sample","Unnamed: 0","ID","id"], True, "distance-matrix index")
    df[id_col] = _normalize_ids(df[id_col])
    df = df.set_index(id_col)
    df.columns = _normalize_ids(pd.Index(df.columns))
    df = df.apply(pd.to_numeric, errors="coerce")
    common = df.index.intersection(df.columns)
    df = df.loc[common, common]
    return df

def _classical_mds(D: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classical MDS (PCoA) on a distance matrix D (NxN).
    Returns:
      coords: (N x k) principal coordinates
      eigvals: (k,) eigenvalues used (for variance explained)
    """
    n = D.shape[0]
    if n == 0:
        return np.zeros((0, k)), np.zeros(k)
    # double-centering
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # eigendecomposition
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    pos = evals > 0
    if not np.any(pos):
        return np.zeros((n, k)), np.zeros(k)
    evals_pos = evals[pos]
    evecs_pos = evecs[:, pos]
    k_use = min(k, len(evals_pos))
    L = np.diag(np.sqrt(evals_pos[:k_use]))
    coords = evecs_pos[:, :k_use] @ L
    # pad if fewer than k components
    if coords.shape[1] < k:
        coords = np.hstack([coords, np.zeros((n, k - coords.shape[1]))])
        eigs = np.pad(evals_pos[:k_use], (0, k - coords.shape[1]), constant_values=0)
    else:
        eigs = evals_pos[:k_use]
    return coords, eigs

def build_coords_from_beta(dist_df: pd.DataFrame, k: int = 2) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run classical MDS on the given distance matrix DataFrame."""
    ids = dist_df.index.to_list()
    coords, eigs = _classical_mds(dist_df.to_numpy(dtype=float), k=k)
    coords_df = pd.DataFrame(coords, columns=[f"pcoa{i+1}" for i in range(k)])
    coords_df.insert(0, "sample_id", ids)
    return coords_df, eigs

# -------------------------- plotting --------------------------
def _plot_pcoa(coords: pd.DataFrame,
               outliers: Set[str],
               eigvals: np.ndarray,
               title: str,
               outfile: Path,
               colors_dict: Dict[str, str],
               groups_series: pd.Series | None = None):
    """Plot PCoA colored by group. Prefer coords['group']; if missing, use groups_series."""
    # variance explained
    if eigvals is not None and eigvals.sum() > 0:
        var = eigvals / eigvals.sum()
        xlab = f"PCo1 ({var[0]*100:.1f}% var)"
        ylab = f"PCo2 ({var[1]*100:.1f}% var)" if len(var) > 1 else "PCo2"
    else:
        xlab, ylab = "PCo1", "PCo2"

    df = coords.copy()
    # if coords has no 'group', try to add from series (indexed by sample_id)
    if "group" not in df.columns and groups_series is not None:
        df = df.join(groups_series.rename("group").astype(object).rename_axis("sample_id")
                     .reindex(df["sample_id"]).reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(9, 7))
    # plot each group
    if "group" in df.columns:
        for g in sorted(df["group"].fillna("Unknown").unique()):
            d = df[df["group"].fillna("Unknown") == g]
            if not d.empty:
                ax.scatter(d["pcoa1"], d["pcoa2"], s=40, alpha=0.95,
                           label=g, c=_color_for_group(g, colors_dict))
    else:
        # fallback (no group info)
        ax.scatter(df["pcoa1"], df["pcoa2"], s=40, alpha=0.95, label="All",
                   c=_color_for_group("Unknown", colors_dict))

    # outline outliers
    if outliers:
        d = df[df["sample_id"].isin(outliers)]
        ax.scatter(d["pcoa1"], d["pcoa2"], s=140, facecolors="none",
                   edgecolors="black", linewidths=2, label="Top outliers")

    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)

# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha-csv", action="append", required=True,
                    help="Alpha CSVs (pass oral, fecal, and optionally healthy).")
    ap.add_argument("--beta-dist-csv", required=True,
                    help="Square Bray–Curtis distance matrix CSV.")
    ap.add_argument("--meta", required=True, help="Clean metadata CSV.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml with colors.")
    ap.add_argument("--beta-top-pct", type=float, default=0.03,
                    help="Top fraction for beta outliers (e.g., 0.03 = top 3%).")
    ap.add_argument("--beta-within-group", action="store_true",
                    help="Also compute within-group outliers (>=1 per group).")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    colors_dict = _load_colors_from_config(args.config)

    # ---------- 1) Alpha ----------
    alpha_frames = [read_alpha_csv(p) for p in args.alpha_csv]
    if not alpha_frames:
        raise RuntimeError("No alpha CSVs provided.")
    alpha_all = pd.concat(alpha_frames, ignore_index=True)
    alpha_all["sample_id"] = _normalize_ids(alpha_all["sample_id"])

    alpha_out = alpha_outliers_iqr(alpha_all)
    alpha_out.to_csv(outdir / "outliers_alpha.csv", index=False)

    # ---------- 2) Beta distances -> PCoA ----------
    dists = read_beta_dist_csv(args.beta_dist_csv)
    coords_df, eigs = build_coords_from_beta(dists, k=2)
    coords_df["sample_id"] = _normalize_ids(coords_df["sample_id"])

    # label groups for coords using alpha presence (Oral / Fecal / Healthy)
    ids_oral    = set(alpha_all.loc[alpha_all["group"]=="Oral",    "sample_id"])
    ids_fecal   = set(alpha_all.loc[alpha_all["group"]=="Fecal",   "sample_id"])
    ids_healthy = set(alpha_all.loc[alpha_all["group"]=="Healthy", "sample_id"])
    def _grp(i):
        if i in ids_oral:    return "Oral"
        if i in ids_fecal:   return "Fecal"
        if i in ids_healthy: return "Healthy"
        return "Unknown"
    coords_df["group"] = coords_df["sample_id"].map(_grp)
    groups = coords_df.set_index("sample_id")["group"]  # Series used by plotting

    # ---------- 3) Missing & duplicates ----------
    meta = pd.read_csv(args.meta)

    # collect all IDs present in metadata (union of possible columns)
    meta_ids = set()
    for c in ["Oral_sample_ID","Fecal_sample_ID","Healthy_sample_ID","SampleID","sample_id","ID"]:
        if c in meta.columns:
            meta_ids |= set(_normalize_ids(meta[c].dropna()))

    ids_alpha  = set(alpha_all["sample_id"])
    ids_beta   = set(dists.index)
    ids_coords = set(coords_df["sample_id"])

    miss_beta   = sorted((ids_alpha | meta_ids) - ids_beta)
    miss_coords = sorted((ids_alpha | meta_ids) - ids_coords)
    miss_alpha  = sorted(ids_beta - ids_alpha)

    missing_long = pd.concat([
        pd.DataFrame({"where_missing": "beta_matrix", "sample_id": miss_beta}),
        pd.DataFrame({"where_missing": "pcoa_coords", "sample_id": miss_coords}),
        pd.DataFrame({"where_missing": "alpha_tables", "sample_id": miss_alpha}),
    ], ignore_index=True)
    if missing_long.empty:
        missing_long = pd.DataFrame(columns=["where_missing","sample_id"])
    missing_long.to_csv(outdir / "missing_ids.csv", index=False)

    dup_alpha = (alpha_all["sample_id"]
                 .value_counts().rename_axis("sample_id")
                 .reset_index(name="count")
                 .query("count > 1")
                 .assign(source="alpha"))
    dup_beta = (pd.Series(list(dists.index))
                .value_counts().rename_axis("sample_id")
                .reset_index(name="count")
                .query("count > 1")
                .assign(source="beta_matrix"))
    dup = pd.concat([dup_alpha, dup_beta], ignore_index=True)
    if dup.empty:
        dup = pd.DataFrame(columns=["sample_id","count","source"])
    dup.to_csv(outdir / "duplicates.csv", index=False)

    # ---------- 4) Beta outliers: GLOBAL ----------
    XY = coords_df[["pcoa1","pcoa2"]].to_numpy(dtype=float)
    centroid = XY.mean(axis=0)
    dist_global = np.linalg.norm(XY - centroid, axis=1)
    k_global = max(1, int(np.ceil(len(coords_df) * args.beta_top_pct)))
    thr_global = np.partition(dist_global, -k_global)[-k_global]
    mask_g = dist_global >= thr_global

    beta_global = coords_df.loc[mask_g, ["sample_id","group","pcoa1","pcoa2"]].copy()
    beta_global["dist_to_centroid_global"] = dist_global[mask_g]
    beta_global["is_outlier"] = 1
    beta_global.to_csv(outdir / "outliers_beta.csv", index=False)

    pcoa_all_fig = outdir / "pcoa_bray_allgroups_outliers_marked.png"
    _plot_pcoa(
        coords_df,
        set(beta_global["sample_id"]),
        eigs,
        "PCoA (Bray-Curtis) — global outliers",
        pcoa_all_fig,
        colors_dict,
        groups_series=coords_df.set_index("sample_id")["group"]
    )

    # ---------- 5) Beta outliers: WITHIN-GROUP ----------
    within_csv = outdir / "outliers_beta_within.csv"
    within_fig = outdir / "pcoa_bray_within_groups_outliers_marked.png"

    if args.beta_within_group:
        rows = []; marked = []
        for g, sub in coords_df.groupby("group"):
            if g == "Unknown" or len(sub) < 2:
                continue
            X = sub[["pcoa1","pcoa2"]].to_numpy(dtype=float)
            c = X.mean(axis=0)
            d = np.linalg.norm(X - c, axis=1)
            k = max(1, int(np.ceil(len(sub) * args.beta_top_pct)))
            thr = np.partition(d, -k)[-k]
            m = d >= thr
            sel = sub.loc[m, ["sample_id","group","pcoa1","pcoa2"]].copy()
            sel["dist_to_centroid_within"] = d[m]
            sel["is_outlier"] = 1
            rows.append(sel)
            marked.extend(sel["sample_id"].tolist())

        beta_within = (pd.concat(rows, ignore_index=True)
                       if rows else pd.DataFrame(columns=["sample_id","group","pcoa1","pcoa2",
                                                          "dist_to_centroid_within","is_outlier"]))
        beta_within.to_csv(within_csv, index=False)

        _plot_pcoa(
            coords_df,
            set(marked),
            eigs,
            "PCoA (Bray-Curtis) — within-group outliers",
            within_fig,
            colors_dict,
            groups_series=coords_df.set_index("sample_id")["group"]
        )

    else:
        # Write empty CSV & a plain figure (no highlighted points)
        pd.DataFrame(columns=["sample_id","group","pcoa1","pcoa2",
                              "dist_to_centroid_within","is_outlier"]).to_csv(within_csv, index=False)
        _plot_pcoa(
            coords_df,
            set(),
            eigs,
            "PCoA (Bray-Curtis) — within-group outliers",
            within_fig,
            colors_dict,
            groups_series=coords_df.set_index("sample_id")["group"]
        )

    print("[OK] QC outliers finished.")

if __name__ == "__main__":
    main()
