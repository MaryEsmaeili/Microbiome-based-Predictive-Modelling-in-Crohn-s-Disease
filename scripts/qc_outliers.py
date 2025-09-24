#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC outlier detection for ML wide tables.

Inputs
------
--wide-clr : required, CLR-wide table (rows=samples, cols=features)
--wide-all : optional, NON-CLR wide table (for zero-fraction)
--rank     : genus|species
--outdir   : output directory

Key logic
---------
- zero_fraction: from --wide-all if provided, else from --wide-clr (≈ zeros by |x|<1e-12)
- norm_zscore : L2 norm of CLR row vector, standardized across samples
- exclude if: zero_fraction >= max_zero_frac OR norm_z > max_norm_z OR (Mahalanobis p < mah_p)
- (optional) Mahalanobis with chi2 test, if --mah-p > 0 and scipy available

Outputs (inside outdir)
-----------------------
ml_{rank}_wide_clr.qc.csv
qc_report_{rank}.csv
qc_meta_{rank}.json
excluded_samples_{rank}.txt
qc_scatter_{rank}.png
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ensure non-interactive backend for PNG saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide-clr", required=True, dest="wide_clr",
                    help="CLR-wide table (rows: samples)")
    ap.add_argument("--wide-all", required=False, dest="wide_all",
                    help="NON-CLR wide table (used to compute zero-fraction)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--rank", choices=["genus", "species"], required=True)

    # thresholds
    ap.add_argument("--max-zero-frac", type=float, default=0.98,
                    help="Samples with zero fraction >= this are excluded (default: 0.98)")
    ap.add_argument("--max-norm-z", type=float, default=5.0,
                    help="Samples with CLR L2-norm z-score > this are excluded (default: 5.0)")
    ap.add_argument("--mah-p", type=float, default=-1.0,
                    help="If >0, apply Mahalanobis chi2 test at p-threshold (e.g., 0.001). Default: disabled")
    ap.add_argument("--mah-p-site", type=float, default=-1.0,
                    help="(Reserved) Within-site Mahalanobis. Not used here; kept for compatibility.")
    ap.add_argument("--no-plots", action="store_true", help="Skip generating scatter plot")
    return ap.parse_args()


def _find_id_col(cols):
    for c in ("Sample_ID", "Sample", "ID", "sample", "id"):
        if c in cols:
            return c
    return None


def _feature_cols(df, extra_drop=None):
    drop = {"Sample", "Sample_ID", "Group", "Site"}
    if extra_drop:
        drop |= set(extra_drop)
    return [c for c in df.columns if c not in drop]


def _row_l2_norms(M):
    # M: ndarray of shape (n_samples, n_features)
    return np.linalg.norm(M, axis=1)


def _zscore(x):
    m = np.nanmean(x)
    s = np.nanstd(x, ddof=0)
    if s <= 0 or not np.isfinite(s):
        return np.zeros_like(x)
    return (x - m) / s


def _mahalanobis_pvals(X):
    """
    Return two arrays: squared Mahalanobis distances and chi2 p-values.
    Uses numpy, falls back to approximate if singular; needs scipy for chi2 cdf.
    """
    n, p = X.shape
    Xc = X - np.nanmean(X, axis=0, keepdims=True)

    # regularized covariance
    cov = np.cov(Xc, rowvar=False)
    # add tiny ridge to diagonal if near-singular
    ridge = 1e-6 * np.eye(p)
    try:
        inv_cov = np.linalg.inv(cov + ridge)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov + ridge)

    # Mahalanobis^2
    md2 = np.einsum("ij,jk,ik->i", Xc, inv_cov, Xc)

    try:
        from scipy.stats import chi2
        pvals = 1.0 - chi2.cdf(md2, df=p)
    except Exception:
        warnings.warn("[qc] SciPy not available; Mahalanobis p-values set to NaN.")
        pvals = np.full_like(md2, np.nan, dtype=float)

    return md2, pvals


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- Load tables ----------
    df_clr = pd.read_csv(args.wide_clr)
    id_col = _find_id_col(df_clr.columns)
    if id_col is None:
        # assume first column is ID if name not found
        id_col = df_clr.columns[0]
        warnings.warn(f"[qc] Could not find an ID column; using first column '{id_col}' as sample ID.")

    # keep ID as first column
    col_order = [id_col] + [c for c in df_clr.columns if c != id_col]
    df_clr = df_clr[col_order]

    df_all = None
    if args.wide_all:
        try:
            df_all = pd.read_csv(args.wide_all)
            if id_col not in df_all.columns:
                # try to align by first column name
                id2 = _find_id_col(df_all.columns) or df_all.columns[0]
                df_all = df_all.rename(columns={id2: id_col})
            # align rows to df_clr by ID (inner join)
            df_all = df_all.set_index(id_col).reindex(df_clr[id_col]).reset_index()
        except Exception as e:
            warnings.warn(f"[qc] Failed to read --wide-all: {e}. Falling back to CLR-only zeros.")
            df_all = None

    # ---------- Compute metrics ----------
    feat_cols_clr = _feature_cols(df_clr)
    X_clr = df_clr[feat_cols_clr].to_numpy(dtype=float)

    norms = _row_l2_norms(X_clr)
    norms_z = _zscore(norms)

    if df_all is not None:
        feat_cols_all = _feature_cols(df_all)
        A = df_all[feat_cols_all].to_numpy(dtype=float)
        zero_frac = (A == 0).sum(axis=1) / max(1, A.shape[1])
    else:
        # fallback: treat absolute ~0 as zeros (CLR rarely exact zero)
        zero_eps = 1e-12
        zero_frac = (np.abs(X_clr) < zero_eps).sum(axis=1) / max(1, X_clr.shape[1])

    # Optional Mahalanobis
    md2 = np.full_like(norms, np.nan, dtype=float)
    mpv = np.full_like(norms, np.nan, dtype=float)
    if args.mah_p is not None and args.mah_p > 0:
        try:
            md2, mpv = _mahalanobis_pvals(X_clr)
        except Exception as e:
            warnings.warn(f"[qc] Mahalanobis failed ({e}); skipping.")

    # ---------- Flags & decision ----------
    flag_zero = zero_frac >= float(args.max_zero_frac)
    flag_norm = norms_z > float(args.max_norm_z)
    if args.mah_p is not None and args.mah_p > 0:
        flag_mah = mpv < float(args.mah_p)
    else:
        flag_mah = np.zeros_like(norms, dtype=bool)

    exclude = flag_zero | flag_norm | flag_mah
    keep_mask = ~exclude

    # ---------- Reports ----------
    rep = pd.DataFrame({
        "Sample_ID": df_clr[id_col].values,
        "zero_fraction": zero_frac,
        "clr_l2_norm": norms,
        "clr_l2_z": norms_z,
        "flag_zero": flag_zero.astype(int),
        "flag_norm": flag_norm.astype(int),
        "flag_mah": flag_mah.astype(int),
        "exclude": exclude.astype(int),
    })

    n_total = rep.shape[0]
    n_excl = int(exclude.sum())
    n_keep = int(keep_mask.sum())

    # ---------- Save filtered CLR matrix ----------
    df_qc = df_clr.loc[keep_mask, [id_col] + feat_cols_clr].reset_index(drop=True)
    qc_csv_path = outdir / f"ml_{args.rank}_wide_clr.qc.csv"
    df_qc.to_csv(qc_csv_path, index=False)

    # ---------- Save report ----------
    rep_path = outdir / f"qc_report_{args.rank}.csv"
    rep.to_csv(rep_path, index=False)

    # ---------- Save meta ----------
    meta = {
        "rank": args.rank,
        "n_in": int(n_total),
        "n_excluded": int(n_excl),
        "n_kept": int(n_keep),
        "thresholds": {
            "max_zero_frac": float(args.max_zero_frac),
            "max_norm_z": float(args.max_norm_z),
            "mah_p": float(args.mah_p) if args.mah_p is not None else None,
        },
        "inputs": {
            "wide_clr": str(Path(args.wide_clr).resolve()),
            "wide_all": str(Path(args.wide_all).resolve()) if args.wide_all else None,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "id_column": id_col,
        "feature_count": int(len(feat_cols_clr)),
    }
    with open(outdir / f"qc_meta_{args.rank}.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---------- Save excluded list ----------
    excl_ids = rep.loc[exclude, "Sample_ID"].astype(str).tolist()
    with open(outdir / f"excluded_samples_{args.rank}.txt", "w") as f:
        for s in excl_ids:
            f.write(f"{s}\n")

    # ---------- Plot ----------
    if not args.no_plots:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            # colors: keep vs exclude
            ax.scatter(zero_frac[keep_mask], norms_z[keep_mask], s=24, alpha=0.9, label="kept")
            if n_excl > 0:
                ax.scatter(zero_frac[exclude], norms_z[exclude], s=28, alpha=0.9, marker="x", label="excluded")

            # thresholds
            ax.axvline(float(args.max_zero_frac), linestyle="--")
            ax.axhline(float(args.max_norm_z), linestyle="--")

            ax.set_xlabel("Zero fraction")
            ax.set_ylabel("CLR norm z-score")
            ax.set_title(f"QC scatter ({args.rank})")

            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            plt.savefig(outdir / f"qc_scatter_{args.rank}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"[qc] Failed to make plot: {e}")

    # ---------- Console summary ----------
    print(f"[QC] {args.rank}: n_in={n_total}  n_excluded={n_excl}  n_kept={n_keep}")
    print(f"[QC] saved: {qc_csv_path.name}, {rep_path.name}, qc_meta_{args.rank}.json, excluded_samples_{args.rank}.txt, qc_scatter_{args.rank}.png")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
