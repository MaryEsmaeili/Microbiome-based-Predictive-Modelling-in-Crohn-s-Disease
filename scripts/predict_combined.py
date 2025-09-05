#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Responder prediction using oral, fecal, and combined features
- Input matrices are "filtered_normalized" wide files (taxa rows × Sxxxx columns)
- Metadata: data/meta/crohn_metadata_extended.csv (has STUDY_ID, Oral_sample_ID, Fecal_sample_ID, Responder)
- Levels: genus and species
- Designs: oral-only, fecal-only, combined
- Models: ElasticNet (logistic, saga) and RandomForest
- Transforms: CLR on each sample (with small pseudocount); then StandardScaler
- Outputs per level/mode: PR.png, ROC.png, scores.csv, top_features.csv, debug.txt
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

# ------------------------------ IO helpers ------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_csv(p):
    return pd.read_csv(p, low_memory=False)

def detect_tax_col(df: pd.DataFrame) -> str:
    """Finds taxonomy column in wide abundance table."""
    cand = [c for c in df.columns if c.lower() in ("clade_name", "taxon", "taxonomy", "name", "feature")]
    if cand:
        return cand[0]
    # otherwise, assume first column is taxonomy (common in your files where header is blank)
    return df.columns[0]

def clean_numeric_block(df: pd.DataFrame, valcols: list) -> pd.DataFrame:
    """Coerce numeric; any weird concatenations become NaN→0."""
    out = df.copy()
    for c in valcols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.fillna(0.0)

def load_raw_matrix(path_csv: str) -> pd.DataFrame:
    """
    Loads the wide abundance matrix (taxa rows × sample columns).
    Accepts both 'clade_name' and unnamed first column for taxonomy.
    """
    df = read_csv(path_csv)
    taxcol = detect_tax_col(df)
    df = df.rename(columns={taxcol: "taxon"})
    valcols = [c for c in df.columns if c != "taxon"]
    df = clean_numeric_block(df, valcols)
    # drop totally-empty taxa
    df = df.loc[(df[valcols] != 0).any(axis=1)].reset_index(drop=True)
    return df

def prep_level(df_raw: pd.DataFrame, level: str) -> pd.DataFrame:
    """
    Extract level-specific matrix from raw (still wide; taxon column exists).
    - species: rows that contain s__..., taxon becomes the trailing s__Name
    - genus  : prefer rows that end at genus (no s__), otherwise aggregate species to genus
    Returns taxa × sample matrix (index: taxon, columns: Sxxxx).
    """
    df = df_raw.copy()
    tax = df["taxon"].astype(str)

    if level == "species":
        mask = tax.str.contains("s__")
        sp = df.loc[mask].copy()
        sp["taxon"] = sp["taxon"].str.extract(r"(s__[^|;]+)$", expand=False)
        mat = sp.set_index("taxon")
        return mat

    # GENUS
    has_gen = tax.str.contains("g__")
    if (has_gen & ~tax.str.contains("s__")).any():
        # keep pure genus rows (no s__)
        gn = df.loc[has_gen & ~tax.str.contains("s__")].copy()
        gn["taxon"] = gn["taxon"].str.extract(r"(g__[^|;]+)", expand=False)
        mat = gn.set_index("taxon")
        return mat
    else:
        # aggregate species (or mixed) to genus
        agg = df.copy()
        agg["gen"] = agg["taxon"].str.extract(r"(g__[^|;]+)", expand=False)
        agg = agg.dropna(subset=["gen"])
        valcols = [c for c in agg.columns if c not in ("taxon", "gen")]
        mat = agg.groupby("gen")[valcols].sum()
        mat.index.name = "taxon"
        return mat

# ------------------------------ Mapping ------------------------------

def build_mapping(meta_fp, oral_cols, fecal_cols) -> pd.DataFrame:
    """
    Returns mapping with STUDY_ID, Responder, Oral_sample_ID, Fecal_sample_ID,
    and flags whether each sample column exists in matrices.
    """
    meta = read_csv(meta_fp)
    need = ["STUDY_ID", "Responder", "Oral_sample_ID", "Fecal_sample_ID"]
    for col in need:
        if col not in meta.columns:
            raise RuntimeError(f"Metadata missing column: {col}")

    m = meta[need].copy()
    m["Responder"] = pd.to_numeric(m["Responder"], errors="coerce").fillna(0).astype(int)

    m["Oral_col_ok"]  = m["Oral_sample_ID"].astype(str).isin(list(oral_cols))
    m["Fecal_col_ok"] = m["Fecal_sample_ID"].astype(str).isin(list(fecal_cols))
    return m

# ------------------------------ Design matrices ------------------------------

def design_from_site(mapping: pd.DataFrame, mat: pd.DataFrame, site: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Builds X (patients × features) and y for a single site.
    Returns (X, y, dropped_info)
    """
    if site == "oral":
        keep = mapping["Oral_col_ok"]
        cols = mapping.loc[keep, "Oral_sample_ID"].astype(str).tolist()
        X = mat[cols].T if cols else pd.DataFrame(index=[], columns=mat.index).astype(float)
        X.index = mapping.loc[keep, "STUDY_ID"].values
    elif site == "fecal":
        keep = mapping["Fecal_col_ok"]
        cols = mapping.loc[keep, "Fecal_sample_ID"].astype(str).tolist()
        X = mat[cols].T if cols else pd.DataFrame(index=[], columns=mat.index).astype(float)
        X.index = mapping.loc[keep, "STUDY_ID"].values
    else:
        raise ValueError(site)

    # responder aligned to STUDY_ID index
    y = mapping.set_index("STUDY_ID").loc[X.index, "Responder"]

    # info on dropped patients for this site
    dropped = mapping.loc[~keep, ["STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID"]].copy()
    dropped["reason"] = f"{site}_sample_missing"

    # prefix feature names to avoid collisions later
    X = X.rename(columns=lambda t: f"{site}|{t}").astype(float)
    return (X, y, dropped)

def design_combined(mapping: pd.DataFrame, oral_mat: pd.DataFrame, fecal_mat: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Combined = concat(oral-features, fecal-features). Keep patients with at least ONE valid sample."""
    any_ok = mapping["Oral_col_ok"] | mapping["Fecal_col_ok"]
    sub = mapping.loc[any_ok].copy()

    # oral block (some patients may miss it)
    Xo = pd.DataFrame(index=sub["STUDY_ID"].values)
    ok_o = sub["Oral_col_ok"]
    if ok_o.any():
        cols_o = sub.loc[ok_o, "Oral_sample_ID"].astype(str).tolist()
        Xo = oral_mat[cols_o].T
        Xo.index = sub.loc[ok_o, "STUDY_ID"].values
        Xo = Xo.rename(columns=lambda t: f"oral|{t}")
    else:
        Xo = pd.DataFrame(index=sub["STUDY_ID"].values)

    # fecal block
    Xf = pd.DataFrame(index=sub["STUDY_ID"].values)
    ok_f = sub["Fecal_col_ok"]
    if ok_f.any():
        cols_f = sub.loc[ok_f, "Fecal_sample_ID"].astype(str).tolist()
        Xf = fecal_mat[cols_f].T
        Xf.index = sub.loc[ok_f, "STUDY_ID"].values
        Xf = Xf.rename(columns=lambda t: f"fecal|{t}")
    else:
        Xf = pd.DataFrame(index=sub["STUDY_ID"].values)

    X = pd.concat([Xo, Xf], axis=1).fillna(0.0).astype(float)
    y = sub.set_index("STUDY_ID")["Responder"].reindex(X.index)
    dropped = mapping.loc[~any_ok, ["STUDY_ID", "Oral_sample_ID", "Fecal_sample_ID"]].copy()
    dropped["reason"] = "both_samples_missing"
    return (X, y, dropped)

# ------------------------------ Transforms & filters ------------------------------

def filter_prevalence(X: pd.DataFrame, min_prev: float) -> pd.DataFrame:
    """Remove columns with prevalence < min_prev; always drop all-zero columns."""
    X = X.loc[:, (X != 0).any(axis=0)]
    if min_prev <= 0:
        return X
    prev = (X > 0).mean(axis=0)
    keep = prev >= min_prev
    return X.loc[:, keep]

def clr_transform(X: pd.DataFrame, pseudo: float = 1e-6) -> pd.DataFrame:
    """Centered log-ratio per sample (row). Robust to zeros via pseudocount."""
    if X.shape[1] == 0:
        return X.copy()
    V = X.clip(lower=0).astype(float)
    row_sum = V.sum(axis=1).replace(0, np.nan)
    V = V.div(row_sum, axis=0)
    V = np.log(V + pseudo)
    gm = V.mean(axis=1)
    V = V.sub(gm, axis=0)
    return V.fillna(0.0)

# ------------------------------ Modeling ------------------------------

def kfold_eval(X: pd.DataFrame, y: pd.Series, outdir: Path, title: str):
    """
    Run 5-fold stratified CV for ElasticNet (logistic) and RandomForest.
    Saves PR/ROC curves (with baseline), scores.csv, and top_features.csv.
    """
    outdir = ensure_dir(outdir)
    debug_lines = []

    # guard rails
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    debug_lines += [f"n={len(y)}, pos={n_pos}, neg={n_neg}",
                    f"X shape (before scale): {X.shape}"]

    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        (outdir / "EMPTY_DESIGN.txt").write_text("Not enough features or single-class labels.\n")
        return

    # transform + scale
    X = clr_transform(X)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X.values)

    # models
    enet = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5,
        max_iter=5000, class_weight="balanced", n_jobs=1, C=1.0
    )
    rf = RandomForestClassifier(
        n_estimators=1000, random_state=42, class_weight="balanced", n_jobs=1
    )
    models = [("elasticnet", enet), ("rf", rf)]

    # folds (avoid impossible splits)
    n_splits = min(5, y.value_counts().min()) if y.value_counts().min() >= 2 else 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, model in models:
        prob = np.zeros(len(y), dtype=float)
        idx = np.arange(len(y))
        for tr, te in skf.split(Xs, y.values):
            model.fit(Xs[tr], y.values[tr])
            prob[te] = model.predict_proba(Xs[te])[:, 1]

        # metrics
        roc_auc = roc_auc_score(y, prob)
        ap = average_precision_score(y, prob)
        pd.DataFrame({"ROC_AUC":[roc_auc], "PR_AUC":[ap]}).to_csv(outdir / f"{name}_scores.csv", index=False)

        # curves
        fpr, tpr, _ = roc_curve(y, prob)
        rc, pr, _ = precision_recall_curve(y, prob)

        # ROC
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        plt.plot([0,1],[0,1],"--",alpha=.5)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(title)
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / f"{name}_roc.png", dpi=130)
        plt.close()

        # PR + baseline
        baseline = y.mean()
        plt.figure(figsize=(6,5))
        plt.plot(rc, pr, label=f"{name} (AP={ap:.3f})")
        plt.hlines(baseline, xmin=0, xmax=1, linestyles="--", label=f"baseline={baseline:.2f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(title)
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / f"{name}_pr.png", dpi=130)
        plt.close()

        # top features
        try:
            if name == "rf":
                imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            else:
                coef = pd.Series(model.coef_.ravel(), index=X.columns)
                imp = coef.abs().sort_values(ascending=False)
            imp.head(50).to_csv(outdir / f"{name}_top_features.csv", header=["importance"])
        except Exception as e:
            debug_lines.append(f"[warn] could not export top features for {name}: {e}")

    # write debug
    (outdir / "debug.txt").write_text("\n".join(debug_lines) + "\n")

# ------------------------------ Runner per level ------------------------------

def run_for_level(level: str, meta_fp: str, oral_fp: str, fecal_fp: str, outroot: Path,
                  min_prev: float = 0.00):
    """
    Builds designs (oral/fecal/combined) for a given level and runs models.
    """
    outroot = ensure_dir(outroot / level)

    # load matrices
    oral_raw  = load_raw_matrix(oral_fp)
    fecal_raw = load_raw_matrix(fecal_fp)
    oral_mat  = prep_level(oral_raw, level)     # taxa × Sxxxx
    fecal_mat = prep_level(fecal_raw, level)    # taxa × Sxxxx

    # mapping
    mapping = build_mapping(meta_fp, oral_mat.columns, fecal_mat.columns)

    # designs
    modes = []

    # oral-only
    Xo, yo, drop_o = design_from_site(mapping, oral_mat, "oral")
    Xo = filter_prevalence(Xo, min_prev)
    modes.append(("oral", Xo, yo, drop_o))

    # fecal-only
    Xf, yf, drop_f = design_from_site(mapping, fecal_mat, "fecal")
    Xf = filter_prevalence(Xf, min_prev)
    modes.append(("fecal", Xf, yf, drop_f))

    # combined
    Xc, yc, drop_c = design_combined(mapping, oral_mat, fecal_mat)
    Xc = filter_prevalence(Xc, min_prev)
    modes.append(("combined", Xc, yc, drop_c))

    # save dropped patients
    pd.concat(m[3] for m in modes).to_csv(outroot / "dropped_patients.csv", index=False)

    # run models per mode
    for mode, X, y, _ in modes:
        modir = ensure_dir(outroot / mode)
        # if empty → mark & continue
        if X.shape[0] == 0 or X.shape[1] == 0 or len(np.unique(y)) < 2:
            (modir / "EMPTY_DESIGN.txt").write_text(
                f"Empty or single-class design: X={X.shape}, unique(y)={np.unique(y)}\n"
            )
            continue

        title = f"{level} — {mode} (n={len(y)}, pos={int(y.sum())})"
        kfold_eval(X, y, modir, title)

# ------------------------------ CLI ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Responder prediction: oral, fecal, combined (genus/species).")
    ap.add_argument("--metadata", required=True, help="data/meta/crohn_metadata_extended.csv")
    ap.add_argument("--oral", required=True, help="filtered oral abundance CSV (wide)")
    ap.add_argument("--fecal", required=True, help="filtered fecal abundance CSV (wide)")
    ap.add_argument("--outdir", required=True, help="results directory")
    ap.add_argument("--min-prevalence", type=float, default=0.00,
                    help="drop features with prevalence < this fraction across patients (default 0.00)")
    args = ap.parse_args()

    outroot = ensure_dir(Path(args.outdir))

    # GENUS & SPECIES
    run_for_level("genus", args.metadata, args.oral, args.fecal, outroot, min_prev=args.min_prevalence)
    run_for_level("species", args.metadata, args.oral, args.fecal, outroot, min_prev=args.min_prevalence)


if __name__ == "__main__":
    main()
