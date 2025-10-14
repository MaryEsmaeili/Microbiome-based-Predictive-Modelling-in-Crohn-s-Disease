#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ILR sensitivity ML run (nested CV).
- Reads X (features, needs Sample_ID) and y (design; must include label column).
- Optional site filter (--subset-site Oral|Fecal).
- Transforms features to ILR balances (D -> D-1) with a small pseudocount.
- Nested CV (StratifiedKFold outer/inner), LogisticRegression grid.
- Saves: cv_metrics.csv, oof_predictions.csv, oof_roc.png, oof_pr.png.

Usage (matches your Snakefile):
python scripts/ml_ilr_sensitivity.py \
  --X results/ml_features/species/X_fecal_clr_z.csv \
  --y results/ml_features/species/y_design.csv \
  --factor disease \
  --subset-site Fecal \
  --outer-folds 5 --inner-folds 5 \
  --outdir results/ml_models_ilr/species/fecal_cd
"""
import os, argparse, json, numpy as np, pandas as pd
from numpy.linalg import qr
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, brier_score_loss, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

# ---------- IO helpers ----------
def read_with_id(p):
    df = pd.read_csv(p)
    if "Sample_ID" not in df.columns:
        first = df.columns[0]
        if str(first).lower() in {"sample_id","id","sample","unnamed: 0"}:
            df = df.rename(columns={first:"Sample_ID"})
        else:
            raise SystemExit(f"[ml_ilr] 'Sample_ID' missing in: {p}")
    df["Sample_ID"] = df["Sample_ID"].astype(str)
    return df

def normalize_site(y):
    cands = [c for c in y.columns if c.lower() in {"site","body_site","location"}]
    if cands:
        raw = y[cands[0]].astype(str).str.strip().str.lower()
        norm = raw.map({"oral":"Oral","mouth":"Oral","fecal":"Fecal","faecal":"Fecal","stool":"Fecal"})
        y = y.drop(columns=[cands[0]]).assign(site=norm)
    return y

# ---------- ILR transform ----------
def helmert_submatrix(D: int) -> np.ndarray:
    """(D x (D-1)) Helmert sub-matrix, columns orthonormal up to scaling."""
    H = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            if j < i:
                H[i, j] = 1.0
            elif j == i:
                H[i, j] = -i
            else:
                H[i, j] = 0.0
    H = H[:, 1:]  # drop first column of full Helmert
    # make columns orthonormal
    Q, _ = qr(H)
    return Q[:, :D-1]

def ilr_transform(Xraw: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    """Row-wise ILR. Xraw must be non-negative. Adds pseudocount and renormalizes to compositions."""
    X = Xraw.copy().astype(float)
    X[X < 0] = 0.0
    X += pseudocount
    X /= X.sum(axis=1, keepdims=True)
    G = helmert_submatrix(X.shape[1])  # D x (D-1)
    # ilr = ln(X) * G
    ilr = np.log(X) @ G
    return ilr

# ---------- Plot helpers ----------
def plot_roc_pr(y_true, y_prob, outdir):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC = {auc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "oof_roc.png"), dpi=150)
    plt.close()
    # PR
    p, r, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6,4))
    plt.plot(r, p)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.ylim(0,1.05)
    plt.title(f"Precision–Recall (AP = {ap:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "oof_pr.png"), dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", required=True)
    ap.add_argument("--y", required=True)
    # accept both names; prefer label_col if supplied
    ap.add_argument("--factor", required=False, help="label column name (alias of --label-col)")
    ap.add_argument("--label-col", required=False, help="label column name")
    ap.add_argument("--subset-site", choices=["Oral","Fecal"], required=False)
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    Xdf = read_with_id(args.X)
    ydf = normalize_site(read_with_id(args.y))

    # resolve label column
    label_col = args.label_col or args.factor
    if not label_col:
        raise SystemExit("[ml_ilr] Please provide --factor (or --label-col).")
    lbl = [c for c in ydf.columns if c.lower()==label_col.lower()]
    if not lbl:
        raise SystemExit(f"[ml_ilr] Label column '{label_col}' not found in y.")
    ydf = ydf.rename(columns={lbl[0]:"label"})

    if args.subset_site:
        if "site" not in ydf.columns:
            print(f"[ml_ilr] site column missing; skipping subset. (requested: {args.subset_site})")
        else:
            ydf = ydf[ydf["site"] == args.subset_site]

    df = Xdf.merge(ydf[["Sample_ID","label"]], on="Sample_ID", how="inner", validate="one_to_one")
    if df.empty:
        raise SystemExit("[ml_ilr] After join, no rows remain. Check Sample_ID alignment / site subset.")

    y = df["label"].astype(int).values
    if len(np.unique(y)) < 2:
        # write a sentinel and exit gracefully (no model can be trained)
        pd.DataFrame({"note":["single-class after subset; skipping ML"],
                      "n":[len(y)], "positives":[int(y.sum())]}).to_csv(
            os.path.join(args.outdir, "cv_metrics.csv"), index=False)
        open(os.path.join(args.outdir, "SKIPPED_single_class.flag"), "w").write("1")
        return

    feats = [c for c in df.columns if c not in {"Sample_ID","label","site"}]
    Xraw = df[feats].values
    X_ilr = ilr_transform(Xraw, pseudocount=1e-6)
    X_ilr = StandardScaler().fit_transform(X_ilr)

    # nested CV: logistic regression (liblinear) grid on C
    outer = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=42)
    inner = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=13)
    base = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)

    param_grid = {"C":[0.01, 0.1, 1.0, 5.0, 10.0]}
    oof_prob = np.zeros(len(y))
    rows = []

    for fold, (tr, te) in enumerate(outer.split(X_ilr, y), start=1):
        gs = GridSearchCV(base, param_grid=param_grid, scoring="roc_auc", cv=inner, n_jobs=1, refit=True)
        gs.fit(X_ilr[tr], y[tr])
        prob = gs.predict_proba(X_ilr[te])[:,1]
        oof_prob[te] = prob
        auc  = roc_auc_score(y[te], prob)
        ap   = average_precision_score(y[te], prob)
        yhat = (prob >= 0.5).astype(int)
        acc  = accuracy_score(y[te], yhat)
        f1   = f1_score(y[te], yhat)
        brier= brier_score_loss(y[te], prob)
        rows.append({
            "fold": fold, "AUC": auc, "AP": ap, "ACC": acc, "F1": f1, "Brier": brier,
            "best_params": gs.best_params_
        })

    # save metrics
    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(args.outdir, "cv_metrics.csv"), index=False)

    # save OOF predictions
    out_pred = pd.DataFrame({
        "Sample_ID": df["Sample_ID"],
        "y_true": y, "y_prob": oof_prob
    })
    out_pred.to_csv(os.path.join(args.outdir, "oof_predictions.csv"), index=False)

    # plots
    plot_roc_pr(y, oof_prob, args.outdir)

    # model card (short)
    card = {
        "transform":"ILR", "outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
        "label_column": label_col, "subset_site": args.subset_site,
        "n_samples": int(len(y)), "n_features_raw": int(len(feats)), "n_features_ilr": int(X_ilr.shape[1])
    }
    with open(os.path.join(args.outdir,"model_card.json"), "w") as f:
        json.dump(card, f, indent=2)

if __name__ == "__main__":
    main()
