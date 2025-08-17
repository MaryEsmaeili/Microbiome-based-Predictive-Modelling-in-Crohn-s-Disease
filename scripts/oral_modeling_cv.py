#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IBD vs Healthy (oral) — Repeated CV with OOF calibration/PR/threshold.

Inputs
------
--crohn   : CSV abundance (rows=taxa, cols=samples, normalized)
--healthy : CSV abundance (rows=taxa, cols=samples, normalized)

Outputs (in --outdir)
---------------------
metrics_cv.csv  (model, fold, accuracy, f1, auc)
summary_mean_sd.csv
preds_logreg_oof.csv
preds_rf_oof.csv
calibration_logreg.png
calibration_rf.png
pr_curve.png
threshold_report.csv
cm_normalized.png
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

plt.rcParams["figure.dpi"] = 160

HEALTHY_COLOR = "#83AAAC"
CROHN_COLOR   = "#516D99"

def load_tables(crohn_file: str, healthy_file: str):
    crohn = pd.read_csv(crohn_file, index_col=0).fillna(0)
    healthy = pd.read_csv(healthy_file, index_col=0).fillna(0)
    # drop all-zero taxa
    crohn = crohn.loc[crohn.sum(axis=1) > 0]
    healthy = healthy.loc[healthy.sum(axis=1) > 0]
    return crohn, healthy

def build_dataset(crohn: pd.DataFrame, healthy: pd.DataFrame):
    taxa_union = crohn.index.union(healthy.index)
    X = pd.concat([crohn.reindex(taxa_union).fillna(0).T,
                   healthy.reindex(taxa_union).fillna(0).T], axis=0)
    y = np.concatenate([np.ones(crohn.shape[1], dtype=int),
                        np.zeros(healthy.shape[1], dtype=int)])
    X = X.fillna(0)
    return X, y

def cv_oof(X: pd.DataFrame, y: np.ndarray, repeats: int, folds: int, seed: int, scale: bool, class_weight):
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True) if scale else "passthrough"),
            ("clf", LogisticRegression(max_iter=3000, class_weight=class_weight, solver="lbfgs"))
        ]),
        "rf": Pipeline([
            ("scaler", "passthrough"),  # RF نیازی به اسکیل ندارد
            ("clf", RandomForestClassifier(n_estimators=500, random_state=seed, class_weight=class_weight))
        ])
    }

    metrics_rows = []
    oof = {m: np.zeros(len(y), dtype=float) for m in models}  # prob of class 1
    for fold_idx, (tr, te) in enumerate(rskf.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]
        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            oof[name][te] = proba
            metrics_rows.append({
                "model": name,
                "fold": fold_idx,
                "accuracy": accuracy_score(y_te, pred),
                "f1": f1_score(y_te, pred, zero_division=0),
                "auc": roc_auc_score(y_te, proba)
            })

    metrics_cv = pd.DataFrame(metrics_rows)
    return metrics_cv, oof

def save_summary(metrics_cv: pd.DataFrame, outdir: Path):
    summary = metrics_cv.groupby("model")[["accuracy","f1","auc"]].agg(["mean","std"])
    summary.columns = [f"{m}_{s}" for m,s in summary.columns]
    summary.to_csv(outdir/"summary_mean_sd.csv")
    metrics_cv.to_csv(outdir/"metrics_cv.csv", index=False)

def plot_calibration(y_true, probs, title, outfile):
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot([0,1],[0,1], "k--", alpha=.4)
    ax.plot(mean_pred, frac_pos, marker="o", lw=2)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(outfile); plt.close(fig)

def plot_pr(y_true, probs_dict: dict, outfile):
    fig, ax = plt.subplots(figsize=(6,4.5))
    for name, p in probs_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)
        ax.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
    ax.set_title("Precision-Recall (OOF)")
    fig.tight_layout(); fig.savefig(outfile); plt.close(fig)

def youden_threshold(y_true, probs):
    # scan thresholds on unique probs
    thr = np.unique(np.r_[0.0, probs, 1.0])
    best_t, best_j, best_stats = 0.5, -1, None
    for t in thr:
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
        j = sens + spec - 1
        acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0.0
        f1 = f1_score(y_true, pred, zero_division=0)
        auc = roc_auc_score(y_true, probs)
        if j > best_j:
            best_j, best_t = j, t
            best_stats = (sens, spec, acc, f1, auc)
    sens, spec, acc, f1, auc = best_stats
    return best_t, {"sensitivity": sens, "specificity": spec, "accuracy": acc, "f1": f1, "auc": auc, "youdenJ": best_j}

def plot_cm_norm(y_true, probs, thr, outfile):
    pred = (probs >= thr).astype(int)
    cm = confusion_matrix(y_true, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(4.5,3.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Healthy","IBD"]); ax.set_yticklabels(["Healthy","IBD"])
    ax.set_title(f"Normalized CM @ threshold={thr:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(outfile); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crohn", required=True)
    ap.add_argument("--healthy", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--no-class-weight", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    crohn, healthy = load_tables(args.crohn, args.healthy)
    X, y = build_dataset(crohn, healthy)

    class_weight = None if args.no_class_weight else "balanced"
    metrics_cv, oof = cv_oof(X, y, args.repeats, args.folds, args.seed, args.scale, class_weight)
    save_summary(metrics_cv, outdir)

    # Save OOF preds
    pd.DataFrame({"sample_id": X.index, "y_true": y, "y_proba": oof["logreg"]}).to_csv(outdir/"preds_logreg_oof.csv", index=False)
    pd.DataFrame({"sample_id": X.index, "y_true": y, "y_proba": oof["rf"]}).to_csv(outdir/"preds_rf_oof.csv", index=False)

    # Calibration & PR using OOF
    plot_calibration(y, oof["logreg"], "Calibration (LogReg, OOF)", outdir/"calibration_logreg.png")
    plot_calibration(y, oof["rf"],     "Calibration (RF, OOF)",     outdir/"calibration_rf.png")
    plot_pr(y, {"logreg": oof["logreg"], "rf": oof["rf"]}, outdir/"pr_curve.png")

    # Threshold selection on best model (higher AUC mean across folds); use OOF logreg by default
    mean_auc = metrics_cv.groupby("model")["auc"].mean()
    best_model = mean_auc.idxmax()
    probs_best = oof[best_model]
    thr, stats = youden_threshold(y, probs_best)
    pd.DataFrame([{"model": best_model, "threshold": thr, **stats}]).to_csv(outdir/"threshold_report.csv", index=False)
    plot_cm_norm(y, probs_best, thr, outdir/"cm_normalized.png")

if __name__ == "__main__":
    main()
