#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SummarizeML.py
- Summarize ML runs into CSV + JSONL
- Extract top-N biomarkers from features_used.txt
- Make ROC/PR/Confusion Matrix plots from cv_predictions.csv
"""

import argparse, json, glob, os, re, math
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------- utils -----------------------------

def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

def parse_features_file(feat_path):
    """Parse features_used.txt flexibly: lines like:
       - 'feature'
       - 'feature,weight'
       - 'feature<TAB>weight'
       - 'weight feature' (less common)
    Return DataFrame with columns: feature, weight (float or None)
    """
    feats, wts = [], []
    if not Path(feat_path).exists():
        return pd.DataFrame(columns=["feature","weight"])
    with open(feat_path, "r", encoding="utf-8") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            m = re.match(r"^\s*([^,\t]+)[,\t]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$", s)
            if m:
                feats.append(m.group(1).strip())
                wts.append(float(m.group(2)))
                continue
            m2 = re.match(r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(.+)$", s)
            if m2:
                wts.append(float(m2.group(1)))
                feats.append(m2.group(2).strip())
                continue
            feats.append(s)
            wts.append(None)
    return pd.DataFrame({"feature": feats, "weight": wts})

def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def safe_get(d, *keys, default=None):
    cur = d if isinstance(d, dict) else {}
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def flatten_scores(scores):
    """Expect structure like:
       {"auroc":{"mean":..,"std":..}, "accuracy":..., ...}
       Return dict with *_mean and *_std keys. If missing, put NaN.
    """
    out = {}
    for metric in ["auroc","accuracy","f1","precision","recall"]:
        m = scores.get(metric, {}) if isinstance(scores, dict) else {}
        out[f"{metric}_mean"] = safe_get(m, "mean", default=np.nan)
        out[f"{metric}_std"]  = safe_get(m, "std",  default=np.nan)
    return out

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ---------------------- plotting helpers -------------------------

def make_plots(preds_csv, out_roc, out_pr, out_cm):
    """Make ROC/PR/CM plots if columns are available."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

    CAND_TRUE = ["y_true", "true", "label", "target"]
    CAND_PRED = ["y_pred", "pred", "prediction"]
    CAND_PROB = ["proba", "proba_1", "prob_1", "p1", "score"]

    if not Path(preds_csv).exists():
        return {"roc": None, "pr": None, "cm": None}

    df = pd.read_csv(preds_csv)
    col_y = find_col(df, CAND_TRUE)
    col_p = find_col(df, CAND_PRED)
    col_s = find_col(df, CAND_PROB)

    if col_y is None or col_p is None:
        return {"roc": None, "pr": None, "cm": None}

    y_true = df[col_y].values
    y_pred = df[col_p].values

    # Confusion Matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    except Exception:
        cm = None

    if cm is not None:
        import numpy as np
        fig = plt.figure(figsize=(4.5, 4))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues")
        for (i,j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["0","1"]); ax.set_yticklabels(["0","1"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        ensure_dir(out_cm)
        fig.savefig(out_cm, dpi=150)
        plt.close(fig)
    else:
        out_cm = None

    out_roc_path, out_pr_path = None, None
    # ROC & PR only if we have scores/probabilities
    if col_s is not None:
        y_score = df[col_s].values.astype(float)

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fig = plt.figure(figsize=(5.5, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0,1], [0,1], lw=1, linestyle="--")
        ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve"); ax.legend(loc="lower right")
        fig.tight_layout()
        ensure_dir(out_roc)
        fig.savefig(out_roc, dpi=150)
        plt.close(fig)
        out_roc_path = out_roc

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        # approximate AP
        ap_score = np.trapz(precision[::-1], recall[::-1])
        fig = plt.figure(figsize=(5.5, 4.5))
        ax = fig.add_subplot(111)
        ax.plot(recall, precision, lw=2, label=f"AP ≈ {ap_score:.3f}")
        ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve"); ax.legend(loc="lower left")
        fig.tight_layout()
        ensure_dir(out_pr)
        fig.savefig(out_pr, dpi=150)
        plt.close(fig)
        out_pr_path = out_pr

    return {"roc": out_roc_path, "pr": out_pr_path, "cm": out_cm}

# ---------------------- biomarkers helper ------------------------

def write_top_biomarkers(features_txt, out_csv, topn=30):
    df = parse_features_file(features_txt)
    ensure_dir(out_csv)
    if df.empty:
        df.to_csv(out_csv, index=False)
        return out_csv
    if df["weight"].notna().any():
        df["abs_weight"] = df["weight"].abs()
        df = df.sort_values("abs_weight", ascending=False)
    else:
        df["abs_weight"] = np.nan
    df = df.head(topn).copy()
    df["rank"] = range(1, len(df)+1)
    df.to_csv(out_csv, index=False)
    return out_csv

# ------------------------------ main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Summarize ML runs + biomarkers + plots")
    ap.add_argument("--runs-glob", required=True,
                    help='Glob for cv_scores.json files, e.g. "results/ml/*/cv_scores.json"')
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--outdir-summary", default="results/ml/summary",
                    help="Base directory for plots/biomarkers")
    ap.add_argument("--topn-biomarkers", type=int, default=30)
    ap.add_argument("--make-plots", action="store_true", help="Make ROC/PR/CM plots per run")
    args = ap.parse_args()

    score_files = sorted(glob.glob(args.runs_glob))
    rows = []

    bio_dir  = os.path.join(args.outdir_summary, "biomarkers")
    plot_dir = os.path.join(args.outdir_summary, "plots")

    for score_file in score_files:
        run_dir = str(Path(score_file).parent)
        rec = {"run_dir": run_dir}

        # meta + scores
        meta = read_json(os.path.join(run_dir, "run_meta.json")) or {}
        scores = read_json(score_file) or {}

        # carry key meta fields (matching your previous schema)
        rec["rank"]                = meta.get("rank")
        rec["cv_mode"]             = meta.get("cv_mode") or meta.get("cv_mode".upper(), None)
        rec["model"]               = meta.get("model")
        rec["seed"]                = meta.get("seed")
        rec["requested_n_folds"]   = meta.get("requested_n_folds")
        rec["effective_n_folds"]   = meta.get("effective_n_folds")
        rec["n_samples"]           = meta.get("n_samples")
        rec["n_features"]          = meta.get("n_features")
        rec["class_counts"]        = meta.get("class_counts")
        rec["input"]               = meta.get("input")

        # flatten metrics
        rec.update(flatten_scores(scores))

        # paths for predictions and features
        preds_csv = os.path.join(run_dir, "cv_predictions.csv")
        feats_txt = os.path.join(run_dir, "features_used.txt")
        rec["predictions_path"] = preds_csv if Path(preds_csv).exists() else None
        rec["features_path"]    = feats_txt if Path(feats_txt).exists() else None

        # biomarkers per run
        bio_out = None
        if Path(feats_txt).exists():
            # build safe filename: replace slashes with _
            safe_run = re.sub(r"[\\/]", "_", run_dir.strip("/"))
            bio_out = os.path.join(bio_dir, f"{safe_run}_top{args.topn_biomarkers}.csv")
            write_top_biomarkers(feats_txt, bio_out, args.topn_biomarkers)
        rec["biomarkers_path"] = bio_out

        # plots per run
        roc_path = pr_path = cm_path = None
        if args.make_plots and Path(preds_csv).exists():
            safe_run = re.sub(r"[\\/]", "_", run_dir.strip("/"))
            roc_path = os.path.join(plot_dir, f"{safe_run}_roc.png")
            pr_path  = os.path.join(plot_dir, f"{safe_run}_pr.png")
            cm_path  = os.path.join(plot_dir, f"{safe_run}_cm.png")
            paths = make_plots(preds_csv, roc_path, pr_path, cm_path)
            roc_path, pr_path, cm_path = paths["roc"], paths["pr"], paths["cm"]

        rec["roc_png"] = roc_path
        rec["pr_png"]  = pr_path
        rec["cm_png"]  = cm_path

        # keep raw for JSONL too
        rec["scores_raw"] = scores
        rows.append(rec)

    # Write outputs
    out_df = pd.DataFrame(rows)
    ensure_dir(args.out_csv)
    out_df.to_csv(args.out_csv, index=False)

    ensure_dir(args.out_jsonl)
    with open(args.out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
