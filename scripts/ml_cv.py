#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def build_model(model_name: str, random_state: int):
    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_split=2,
            n_jobs=-1, random_state=random_state, class_weight="balanced"
        )
    elif model_name == "l1":
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0,
            class_weight="balanced", max_iter=2000, random_state=random_state
        )
    else:
        clf = LogisticRegression(
            penalty="l2", solver="liblinear", C=1.0,
            class_weight="balanced", max_iter=2000, random_state=random_state
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

def safe_n_splits(requested, y):
    _, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min()) if counts.size else 0
    return max(2, min(requested, min_class))

def metrics_dict(y_true, y_prob, y_pred):
    out = {}
    try: out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception: out["auroc"] = float("nan")
    out["accuracy"]  = float(accuracy_score(y_true, y_pred))
    out["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
    return out

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", choices=["genus","species"], required=True)
    ap.add_argument("--in-wide-clr", dest="in_wide_clr", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--cv-mode", choices=["stratified_kfold","leave_site_out"], default="stratified_kfold")
    ap.add_argument("--model", choices=["logreg","l1","rf"], default="logreg")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.in_wide_clr)
    if args.cv_mode == "leave_site_out":
        if "Site" not in df.columns:
            print("[ml_cv] WARNING: 'Site' column missing; falling back to stratified_kfold.")
            args.cv_mode = "stratified_kfold"
        else:
            uniq_sites = df["Site"].nunique(dropna=True)
            if uniq_sites < 2:
                print(f"[ml_cv] WARNING: only {uniq_sites} site present; falling back to stratified_kfold.")
                args.cv_mode = "stratified_kfold"
    if "Sample" not in df.columns:
        df.insert(0, "Sample", [f"s{i}" for i in range(len(df))])
    if "Group" not in df.columns:
        raise ValueError("Input must contain a 'Group' column (Crohn/Healthy). Use qc_attach.py.")
    if "Site" not in df.columns:
        df["Site"] = "NA"

    ymap = {"Crohn":1, "Healthy":0, "C":1, "H":0, 1:1, 0:0}
    y = df["Group"].map(ymap)
    mask = y.notna()
    if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
        raise ValueError("After mapping labels, need at least 3 samples and 2 classes.")

    meta_cols = ["Sample","Group","Site"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df.loc[mask, feat_cols].astype(float).values
    y = df.loc[mask, "Group"].map(ymap).astype(int).values
    samples = df.loc[mask, "Sample"].astype(str).values
    sites   = df.loc[mask, "Site"].astype(str).values

    # splitter
    if args.cv_mode == "leave_site_out":
        unique_sites = np.unique(sites)
        if unique_sites.size < 2:
            warnings.warn("[CV] leave_site_out requested but only one Site present; falling back to stratified_kfold.")
            args.cv_mode = "stratified_kfold"

    if args.cv_mode == "leave_site_out":
        splitter = GroupKFold(n_splits=np.unique(sites).size)
        split_iter = splitter.split(X, y, sites)
    else:
        n_splits = safe_n_splits(args.n_folds, y)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(X, y)

    pipe = build_model(args.model, args.seed)
    rows_pred, fold_scores = [], []
    fold_id = 0

    for tr_idx, te_idx in split_iter:
        fold_id += 1
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        samp_te = samples[te_idx]

        if np.unique(y_tr).size < 2:
            warnings.warn(f"[CV] Fold {fold_id} skipped (single-class train).")
            continue

        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:,1]
        pred = (prob >= 0.5).astype(int)

        m = metrics_dict(y_te, prob, pred); m["fold"]=fold_id
        fold_scores.append(m)
        for s, yt, yp, pr in zip(samp_te, y_te, pred, prob):
            rows_pred.append({"fold":fold_id,"Sample":s,"y_true":int(yt),"y_pred":int(yp),"prob":float(pr)})

    # write outputs
    pd.DataFrame(rows_pred).to_csv(os.path.join(args.outdir, "cv_predictions.csv"), index=False)

    df_scores = pd.DataFrame(fold_scores)
    summary = {}
    for k in ["auroc","accuracy","f1","precision","recall"]:
        if k in df_scores.columns and len(df_scores):
            summary[k] = {"mean": float(np.nanmean(df_scores[k])),
                          "std":  float(np.nanstd(df_scores[k])),
                          "per_fold": df_scores[k].tolist()}
        else:
            summary[k] = {"mean": float("nan"), "std": float("nan"), "per_fold": []}
    with open(os.path.join(args.outdir, "cv_scores.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.outdir, "features_used.txt"), "w") as f:
        for c in feat_cols: f.write(str(c)+"\n")

    meta = {
        "rank": args.rank,
        "cv_mode": args.cv_mode,
        "model": args.model,
        "seed": args.seed,
        "requested_n_folds": args.n_folds,
        "effective_n_folds": int(np.unique([r["fold"] for r in rows_pred]).size) if rows_pred else 0,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_counts": {int(k):int(v) for k,v in zip(*np.unique(y, return_counts=True))},
        "input": args.in_wide_clr
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Done. Wrote outputs to: {args.outdir}")

if __name__ == "__main__":
    main()
