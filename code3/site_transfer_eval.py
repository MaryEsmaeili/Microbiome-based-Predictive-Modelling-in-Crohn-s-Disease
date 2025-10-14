#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train on one site, evaluate on the other (species matrix).
- Align features by intersection.
- Gracefully handle empty or single-class test sets.

Outputs
-------
transfer_metrics.json with keys:
  n_train, n_test, pos_train, pos_test, auc, ap, note
"""
import os, json, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

def read_with_id(p):
    df = pd.read_csv(p)
    if "Sample_ID" not in df.columns:
        first = df.columns[0]
        if str(first).lower() in {"sample_id","id","sample","unnamed: 0"}:
            df = df.rename(columns={first:"Sample_ID"})
        else:
            raise SystemExit(f"'Sample_ID' missing in {p}")
    df["Sample_ID"] = df["Sample_ID"].astype(str)
    return df

def normalize_site(y):
    cand = [c for c in y.columns if c.lower() in {"site","body_site","location"}]
    if cand:
        s = y[cand[0]].astype(str).str.strip().str.lower()
        s = s.map({"oral":"Oral","mouth":"Oral","fecal":"Fecal","faecal":"Fecal","stool":"Fecal"}).fillna(np.nan)
        y = y.drop(columns=[cand[0]]).assign(site=s)
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--label-col", required=True, choices=["disease","ppi_use"])
    ap.add_argument("--train-site", required=True, choices=["Oral","Fecal"])
    ap.add_argument("--test-site", required=True, choices=["Oral","Fecal"])
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    X = read_with_id(args.X)
    y = read_with_id(args.y)
    y = normalize_site(y)

    lbl = [c for c in y.columns if c.lower()==args.label_col.lower()]
    if not lbl:
        raise SystemExit(f"Label '{args.label_col}' not in y: {list(y.columns)}")
    y = y.rename(columns={lbl[0]:"label"})

    df = X.merge(y[["Sample_ID","label","site"]], on="Sample_ID", how="inner", validate="one_to_one")
    feats = [c for c in df.columns if c not in {"Sample_ID","label","site"}]

    tr = df[df["site"]==args.train_site]
    te = df[df["site"]==args.test_site]

    note = ""
    metrics = {
        "n_train": int(len(tr)),
        "n_test":  int(len(te)),
        "pos_train": int((tr["label"]==1).sum()),
        "pos_test":  int((te["label"]==1).sum()),
        "auc": None,
        "ap":  None,
        "note": ""
    }

    if len(te)==0 or len(tr)==0:
        note = "empty split after site filtering"
    elif tr["label"].nunique()<2:
        note = "train has single class"
    elif te["label"].nunique()<2:
        note = "test has single class; AUC undefined"
        # we can still compute AP vs prevalence
        # but if all positives or all negatives, AP is 1 or 0 by definition later

    if note:
        metrics["note"] = note
        with open(os.path.join(args.outdir,"transfer_metrics.json"),"w") as f:
            json.dump(metrics, f, indent=2)
        return

    # fit simple XGB (same defaults as training)
    model = XGBClassifier(
        n_estimators=500, max_depth=3, subsample=0.8, colsample_bytree=0.8,
        learning_rate=0.05, reg_lambda=1.0, n_jobs=-1, random_state=42,
        eval_metric="logloss"
    )
    model.fit(tr[feats].to_numpy(), tr["label"].to_numpy())
    prob = model.predict_proba(te[feats].to_numpy())[:,1]

    try:
        auc = roc_auc_score(te["label"], prob)
    except Exception:
        auc = None
        note = (metrics["note"] + "; " if metrics["note"] else "") + "AUC undefined"

    ap = average_precision_score(te["label"], prob)

    metrics.update({"auc": auc, "ap": float(ap), "note": note})
    with open(os.path.join(args.outdir,"transfer_metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
