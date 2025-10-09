#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, accuracy_score

def read_table(p): return pd.read_csv(p)

def get_sample_col(df):
    for c in df.columns:
        cl = c.lower()
        if cl in ("sample","sample_id"): return c
    return None

def ensure_disease(df: pd.DataFrame) -> pd.DataFrame:
    if "disease" in df.columns: return df
    if "Group" in df.columns:
        g = df["Group"].astype(str).str.lower().str.strip()
        df["disease"] = g.replace({"crohn":1,"case":1,"patient":1,"cd":1,"1":1,
                                   "healthy":0,"control":0,"0":0}).astype(int)
        return df
    raise ValueError("Input must contain 'disease' (0/1) or 'Group' (Crohn/Healthy).")

def unify_site(df: pd.DataFrame, group_col: str | None):
    if group_col: return group_col, df
    if "site" in df.columns: 
        df["site"] = df["site"].astype(str).str.lower().str.strip(); return "site", df
    if "Site" in df.columns:
        df = df.rename(columns={"Site":"site"})
        df["site"] = df["site"].astype(str).str.lower().str.strip()
        return "site", df
    return None, df

def build_model(name: str):
    name = name.lower()
    if name in ("logreg","l2","lr"):
        clf = LogisticRegression(penalty="l2", solver="saga", max_iter=5000)
    elif name in ("l1","lasso"):
        clf = LogisticRegression(penalty="l1", solver="saga", max_iter=5000)
    elif name in ("rf","random_forest","random-forest"):
        clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler(with_mean=True, with_std=True)),
                     ("clf", clf)])

def choose_splitter(cv_mode: str, n_folds: int, groups: pd.Series | None, seed: int):
    cv_mode = cv_mode.lower()
    if cv_mode == "stratified_kfold":
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed), False
    if cv_mode in ("groupkfold_site","groupkfold"):
        if groups is None: raise ValueError("groupkfold selected but no groups provided.")
        n_groups = groups.astype(str).nunique()
        return GroupKFold(n_splits=min(max(n_groups,2), n_folds)), True
    if cv_mode == "leave_site_out":
        if groups is None: raise ValueError("leave_site_out selected but no groups provided.")
        return LeaveOneGroupOut(), True
    raise ValueError(f"Unknown cv-mode: {cv_mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", required=True, choices=["genus","species"])
    ap.add_argument("--in-wide-clr", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--cv-mode", required=True,
                    choices=["stratified_kfold","groupkfold_site","leave_site_out","groupkfold"])
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--model", required=True,
                    choices=["logreg","l1","rf","random_forest"])
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.in_wide_clr)
    sample_col = get_sample_col(df)
    if sample_col: df[sample_col] = df[sample_col].astype(str)

    df = ensure_disease(df)
    group_col, df = unify_site(df, args.group_col if args.group_col else None)

    y = df["disease"].astype(int).to_numpy()
    drop = {"disease"}
    if sample_col: drop.add(sample_col)
    if group_col:  drop.add(group_col)
    Xnum = df.drop(columns=list(drop), errors="ignore").select_dtypes(include=[np.number])
    if Xnum.shape[1]==0: raise ValueError("No numeric feature columns for X.")
    X = Xnum.to_numpy(dtype=float)

    groups = df[group_col].astype(str) if group_col and args.cv_mode in ("groupkfold_site","groupkfold","leave_site_out") else None
    splitter, grouped = choose_splitter(args.cv_mode, args.n_folds, groups, args.seed)

    pipe = build_model("rf" if args.model=="random_forest" else args.model)

    preds_list, scores = [], []
    folds = splitter.split(X, y, groups=groups) if grouped else splitter.split(X, y)

    for fid, (tr, te) in enumerate(folds, start=1):
        pipe.fit(X[tr], y[tr])
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_score = pipe.predict_proba(X[te])[:,1]
        elif hasattr(pipe.named_steps["clf"], "decision_function"):
            z = pipe.decision_function(X[te]); y_score = 1/(1+np.exp(-z))
        else:
            y_score = pipe.predict(X[te]).astype(float)

        y_pred = (y_score>=0.5).astype(int)
        try: auc = roc_auc_score(y[te], y_score)
        except Exception: auc = float("nan")
        acc = accuracy_score(y[te], y_pred)
        scores.append({"fold":fid,"auc":float(auc),"acc":float(acc)})

        df_fold = pd.DataFrame({"y_true":y[te],"y_score":y_score,"y_pred":y_pred,"fold":fid})
        if sample_col: df_fold.insert(0, sample_col, df.iloc[te][sample_col].values)
        preds_list.append(df_fold)

    preds = pd.concat(preds_list, ignore_index=True)
    preds.to_csv(outdir/"cv_predictions.csv", index=False)

    s = pd.DataFrame(scores)
    cv_scores = {
        "per_fold": scores,
        "auc_mean": float(np.nanmean(s["auc"])) if not s.empty else None,
        "auc_std":  float(np.nanstd(s["auc"]))  if not s.empty else None,
        "acc_mean": float(np.nanmean(s["acc"])) if not s.empty else None,
        "acc_std":  float(np.nanstd(s["acc"]))  if not s.empty else None,
    }
    (outdir/"cv_scores.json").write_text(json.dumps(cv_scores, indent=2), encoding="utf-8")

    (outdir/"features_used.txt").write_text("\n".join(map(str, Xnum.columns)), encoding="utf-8")
    run_meta = {
        "model": args.model, "rank": args.rank, "cv_mode": args.cv_mode,
        "n_folds_requested": args.n_folds,
        "n_folds_effective": int(s["fold"].nunique()) if not s.empty else None,
        "seed": args.seed, "group_col": group_col, "n_features": int(X.shape[1]),
        "in_file": args.in_wide_clr
    }
    (outdir/"run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
