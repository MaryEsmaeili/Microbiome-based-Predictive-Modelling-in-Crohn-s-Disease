#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning curve for binary classification with robust handling of tiny/imbalanced data.
- Builds a feasible size grid from class counts.
- For each train size m:
    * Draw an exact-size stratified subset via StratifiedShuffleSplit.
    * Optionally cap majority:minority ratio inside the subset.
    * If minority>=2 → StratifiedKFold CV; else fall back to Repeated SSS evaluation.
- Reports AUC/AP plus baselines and lift.

Output: learning_curve.csv with
[train_size, pos_in_subset, neg_in_subset, mean_auc, std_auc,
 mean_ap, std_ap, n_folds_total, pos_rate, ap_baseline, auc_baseline, ap_lift]
"""
import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier


# ----------------- IO helpers -----------------
def read_with_id(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Sample_ID" not in df.columns:
        first = df.columns[0]
        if str(first).lower() in {"sample_id", "id", "sample", "unnamed: 0", "index"}:
            df = df.rename(columns={first: "Sample_ID"})
        else:
            raise SystemExit(f"'Sample_ID' missing in {path}")
    df["Sample_ID"] = df["Sample_ID"].astype(str)
    return df

def normalize_site(y: pd.DataFrame) -> pd.DataFrame:
    cand = [c for c in y.columns if c.lower() in {"site","body_site","location"}]
    if cand:
        s = y[cand[0]].astype(str).str.strip().str.lower()
        s = s.map({"oral":"Oral","mouth":"Oral","fecal":"Fecal","faecal":"Fecal","stool":"Fecal"})
        y = y.drop(columns=[cand[0]]).assign(site=s)
    return y


# ----------------- Grid logic -----------------
def feasible_sizes(y: np.ndarray, min_step: int, max_steps: int, test_frac: float) -> list[int]:
    """Build size grid that leaves ~test_frac for test in SSS and respects class feasibility."""
    y = pd.Series(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    # keep at least test_frac of each class for test
    keep_pos = max(n_pos - max(1, int(test_frac * n_pos)), 1)
    keep_neg = max(n_neg - max(1, int(test_frac * n_neg)), 1)
    max_per_class = max(1, min(keep_pos, keep_neg))
    cap = 2 * max_per_class

    grid = sorted(set([min_step * k for k in range(1, max_steps + 1)] + [cap]))
    sizes = [s for s in grid if s >= 2 and s < cap] + ([cap - 1] if cap > 2 else [])
    return sorted(set([s for s in sizes if s >= 2]))


# ----------------- Subset builders -----------------
def draw_stratified_subset(X, y, m, random_state=42):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=m, random_state=random_state)
    idx, _ = next(sss.split(X, y))
    return idx

def cap_ratio(Xs, ys, max_ratio=2.0, random_state=42):
    """Optional: cap majority:minority ratio inside subset."""
    if max_ratio is None:
        return Xs, ys
    ys = np.asarray(ys)
    cls, cnt = np.unique(ys, return_counts=True)
    if len(cls) < 2:
        return Xs, ys
    maj = cls[np.argmax(cnt)]
    mino = cls[np.argmin(cnt)]
    maj_idx = np.where(ys == maj)[0]
    min_idx = np.where(ys == mino)[0]
    target_maj = int(min(max_ratio * len(min_idx), len(maj_idx)))
    if target_maj < len(maj_idx):
        rng = np.random.RandomState(random_state)
        take = rng.choice(maj_idx, size=target_maj, replace=False)
        keep = np.sort(np.concatenate([take, min_idx]))
        return Xs[keep], ys[keep]
    return Xs, ys


# ----------------- Eval engines -----------------
def eval_kfold(Xs, ys, max_depth=3, folds=None, random_state=42):
    """Standard StratifiedKFold when feasible (minority >= folds >= 2)."""
    if folds is None:
        # folds limited by minority count
        _, counts = np.unique(ys, return_counts=True)
        folds = min(5, int(counts.min()))
    if folds < 2:
        return None

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    aucs, aps = [], []
    for tr, te in skf.split(Xs, ys):
        clf = XGBClassifier(
            n_estimators=500, max_depth=max_depth,
            subsample=0.8, colsample_bytree=0.8,
            learning_rate=0.05, reg_lambda=1.0,
            n_jobs=-1, random_state=random_state, eval_metric="logloss"
        )
        clf.fit(Xs[tr], ys[tr])
        prob = clf.predict_proba(Xs[te])[:, 1]
        # AUC may fail if test set becomes single-class (rare but guard it)
        try:
            aucs.append(roc_auc_score(ys[te], prob))
        except Exception:
            pass
        aps.append(average_precision_score(ys[te], prob))
    if len(aucs) == 0 and len(aps) == 0:
        return None
    return {
        "mean_auc": np.mean(aucs) if aucs else None,
        "std_auc": np.std(aucs) if aucs else None,
        "mean_ap": np.mean(aps) if aps else None,
        "std_ap": np.std(aps) if aps else None,
        "n_folds_total": folds
    }

def eval_repeated_sss(Xs, ys, repeats=25, test_frac=0.2, random_state=42):
    """Monte-Carlo CV when KFold is infeasible. Returns means over repeats."""
    if len(np.unique(ys)) < 2:
        return None
    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=test_frac, random_state=random_state)
    aucs, aps = [], []
    for tr, te in sss.split(Xs, ys):
        # Guard single-class test
        if len(np.unique(ys[te])) < 2:
            # AUC undefined; AP still قابل محاسبه است
            clf = XGBClassifier(
                n_estimators=500, max_depth=3,
                subsample=0.8, colsample_bytree=0.8,
                learning_rate=0.05, reg_lambda=1.0,
                n_jobs=-1, random_state=random_state, eval_metric="logloss"
            )
            clf.fit(Xs[tr], ys[tr])
            prob = clf.predict_proba(Xs[te])[:, 1]
            aps.append(average_precision_score(ys[te], prob))
            continue

        clf = XGBClassifier(
            n_estimators=500, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            learning_rate=0.05, reg_lambda=1.0,
            n_jobs=-1, random_state=random_state, eval_metric="logloss"
        )
        clf.fit(Xs[tr], ys[tr])
        prob = clf.predict_proba(Xs[te])[:, 1]
        try:
            aucs.append(roc_auc_score(ys[te], prob))
        except Exception:
            pass
        aps.append(average_precision_score(ys[te], prob))

    if len(aucs) == 0 and len(aps) == 0:
        return None
    return {
        "mean_auc": np.mean(aucs) if aucs else None,
        "std_auc": np.std(aucs) if aucs else None,
        "mean_ap": np.mean(aps) if aps else None,
        "std_ap": np.std(aps) if aps else None,
        "n_folds_total": repeats
    }


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--label-col", required=True, choices=["disease","ppi_use"])
    ap.add_argument("--subset-site", default=None, choices=[None,"Oral","Fecal"])
    ap.add_argument("--outdir", required=True)

    # tuning knobs
    ap.add_argument("--min-step", type=int, default=8, help="Base step for size grid")
    ap.add_argument("--max-steps", type=int, default=8, help="How many steps to try")
    ap.add_argument("--test-frac", type=float, default=0.2, help="Holdout fraction inside SSS")
    ap.add_argument("--repeats", type=int, default=25, help="Repeats for Monte-Carlo CV fallback")
    ap.add_argument("--max-ratio", type=float, default=None, help="Optional cap on majority:minority in subset, e.g. 2.0")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X = read_with_id(args.X)
    y = read_with_id(args.y)
    y = normalize_site(y)

    # label to 'label'
    lbl = [c for c in y.columns if c.lower() == args.label_col.lower()]
    if not lbl:
        raise SystemExit(f"label '{args.label_col}' not in {args.y}")
    y = y.rename(columns={lbl[0]:"label"})
    keep = ["Sample_ID","label"] + (["site"] if "site" in y.columns else [])
    y = y[keep].dropna(subset=["label"])

    if args.subset_site:
        y = y[y.get("site")==args.subset_site]

    df = X.merge(y, on="Sample_ID", how="inner", validate="one_to_one")
    feats = [c for c in df.columns if c not in {"Sample_ID","label","site"}]
    Xn = df[feats].to_numpy(dtype=float)
    yn = df["label"].astype(int).to_numpy()

    if len(df) < 5:
        raise SystemExit(f"Too few samples (n={len(df)})")

    sizes = feasible_sizes(yn, args.min_step, args.max_steps, args.test_frac)

    rows = []
    for m in sizes:
        # draw exact-size stratified subset
        idx = draw_stratified_subset(Xn, yn, m, random_state=42)
        Xs, ys = Xn[idx], yn[idx]

        # optional rebalancing inside subset
        Xs, ys = cap_ratio(Xs, ys, max_ratio=args.max_ratio, random_state=42)

        pos = int((ys==1).sum()); neg = int((ys==0).sum())
        pos_rate = pos / max(1, (pos+neg))
        ap_baseline = pos_rate
        auc_baseline = 0.5

        # First try KFold if feasible
        _, counts = np.unique(ys, return_counts=True)
        minority = int(counts.min()) if len(counts)>0 else 0

        result = None
        if len(np.unique(ys)) == 2 and minority >= 2:
            result = eval_kfold(Xs, ys, folds=None, random_state=42)

        # Fallback: Repeated SSS (Monte Carlo)
        if result is None:
            result = eval_repeated_sss(Xs, ys, repeats=args.repeats, test_frac=args.test_frac, random_state=42)

        if result is None:
            # Truly infeasible (e.g., total minority==1)
            rows.append({
                "train_size": int(m),
                "pos_in_subset": pos, "neg_in_subset": neg,
                "mean_auc": None, "std_auc": None,
                "mean_ap": None, "std_ap": None,
                "n_folds_total": 0,
                "pos_rate": pos_rate, "ap_baseline": ap_baseline,
                "auc_baseline": auc_baseline, "ap_lift": None
            })
            continue

        ap_lift = (result["mean_ap"]/ap_baseline) if (result["mean_ap"] is not None and ap_baseline>0) else None
        rows.append({
            "train_size": int(m),
            "pos_in_subset": pos, "neg_in_subset": neg,
            "mean_auc": result["mean_auc"], "std_auc": result["std_auc"],
            "mean_ap": result["mean_ap"], "std_ap": result["std_ap"],
            "n_folds_total": result["n_folds_total"],
            "pos_rate": pos_rate, "ap_baseline": ap_baseline,
            "auc_baseline": auc_baseline, "ap_lift": ap_lift
        })

    out_csv = os.path.join(args.outdir, "learning_curve.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
