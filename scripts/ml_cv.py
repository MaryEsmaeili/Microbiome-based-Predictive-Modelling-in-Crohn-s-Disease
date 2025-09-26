#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, accuracy_score


def read_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_sample_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = c.lower()
        if cl in ("sample_id", "sample"):
            return c
    return None


def build_model(name: str) -> Pipeline:
    name = name.lower()
    if name in ("logreg", "l2", "lr"):
        clf = LogisticRegression(
            penalty="l2", solver="saga", max_iter=5000, n_jobs=None
        )
    elif name in ("l1", "lasso"):
        clf = LogisticRegression(
            penalty="l1", solver="saga", max_iter=5000, n_jobs=None
        )
    elif name in ("rf", "random_forest", "random-forest"):
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=None, random_state=42, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf),
    ])
    return pipe


def choose_splitter(cv_mode: str, n_folds: int, y: np.ndarray, groups: pd.Series | None, seed: int):
    cv_mode = cv_mode.lower()
    if cv_mode == "stratified_kfold":
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed), False
    if cv_mode in ("groupkfold_site", "groupkfold"):
        if groups is None:
            raise ValueError("groupkfold selected but no groups provided.")
        n_groups = groups.astype(str).nunique()
        n_splits = min(n_folds, n_groups) if n_groups > 1 else 2  # حداقل 2 برای اسکیکت
        if n_splits < 2:
            raise ValueError("Need at least 2 groups for GroupKFold.")
        return GroupKFold(n_splits=n_splits), True
    if cv_mode == "leave_site_out":
        if groups is None:
            raise ValueError("leave_site_out selected but no groups provided.")
        return LeaveOneGroupOut(), True
    raise ValueError(f"Unknown cv-mode: {cv_mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", required=True, choices=["genus", "species"])
    ap.add_argument("--in-wide-clr", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--cv-mode", required=True,
                    choices=["stratified_kfold", "groupkfold_site", "leave_site_out", "groupkfold"])
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--model", required=True,
                    choices=["logreg", "l1", "rf", "random_forest"])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    df = read_table(args.in_wide_clr)
    sample_col = get_sample_col(df)  # Sample_ID یا Sample اگر موجود
    # disease باید وجود داشته باشد
    if "disease" not in df.columns:
        raise ValueError("Input must contain 'disease' column (0/1).")

    # groups (برای سایت)
    group_col = args.group_col
    if group_col is None:
        # تلاش برای پیدا کردن site
        if "site" in df.columns:
            group_col = "site"
        elif "Site" in df.columns:
            # هماهنگ‌سازی
            df = df.rename(columns={"Site": "site"})
            group_col = "site"

    groups = None
    if args.cv_mode in ("groupkfold_site", "groupkfold", "leave_site_out"):
        if group_col is None or group_col not in df.columns:
            raise ValueError("Grouping mode selected but group column not found.")
        groups = df[group_col].astype(str)

    # هدف
    y = df["disease"].astype(int).to_numpy()

    # فیچرهای عددی فقط (ستون‌های غیرعددی بیرون)
    # disease و ستون نمونه و ستون گرو‌ه را حذف می‌کنیم
    drop_cols = {"disease"}
    if sample_col:
        drop_cols.add(sample_col)
    if group_col:
        drop_cols.add(group_col)

    X_num = df.drop(columns=list(drop_cols), errors="ignore")
    X_num = X_num.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] == 0:
        raise ValueError("No numeric feature columns found for X.")

    X = X_num.to_numpy(dtype=float)

    # تقسیم‌بند
    splitter, is_grouped = choose_splitter(args.cv_mode, args.n_folds, y, groups, args.seed)

    # مدل
    model_name = "rf" if args.model == "random_forest" else args.model
    pipe = build_model(model_name)

    preds_records = []
    scores = []

    # تولید ایندکس‌های فولد
    if is_grouped:
        split_iter = splitter.split(X, y, groups=groups)
    else:
        split_iter = splitter.split(X, y)

    fold_id = 0
    for tr_idx, te_idx in split_iter:
        fold_id += 1
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        pipe.fit(Xtr, ytr)

        # y_score = احتمال کلاس 1
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_score = pipe.predict_proba(Xte)[:, 1]
        elif hasattr(pipe.named_steps["clf"], "decision_function"):
            # تبدیل به احتمال لجستیکی
            df_raw = pipe.decision_function(Xte)
            # نگاشت سیگموید امن
            y_score = 1.0 / (1.0 + np.exp(-df_raw))
        else:
            # fallback: از پیش‌بینی دودویی استفاده می‌کنیم (Weak)
            y_score = pipe.predict(Xte).astype(float)

        y_pred = (y_score >= 0.5).astype(int)

        # متریک‌ها
        try:
            auc = roc_auc_score(yte, y_score)
        except Exception:
            auc = np.nan
        acc = accuracy_score(yte, y_pred)

        scores.append({"fold": fold_id, "auc": float(auc), "acc": float(acc)})

        # ذخیرهٔ پیش‌بینی‌ها
        # سعی می‌کنیم Sample_ID را هم ضمیمه کنیم اگر قابل بازیابی باشد
        row = pd.DataFrame({
            "y_true": yte,
            "y_score": y_score,
            "y_pred": y_pred,
        })
        if sample_col:
            # ایندکس تست را روی df اصلی مپ می‌کنیم
            row[sample_col] = df.iloc[te_idx][sample_col].values
            # ستون نمونه را بیاور اول
            cols = [sample_col, "y_true", "y_score", "y_pred"]
            row = row[cols]

        row["fold"] = fold_id
        preds_records.append(row)

    preds = pd.concat(preds_records, ignore_index=True)
    scores_df = pd.DataFrame(scores)
    summary = {
        "model": model_name,
        "rank": args.rank,
        "cv_mode": args.cv_mode,
        "n_folds_requested": args.n_folds,
        "n_folds_effective": int(scores_df["fold"].nunique()) if not scores_df.empty else None,
        "seed": args.seed,
        "group_col": group_col,
        "n_features": int(X.shape[1]),
        "features_file": str((outdir / "features_used.txt").resolve()),
    }

    # فایل‌ها
    preds.to_csv(outdir / "cv_predictions.csv", index=False)
    # امتیازها
    cv_scores = {
        "per_fold": scores,
        "auc_mean": float(np.nanmean(scores_df["auc"])) if not scores_df.empty else None,
        "auc_std": float(np.nanstd(scores_df["auc"])) if not scores_df.empty else None,
        "acc_mean": float(np.nanmean(scores_df["acc"])) if not scores_df.empty else None,
        "acc_std": float(np.nanstd(scores_df["acc"])) if not scores_df.empty else None,
    }
    Path(outdir / "cv_scores.json").write_text(json.dumps(cv_scores, ensure_ascii=False, indent=2), encoding="utf-8")

    # لیست فیچرها
    features_used = list(X_num.columns)
    Path(outdir / "features_used.txt").write_text("\n".join(map(str, features_used)), encoding="utf-8")

    # متای ران
    Path(outdir / "run_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
