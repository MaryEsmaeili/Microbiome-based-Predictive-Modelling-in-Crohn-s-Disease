#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ----------------- Robust loader -----------------
def _detect_clade_col(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if df.index.name and df.index.name.lower() in {"clade_name", "taxon"}:
        df = df.reset_index()
    for c in ["clade_name", "taxon", "Taxon", "Unnamed: 0"]:
        if c in df.columns:
            return df, c
    return df, df.columns[0]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    valcols = [c for c in df.columns if c != "clade_name"]
    for c in valcols:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(r"\s+", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def load_processed(path_csv: str, level: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, low_memory=False)
    df, clade_col = _detect_clade_col(df)
    df = df.rename(columns={clade_col: "clade_name"})

    # اگر احتمال transpose وجود داشته باشد
    numeric_like = df.drop(columns=["clade_name"], errors="ignore")
    if numeric_like.shape[1] < 5 and df.shape[0] > 50:
        df = df.set_index("clade_name").T.reset_index().rename(columns={"index": "clade_name"})

    df = _coerce_numeric(df)

    # سطح مورد نظر (species/genus)
    df["taxon"] = df["clade_name"].astype(str).apply(
        lambda x: next(
            (p for p in str(x).split("|") if p.startswith("s__" if level == "species" else "g__")),
            None,
        )
    )
    df = df[~df["taxon"].isna()].drop(columns=["clade_name"]).set_index("taxon")
    df = df.groupby(df.index, sort=False).sum()
    return df  # taxon × samples


# ----------------- ML part -----------------
def make_feature_matrix(meta, fecal_level):
    if "Fecal_sample_ID" not in meta.columns:
        raise RuntimeError("Metadata must contain Fecal_sample_ID.")

    ids = [c for c in meta["Fecal_sample_ID"].dropna().unique() if c in fecal_level.columns]
    if not ids:
        raise RuntimeError("هیچ fecal نمونه‌ی قابل‌استفاده برای Responder پیدا نشد.")

    sub = meta.set_index("Fecal_sample_ID").loc[ids]
    X = fecal_level[ids].T  # samples × taxon

    y = sub["Responder"].astype(float).astype(int).reindex(X.index)
    return X, y


def kfold_eval(X, y, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)

    lr = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=5000)
    rf = RandomForestClassifier(n_estimators=500, random_state=1)

    models = [("elasticnet", lr), ("rf", rf)]
    curves = {}

    for name, model in models:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        probs = np.zeros(len(y))
        for tr, te in skf.split(Xs, y):
            model.fit(Xs[tr], y.iloc[tr])
            probs[te] = model.predict_proba(Xs[te])[:, 1]

        auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else np.nan
        ap = average_precision_score(y, probs) if len(np.unique(y)) > 1 else np.nan

        fpr, tpr, _ = roc_curve(y, probs) if len(np.unique(y)) > 1 else (np.array([0, 1]), np.array([0, 1]), None)
        pr, rc, _ = precision_recall_curve(y, probs) if len(np.unique(y)) > 1 else (np.array([0, 1]), np.array([0, 1]), None)
        curves[name] = {"fpr": fpr, "tpr": tpr, "pr": pr, "rc": rc, "auc": auc, "ap": ap}
        pd.Series({"ROC_AUC": auc, "PR_AUC": ap}).to_csv(outdir / f"{name}_scores.csv")

    # ROC
    plt.figure()
    for name, c in curves.items():
        plt.plot(c["fpr"], c["tpr"], label=f"{name} (AUC={c['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", alpha=.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "roc.png", dpi=150)
    plt.close()

    # PR
    plt.figure()
    for name, c in curves.items():
        plt.plot(c["rc"], c["pr"], label=f"{name} (AP={c['ap']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pr.png", dpi=150)
    plt.close()


def run(level, meta_fp, oral_fp, fecal_fp, outdir):
    meta = pd.read_csv(meta_fp, dtype=str)
    fecal = load_processed(fecal_fp, level)
    X, y = make_feature_matrix(meta, fecal)
    # مطمئن شو همه عددی‌اند و ستون‌های تمام‌صفر حذف شوند
    X = X.astype(float)
    X = X.loc[:, (X > 0).any(axis=0)]
 # حذف ستون‌های صفر
    if len(np.unique(y)) < 2 or X.shape[1] == 0:
        print(f"[WARN] Not enough variation for {level}.")
        return
    kfold_eval(X, y, Path(outdir) / level)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--oral", required=True)  # فعلاً فقط برای امضا
    ap.add_argument("--fecal", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    run("species", args.metadata, args.oral, args.fecal, args.outdir)
    run("genus", args.metadata, args.oral, args.fecal, args.outdir)


if __name__ == "__main__":
    main()
