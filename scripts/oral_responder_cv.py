#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Responder vs Non-responder (oral) — Repeated CV with OOF calibration/PR/threshold
+ Explainability: L1-LogReg bootstrap stability, RF permutation importance across CV.

Inputs
------
--abund : CSV abundance (rows=taxa, cols=samples)  [oral]
--meta  : clinical_clean.csv  (columns: SampleID/Oral_sample_ID, Responder, ...)

Outputs (in --outdir)
---------------------
metrics_cv.csv, summary_mean_sd.csv
preds_logregL1_oof.csv, preds_rf_oof.csv
calibration_logreg.png, calibration_rf.png, pr_curve.png
threshold_report.csv, cm_normalized.png
selected_features.csv (اگر FS روشن باشد)
coef_stability.csv
rf_permutation_importance_cv.csv
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
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

plt.rcParams["figure.dpi"] = 160

def load_abundance(abund_path: str, level: str) -> pd.DataFrame:
    df = pd.read_csv(abund_path, index_col=0).fillna(0)
    df = df.loc[df.sum(axis=1) > 0]  # drop all-zero taxa
    if level.lower() not in {"all","genus","species"}:
        raise ValueError("--level must be one of all/genus/species")
    if level in {"genus","species"}:
        def extract_level(name: str, level: str):
            parts = str(name).split("|")
            if level=="genus":
                xs=[p for p in parts if p.startswith("g__")]; return xs[0].replace("g__","") if xs else None
            else:
                xs=[p for p in parts if p.startswith("s__")]; return xs[0].replace("s__","") if xs else None
        mask = df.index.map(lambda x: extract_level(x, level) is not None)
        df = df.loc[mask].copy()
        df.index = df.index.map(lambda x: extract_level(x, level))
    return df

def load_meta(meta_path: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)
    meta.columns = meta.columns.astype(str).str.strip()
    # Accept either 'SampleID' or 'Oral_sample_ID'
    idcol = "SampleID" if "SampleID" in meta.columns else ("Oral_sample_ID" if "Oral_sample_ID" in meta.columns else None)
    if idcol is None:
        raise KeyError("Expected 'SampleID' or 'Oral_sample_ID' in clinical_clean.")
    meta = meta.rename(columns={idcol: "SampleID"})
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip().str.upper()
    if "Responder" not in meta.columns:
        raise KeyError("Expected 'Responder' column in clinical_clean.")
    # Ensure numeric 0/1 (will preserve Int64)
    meta["Responder"] = pd.to_numeric(meta["Responder"], errors="coerce")
    return meta[["SampleID","Responder"]].dropna()

def align(X_abund: pd.DataFrame, meta: pd.DataFrame):
    X_abund.columns = pd.Index([str(c).strip().upper() for c in X_abund.columns])
    common = [s for s in meta["SampleID"] if s in X_abund.columns]
    if len(common) < 8:
        print(f"[WARN] only {len(common)} samples match between abundance and metadata.", file=sys.stderr)
    X = X_abund[common].T  # samples x features
    y = meta.set_index("SampleID").loc[common, "Responder"].astype(int).values
    return X, y, common

def feature_select(X: pd.DataFrame, y: np.ndarray, vt: float, kbest: int|None):
    cols = X.columns
    if vt > 0:
        vt_sel = VarianceThreshold(threshold=vt)
        X_v = vt_sel.fit_transform(X)
        cols = cols[vt_sel.get_support()]
        X = pd.DataFrame(X_v, index=X.index, columns=cols)
    if kbest and kbest > 0 and kbest < X.shape[1]:
        skb = SelectKBest(f_classif, k=kbest)
        X_k = skb.fit_transform(X, y)
        cols = cols[skb.get_support()]
        X = pd.DataFrame(X_k, index=X.index, columns=cols)
    return X, list(cols)

def cv_oof(X: pd.DataFrame, y: np.ndarray, repeats: int, folds: int, seed: int, scale: bool):
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    models = {
        "logregL1": Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True) if scale else "passthrough"),
            ("clf", LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=4000, class_weight="balanced", random_state=seed))
        ]),
        "rf": Pipeline([
            ("scaler", "passthrough"),
            ("clf", RandomForestClassifier(n_estimators=500, random_state=seed, class_weight="balanced"))
        ])
    }
    metrics_rows = []
    oof = {m: np.zeros(len(y), dtype=float) for m in models}
    fold_preds = {}  # store per-fold fitted estimators for explainability
    for name in models:
        fold_preds[name] = []

    for fold_idx, (tr, te) in enumerate(rskf.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]
        for name, pipe in models.items():
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            oof[name][te] = proba
            metrics_rows.append({
                "model": name, "fold": fold_idx,
                "accuracy": accuracy_score(y_te, pred),
                "f1": f1_score(y_te, pred, zero_division=0),
                "auc": roc_auc_score(y_te, proba)
            })
            fold_preds[name].append((pipe, tr, te))
    metrics_cv = pd.DataFrame(metrics_rows)
    return metrics_cv, oof, fold_preds

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
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction positive")
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
    ax.set_xticklabels(["Non-Resp","Resp"]); ax.set_yticklabels(["Non-Resp","Resp"])
    ax.set_title(f"Normalized CM @ threshold={thr:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(outfile); plt.close(fig)

def coef_stability_bootstrap(X: pd.DataFrame, y: np.ndarray, n_boot: int, seed: int):
    rng = np.random.RandomState(seed)
    feats = np.array(X.columns)
    counts = np.zeros(len(feats), dtype=int)
    for b in range(n_boot):
        idx = rng.choice(np.arange(len(y)), size=len(y), replace=True)
        Xb, yb = X.iloc[idx], y[idx]
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=4000, class_weight="balanced", random_state=seed+b))
        ])
        pipe.fit(Xb, yb)
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "coef_"):
            nz = (clf.coef_.ravel() != 0)
            counts[nz] += 1
    pct = (counts / n_boot) * 100.0
    return pd.DataFrame({"feature": feats, "selected_pct": pct}).sort_values("selected_pct", ascending=False)

def rf_perm_importance_across_cv(
    fold_preds, X: pd.DataFrame, y: np.ndarray, seed: int,
    topk: int = 50, max_folds: int = 8, n_repeats: int = 5,
    metric: str = "roc_auc",
):
    """
    Permutation importance فقط روی Top-K فیچرها و چند فولد محدود.
    نکتهٔ مهم: همیشه کل X ولیدیشن را به RF می‌دهیم و فقط ستونِ هدف را permute می‌کنیم،
    تا mismatch تعداد/اسم فیچرها پیش نیاید.
    """
    rng = np.random.RandomState(seed)
    feats_all = np.array(X.columns)

    # 1) انتخاب Top-K با میانگین Gini روی مدل‌های فولد
    gini_sum = np.zeros(len(feats_all), dtype=float)
    for (pipe, tr, te) in fold_preds["rf"]:
        gini_sum += pipe.named_steps["clf"].feature_importances_
    order = np.argsort(gini_sum)[::-1]
    if topk and 0 < topk < len(feats_all):
        feats_sel = feats_all[order[:topk]]
    else:
        feats_sel = feats_all

    # جمع‌کنندهٔ امتیازها برای هر فیچر در فولدهای مختلف
    per_feat_scores = {f: [] for f in feats_sel}

    # 2) روی چند فولد اول ولیدیشن، permutation انجام بده
    for i, (pipe, tr, te) in enumerate(fold_preds["rf"], start=1):
        if max_folds and max_folds > 0 and i > max_folds:
            break

        rf = pipe.named_steps["clf"]
        Xv = X.iloc[te]        # مهم: کل ستون‌ها
        yv = y[te]

        base_proba = rf.predict_proba(Xv)[:, 1]
        if metric == "roc_auc":
            from sklearn.metrics import roc_auc_score
            base_score = roc_auc_score(yv, base_proba)
            def _score(Xtmp):
                return roc_auc_score(yv, rf.predict_proba(Xtmp)[:, 1])
        else:
            from sklearn.metrics import accuracy_score
            base_score = accuracy_score(yv, (base_proba >= 0.5).astype(int))
            def _score(Xtmp):
                proba = rf.predict_proba(Xtmp)[:, 1]
                return accuracy_score(yv, (proba >= 0.5).astype(int))

        # برای هر فیچر انتخابی، n_repeats بار permute و امتیاز کاهش را حساب کن
        for f in feats_sel:
            deltas = []
            for r in range(n_repeats):
                Xp = Xv.copy()
                Xp[f] = rng.permutation(Xp[f].values)
                sc = _score(Xp)
                deltas.append(base_score - sc)  # کاهش امتیاز = اهمیت
            per_feat_scores[f].append(np.mean(deltas))

    # 3) میانگین و SD اهمیت در فولدهای استفاده‌شده
    rows = []
    for f, vals in per_feat_scores.items():
        if len(vals) == 0:
            continue
        rows.append({"feature": f, "mean": float(np.mean(vals)), "std": float(np.std(vals))})
    imp = pd.DataFrame(rows).sort_values("mean", ascending=False)
    return imp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abund", required=True)
    ap.add_argument("--meta", required=True)  # clinical_clean.csv
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--level", choices=["all","genus","species"], default="all")
    ap.add_argument("--vt", type=float, default=1e-4)
    ap.add_argument("--kbest", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--perm-topk", type=int, default=50,
                help="Run permutation importance only on top-K RF features (by mean Gini across folds). 0=all")
    ap.add_argument("--perm-max-folds", type=int, default=8,
                    help="Limit number of validation folds used for permutation importance. 0=all")
    ap.add_argument("--perm-n-repeats", type=int, default=5,
                    help="Permutation repeats per fold (scikit-learn n_repeats).")

    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    abund = load_abundance(args.abund, args.level)
    meta  = load_meta(args.meta)
    X_raw, y, sample_ids = align(abund, meta)

    # Optional feature selection
    X, kept = feature_select(X_raw, y, vt=args.vt, kbest=(args.kbest if args.kbest>0 else None))
    pd.Series(kept, name="selected_features").to_csv(outdir/"selected_features.csv", index=False)

    metrics_cv, oof, fold_preds = cv_oof(X, y, args.repeats, args.folds, args.seed, args.scale)
    save_summary(metrics_cv, outdir)

    # Save OOF preds
    pd.DataFrame({"sample_id": sample_ids, "y_true": y, "y_proba": oof["logregL1"]}).to_csv(outdir/"preds_logregL1_oof.csv", index=False)
    pd.DataFrame({"sample_id": sample_ids, "y_true": y, "y_proba": oof["rf"]}).to_csv(outdir/"preds_rf_oof.csv", index=False)

    # Calibration & PR
    plot_calibration(y, oof["logregL1"], "Calibration (LogReg L1, OOF)", outdir/"calibration_logreg.png")
    plot_calibration(y, oof["rf"],       "Calibration (RF, OOF)",        outdir/"calibration_rf.png")
    plot_pr(y, {"logregL1": oof["logregL1"], "rf": oof["rf"]}, outdir/"pr_curve.png")

    # Threshold (best model by mean AUC)
    best_model = metrics_cv.groupby("model")["auc"].mean().idxmax()
    probs_best = oof[best_model]
    thr, stats = youden_threshold(y, probs_best)
    pd.DataFrame([{"model": best_model, "threshold": thr, **stats}]).to_csv(outdir/"threshold_report.csv", index=False)
    plot_cm_norm(y, probs_best, thr, outdir/"cm_normalized.png")

    # Explainability
    stab = coef_stability_bootstrap(X, y, n_boot=args.n_boot, seed=args.seed)
    stab.to_csv(outdir/"coef_stability.csv", index=False)
    imp = rf_perm_importance_across_cv(
        fold_preds, X, y, seed=args.seed,
        topk=args.perm_topk,
        max_folds=args.perm_max_folds,
        n_repeats=args.perm_n_repeats
    )
    imp.to_csv(outdir/"rf_permutation_importance_cv.csv", index=False)

if __name__ == "__main__":
    main()
