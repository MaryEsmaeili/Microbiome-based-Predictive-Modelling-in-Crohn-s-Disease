#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def safe_read_csv(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None

def safe_read_json(p):
    p = Path(p)
    if not p.exists(): return None
    with open(p) as f: return json.load(f)

def confusion_from_preds(df, thr=0.5):
    # df باید شامل y_true و یکی از y_score یا y_pred باشد
    if df is None or df.empty:
        return {"tp":0,"fp":0,"tn":0,"fn":0,
                "accuracy":np.nan,"precision":np.nan,"recall":np.nan,"specificity":np.nan}
    if "y_true" not in df.columns:
        raise ValueError("predictions CSV must contain 'y_true' column")
    y_true = df["y_true"].astype(int).values

    # اگر y_score هست، آستانه‌گذاری کن؛ وگرنه از y_pred استفاده کن
    if "y_score" in df.columns:
        y_pred = (df["y_score"].values >= thr).astype(int)
        have_score = True
    elif "y_pred" in df.columns:
        y_pred = df["y_pred"].astype(int).values
        have_score = False
    else:
        raise ValueError("predictions CSV must contain 'y_score' or 'y_pred'")

    tp = int(((y_true==1) & (y_pred==1)).sum())
    tn = int(((y_true==0) & (y_pred==0)).sum())
    fp = int(((y_true==0) & (y_pred==1)).sum())
    fn = int(((y_true==1) & (y_pred==0)).sum())
    acc = (tp+tn)/max(tp+tn+fp+fn, 1)
    prec = tp/max(tp+fp, 1) if (tp+fp)>0 else np.nan
    rec = tp/max(tp+fn, 1) if (tp+fn)>0 else np.nan
    spec = tn/max(tn+fp, 1) if (tn+fp)>0 else np.nan
    out = {"tp":tp,"fp":fp,"tn":tn,"fn":fn,
           "accuracy":acc,"precision":prec,"recall":rec,"specificity":spec}
    out["_have_score"] = have_score
    return out

def try_auc(df):
    # اگر y_score داشته باشیم ROC-AUC حساب می‌کنیم، وگرنه NaN
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return np.nan
    if df is None or df.empty or "y_score" not in df.columns:
        return np.nan
    y_true = df["y_true"].astype(int).values
    y_score = df["y_score"].astype(float).values
    # اگر همهٔ y_true یک کلاس باشند، AUC تعریف ندارد
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan

def pack_one(model_name, d):
    d = Path(d)
    preds = safe_read_csv(d/"cv_predictions.csv")
    scores = safe_read_json(d/"cv_scores.json")
    meta = safe_read_json(d/"run_meta.json")
    feats = None
    fp = d/"features_used.txt"
    if fp.exists():
        with open(fp) as f:
            feats = [ln.strip() for ln in f if ln.strip()]
    return {"name": model_name, "dir": str(d),
            "preds": preds, "scores": scores, "meta": meta, "features": feats}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genus-dir", required=True)
    ap.add_argument("--species-dir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = [
        pack_one("genus_logreg", args.genus_dir),
        pack_one("species_logreg", args.species_dir),
    ]

    # خلاصهٔ کلی
    rows_summary = []
    rows_conf = []
    all_meta = {}

    for r in runs:
        preds = r["preds"]
        conf = confusion_from_preds(preds)
        auc = try_auc(preds)

        # سطر خلاصه
        rows_summary.append({
            "run": r["name"],
            "dir": r["dir"],
            "accuracy": conf["accuracy"],
            "precision": conf["precision"],
            "recall": conf["recall"],
            "specificity": conf["specificity"],
            "roc_auc": auc,
            "have_score": conf.get("_have_score", False),
            "n_rows": 0 if preds is None else len(preds)
        })
        # ماتریس سردرگمی
        rows_conf.append({
            "run": r["name"], "tp": conf["tp"], "fp": conf["fp"],
            "tn": conf["tn"], "fn": conf["fn"]
        })
        # متادیتای اجرا
        if r["meta"] is not None:
            all_meta[r["name"]] = r["meta"]

    # ویژگی‌های برتر (اگر وجود داشت)
    top_rows = []
    for r in runs:
        feats = r["features"]
        if feats:
            for rank, feat in enumerate(feats, start=1):
                top_rows.append({"run": r["name"], "rank": rank, "feature": feat})

    # ذخیرهٔ خروجی‌ها
    pd.DataFrame(rows_summary).to_csv(outdir/"summary_overview.csv", index=False)
    pd.DataFrame(rows_conf).to_csv(outdir/"confusion_matrices.csv", index=False)
    pd.DataFrame(top_rows).to_csv(outdir/"top_features.csv", index=False)

    with open(outdir/"run_meta_collated.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    print({
        "written": [str(outdir/"summary_overview.csv"),
                    str(outdir/"confusion_matrices.csv"),
                    str(outdir/"top_features.csv"),
                    str(outdir/"run_meta_collated.json")]
    })

if __name__ == "__main__":
    main()
