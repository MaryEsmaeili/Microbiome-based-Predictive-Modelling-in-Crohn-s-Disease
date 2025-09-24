#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, glob
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    rows = []
    for score_file in glob.glob(args.runs_glob):
        run_dir = Path(score_file).parent
        rec = {"run_dir": str(run_dir)}

        meta_p = run_dir / "run_meta.json"
        if meta_p.exists():
            try:
                rec.update(json.loads(meta_p.read_text()))
            except Exception:
                pass

        try:
            rec["scores"] = json.loads(Path(score_file).read_text())
        except Exception:
            rec["scores"] = None

        preds_p = run_dir / "cv_predictions.csv"
        if preds_p.exists():
            rec["predictions_path"] = str(preds_p)

        feats_p = run_dir / "features_used.txt"
        if feats_p.exists():
            rec["features_path"] = str(feats_p)

        rows.append(rec)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    with open(args.out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
