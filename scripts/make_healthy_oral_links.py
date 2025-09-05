#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re
from pathlib import Path
import pandas as pd

def read_any(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, dtype=str)
    except Exception:
        try:
            return pd.read_csv(p, sep="\t", dtype=str)
        except Exception:
            return pd.read_csv(p, sep=None, engine="python", dtype=str)

def last6_digits_from_sample(x: str | None) -> str | None:
    # فقط رقم‌ها را برمی‌داریم و ۶ رقم آخر را می‌گیریم
    s = re.sub(r"\D", "", str(x) if x is not None else "")
    if len(s) < 6:
        return None
    return s[-6:]

def main(inp: str, outp: str):
    df = read_any(Path(inp))
    cols = {c.lower(): c for c in df.columns}

    # باید دقیقاً این دو ستون را داشته باشیم:
    if "sample" not in cols:
        raise SystemExit("ستون 'Sample' در ورودی پیدا نشد.")
    dag3_col = cols.get("dag3_sampleid") or ("DAG3_sampleID" if "DAG3_sampleID" in df.columns else None)
    if dag3_col is None:
        raise SystemExit("ستون 'DAG3_sampleID' / 'dag3_sampleid' در ورودی پیدا نشد.")

    out = pd.DataFrame({
        "DAG3_sampleID": df[dag3_col].astype(str).str.strip(),
        "OralID": df[cols["sample"]].map(last6_digits_from_sample)
    })

    # حذف ردیف‌های بدون OralID معتبر و تمیزکاری
    out = out.dropna(subset=["OralID"]).copy()
    out["OralID"] = out["OralID"].str.zfill(6)
    out = out.drop_duplicates()

    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)
    print(f"[OK] wrote {outp} rows={len(out)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", "-i", required=True)
    ap.add_argument("--out", "-o", required=True)
    args = ap.parse_args()
    main(args.infile, args.out)
