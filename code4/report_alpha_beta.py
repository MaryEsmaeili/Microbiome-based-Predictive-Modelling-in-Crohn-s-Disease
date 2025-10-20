#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds a concise Markdown report for alpha/beta diversity results across ranks.
Inputs are discovered under results/{alpha_models,beta_models}/{genus,species}.
Colors are read from config/colors.yml (optional).
"""

import os, glob, argparse, json, textwrap
import pandas as pd
import yaml

def load_yaml(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

def summarize_alpha(dir_level):
    out = {}
    p_glob = os.path.join(dir_level, "paired_wilcoxon_summary.csv")
    if os.path.exists(p_glob):
        out["paired"] = pd.read_csv(p_glob)
    for fn in ["alpha_with_covariates.csv",
               "alpha_models_oral.csv", "alpha_models_fecal.csv",
               "alpha_site_disease_interaction.csv"]:
        p = os.path.join(dir_level, fn)
        if os.path.exists(p):
            out[fn] = pd.read_csv(p)
    return out

def summarize_beta(dir_level):
    out = {}
    base = dir_level
    # core permanovas
    for fn in ["permanova_oral_bray_with_cov.csv",
               "permanova_fecal_bray_with_cov.csv",
               "permanova_interaction_bray.csv",
               "permdisp_oral_bray.csv",
               "permdisp_fecal_bray.csv",
               "beta_group_distances.csv",
               "pairwise_oral_fecal_summary.csv"]:
        p = os.path.join(base, fn)
        if os.path.exists(p):
            out[fn] = pd.read_csv(p)
    # figures we’ll reference
    figs = [ "pcoa_bray_allgroups.png",
             "pcoa_bray_oral_CH.png", "pcoa_bray_fecal_CH.png" ]
    out["figs"] = [os.path.join(base, f) for f in figs if os.path.exists(os.path.join(base, f))]
    return out

def table_to_md(df, max_rows=12):
    if df is None or df.empty: return "_(empty)_\n"
    dfx = df.copy()
    if len(dfx) > max_rows: dfx = dfx.head(max_rows)
    return dfx.to_markdown(index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results/alpha_beta/summary.md")
    ap.add_argument("--colors", default="config/colors.yml")
    ap.add_argument("--levels", nargs="+", default=["genus","species"])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    colors = load_yaml(args.colors)

    lines = []
    lines += ["# Alpha/Beta diversity — summary\n"]
    if colors:
        lines += ["_Colors loaded from `config/colors.yml`._\n"]

    for lvl in args.levels:
        lines += [f"## Level: **{lvl}**\n"]
        a_dir = f"results/alpha_models/{lvl}"
        b_dir = f"results/beta_models/{lvl}"

        # Alpha
        lines += ["### Alpha diversity\n"]
        a = summarize_alpha(a_dir)
        if "alpha_with_covariates.csv" in a:
            lines += ["**Models with covariates (subset):**\n",
                      table_to_md(a["alpha_with_covariates.csv"]), "\n"]
        if "paired" in a:
            lines += ["**Paired Wilcoxon (Crohn oral↔fecal):**\n",
                      table_to_md(a["paired"]), "\n"]

        # Beta
        lines += ["### Beta diversity & interactions (Bray)\n"]
        b = summarize_beta(b_dir)
        for k in ["permanova_oral_bray_with_cov.csv",
                  "permanova_fecal_bray_with_cov.csv",
                  "permanova_interaction_bray.csv",
                  "permdisp_oral_bray.csv",
                  "permdisp_fecal_bray.csv"]:
            if k in b:
                lines += [f"**{k}**\n", table_to_md(b[k]), "\n"]

        # Distances
        for k in ["beta_group_distances.csv","pairwise_oral_fecal_summary.csv"]:
            if k in b:
                lines += [f"**{k}**\n", table_to_md(b[k]), "\n"]

        # Figures (just list paths to embed later)
        if b.get("figs"):
            lines += ["**Key figures:**\n"]
            for fp in b["figs"]:
                lines += [f"- {fp}"]
            lines += ["\n"]

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print("[INFO] Wrote", args.out)

if __name__ == "__main__":
    main()
