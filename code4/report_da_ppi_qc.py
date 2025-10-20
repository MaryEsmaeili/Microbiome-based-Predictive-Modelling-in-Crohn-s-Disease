#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds a Markdown report summarizing DA with covariates, PPI interactions,
and targeted QC plots/tables. Works for genus & species if present.
"""

import os, argparse, pandas as pd, yaml

def load_yaml(path):
    if path and os.path.exists(path):
        with open(path, "r") as f: return yaml.safe_load(f)
    return {}

def read_if(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def da_block(level):
    base = f"results/da_models/{level}"
    return {
        "oral_unadj":  read_if(f"{base}/da_oral_unadj_{level}.csv"),
        "fecal_unadj": read_if(f"{base}/da_fecal_unadj_{level}.csv"),
        "all_adj":     read_if(f"{base}/da_all_adj_{level}.csv"),
        "oral_adj":    read_if(f"{base}/da_oral_adj_{level}.csv"),
        "fecal_adj":   read_if(f"{base}/da_fecal_adj_{level}.csv"),
        "figs": [p for p in [
            f"{base}/volcano_oral_unadj_{level}.png",
            f"{base}/volcano_fecal_unadj_{level}.png",
            f"{base}/volcano_all_adj_{level}.png",
            f"{base}/volcano_oral_adj_{level}.png",
            f"{base}/volcano_fecal_adj_{level}.png",
        ] if os.path.exists(p)]
    }

def ppi_block(level):
    base = f"results/ppi_interactions/{level}"
    return {
        "ppi_all":   read_if(f"{base}/ppi_effects_all.csv"),
        "ppi_oral":  read_if(f"{base}/ppi_effects_oral.csv"),
        "ppi_fecal": read_if(f"{base}/ppi_effects_fecal.csv"),
        "figs": [p for p in [
            f"{base}/volcano_ppi_all.png",
            f"{base}/volcano_ppi_oral.png",
            f"{base}/volcano_ppi_fecal.png",
            f"{base}/volcano_int_disease_ppi.png",
            f"{base}/volcano_int_disease_site.png",
        ] if os.path.exists(p)],
        "design": read_if(f"{base}/ppi_design_counts.csv")
    }

def qc_block(level):
    base = f"results/qc_targeted/{level}"
    return {
        "top": read_if(f"{base}/top_taxa_selected.csv"),
        "figs": [p for p in [
            f"{base}/box_by_ppi_oral.png",
            f"{base}/box_by_ppi_fecal.png",
            f"{base}/interaction_means.png",
            f"{base}/paired_spaghetti_crohn.png",
            f"{base}/density_top_taxa_oral.png",
            f"{base}/density_top_taxa_fecal.png",
        ] if os.path.exists(p)]
    }

def to_md(df, max_rows=15):
    if df is None or df.empty: return "_(empty)_\n"
    d = df.copy()
    if len(d)>max_rows: d = d.head(max_rows)
    return d.to_markdown(index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results/da_ppi_qc/summary.md")
    ap.add_argument("--levels", nargs="+", default=["genus","species"])
    ap.add_argument("--colors", default="config/colors.yml")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    colors = load_yaml(args.colors)

    lines = []
    lines += ["# DA + PPI interactions + Targeted QC — summary\n"]
    if colors: lines += ["_Colors loaded from `config/colors.yml`._\n"]

    for lvl in args.levels:
        lines += [f"## Level: **{lvl}**\n"]

        # DA
        lines += ["### Differential abundance (OLS per-taxon)\n"]
        da = da_block(lvl)
        for k in ["all_adj","oral_adj","fecal_adj"]:
            if da[k] is not None:
                lines += [f"**{k} (head):**\n", to_md(da[k]), "\n"]
        if da["figs"]:
            lines += ["**Volcano figures:**\n"] + [f"- {p}" for p in da["figs"]] + ["\n"]

        # PPI
        lines += ["### PPI main & interactions\n"]
        ppi = ppi_block(lvl)
        if ppi["ppi_all"] is not None:
            lines += ["**ppi_effects_all (head):**\n", to_md(ppi["ppi_all"]), "\n"]
        if ppi["design"] is not None:
            lines += ["**design counts:**\n", to_md(ppi["design"]), "\n"]
        if ppi["figs"]:
            lines += ["**PPI volcanoes:**\n"] + [f"- {p}" for p in ppi["figs"]] + ["\n"]

        # QC targeted
        lines += ["### Targeted QC (top taxa from PPI / variance)\n"]
        qc = qc_block(lvl)
        if qc["top"] is not None:
            lines += ["**Top taxa (selected):**\n", to_md(qc["top"]), "\n"]
        if qc["figs"]:
            lines += ["**QC figures:**\n"] + [f"- {p}" for p in qc["figs"]] + ["\n"]

    # minimal methods paragraph
    lines += [
        "\n---\n",
        "### Methods (brief)\n",
        "- OLS per-taxon with covariates; p-values BH-adjusted to q.\n",
        "- PPI models include main effect of `ppi_use` and interactions (`disease:ppi_use`, `disease:site`).\n",
        "- Targeted QC selects top taxa from PPI q-values (or variance fallback) and shows PPI-stratified boxplots, disease×site means, paired Crohn oral↔fecal spaghetti, and CLR densities.\n"
    ]

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print("[INFO] Wrote", args.out)

if __name__ == "__main__":
    main()
