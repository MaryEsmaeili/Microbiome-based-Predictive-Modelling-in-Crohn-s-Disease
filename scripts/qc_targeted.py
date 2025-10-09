#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- NEW: YAML colors ----------
try:
    import yaml
except ImportError:
    yaml = None

def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0,2,4))

def _rgb_to_hex(rgb):
    r,g,b = [max(0,min(1,x)) for x in rgb]
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

def _mix(c1, c2, w=0.5):
    a = np.array(_hex_to_rgb(c1)); b = np.array(_hex_to_rgb(c2))
    return _rgb_to_hex(a*(1-w)+b*w)

def _lighten(hexcolor, amt=0.25):  # 0..1
    return _mix(hexcolor, "#ffffff", amt)

def _darken(hexcolor, amt=0.20):
    return _mix(hexcolor, "#000000", amt)

def load_colors(yaml_path=None):
    """
    Reads group colors + synonyms. Falls back to sensible defaults if needed.
    Returns:
      group_colors: dict with keys in {'Fecal_Crohn','Oral_Crohn','Fecal_Healthy','Oral_Healthy'}
      site_base:    {'Oral': hex, 'Fecal': hex} (bases used to derive PPI colors)
      ppi_palette:  function(site)->(c0,c1) for PPI=0/1
      line_colors:  {'Healthy': hex, 'Crohn': hex} for interaction plot
      qual_palette: list of ~12 qualitative colors for KDE (taxa)
    """
    # defaults (in case YAML or PyYAML is missing)
    defaults = {
        "Fecal_Crohn":   "#30638e",
        "Oral_Crohn":    "#edae49",
        "Fecal_Healthy": "#d1495b",
        "Oral_Healthy":  "#00798c",
    }
    cfg = None
    if yaml_path and os.path.exists(yaml_path) and yaml is not None:
        with open(yaml_path,"r") as f:
            cfg = yaml.safe_load(f) or {}
    g = ((cfg or {}).get("group") or {}) if cfg else {}
    group_colors = {
        "Fecal_Crohn":   g.get("Fecal_Crohn",   defaults["Fecal_Crohn"]),
        "Oral_Crohn":    g.get("Oral_Crohn",    defaults["Oral_Crohn"]),
        "Fecal_Healthy": g.get("Fecal_Healthy", defaults["Fecal_Healthy"]),
        "Oral_Healthy":  g.get("Oral_Healthy",  defaults["Oral_Healthy"]),
    }
    # base per site (هماهنگ با YAML)
    site_base = {
        "Oral":  group_colors["Oral_Healthy"],
        "Fecal": group_colors["Fecal_Healthy"],
    }
    # PPI=0/1 palette per site (روشن/تیره از رنگ پایه همان سایت)
    def ppi_palette(site):
        base = site_base.get(site, "#777777")
        return (_lighten(base, 0.35), _darken(base, 0.15))  # (PPI=0, PPI=1)
    # Interaction plot: Healthy vs Crohn
    line_colors = {
        "Healthy": group_colors["Oral_Healthy"],
        "Crohn":   group_colors["Fecal_Crohn"],
    }
    # Qualitative 12-color palette برای KDE و موارد دیگر (چرخه روشن/تیره)
    bases = [
        group_colors["Oral_Healthy"],
        group_colors["Oral_Crohn"],
        group_colors["Fecal_Healthy"],
        group_colors["Fecal_Crohn"],
    ]
    qual = []
    for b in bases:
        qual += [_lighten(b,0.15), b, _darken(b,0.18)]
    return group_colors, site_base, ppi_palette, line_colors, qual[:12]

# ---------- existing data helpers ----------
def clr_transform(pct, pseudocount=1e-6):
    X = (pct.astype(float)/100.0)+pseudocount
    logX = np.log(X)
    gm = logX.mean(axis=0)
    return logX.sub(gm, axis=1)

def norm_meta(meta):
    m = meta.copy()
    m.columns = [c.strip() for c in m.columns]
    lower = {c.lower(): c for c in m.columns}
    def pick(*ks): 
        for k in ks:
            if k in lower: return lower[k]
        return None
    out = pd.DataFrame({
        "sample_id": m[pick("sample_id","id","sample","sampleid")],
        "site":      m[pick("site","body_site","location")],
        "disease":   m[pick("disease","status","group")],
        "ppi_use":   m[pick("ppi_use","ppi","ppi_current","ppi3m")] if pick("ppi_use","ppi","ppi_current","ppi3m") else np.nan,
    })
    out["sample_id"] = out["sample_id"].astype(str)
    out["site"] = out["site"].astype(str).str.capitalize().replace({"Faecal":"Fecal"})
    out.loc[~out["site"].isin(["Oral","Fecal"]), "site"] = np.nan
    def to01(x):
        if pd.isna(x): return np.nan
        s=str(x).strip().lower()
        if s in {"1","yes","true","y","crohn","cd","case"}: return 1
        if s in {"0","no","false","n","healthy","control","hc"}: return 0
        try:
            v=int(float(s)); 
            if v in (0,1): return v
        except: pass
        return np.nan
    out["disease"] = out["disease"].map(to01)
    if "ppi_use" in out: out["ppi_use"] = out["ppi_use"].map(to01)
    return out

def pick_top_taxa(ppi_all_csv, n=12, fallback_by_var=None):
    if ppi_all_csv and os.path.exists(ppi_all_csv):
        df = pd.read_csv(ppi_all_csv)
        prio = []
        for term in ["disease:site","disease:ppi_use","ppi_use"]:
            sub = df[df["term"].astype(str).str.startswith(term, na=False)].copy()
            if "q" in sub: sub = sub.sort_values("q")
            elif "p" in sub: sub = sub.sort_values("p")
            prio.append(sub.head(n*2))
        cand = pd.concat(prio, ignore_index=True) if prio else pd.DataFrame()
        if not cand.empty:
            top = cand.groupby("taxon")["p"].min().sort_values().head(n).index.tolist()
            return top
    if fallback_by_var is not None:
        return fallback_by_var.sort_values(ascending=False).head(n).index.tolist()
    return []

# ---------- PLOTS (now color-aware) ----------
def box_by_ppi(long, taxa, site, out_png, ppi_palette_fn):
    d = long[(long["site"]==site) & (long["taxon"].isin(taxa))].copy()
    ensure_dir(os.path.dirname(out_png))
    if d.empty:
        fig,ax = plt.subplots(figsize=(7,4)); ax.axis("off"); ax.text(0.5,0.5,"No data", ha="center"); fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig); return
    fig, ax = plt.subplots(figsize=(min(14, 1.2*len(taxa)+4), 5))
    c0, c1 = ppi_palette_fn(site)  # PPI=0, PPI=1
    order = taxa
    data0 = [d[(d["taxon"]==t) & (d["ppi_use"]==0)]["CLR"].values for t in order]
    data1 = [d[(d["taxon"]==t) & (d["ppi_use"]==1)]["CLR"].values for t in order]
    pos = np.arange(len(order))
    bp0 = ax.boxplot(data0, positions=pos-0.18, widths=0.32, patch_artist=True)
    bp1 = ax.boxplot(data1, positions=pos+0.18, widths=0.32, patch_artist=True)
    for b in bp0['boxes']: b.set(facecolor=c0, edgecolor=_darken(c0,0.35), alpha=0.95)
    for b in bp1['boxes']: b.set(facecolor=c1, edgecolor=_darken(c1,0.35), alpha=0.95)
    for part in ["whiskers","caps","medians","fliers"]:
        for l in bp0.get(part, []): l.set(color=_darken(c0,0.45))
        for l in bp1.get(part, []): l.set(color=_darken(c1,0.45))
    ax.set_xticks(pos); ax.set_xticklabels(order, rotation=60, ha="right")
    ax.set_title(f"CLR by PPI — {site}")
    ax.set_ylabel("CLR")
    # legend patches
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor=c0), Patch(facecolor=c1)], ["PPI=0","PPI=1"], loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)

def interaction_means(long, taxa, out_png, line_colors):
    d = long[long["taxon"].isin(taxa)].copy()
    ensure_dir(os.path.dirname(out_png))
    if d.empty:
        fig,ax = plt.subplots(); ax.axis("off"); fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig); return
    grp = d.groupby(["taxon","disease","site"])["CLR"].mean().unstack(["disease","site"])
    fig, ax = plt.subplots(figsize=(min(14, 1.2*len(taxa)+4), 5))
    for i,t in enumerate(taxa):
        if t not in grp.index: continue
        row = grp.loc[t]
        xs = [0,1]  # Oral→Fecal
        y_h = [row.get((0,"Oral"), np.nan), row.get((0,"Fecal"), np.nan)]
        y_c = [row.get((1,"Oral"), np.nan), row.get((1,"Fecal"), np.nan)]
        ax.plot([i+xx for xx in xs], y_h, marker="o", lw=2.0, color=line_colors["Healthy"], alpha=0.95, label="Healthy" if i==0 else None)
        ax.plot([i+xx for xx in xs], y_c, marker="o", lw=2.0, color=line_colors["Crohn"],   alpha=0.95, label="Crohn"   if i==0 else None)
    ax.set_xticks([i+0.5 for i in range(len(taxa))])
    ax.set_xticklabels(taxa, rotation=60, ha="right")
    ax.set_title("Interaction means (disease × site)")
    ax.set_ylabel("Mean CLR"); ax.legend(loc="best", frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)

def spaghetti_crohn(long, taxa, pairs_csv, out_png):
    ensure_dir(os.path.dirname(out_png))
    if (pairs_csv is None) or (not os.path.exists(pairs_csv)):
        fig,ax = plt.subplots(); ax.axis("off"); ax.text(0.5,0.5,"No pairs file",ha="center"); fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig); return
    pairs = pd.read_csv(pairs_csv)
    cols = [c.lower() for c in pairs.columns]
    def pick(*opts):
        return next((pairs.columns[cols.index(o.lower())] for o in opts if o.lower() in cols), None)
    oc = pick("Oral_ID","Oral","OC","oral_id","oral")
    fc = pick("Fecal_ID","Fecal","FC","fecal_id","fecal")
    if oc is None or fc is None:
        fig,ax = plt.subplots(); ax.axis("off"); ax.text(0.5,0.5,"Pairs columns not found",ha="center"); fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig); return
    crohn = long[long["disease"]==1].copy()
    crohn_ids = set(crohn["Sample_ID"].astype(str))
    usable = pairs[[oc,fc]].dropna().astype(str).values.tolist()
    usable = [(o,f) for (o,f) in usable if o in crohn_ids and f in crohn_ids]
    if len(usable)<5:
        fig,ax = plt.subplots(); ax.axis("off"); ax.text(0.5,0.5,"Not enough Crohn pairs",ha="center"); fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig); return

    fig, axes = plt.subplots(nrows=len(taxa), ncols=1, figsize=(8, 2.2*len(taxa)), sharex=True)
    if len(taxa)==1: axes=[axes]
    for ax,t in zip(axes,taxa):
        dd = crohn[crohn["taxon"]==t]
        dd = dd.set_index(["Sample_ID","site"])["CLR"].unstack("site")
        for (o,f) in usable:
            if (o in dd.index) and (f in dd.index):
                xo, xf = dd.loc[o,"Oral"], dd.loc[f,"Fecal"]
                if pd.notna(xo) and pd.notna(xf):
                    ax.plot([0,1],[xo,xf], marker="o", alpha=0.35, color="#9aa0a6")  # harmonious neutral
        ax.set_title(t); ax.set_ylabel("CLR")
    axes[-1].set_xticks([0,1]); axes[-1].set_xticklabels(["Oral","Fecal"])
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)

def kde_by_site(long, taxa, site, out_png, qual_palette):
    d = long[(long["site"]==site) & (long["taxon"].isin(taxa))].copy()
    ensure_dir(os.path.dirname(out_png))
    if d.empty:
        fig,ax = plt.subplots(); ax.axis("off"); fig.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close(fig); return
    fig,ax = plt.subplots(figsize=(10,6))
    from scipy.stats import gaussian_kde
    colors = (qual_palette * ((len(taxa)+len(qual_palette)-1)//len(qual_palette)))[:len(taxa)]
    for t,c in zip(taxa, colors):
        v = d.loc[d["taxon"]==t, "CLR"].astype(float).values
        v = v[np.isfinite(v)]
        if len(v) > 1:
            xs = np.linspace(np.nanmin(v)-1, np.nanmax(v)+1, 200)
            kde = gaussian_kde(v)
            ax.plot(xs, kde(xs), alpha=0.95, lw=2.0, label=t, color=c)
    ax.set_title(f"CLR density — {site} (top taxa)")
    ax.set_xlabel("CLR"); ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- CLI & main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct-all", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ppi-effects-all", dest="ppi_effects_all", default=None,
                    help="Path to results/ppi_interactions/<rank>/ppi_effects_all.csv")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--rank", required=True, choices=["genus","species"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--colors", default="config/colors.yml",
                    help="YAML file with group colors (default: config/colors.yml)")
    args = ap.parse_args()

    # load colors
    group_colors, site_base, ppi_palette_fn, line_colors, qual_palette = load_colors(args.colors)

    out_rank = os.path.join(args.outdir, args.rank); ensure_dir(out_rank)

    pct = pd.read_csv(args.pct_all, index_col=0)
    pct.columns = pct.columns.astype(str)
    clr = clr_transform(pct, 1e-6)     # taxa × samples
    clr_T = clr.T                      # samples × taxa
    clr_T.index.name = "Sample_ID"

    meta_raw = pd.read_csv(args.meta)
    meta = norm_meta(meta_raw)
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta = meta[meta["sample_id"].isin(clr.columns.astype(str))].copy()

    long = clr_T.stack().reset_index()
    long.columns = ["Sample_ID","taxon","CLR"]
    long = long.merge(meta.rename(columns={"sample_id":"Sample_ID"}), on="Sample_ID", how="left")

    # top taxa انتخاب
    var_series = clr_T.var(axis=0)
    top_taxa = pick_top_taxa(args.ppi_effects_all, n=args.topk, fallback_by_var=var_series)
    if not top_taxa:
        top_taxa = var_series.sort_values(ascending=False).head(args.topk).index.tolist()
    pd.DataFrame({"taxon": top_taxa}).to_csv(os.path.join(out_rank,"top_taxa_selected.csv"), index=False)

    # plots with YAML colors
    box_by_ppi(long, top_taxa, "Oral",  os.path.join(out_rank,"box_by_ppi_oral.png"),  ppi_palette_fn)
    box_by_ppi(long, top_taxa, "Fecal", os.path.join(out_rank,"box_by_ppi_fecal.png"), ppi_palette_fn)
    interaction_means(long, top_taxa,   os.path.join(out_rank,"interaction_means.png"), line_colors)
    spaghetti_crohn(long, top_taxa, args.pairs, os.path.join(out_rank,"paired_spaghetti_crohn.png"))
    kde_by_site(long, top_taxa, "Oral",  os.path.join(out_rank, "density_top_taxa_oral.png"),  qual_palette)
    kde_by_site(long, top_taxa, "Fecal", os.path.join(out_rank, "density_top_taxa_fecal.png"), qual_palette)

    print("[INFO] QC targeted saved →", out_rank)

if __name__ == "__main__":
    main()
