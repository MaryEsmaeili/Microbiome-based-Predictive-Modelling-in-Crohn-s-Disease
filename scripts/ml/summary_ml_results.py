# scripts/ml/summary_ml_results.py

"""
Summaries for microbiome ML experiments.

This module aggregates and visualizes results from the per-model runs under
`results/ml` and writes combined figures/tables under `results/ml/ml_summary`.

It performs three main tasks:

1) Within-site summaries (nested CV)
   - For each combination of:
       * site   ∈ {"fecal", "oral"}
       * target ∈ {"disease", "ppi", "responder"}
       * rank   ∈ {"species", "genus"}
     it looks for per-model outputs at:
       results/ml/within/{site}/{target}/{model}/{rank}/
     and, if available, combines:
       - cv_predictions.csv  → multi-model ROC, PR, calibration plots
                               and a summary CSV (AUC, AP, Brier, etc.).
       - learning_curve.csv  → multi-model learning curve plot
                               and a small summary CSV.

2) Transfer summaries (cross-site)
   - For each task in {"fecal_to_oral", "oral_to_fecal"} and rank in
     {"species", "genus"} it reads:
       - External test predictions:
           results/ml/transfer/{task}/{model}/{rank}/external_predictions.csv
         → combined ROC/PR/Calibration for external test.
       - Train-site nested CV outputs:
           results/ml/transfer/{task}/{model}/{rank}/trainCV/cv_predictions.csv
           results/ml/transfer/{task}/{model}/{rank}/trainCV/learning_curve.csv
         → combined ROC/PR/Calibration and learning curves for train-site CV.

3) Consensus feature importance (across models)
   - Scans all `importances.csv` files under `results/ml` for any task where
     feature importances have been saved (e.g. within-site and transfer trainCV).
   - For each logical task/group key (e.g. "within_fecal_disease_species",
     "transfer_fecal_to_oral_species") it:
       * merges importances from all models (logit, rf, svm, xgb),
       * keeps features whose absolute weight ≥ ABS_WEIGHT_THRESHOLD
         in at least MIN_MODELS models,
       * sorts by total absolute weight across models,
       * writes the filtered table to:
           results/ml/ml_summary/consensus_threshold/{gkey}/consensus_filtered.csv
       * creates a grouped bar plot (taxa × models) with short taxon labels
         for the top TOP_N features:
           results/ml/ml_summary/consensus_threshold/{gkey}/consensus_plot.png

This script is designed to be executable without arguments, so that a Snakemake
rule can simply run:

    python scripts/ml/summary_ml_results.py

and receive all summary outputs in `results/ml/ml_summary/`.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

# Global settings
MODELS = ["logit", "rf", "svm", "xgb"]

# Color palette for models (used across plots)
MODEL_COLORS = {
    "logit": "#1f77b4",
    "rf":    "#ff7f0e",
    "svm":   "#2ca02c",
    "xgb":   "#d62728",
}

ROOT_ML = Path("results/ml")
OUT_ROOT = ROOT_ML / "ml_summary"

# Consensus-feature settings
CONSENSUS_OUT_ROOT = OUT_ROOT / "consensus_threshold"
ABS_WEIGHT_THRESHOLD = 0.02   # absolute weight threshold used for selection
MIN_MODELS = 4                # minimum number of models above threshold
TOP_N = 10                    # maximum number of taxa in each consensus plot

def _ensure_dir(p: Path) -> None:
    """Ensure that a directory exists."""
    p.mkdir(parents=True, exist_ok=True)

# COMBINED ROC + PR + Calibration
def plot_combined_predictions(pred_paths, title_prefix, outdir, tag, rank):
    """
    Create combined ROC, PR, and calibration curves from per-model prediction CSVs.

    Each CSV must have at least:
        y_true, y_prob
    """
    if not pred_paths:
        return

    _ensure_dir(outdir)

    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 6))
    fig_cal, ax_cal = plt.subplots(figsize=(6, 6))

    rows = []

    for model, csv_path in pred_paths.items():
        df = pd.read_csv(csv_path)

        # Required columns
        if "y_true" not in df.columns or "y_prob" not in df.columns:
            continue

        y_true = df["y_true"].values
        y_prob = df["y_prob"].values

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)

        # Calibration
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="quantile"
        )
        brier = brier_score_loss(y_true, y_prob)

        color = MODEL_COLORS.get(model)

        # ROC curve
        ax_roc.plot(
            fpr,
            tpr,
            label=f"{model} (AUC={auc_val:.2f})",
            lw=2,
            color=color,
        )

        # PR curve
        ax_pr.plot(
            rec,
            prec,
            label=f"{model} (AP={ap_val:.2f})",
            lw=2,
            color=color,
        )

        # Calibration curve
        ax_cal.plot(
            prob_pred,
            prob_true,
            marker="o",
            lw=1.5,
            label=f"{model} (Brier={brier:.2f})",
            color=color,
        )

        rows.append(
            {
                "model": model,
                "n": len(y_true),
                "positive": int(y_true.sum()),
                "pos_frac": float(y_true.mean()),
                "auc": float(auc_val),
                "ap": float(ap_val),
                "brier": float(brier),
            }
        )

    if not rows:
        plt.close(fig_roc)
        plt.close(fig_pr)
        plt.close(fig_cal)
        return

    # ROC: final formatting
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{title_prefix} — ROC ({rank})")
    ax_roc.grid(alpha=0.25)
    ax_roc.legend()
    fig_roc.tight_layout()
    fig_roc.savefig(outdir / f"{tag}_ROC_{rank}.png", dpi=200)
    plt.close(fig_roc)

    # PR: final formatting
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"{title_prefix} — PR ({rank})")
    ax_pr.grid(alpha=0.25)
    ax_pr.legend()
    fig_pr.tight_layout()
    fig_pr.savefig(outdir / f"{tag}_PR_{rank}.png", dpi=200)
    plt.close(fig_pr)

    # Calibration: final formatting
    ax_cal.plot([0, 1], [0, 1], "k--", lw=1)
    ax_cal.set_xlabel("Predicted Probability")
    ax_cal.set_ylabel("Observed Frequency")
    ax_cal.set_title(f"{title_prefix} — Calibration ({rank})")
    ax_cal.grid(alpha=0.25)
    ax_cal.legend()
    fig_cal.tight_layout()
    fig_cal.savefig(outdir / f"{tag}_calibration_{rank}.png", dpi=200)
    plt.close(fig_cal)

    # Summary table
    pd.DataFrame(rows).to_csv(outdir / f"{tag}_summary_{rank}.csv", index=False)

# COMBINED Learning Curves
def plot_combined_learning_curves(lc_paths, title_prefix, outdir, tag, rank):
    """
    Combine learning curves from per-model learning_curve.csv files.

    Expected columns in each CSV:
        train_size, cv_auc_mean  (required)
        cv_auc_std               (optional)
    """
    if not lc_paths:
        return

    _ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(6, 6))
    rows = []

    for model, csv_path in lc_paths.items():
        df = pd.read_csv(csv_path)

        if "train_size" not in df.columns or "cv_auc_mean" not in df.columns:
            continue

        color = MODEL_COLORS.get(model)

        ax.plot(
            df["train_size"],
            df["cv_auc_mean"],
            marker="o",
            lw=2,
            color=color,
            label=f"{model}",
        )

        rows.append(
            {
                "model": model,
                "max_train_size": float(df["train_size"].max()),
                "final_cv_auc": float(df["cv_auc_mean"].iloc[-1]),
            }
        )

    if not rows:
        plt.close(fig)
        return

    ax.set_xlabel("Training Samples")
    ax.set_ylabel("CV ROC-AUC")
    ax.set_title(f"{title_prefix} — Learning Curve ({rank})")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_learning_{rank}.png", dpi=200)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(outdir / f"{tag}_learning_summary_{rank}.csv", index=False)

# Summaries: WITHIN-site
def summarize_within():
    """Build summaries for within-oral and within-fecal nested CV tasks."""
    for site in ["fecal", "oral"]:
        for target in ["disease", "ppi", "responder"]:
            for rank in ["species", "genus"]:

                pred_paths = {}
                lc_paths = {}

                for model in MODELS:
                    base = ROOT_ML / "within" / site / target / model / rank

                    cv_file = base / "cv_predictions.csv"
                    if cv_file.exists():
                        pred_paths[model] = cv_file

                    lc_file = base / "learning_curve.csv"
                    if lc_file.exists():
                        lc_paths[model] = lc_file

                if not pred_paths and not lc_paths:
                    continue

                outdir = OUT_ROOT / f"within_{site}_{target}"
                title = f"Nested CV — within {site} ({target})"

                # Combined ROC/PR/Calibration
                if pred_paths:
                    plot_combined_predictions(
                        pred_paths=pred_paths,
                        title_prefix=title,
                        outdir=outdir,
                        tag="cv",
                        rank=rank,
                    )

                # Combined learning curves
                if lc_paths:
                    plot_combined_learning_curves(
                        lc_paths=lc_paths,
                        title_prefix=title,
                        outdir=outdir,
                        tag="cv",
                        rank=rank,
                    )

# Summaries: TRANSFER learning
def summarize_transfer():
    """Build summaries for oral→fecal and fecal→oral transfer tasks."""
    for task in ["fecal_to_oral", "oral_to_fecal"]:
        for rank in ["species", "genus"]:

            outdir = OUT_ROOT / f"transfer_{task}"

            # ----- External predictions -----
            pred_ext = {}
            for model in MODELS:
                file_ext = (
                    ROOT_ML
                    / "transfer"
                    / task
                    / model
                    / rank
                    / "external_predictions.csv"
                )
                if file_ext.exists():
                    pred_ext[model] = file_ext

            if pred_ext:
                title = f"External test — transfer {task}"
                plot_combined_predictions(
                    pred_paths=pred_ext,
                    title_prefix=title,
                    outdir=outdir,
                    tag="external",
                    rank=rank,
                )

            # ----- Train-site nested CV -----
            pred_cv = {}
            lc_paths = {}
            for model in MODELS:
                base = (
                    ROOT_ML
                    / "transfer"
                    / task
                    / model
                    / rank
                    / "trainCV"
                )

                file_cv = base / "cv_predictions.csv"
                if file_cv.exists():
                    pred_cv[model] = file_cv

                file_lc = base / "learning_curve.csv"
                if file_lc.exists():
                    lc_paths[model] = file_lc

            if pred_cv:
                title = f"Nested CV — transfer {task}"
                plot_combined_predictions(
                    pred_paths=pred_cv,
                    title_prefix=title,
                    outdir=outdir,
                    tag="cv",
                    rank=rank,
                )

            if lc_paths:
                title = f"Learning curves — transfer {task}"
                plot_combined_learning_curves(
                    lc_paths=lc_paths,
                    title_prefix=title,
                    outdir=outdir,
                    tag="cv",
                    rank=rank,
                )

# Consensus feature importance
def _load_importances(csv_path: Path, model: str) -> pd.DataFrame:
    """
    Load a single importances.csv and normalize column names to:
        feature, weight, model
    """
    df = pd.read_csv(csv_path)
    fcol = "feature" if "feature" in df.columns else df.columns[0]
    wcol = "weight" if "weight" in df.columns else df.columns[1]

    out = df[[fcol, wcol]].copy()
    out.columns = ["feature", "weight"]
    out["model"] = model
    return out

def _build_group_key(path: Path) -> tuple[str, str]:
    """
    Build a logical group key and detect the model name from the path.

    Example:
      results/ml/within/fecal/disease/logit/species/importances.csv
      → group key: "within_fecal_disease_species", model: "logit"

      results/ml/transfer/fecal_to_oral/logit/species/trainCV/importances.csv
      → group key: "transfer_fecal_to_oral_species", model: "logit"
    """
    rel = path.relative_to(ROOT_ML)
    parts = rel.parts

    model_idx = None
    for i, p in enumerate(parts):
        if p in MODELS:
            model_idx = i
            break

    if model_idx is None:
        raise ValueError(f"Cannot detect model name from path: {path}")

    model = parts[model_idx]
    rank = parts[model_idx + 1] if (model_idx + 1) < len(parts) else "unknown"

    group_parts = parts[:model_idx] + (rank,)
    group_key = "_".join(group_parts)
    return group_key, model

def _pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format importances into feature × model wide format."""
    wide = df.pivot(index="feature", columns="model", values="weight")
    return wide.reindex(columns=MODELS)

def _filter_features(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Apply consensus filtering:
      - Keep features with abs(weight) ≥ ABS_WEIGHT_THRESHOLD in at least
        MIN_MODELS models.
      - Compute:
          * models_support = number of models above threshold
          * total_abs      = sum of absolute weights across models
      - Sort by total_abs descending.
    """
    abs_w = wide.abs()
    important_mask = abs_w >= ABS_WEIGHT_THRESHOLD
    support = important_mask.sum(axis=1)

    keep = support >= MIN_MODELS
    selected = wide.loc[keep].copy()

    selected["models_support"] = support[keep]
    selected["total_abs"] = abs_w.loc[keep].sum(axis=1)

    selected = selected.sort_values("total_abs", ascending=False)
    return selected

def _extract_short_taxon(taxon: str) -> str:
    """
    Extract a clean, short label for plotting:
      - If species-level (contains '|s__'): return only the species part.
      - If genus-level (contains 'g__'): return only the genus part.
      - Otherwise: return the original string.
    """
    if "|s__" in taxon:
        return taxon.split("|s__", 1)[1]
    if "g__" in taxon:
        return taxon.split("g__", 1)[1]
    return taxon

def _plot_consensus_group(df_wide: pd.DataFrame, task_name: str, outpath: Path) -> None:
    """
    Create a grouped bar plot for a single consensus group.

    - X-axis: short taxon labels
    - Each taxon: one cluster of bars, one per model
    - Y-axis: model weights
    """
    df = df_wide.head(TOP_N)
    taxa_full = df.index.tolist()
    taxa_short = [_extract_short_taxon(t) for t in taxa_full]

    n_taxa = len(taxa_full)
    n_models = len(MODELS)

    bar_width = 0.14
    cluster_gap = 0.30
    cluster_span = n_models * bar_width

    centers = np.arange(n_taxa) * (cluster_span + cluster_gap)
    fig_width = max(10.0, 0.8 * n_taxa)

    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for i, m in enumerate(MODELS):
        offset = (i - (n_models - 1) / 2.0) * bar_width
        values = df[m].values
        ax.bar(
            centers + offset,
            values,
            bar_width,
            label=m,
            color=MODEL_COLORS.get(m),
        )

    ax.set_xticks(centers)
    ax.set_xticklabels(taxa_short, rotation=45, ha="right")

    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_ylabel("Model weight")
    ax.set_title(f"Consensus feature importances — {task_name}")
    ax.legend(title="Models", ncol=len(MODELS))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.subplots_adjust(bottom=0.35)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def summarize_consensus():
    """
    Build consensus feature importance plots and tables across models.

    The function scans all `importances.csv` under `results/ml`, groups them by
    logical task (group key), merges weights across models, applies consensus
    filtering, and writes outputs under `results/ml/ml_summary/consensus_threshold/`.
    """
    _ensure_dir(CONSENSUS_OUT_ROOT)

    csv_files = list(ROOT_ML.rglob("importances.csv"))
    if not csv_files:
        return

    grouped: dict[str, list[tuple[Path, str]]] = {}

    for p in csv_files:
        try:
            gkey, model = _build_group_key(p)
        except Exception:
            # Ignore files whose path does not match the expected pattern
            continue

        grouped.setdefault(gkey, []).append((p, model))

    for gkey, items in grouped.items():
        df_list = []
        for path, model in items:
            try:
                df_list.append(_load_importances(path, model))
            except Exception:
                # If a given importances file is malformed, skip it
                continue

        if not df_list:
            continue

        df_all = pd.concat(df_list, ignore_index=True)
        wide = _pivot_wide(df_all)

        selected = _filter_features(wide)

        out_dir = CONSENSUS_OUT_ROOT / gkey
        _ensure_dir(out_dir)

        selected.to_csv(out_dir / "consensus_filtered.csv")

        if selected.empty:
            print(f"[CONSENSUS] {gkey}: no features passed the threshold.")
            continue

        fig_path = out_dir / "consensus_plot.png"
        _plot_consensus_group(
            df_wide=selected[MODELS],
            task_name=gkey,
            outpath=fig_path,
        )

        print(f"[CONSENSUS] {gkey}: {len(selected)} features → plot saved.")

# MAIN
def main():
    summarize_within()
    summarize_transfer()
    summarize_consensus()
    print("Finished ML summary generation (within, transfer, consensus).")


if __name__ == "__main__":
    main()
