# ml/summary_ml_results_fixed.py
# ------------------------------------------------------------
# Combined ML summary script:
# - ROC curves (combined across models)
# - Precision-Recall curves (combined)
# - Calibration curves (combined)
# - Learning curves (combined, if available)
# - Summary tables (AUC, AP, Brier score)
#
# Works for:
#   (1) Within-site models (oral/fecal)
#   (2) Transfer learning (oral→fecal, fecal→oral)
#
# Input file structure (Snakemake Results):
#   results/ml/within/<site>/<target>/<model>/<rank>/cv_predictions.csv
#   results/ml/within/.../learning_curve.csv         (optional)
#   results/ml/transfer/<task>/<model>/<rank>/trainCV/cv_predictions.csv
#   results/ml/transfer/<task>/<model>/<rank>/trainCV/learning_curve.csv
#   results/ml/transfer/<task>/<model>/<rank>/external_predictions.csv
#
# Output:
#   ml_summary/<task>/<tag>_ROC_<rank>.png
#   ml_summary/<task>/<tag>_PR_<rank>.png
#   ml_summary/<task>/<tag>_calibration_<rank>.png
#   ml_summary/<task>/<tag>_learning_<rank>.png
#   ml_summary/<task>/<tag>_summary_<rank>.csv
#
# ------------------------------------------------------------

from pathlib import Path
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

# ------------------------------------------------------------
# Fixed settings
# ------------------------------------------------------------

MODELS = ["logit", "rf", "svm", "xgb"]

COLORS = {
    "logit": "#1f77b4",
    "rf":    "#ff7f0e",
    "svm":   "#2ca02c",
    "xgb":   "#d62728",
}

ROOT_ML = Path("results/ml")
OUT_ROOT = Path("ml_summary")


def _ensure_dir(p: Path):
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# COMBINED ROC + PR + Calibration
# ------------------------------------------------------------

def plot_combined_predictions(pred_paths, title_prefix, outdir, tag, rank):
    """
    Create combined ROC, PR, and calibration curves from per-model prediction CSVs.

    Each CSV must have columns:
        y_true, y_prob
    """

    if not pred_paths:
        return

    _ensure_dir(outdir)

    # Figures
    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 6))
    fig_cal, ax_cal = plt.subplots(figsize=(6, 6))

    rows = []

    for model, csv_path in pred_paths.items():
        df = pd.read_csv(csv_path)

        # Must contain required columns
        if "y_true" not in df.columns or "y_prob" not in df.columns:
            continue

        y_true = df["y_true"].values
        y_prob = df["y_prob"].values

        # --- ROC ---
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)

        # --- PR ---
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)

        # --- Calibration ---
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="quantile"
        )
        brier = brier_score_loss(y_true, y_prob)

        color = COLORS.get(model)

        # ROC
        ax_roc.plot(fpr, tpr, label=f"{model} (AUC={auc_val:.2f})", lw=2, color=color)

        # PR
        ax_pr.plot(rec, prec, label=f"{model} (AP={ap_val:.2f})", lw=2, color=color)

        # Calibration
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
                "pos_frac": y_true.mean(),
                "auc": auc_val,
                "ap": ap_val,
                "brier": brier,
            }
        )

    if not rows:
        plt.close(fig_roc)
        plt.close(fig_pr)
        plt.close(fig_cal)
        return

    # ----- ROC final touches -----
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{title_prefix} — ROC ({rank})")
    ax_roc.grid(alpha=0.25)
    ax_roc.legend()
    fig_roc.tight_layout()
    fig_roc.savefig(outdir / f"{tag}_ROC_{rank}.png", dpi=200)
    plt.close(fig_roc)

    # ----- PR -----
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"{title_prefix} — PR ({rank})")
    ax_pr.grid(alpha=0.25)
    ax_pr.legend()
    fig_pr.tight_layout()
    fig_pr.savefig(outdir / f"{tag}_PR_{rank}.png", dpi=200)
    plt.close(fig_pr)

    # ----- Calibration -----
    ax_cal.plot([0, 1], [0, 1], "k--", lw=1)
    ax_cal.set_xlabel("Predicted Probability")
    ax_cal.set_ylabel("Observed Frequency")
    ax_cal.set_title(f"{title_prefix} — Calibration ({rank})")
    ax_cal.grid(alpha=0.25)
    ax_cal.legend()
    fig_cal.tight_layout()
    fig_cal.savefig(outdir / f"{tag}_calibration_{rank}.png", dpi=200)
    plt.close(fig_cal)

    # ----- Summary table -----
    pd.DataFrame(rows).to_csv(outdir / f"{tag}_summary_{rank}.csv", index=False)


# ------------------------------------------------------------
# COMBINED Learning Curves
# ------------------------------------------------------------

def plot_combined_learning_curves(lc_paths, title_prefix, outdir, tag, rank):
    """
    If per-model learning_curve.csv exists, combine them into a single plot.

    Expected columns:
        train_size, cv_auc_mean (required)
        cv_auc_std (optional)
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

        color = COLORS.get(model)

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
                "max_train_size": df["train_size"].max(),
                "final_cv_auc": df["cv_auc_mean"].iloc[-1],
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


# ------------------------------------------------------------
# Summaries: WITHIN-site
# ------------------------------------------------------------

def summarize_within():
    """Summaries for within-oral and within-fecal tasks."""
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
                        pred_paths,
                        title_prefix=title,
                        outdir=outdir,
                        tag="cv",
                        rank=rank,
                    )

                # Combined Learning curves
                if lc_paths:
                    plot_combined_learning_curves(
                        lc_paths,
                        title_prefix=title,
                        outdir=outdir,
                        tag="cv",
                        rank=rank,
                    )


# ------------------------------------------------------------
# Summaries: TRANSFER learning
# ------------------------------------------------------------

def summarize_transfer():
    """Summaries for oral→fecal and fecal→oral transfer tasks."""
    for task in ["fecal_to_oral", "oral_to_fecal"]:
        for rank in ["species", "genus"]:

            outdir = OUT_ROOT / f"transfer_{task}"

            # ---- External predictions ----
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
                    pred_ext,
                    title_prefix=title,
                    outdir=outdir,
                    tag="external",
                    rank=rank,
                )

            # ---- Train-site Nested CV ----
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
                    pred_cv,
                    title_prefix=title,
                    outdir=outdir,
                    tag="cv",
                    rank=rank,
                )

            if lc_paths:
                title = f"Learning curves — transfer {task}"
                plot_combined_learning_curves(
                    lc_paths,
                    title_prefix=title,
                    outdir=outdir,
                    tag="cv",
                    rank=rank,
                )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    summarize_within()
    summarize_transfer()
    print("Finished ML summary generation.")


if __name__ == "__main__":
    main()