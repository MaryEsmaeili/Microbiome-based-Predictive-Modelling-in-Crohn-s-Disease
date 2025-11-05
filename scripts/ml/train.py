# ml/train.py
import os
import argparse
import pandas as pd

from cv_utils import (
    nested_cv_evaluate,
    run_permutation_test,
    save_learning_curve,
    get_model_and_grid,
    plot_roc,
    plot_pr,
    plot_calibration,
)
from data import load_X_y


# -------------------------------
# Helpers
# -------------------------------

def _intersect_features(trainX: pd.DataFrame, testX: pd.DataFrame):
    """
    Align train and test matrices to the intersection of columns (features).
    Returns aligned train/test plus lists of kept/dropped features.
    """
    keep = trainX.columns.intersection(testX.columns)
    drop_train = [c for c in trainX.columns if c not in keep]
    drop_test = [c for c in testX.columns if c not in keep]
    return trainX[keep].copy(), testX[keep].copy(), keep, drop_train, drop_test


def _write_features_report(outdir: str, keep, drop_train, drop_test):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "features_used.txt"), "w") as f:
        f.write("# Kept features (intersection)\n")
        for c in keep:
            f.write(str(c) + "\n")
        f.write("\n# Dropped in TRAIN (not in TEST)\n")
        for c in drop_train:
            f.write(str(c) + "\n")
        f.write("\n# Dropped in TEST (not in TRAIN)\n")
        for c in drop_test:
            f.write(str(c) + "\n")


# -------------------------------
# Main modes
# -------------------------------

def mode_within(args):
    """
    Nested CV داخل یک سایت (oral یا fecal).
    """
    # Build X,y for a single site
    add_covars = [
        c.strip() for c in args.add_covars.split(",") if c.strip()
    ] if args.add_covars else []

    X, y, meta, groups = load_X_y(
        taxa_csv=args.taxa,
        meta_csv=args.meta,
        site=args.site,
        target=args.target,
        add_covars=add_covars,
        rank=args.rank,
        subject_id_col=args.subject_id_col,
        sample_id_col=args.sample_id_col,
        site_col=args.site_col,
        disease_col=args.disease_col,
        ppi_col=args.ppi_col,
        responder_col=args.responder_col
    )

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    res = nested_cv_evaluate(
        X=X,
        y=y,
        model_name=args.model,
        outdir=outdir,
        groups=groups,
        n_splits_outer=args.outer,
        n_splits_inner=args.inner
    )

    # Optional permutation test
    if args.permutation > 0:
        run_permutation_test(
            X, y, args.model, outdir, n_perm=args.permutation
        )

    # Learning curve
    if args.learningcurve:
        save_learning_curve(
            X, y, args.model,
            out_png=os.path.join(outdir, "learning_curve.png"),
            out_csv=os.path.join(outdir, "learning_curve.csv"),
            cv_splits=args.outer
        )


def mode_transfer(args):
    """
    Train on one site, evaluate on the other (external test).
    Optionally restrict to paired subjects only.
    """
    add_covars = [
        c.strip() for c in args.add_covars.split(",") if c.strip()
    ] if args.add_covars else []

    # Train site
    Xtr, ytr, metatr, grouptr = load_X_y(
        taxa_csv=args.taxa,
        meta_csv=args.meta,
        site=args.train_site,
        target=args.target,
        add_covars=add_covars,
        rank=args.rank,
        subject_id_col=args.subject_id_col,
        sample_id_col=args.sample_id_col,
        site_col=args.site_col,
        disease_col=args.disease_col,
        ppi_col=args.ppi_col,
        responder_col=args.responder_col
    )
    # Test site
    Xte, yte, metate, groupte = load_X_y(
        taxa_csv=args.taxa,
        meta_csv=args.meta,
        site=args.test_site,
        target=args.target,
        add_covars=add_covars,
        rank=args.rank,
        subject_id_col=args.subject_id_col,
        sample_id_col=args.sample_id_col,
        site_col=args.site_col,
        disease_col=args.disease_col,
        ppi_col=args.ppi_col,
        responder_col=args.responder_col
    )

    # If paired only: keep subjects that appear in both sites
    if args.paired:
        if "subject_id" not in metatr.columns or "subject_id" not in metate.columns:
            raise ValueError(
                "Paired mode requires 'subject_id' column in metadata."
            )
        common_subj = set(metatr["subject_id"]).intersection(
            set(metate["subject_id"])
        )
        keep_tr_idx = metatr.loc[
            metatr["subject_id"].isin(common_subj)
        ].index
        keep_te_idx = metate.loc[
            metate["subject_id"].isin(common_subj)
        ].index

        Xtr = Xtr.loc[keep_tr_idx]
        ytr = ytr.loc[keep_tr_idx]
        metatr = metatr.loc[keep_tr_idx]

        Xte = Xte.loc[keep_te_idx]
        yte = yte.loc[keep_te_idx]
        metate = metate.loc[keep_te_idx]

    # Align features
    Xtr2, Xte2, keep, drop_train, drop_test = _intersect_features(Xtr, Xte)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    _write_features_report(outdir, keep, drop_train, drop_test)

    # Nested CV summary on train-site (for reporting)
    nested_cv_evaluate(
        Xtr2, ytr, args.model, os.path.join(outdir, "trainCV")
    )

    # Final refit on full train with inner CV to choose hyperparams, then evaluate on test
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    model, grid = get_model_and_grid(args.model)
    inner = StratifiedKFold(
        n_splits=args.inner,
        shuffle=True,
        random_state=42
    )
    gscv = GridSearchCV(
        model, grid, scoring="roc_auc", cv=inner, n_jobs=-1
    )
    gscv.fit(Xtr2, ytr)
    best = gscv.best_estimator_

    # Probabilities on the external test site
    if hasattr(best, "predict_proba"):
        prob = best.predict_proba(Xte2)[:, 1]
    else:
        prob = best.decision_function(Xte2)
        pmin, pmax = prob.min(), prob.max()
        if pmax > pmin:
            prob = (prob - pmin) / (pmax - pmin)

    # Save per-sample predictions (subject-aware if available)
    pred_df = pd.DataFrame({
        "sample_id": Xte2.index,
        "y_true": yte.loc[Xte2.index].values,
        "y_prob": prob
    })
    if "subject_id" in metate.columns:
        pred_df["subject_id"] = metate.loc[
            Xte2.index, "subject_id"
        ].values
    pred_df.to_csv(
        os.path.join(outdir, "external_predictions.csv"), index=False
    )

    title = f"External test ({args.train_site}→{args.test_site}) — model={args.model}"
    subtitle = (
        f"test n={len(pred_df)} | pos={int(pred_df.y_true.sum())} "
        f"({pred_df.y_true.mean():.2%})"
    )

    plot_roc(
        pred_df.y_true, pred_df.y_prob,
        os.path.join(outdir, "external_roc.png"),
        title, subtitle
    )
    plot_pr(
        pred_df.y_true, pred_df.y_prob,
        os.path.join(outdir, "external_pr.png"),
        title, subtitle
    )
    plot_calibration(
        pred_df.y_true, pred_df.y_prob,
        os.path.join(outdir, "external_calibration.png"),
        title, subtitle
    )


# -------------------------------
# CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Microbiome ML runner (within / transfer)."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # Common args
    def add_common_args(p):
        p.add_argument("--taxa", required=True)
        p.add_argument("--meta", required=True)
        p.add_argument("--rank", default="species")
        p.add_argument(
            "--target",
            required=True,
            choices=["disease", "ppi", "responder"]
        )
        p.add_argument(
            "--model",
            default="logit",
            choices=["logit", "svm", "rf", "xgb"]
        )
        p.add_argument(
            "--add-covars",
            default="",
            help="Comma-separated covariate names (leave empty to use microbiome only)."
        )
        p.add_argument("--outdir", required=True)
        p.add_argument("--outer", type=int, default=5)
        p.add_argument("--inner", type=int, default=4)
        # column names
        p.add_argument("--sample-id-col", default="sample_id")
        p.add_argument("--subject-id-col", default="subject_id")
        p.add_argument("--site-col",     default="site")
        p.add_argument("--disease-col",  default="disease")
        p.add_argument("--ppi-col",      default="PPI_use")
        p.add_argument("--responder-col", default="responder")

    # within-site mode
    p_within = sub.add_parser(
        "within", help="Nested CV within one site (oral or fecal)."
    )
    add_common_args(p_within)
    p_within.add_argument(
        "--site",
        required=True,
        choices=["oral", "fecal"]
    )
    p_within.add_argument(
        "--learningcurve",
        action="store_true",
        help="Also compute learning curve."
    )
    p_within.add_argument(
        "--permutation",
        type=int,
        default=0,
        help="Number of label permutations for null AUC distribution (0 = skip)."
    )

    # transfer mode
    p_transfer = sub.add_parser(
        "transfer",
        help="Train on one site and evaluate on the other (oral ↔ fecal)."
    )
    add_common_args(p_transfer)
    p_transfer.add_argument(
        "--train-site",
        required=True,
        choices=["oral", "fecal"]
    )
    p_transfer.add_argument(
        "--test-site",
        required=True,
        choices=["oral", "fecal"]
    )
    p_transfer.add_argument(
        "--paired",
        action="store_true",
        help="Use only subjects present in both sites (paired only)."
    )

    args = parser.parse_args()

    if args.mode == "within":
        mode_within(args)
    elif args.mode == "transfer":
        mode_transfer(args)
    else:
        raise ValueError("Unsupported mode.")


if __name__ == "__main__":
    main()
