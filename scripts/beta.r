#!/usr/bin/env Rscript
# Beta diversity with clinical integration (PCoA + PERMANOVA + dispersion)
# Supports:
#   - Tidy metadata (rows=samples, cols=vars, has Sample_ID)
#   - Excel "matrix" metadata (variables in col A, samples across columns) via --excel_matrix
#   - Optional ID mapping via --map_csv (columns: feature_id,Sample_ID)

ensure_pkg <- function(p){
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
  suppressPackageStartupMessages(library(p, character.only = TRUE))
}
invisible(lapply(c("optparse","readr","readxl","dplyr","tibble","tidyr","vegan","ggplot2"), ensure_pkg))

# ---------------- CLI ----------------
option_list <- list(
  optparse::make_option("--features", type="character", help="Feature CSV: rows=features, cols=samples; first col=feature IDs"),
  optparse::make_option("--metadata", type="character", help="Metadata CSV/XLSX"),
  optparse::make_option("--assume_relative", action="store_true", default=FALSE, help="Features already relative"),
  optparse::make_option("--outdir", type="character", default="results/beta", help="Output dir [default: %default]"),

  # Optional mapping file to rename feature columns to Sample_ID
  optparse::make_option("--map_csv", type="character", default=NULL,
                        help="CSV with columns: feature_id,Sample_ID to map feature columns to metadata Sample_ID"),

  # Tidy-mode options
  optparse::make_option("--sample_col", type="character", default="Sample_ID", help="Sample ID column (tidy mode)"),
  optparse::make_option("--group_col",  type="character", default="responder_study", help="Grouping column (tidy mode)"),
  optparse::make_option("--age_col",    type="character", default="Age", help="Age column (tidy mode)"),
  optparse::make_option("--bmi_col",    type="character", default="BMI", help="BMI column (tidy mode)"),

  # Excel-matrix mode
  optparse::make_option("--excel_matrix", action="store_true", default=FALSE,
                        help="Metadata is an Excel-style matrix (variables in col A, samples across columns)"),
  optparse::make_option("--id_row",   type="character", default="Oral_sample_ID",
                        help="Row name containing sample IDs (excel_matrix mode) [default: %default]"),
  optparse::make_option("--group_row",type="character", default="Responder_study",
                        help="Row name for grouping (excel_matrix mode) [default: %default]"),
  optparse::make_option("--age_row",  type="character", default="V1_AgeAtFecalSampling",
                        help="Row name for Age (excel_matrix mode) [default: %default]"),
  optparse::make_option("--bmi_row",  type="character", default="V1_BMI",
                        help="Row name for BMI (excel_matrix mode) [default: %default]")
)
opt <- optparse::parse_args(optparse::OptionParser(option_list=option_list))
dir.create(opt$outdir, showWarnings = FALSE, recursive = TRUE)

# ------------- features -------------
ft <- readr::read_csv(opt$features, show_col_types = FALSE)
if (ncol(ft) < 2) stop("Feature table must have >=2 columns (feature id + >=1 sample).")
ft <- tibble::column_to_rownames(ft, var = colnames(ft)[1])
# keep raw feature column names for potential mapping
feat_cols_raw <- colnames(ft)

# ---------- helpers ----------
norm_names <- function(x){
  x <- gsub("\\s+", "_", x)
  x <- gsub("[.]", "_", x)
  x <- gsub("[^A-Za-z0-9_]", "", x)
  trimws(x)
}
to_numeric <- function(x){
  suppressWarnings(as.numeric(gsub(",", ".", gsub("[^0-9.+-]", "", as.character(x)))))
}
to_rel <- function(m){
  cs <- colSums(m); cs[cs==0] <- 1
  sweep(m, 2, cs, "/")
}

# ---------- optional mapping ----------
if (!is.null(opt$map_csv)) {
  map <- readr::read_csv(opt$map_csv, show_col_types = FALSE)
  req <- c("feature_id","Sample_ID")
  if (!all(req %in% names(map))) stop("map_csv must have columns: feature_id,Sample_ID")
  map$feature_id <- as.character(map$feature_id)
  map$Sample_ID  <- as.character(map$Sample_ID)
  ren <- setNames(map$Sample_ID, map$feature_id)
  colnames(ft) <- ifelse(colnames(ft) %in% names(ren), ren[colnames(ft)], colnames(ft))
}

# ---------- load metadata ----------
is_excel <- grepl("\\.xlsx?$", opt$metadata, ignore.case = TRUE)
if (opt$excel_matrix) {
  meta_raw <- if (is_excel) readxl::read_excel(opt$metadata, col_names = FALSE) else
                           readr::read_csv(opt$metadata, col_names = FALSE, show_col_types = FALSE)
  vars <- as.character(meta_raw[[1]])
  data <- as.data.frame(meta_raw[ , -1, drop=FALSE], check.names = FALSE)

  id_idx <- which(vars == opt$id_row)
  if (length(id_idx) != 1) stop(paste0("Could not find a unique '", opt$id_row, "' row in metadata. Found: ", length(id_idx)))
  sample_ids <- as.character(unlist(data[id_idx, , drop=TRUE]))

  pick_row <- function(row_name, candidates = NULL){
    idx <- which(vars == row_name)
    if (length(idx) == 0 && !is.null(candidates)) {
      for (cand in candidates) { idx <- which(vars == cand); if (length(idx) > 0) break }
    }
    if (length(idx) == 0) return(rep(NA, ncol(data)))
    as.character(unlist(data[idx[1], , drop=TRUE]))
  }

  group_vals <- pick_row(opt$group_row, c("Responder_study","Responder","Response","Treatment_response"))
  age_vals   <- to_numeric(pick_row(opt$age_row,  c("V1_AgeAtFecalSampling","ANTHRO.AGE","Age")))
  bmi_vals   <- to_numeric(pick_row(opt$bmi_row,  c("V1_BMI","ANTHRO.BMI","BMI")))

  meta <- data.frame(Sample_ID = sample_ids,
                     responder_study = group_vals,
                     Age = age_vals,
                     BMI = bmi_vals,
                     stringsAsFactors = FALSE)
  colnames(meta) <- norm_names(colnames(meta))

} else {
  meta <- if (is_excel) readxl::read_excel(opt$metadata) else readr::read_csv(opt$metadata, show_col_types = FALSE)
  colnames(meta) <- norm_names(colnames(meta))
  remap <- function(x){
    nx <- norm_names(x)
    cands <- c(nx, "Sample_ID","SampleID","sample_id","DAG3_sampleID","DAG3_sample_ID","DAG3sampleID",
               "responder_study","Responder","Response","Treatment_response",
               "Age","ANTHRO_AGE","ANTHROAGE",
               "BMI","ANTHRO_BMI","ANTHROBMI")
    hit <- cands[cands %in% colnames(meta)]
    if (length(hit)==0) nx else hit[1]
  }
  opt$sample_col <- remap(opt$sample_col)
  opt$group_col  <- remap(opt$group_col)
  opt$age_col    <- remap(opt$age_col)
  opt$bmi_col    <- remap(opt$bmi_col)

  if (!(opt$sample_col %in% colnames(meta))) stop("Sample_ID column not found in tidy metadata.")
  colnames(meta)[colnames(meta)==opt$sample_col] <- "Sample_ID"
  if (opt$group_col %in% colnames(meta)) colnames(meta)[colnames(meta)==opt$group_col] <- "responder_study"
  if (opt$age_col   %in% colnames(meta)) colnames(meta)[colnames(meta)==opt$age_col]   <- "Age"
  if (opt$bmi_col   %in% colnames(meta)) colnames(meta)[colnames(meta)==opt$bmi_col]   <- "BMI"

  if ("Age" %in% colnames(meta)) meta$Age <- to_numeric(meta$Age)
  if ("BMI" %in% colnames(meta)) meta$BMI <- to_numeric(meta$BMI)
}

# de-duplicate
meta <- meta |> dplyr::distinct(Sample_ID, .keep_all = TRUE)

# -------- align samples --------
common <- intersect(colnames(ft), meta$Sample_ID)
if (length(common) == 0) {
  readr::write_csv(tibble::tibble(Feature_samples = colnames(ft)), file.path(opt$outdir, "feature_sample_names.csv"))
  readr::write_csv(tibble::tibble(Metadata_Sample_ID = meta$Sample_ID), file.path(opt$outdir, "metadata_sample_ids.csv"))
  stop("No overlap between feature columns and metadata Sample_ID. Wrote feature_sample_names.csv & metadata_sample_ids.csv for debugging.")
}
# report unmatched
readr::write_csv(tibble::tibble(Sample_in_features_not_in_meta = setdiff(colnames(ft), meta$Sample_ID)),
                 file.path(opt$outdir, "unmatched_features_vs_meta.csv"))
readr::write_csv(tibble::tibble(Sample_in_meta_not_in_features = setdiff(meta$Sample_ID, colnames(ft))),
                 file.path(opt$outdir, "unmatched_meta_vs_features.csv"))

# subset + order
ft   <- ft[, common, drop = FALSE]
meta <- meta |> dplyr::filter(Sample_ID %in% common) |> dplyr::slice(match(common, Sample_ID))

# -------- relative abundances --------
rel <- if (opt$assume_relative) as.matrix(ft) else {
  cs <- colSums(ft); med <- stats::median(cs)
  if (med > 1.2 || med < 0.8) to_rel(as.matrix(ft)) else as.matrix(ft)
}

# -------- distances --------
bray <- vegan::vegdist(t(rel), method = "bray")
jac  <- vegan::vegdist(t(rel > 0), method = "jaccard")

# -------- PCoA plots --------
pcoa_plot <- function(d, name, outdir, meta){
  p <- stats::cmdscale(d, eig=TRUE, k=2)
  coords <- as.data.frame(p$points); colnames(coords) <- c("PC1","PC2")
  coords$Sample_ID <- rownames(coords)
  df <- dplyr::left_join(coords, meta, by="Sample_ID")
  var_exp <- tryCatch(round(100 * p$eig[1:2] / sum(p$eig[p$eig>0]), 1), error=function(e)c(NA,NA))

  gg <- ggplot2::ggplot(df, ggplot2::aes(PC1, PC2, color = responder_study)) +
    ggplot2::geom_point(size=3, alpha=0.85) +
    ggplot2::labs(title=paste0(name, " PCoA"),
                  x=paste0("PC1 (", ifelse(is.na(var_exp[1]), "NA", var_exp[1]), "%)"),
                  y=paste0("PC2 (", ifelse(is.na(var_exp[2]), "NA", var_exp[2]), "%)"),
                  color="responder_study") +
    ggplot2::theme_minimal()
  ggplot2::ggsave(file.path(outdir, paste0("PCoA_", name, ".png")), gg, dpi=200, width=7, height=5)
}
pcoa_plot(bray, "BrayCurtis", opt$outdir, meta)
pcoa_plot(jac,  "Jaccard",    opt$outdir, meta)

# -------- PERMANOVA (no 'd=' trick) --------
make_formula <- function(lhs, terms) as.formula(paste(lhs, "~", paste(terms, collapse = " + ")))
terms <- c()
if ("responder_study" %in% colnames(meta)) terms <- c(terms, "responder_study")
if ("Age" %in% colnames(meta))             terms <- c(terms, "Age")
if ("BMI" %in% colnames(meta))             terms <- c(terms, "BMI")
if (length(terms) == 0) terms <- "1"  # intercept-only

# Put distance objects in env so formula can see them
bray_env <- list2env(list(bray=bray), parent=globalenv())
jac_env  <- list2env(list(jac=jac),   parent=globalenv())

set.seed(42)
res_bray <- with(bray_env, vegan::adonis2(make_formula("bray", terms), data = meta, permutations = 999))
res_jac  <- with(jac_env,  vegan::adonis2(make_formula("jac",  terms), data = meta, permutations = 999))

adonis_to_df <- function(res){ as.data.frame(res$aov.tab) |> tibble::rownames_to_column("Term") }
db_bray <- adonis_to_df(res_bray); db_bray$Distance <- "BrayCurtis"
db_jac  <- adonis_to_df(res_jac);  db_jac$Distance  <- "Jaccard"
res_all <- dplyr::bind_rows(db_bray, db_jac)
readr::write_csv(res_all, file.path(opt$outdir, "permanova_results.csv"))

# -------- dispersion --------
if ("responder_study" %in% colnames(meta)) {
  grp <- meta$responder_study
  bd_bray <- vegan::betadisper(bray, grp)
  bd_jac  <- vegan::betadisper(jac,  grp)
  sink(file.path(opt$outdir, "betadisper.txt"))
  cat("Bray-Curtis betadisper:\n"); print(anova(bd_bray)); cat("\n")
  cat("Jaccard betadisper:\n");    print(anova(bd_jac));  cat("\n")
  sink()
  disp_df <- data.frame(
    Sample_ID = names(bd_bray$distances),
    Dist_to_Centroid_Bray = bd_bray$distances,
    Dist_to_Centroid_Jacc = bd_jac$distances
  ) |> dplyr::left_join(meta, by="Sample_ID")
  readr::write_csv(disp_df, file.path(opt$outdir, "distance_to_centroid.csv"))
}

cat("[OK] Saved outputs to ", opt$outdir, "\n", sep = "")
