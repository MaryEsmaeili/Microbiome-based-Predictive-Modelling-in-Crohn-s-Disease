#!/usr/bin/env Rscript

# ------------------------------------------------------------------------------
# Differential abundance with covariates using ANCOM-BC (v1 or v2)
# Features input MUST be in PERCENT (0–100), not CLR.
# This script:
#   - Accepts either ml_{rank}_wide_all.csv (percent) or, if you accidentally pass
#     ml_{rank}_wide_clr.csv, it will try to auto-swap to the sibling _wide_all.csv.
#   - Standardizes the ID to Sample_ID, joins meta on Sample_ID, and builds
#     pseudo-counts (sum per sample = 10,000) for ANCOM-BC.
#   - Uses safe guards and writes empty outputs with a reason if data are insufficient.
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(tibble)
  library(ggplot2)
  library(phyloseq)
  library(data.table)
})

# ----------------------------- CLI --------------------------------------------
opt_list <- list(
  make_option("--features", type="character"),
  make_option("--meta",     type="character"),
  make_option("--exclude",  type="character", default=NA),
  make_option("--site",     type="character"),
  make_option("--rank",     type="character"),
  make_option("--outbase",  type="character")
)
opt <- parse_args(OptionParser(option_list = opt_list))
dir.create(dirname(opt$outbase), recursive = TRUE, showWarnings = FALSE)

# ------------------------- helpers --------------------------------------------
write_empty_and_exit <- function(msg="No data") {
  data.table::fwrite(data.frame(), paste0(opt$outbase, ".csv"))
  png(paste0(opt$outbase, "_volcano.png"), width=1200, height=650, res=150)
  plot.new(); text(0.5,0.5,msg); dev.off()
  quit(save="no")
}

read_features_percent <- function(path) {
  # If a CLR file is passed by mistake, auto-swap to _wide_all.csv if present.
  p <- path
  if (grepl("_wide_clr\\.csv$", p) && file.exists(sub("_wide_clr\\.csv$", "_wide_all.csv", p))) {
    p <- sub("_wide_clr\\.csv$", "_wide_all.csv", p)
  }
  X <- readr::read_csv(p, show_col_types = FALSE)

  # Standardize ID: create Sample_ID no matter what original column name was
  id_candidates <- c("Sample_ID","Sample","sample_id","ID","id")
  id_col <- id_candidates[id_candidates %in% names(X)][1]
  if (is.na(id_col)) stop("features must include one of: Sample_ID / Sample / ID")
  X <- X %>% mutate(Sample_ID = as.character(.data[[id_col]]))

  # If Site is absent, create a neutral Site to avoid filtering everything out
  if (!("Site" %in% names(X))) X$Site <- NA_character_

  return(X)
}

map_disease_readable <- function(x) {
  z <- tolower(as.character(x))
  out <- ifelse(z %in% c("1","crohn","cd","case","ibd"), "Crohn",
         ifelse(z %in% c("0","healthy","control","ctr","ctl","hc","non-ibd"), "Healthy", NA))
  factor(out, levels = c("Healthy","Crohn"))
}

# ------------------------- Read features (PERCENT) ----------------------------
X <- tryCatch(read_features_percent(opt$features),
              error = function(e) stop("Failed to read features: ", e$message))

# Exclusion by Sample_ID (if file exists)
if (!is.na(opt$exclude) && file.exists(opt$exclude)) {
  excl <- readLines(opt$exclude, warn = FALSE)
  X <- X %>% filter(!(as.character(Sample_ID) %in% excl))
}

# Site filter with synonyms (case-insensitive)
site_map <- list(
  oral  = c("oral","saliva","mouth","buccal"),
  fecal = c("fecal","faecal","stool","faeces","feces")
)
site_key <- tolower(opt$site)
if ("Site" %in% names(X)) {
  X <- X %>%
    mutate(.site_l = tolower(as.character(Site))) %>%
    filter(.site_l %in% c(site_key, site_map[[site_key]])) %>%
    select(-.site_l)
}
# One row per Sample_ID
X <- X %>%
  filter(!is.na(Sample_ID) & Sample_ID != "") %>%
  distinct(Sample_ID, .keep_all = TRUE)

if (nrow(X) == 0) write_empty_and_exit("No samples after site/exclusion filters")

# ----------------------- Join META on Sample_ID -------------------------------
meta <- readr::read_csv(opt$meta, show_col_types = FALSE)
id_candidates_meta <- c("Sample_ID","Sample","sample_id","ID","id")
id_meta <- id_candidates_meta[id_candidates_meta %in% names(meta)][1]
if (!is.na(id_meta)) {
  meta <- meta %>% mutate(Sample_ID = as.character(.data[[id_meta]]))
  XX   <- inner_join(X, meta, by = "Sample_ID")
} else {
  XX <- X
}

# Allowed covariates
allowed_cov_all <- c("Age","Sex","BMI","Smoking","Antibiotics_3m",
                     "PPI_use","Steroids_ongoing","Immuno_ongoing")

# Identify taxa columns = all numeric except known meta
known_meta <- c("Sample_ID","disease","Site", allowed_cov_all)
tax_cols <- setdiff(names(XX), known_meta)
if (length(tax_cols) > 0) {
  XX[, tax_cols] <- lapply(XX[, tax_cols, drop=FALSE], function(col) suppressWarnings(as.numeric(col)))
  keep_tax <- tax_cols[sapply(XX[, tax_cols, drop=FALSE], function(col) any(!is.na(col)))]
} else {
  keep_tax <- character(0)
}
if (length(keep_tax) == 0) write_empty_and_exit("No numeric taxa features")

# Ensure 'disease' exists; fallback to Group if present in either side
if (!("disease" %in% names(XX))) {
  if ("Group" %in% names(XX)) {
    XX$disease <- XX$Group
  } else if ("disease.x" %in% names(XX)) {
    XX$disease <- XX[["disease.x"]]
  } else if ("disease.y" %in% names(XX)) {
    XX$disease <- XX[["disease.y"]]
  } else {
    XX$disease <- NA_character_
  }
}

# Convert % → pseudo-counts (sum per sample = 10,000)
XX_counts <- XX
row_tot <- rowSums(XX_counts[, keep_tax, drop=FALSE], na.rm=TRUE)
scaler  <- ifelse(row_tot > 0, 10000 / pmax(row_tot, 1e-9), 0)
XX_counts[, keep_tax] <- sweep(XX_counts[, keep_tax, drop=FALSE], 1, scaler, `*`)
XX_counts[, keep_tax] <- lapply(XX_counts[, keep_tax, drop=FALSE], function(col) round(pmax(col, 0)))

# ------------------------- Build phyloseq -------------------------------------
otu <- as.matrix(t(as.data.frame(XX_counts[, keep_tax, drop = FALSE])))
if (ncol(otu) != nrow(XX_counts)) stop("OTU columns != samples; check duplicate/NA Sample_ID.")
rownames(otu) <- keep_tax
colnames(otu) <- XX_counts$Sample_ID

OTU <- phyloseq::otu_table(otu, taxa_are_rows = TRUE)
TAX <- phyloseq::tax_table(matrix(keep_tax, ncol = 1))
rownames(TAX) <- keep_tax
colnames(TAX) <- opt$rank

# Sample data: disease + allowed covariates
allowed_cov <- intersect(allowed_cov_all, names(XX_counts))
sdat <- XX_counts %>%
  mutate(disease = map_disease_readable(disease)) %>%
  select(Sample_ID, disease, all_of(allowed_cov)) %>%
  tibble::column_to_rownames("Sample_ID")
for (cn in allowed_cov) sdat[[cn]] <- as.factor(sdat[[cn]])

SD <- sample_data(sdat)
PS <- phyloseq(OTU, TAX, SD)

# Guards: class balance & variance
cls <- table(sdat$disease)
ok_classes <- all(c("Healthy","Crohn") %in% names(cls)) && min(cls) > 0
var_ok <- apply(otu, 1, function(v) sd(v, na.rm=TRUE) > 0)
if (!ok_classes || !any(var_ok)) write_empty_and_exit("Not enough class/variance")
PS <- prune_taxa(names(var_ok)[var_ok], PS)

# ---------- ANCOM-BC v1/v2 ----------
use_v2 <- requireNamespace("ANCOMBC", quietly = TRUE) &&
          ("ancombc2" %in% getNamespaceExports("ANCOMBC"))

if (use_v2) {
  suppressPackageStartupMessages({ library(ANCOMBC); library(mia) })
  tse <- tryCatch(
    mia::convertFromPhyloseq(PS),
    error = function(e) {
      message("convertFromPhyloseq failed; using fallback: ", e$message)
      mia::makeTreeSummarizedExperimentFromPhyloseq(PS)
    }
  )
  covars <- allowed_cov
  formula_str <- paste(c("disease", covars), collapse = " + ")
  fit <- ANCOMBC::ancombc2(
    data = tse,
    assay_name = "counts",
    fix_formula = formula_str,
    p_adj_method = "BH",
    prv_cut = 0.10,
    lib_cut = 0,
    global = FALSE
  )
  res <- fit$res
  eff_candidates <- colnames(res$lfc)
  patterns <- c("^disease.*crohn$", "^disease.*crohn.*(true|1)$", "crohn", "^disease$")
  eff_col <- NA_character_
  for (pat in patterns) {
    idx <- grep(pat, eff_candidates, ignore.case = TRUE)
    if (length(idx) >= 1) { eff_col <- eff_candidates[idx[1]]; break }
  }
  if (is.na(eff_col) || length(eff_col)==0) write_empty_and_exit("No disease effect column (v2)")
  tab <- tibble::tibble(
    taxon  = rownames(res$lfc),
    effect = as.numeric(res$lfc[, eff_col]),
    se     = as.numeric(res$se[,  eff_col]),
    p      = as.numeric(res$p_val[, eff_col]),
    q      = as.numeric(res$q_val[, eff_col])
  )
} else {
  suppressPackageStartupMessages(library(ANCOMBC))
  covars <- allowed_cov
  formula_str <- paste(c("disease", covars), collapse = " + ")
  fit <- ANCOMBC::ancombc(
    phyloseq = PS, formula = formula_str, p_adj_method = "BH",
    zero_cut = 0.90, lib_cut = 0, tol = 1e-5, max_iter = 100,
    conserve = TRUE, global = FALSE
  )
  res <- fit$res
  use_col <- if ("diseaseCrohn" %in% colnames(res$beta)) "diseaseCrohn" else {
    beta_cols <- colnames(res$beta)
    idx <- grep("(disease|crohn)", beta_cols, ignore.case = TRUE)
    if (length(idx) == 0) NA_character_ else beta_cols[idx[1]]
  }
  if (is.na(use_col)) write_empty_and_exit("No disease column (v1)")
  tab <- tibble::tibble(
    taxon  = rownames(res$beta),
    effect = as.numeric(res$beta[, use_col]),
    se     = as.numeric(res$se[,   use_col]),
    p      = as.numeric(res$p_val[, use_col]),
    q      = as.numeric(res$q_val[, use_col])
  )
}

# ----------------------------- Write & Volcano --------------------------------
data.table::fwrite(tab, paste0(opt$outbase, ".csv"))

tab$neglog10q <- -log10(pmax(tab$q, 1e-300))
thr <- 0.05
p <- ggplot(tab, aes(x = effect, y = neglog10q)) +
  geom_point(aes(color = q <= thr), alpha = 0.85, size = 2) +
  geom_hline(yintercept = -log10(thr), linetype = "dashed") +
  geom_vline(xintercept = 0,            linetype = "dashed") +
  scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "tomato")) +
  labs(
    title = paste0(toupper(substr(opt$site,1,1)), substr(opt$site,2,99),
                   " (", if (exists("use_v2") && use_v2) "ANCOM-BC2" else "ANCOM-BC", ", ", opt$rank, ")"),
    x = "adjusted log-fold change (Crohn vs Healthy)", y = "-log10(q)"
  ) +
  theme_minimal(base_size = 13) + theme(legend.position = "none")

ggsave(paste0(opt$outbase, "_volcano.png"), p, width = 11, height = 6, dpi = 300)
