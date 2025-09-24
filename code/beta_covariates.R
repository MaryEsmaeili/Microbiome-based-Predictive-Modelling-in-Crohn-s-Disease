#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(stringr); library(purrr)
  library(ggplot2)
  library(vegan)
  library(yaml)
})

# ============================ CLI ============================
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(f, d=NULL){ i <- which(args==f); if(length(i)==0||i==length(args)) d else args[i+1] }

oral_crohn_path    <- get_arg("--oral-crohn")
fecal_crohn_path   <- get_arg("--fecal-crohn")
oral_healthy_path  <- get_arg("--oral-healthy")
fecal_healthy_path <- get_arg("--fecal-healthy")
covars_path        <- get_arg("--covariates")
outdir             <- get_arg("--outdir", "results/beta-covariates")  # match Snakemake’s path
pseudo             <- as.numeric(get_arg("--pseudocount", "1e-06"))

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)


# ======================== Utilities ==========================
# Normalize sample IDs so metadata and abundance headers match
# - Remove replicate suffixes like ".1"
# - For numeric-only IDs, drop leading zeros (010083 -> 10083)
# - Uppercase everything (safe for S00xx etc.)
normalize_id <- function(x) {
  x <- gsub("\\.\\d+$", "", x)
  if (grepl("^[0-9]+$", x)) x <- sub("^0+", "", x)
  toupper(x)
}

`%||%` <- function(a,b) if (is.null(a) || is.na(a)) b else a

# Colors (config.yaml optional)
load_colors <- function(path = "config.yaml") {
  if (!file.exists(path)) {
    warning("config.yaml not found; using defaults.")
    return(list(
      oral  = c("Healthy"="#6aaed6","Crohn"="#1b6ca8"),
      fecal = c("Healthy"="#90c987","Crohn"="#1a9850")
    ))
  }
  cfg  <- yaml::read_yaml(path)
  cols <- cfg$colors
  req  <- c("Crohn-Oral","Healthy-Oral","Crohn-Fecal","Healthy-Fecal")
  miss <- setdiff(req, names(cols))
  if (length(miss)) warning("Missing colors in config.yaml: ", paste(miss, collapse=", "))
  list(
    oral  = c("Healthy" = cols[["Healthy-Oral"]]  %||% "#6aaed6",
              "Crohn"   = cols[["Crohn-Oral"]]    %||% "#1b6ca8"),
    fecal = c("Healthy" = cols[["Healthy-Fecal"]] %||% "#90c987",
              "Crohn"   = cols[["Crohn-Fecal"]]   %||% "#1a9850")
  )
}
pal <- load_colors("config.yaml")

safe_read_csv <- function(p){
  if (is.null(p) || is.na(p) || !file.exists(p)) return(NULL)
  suppressMessages(readr::read_csv(p, show_col_types = FALSE))
}

# Read abundance file → matrix [features x samples], numeric, no all-zero rows
read_abund <- function(path){
  df <- safe_read_csv(path)
  if (is.null(df) || nrow(df)==0) return(NULL)

  # Heuristic: if first column looks like feature names, peel it off
  first <- names(df)[1]
  if (is.na(first) || first=="" || first=="...1" ||
      grepl("^(feature|taxon|clade|species|id|name)$", tolower(first))) {
    feat <- df[[1]]
    df   <- df[,-1, drop=FALSE]
  } else {
    cand <- names(df)[str_detect(tolower(names(df)), "feature|taxon|clade|name|id")]
    if (length(cand)) {
      feat <- df[[cand[1]]]
      df   <- df[ , setdiff(names(df), cand[1]), drop=FALSE]
    } else {
      feat <- seq_len(nrow(df))
    }
  }

  # Coerce to numeric
  for (j in seq_along(df)) df[[j]] <- suppressWarnings(as.numeric(df[[j]]))
  df[is.na(df)] <- 0

  # Build matrix
  M <- as.matrix(df)
  rownames(M) <- make.unique(as.character(feat))
  colnames(M) <- make.unique(colnames(M))

  # Drop all-zero features
  keep <- rowSums(M, na.rm=TRUE) > 0
  if (any(!keep)) M <- M[keep, , drop=FALSE]
  if (ncol(M)==0 || nrow(M)==0) return(NULL)

  # *** Normalize sample headers here (where M exists) ***
  colnames(M) <- vapply(colnames(M), normalize_id, character(1))
  M
}

# Union of rows (features), then cbind(A, B)
union_rows_cbind <- function(A, B){
  if (is.null(A) && is.null(B)) return(NULL)
  if (is.null(A)) return(B)
  if (is.null(B)) return(A)
  rn <- union(rownames(A), rownames(B))
  A2 <- matrix(0, nrow=length(rn), ncol=ncol(A), dimnames=list(rn, colnames(A)))
  B2 <- matrix(0, nrow=length(rn), ncol=ncol(B), dimnames=list(rn, colnames(B)))
  if (nrow(A)>0 && ncol(A)>0) A2[rownames(A), colnames(A)] <- A
  if (nrow(B)>0 && ncol(B)>0) B2[rownames(B), colnames(B)] <- B
  cbind(A2, B2)
}

# CLR on samples x features
clr_matrix <- function(X, pseudo=1e-6){
  X2 <- X + pseudo
  L  <- log(X2)
  gm <- rowMeans(L)
  sweep(L, 1, gm)
}

# ================== Load data & covariates =====================
wish_cov <- c("Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing")

# Covariates: allow flexible column names
covars <- safe_read_csv(covars_path) %>%
  mutate(across(everything(), ~.x)) %>%
  rename_with(~"Sample_ID", .cols = tidyselect::any_of(c("Sample_ID","SampleID","sample_id","sample","id"))) %>%
  rename_with(~"site",      .cols = tidyselect::any_of(c("site","Site"))) %>%
  rename_with(~"disease",   .cols = tidyselect::any_of(c("disease","Disease"))) %>%
  mutate(
    Sample_ID = as.character(Sample_ID),
    site      = as.character(site),
    disease   = suppressWarnings(as.numeric(disease))
  )

# Abundances
X_oral_c  <- read_abund(oral_crohn_path)
X_oral_h  <- read_abund(oral_healthy_path)
X_fecal_c <- read_abund(fecal_crohn_path)
X_fecal_h <- read_abund(fecal_healthy_path)

# ================== Prep per-site matrices =====================
prep_site <- function(site_name, M_crohn, M_healthy){
  # Combine features, bind columns (Crohn first, then Healthy)
  M <- union_rows_cbind(M_crohn, M_healthy)
  if (is.null(M)) return(list(M=NULL, md=NULL, usable_cov=NULL, pal=NULL))

  samples <- colnames(M)  # already normalized

  # 1) Select site rows first
  md <- covars %>%
    filter(tolower(site) == tolower(site_name))

  if (nrow(md)==0) return(list(M=NULL, md=NULL, usable_cov=NULL, pal=NULL))

  # 2) Normalize metadata Sample_ID BEFORE intersecting
  md <- md %>%
    mutate(Sample_ID = vapply(as.character(Sample_ID), normalize_id, character(1)))

  # 3) Only keep IDs that exist in abundance
  md <- md %>%
    filter(Sample_ID %in% samples) %>%
    select(any_of(c("Sample_ID","site","disease", wish_cov)))

  if (nrow(md) < 3) return(list(M=NULL, md=NULL, usable_cov=NULL, pal=NULL))

  # Ensure numeric covariates where possible
  for (v in intersect(wish_cov, names(md))) {
    md[[v]] <- suppressWarnings(as.numeric(md[[v]]))
  }

  # Keep covariates with actual variation and at least a few non-NA values
  usable_cov <- c()
  for (v in intersect(wish_cov, names(md))) {
    vals <- md[[v]]
    if (sum(!is.na(vals)) >= 3 && dplyr::n_distinct(vals, na.rm=TRUE) >= 2) {
      usable_cov <- c(usable_cov, v)
    }
  }

  # Drop rows with missing disease label
  md <- md %>% drop_na(disease)

  # Re-order abundance columns to metadata
  common <- intersect(colnames(M), md$Sample_ID)
  md <- md %>% filter(Sample_ID %in% common)
  if (nrow(md) < 3) return(list(M=NULL, md=NULL, usable_cov=NULL, pal=NULL))
  M  <- M[, md$Sample_ID, drop=FALSE]

  # Debug sanity dump
  matched <- length(intersect(colnames(M), md$Sample_ID))
  write_csv(
    tibble(site = site_name,
           n_abund_cols = ncol(M),
           n_md_rows = nrow(md),
           n_matched = matched),
    file.path(outdir, paste0("debug_matches_", site_name, ".csv"))
  )

  list(M=M, md=md, usable_cov=usable_cov,
       pal = if (tolower(site_name)=="oral") pal$oral else pal$fecal)
}

oral  <- prep_site("oral",  X_oral_c,  X_oral_h)
fecal <- prep_site("fecal", X_fecal_c, X_fecal_h)

# =================== Analysis per site =========================
run_all <- function(site_name, obj) {
  f_permanova_bray <- file.path(outdir, paste0("permanova_", site_name, "_bray_with_covariates.csv"))
  f_permanova_ait  <- file.path(outdir, paste0("permanova_", site_name, "_aitchison_with_covariates.csv"))
  f_permdisp_bray  <- file.path(outdir, paste0("permdisp_",  site_name, "_bray_with_covariates.csv"))
  f_pcoa_bray      <- file.path(outdir, paste0("pcoa_", site_name, "_bray_with_ellipses.png"))
  f_pcoa_ait       <- file.path(outdir, paste0("pcoa_", site_name, "_aitchison_with_ellipses.png"))

  # If we don't have enough to proceed, emit placeholders and exit cleanly
  if (is.null(obj$M) || is.null(obj$md)) {
    write_csv(tibble(note="no_data_for_site"), f_permanova_bray)
    write_csv(tibble(note="no_data_for_site"), f_permanova_ait)
    write_csv(tibble(note="no_data_for_site"), f_permdisp_bray)
    g <- ggplot() + theme_void() + annotate("text", x=0, y=0, label=paste("No data for", site_name), size=6)
    ggsave(f_pcoa_bray, g, width=5.5, height=4.3, dpi=300)
    ggsave(f_pcoa_ait,  g, width=5.5, height=4.3, dpi=300)
    message("[beta_covariates] No data for site: ", site_name, " → wrote placeholders.")
    return(invisible(NULL))
  }

  M        <- obj$M
  md       <- obj$md
  covs     <- obj$usable_cov
  pal_site <- obj$pal

  # Distances
  D_bray <- tryCatch(vegdist(t(M), method="bray"), error=function(e) NULL)

  Xsf  <- t(M)
  Xclr <- clr_matrix(Xsf, pseudo)
  D_ait <- tryCatch(stats::dist(Xclr, method="euclidean"), error=function(e) NULL)

  # Build RHS with disease + usable covariates actually present
  build_formula <- function(){
    rhs <- c("disease", covs)
    rhs <- rhs[rhs %in% names(md)]
    as.formula(paste("~", paste(rhs, collapse=" + ")))
  }

  run_permanova <- function(D, out_csv){
    if (is.null(D)) { write_csv(tibble(note="dist_null"), out_csv); return() }
    terms <- all.vars(build_formula())[-1]  # exclude response placeholder
    md2   <- md %>% dplyr::select(Sample_ID, all_of(c("disease", terms))) %>% drop_na(disease)

    idx <- match(labels(D), md2$Sample_ID)
    ok  <- which(!is.na(idx))
    if (length(ok) < 3 || length(unique(md2$disease[idx[ok]])) < 2) {
      write_csv(tibble(note="insufficient_groups_or_n"), out_csv); return()
    }

    md3 <- md2[idx[ok], , drop=FALSE]
    D2  <- as.dist(as.matrix(D)[ok, ok, drop=FALSE])

    usable <- c()
    for (v in intersect(covs, names(md3))) {
      vv <- md3[[v]]
      if (sum(!is.na(vv)) >= 3 && dplyr::n_distinct(vv, na.rm=TRUE) >= 2) usable <- c(usable, v)
    }
    rhs <- paste(c("disease", usable), collapse=" + ")
    fml <- as.formula(paste("D2 ~", rhs))

    res <- vegan::adonis2(fml, data=md3, permutations=999, by="margin")
    out <- as.data.frame(res)
    out$term <- rownames(out); rownames(out) <- NULL
    out <- out %>% rename(Df = Df, SumOfSqs = SumOfSqs, R2 = R2, F = `F`, p = `Pr(>F)`) %>%
      dplyr::select(term, Df, SumOfSqs, R2, F, p)

    write_csv(out, out_csv)
  }

  run_permdisp <- function(D, out_csv){
    if (is.null(D)) { write_csv(tibble(note="dist_null"), out_csv); return() }
    idx <- match(labels(D), md$Sample_ID)
    ok  <- which(!is.na(idx))
    if (length(ok) < 3 || length(unique(md$disease[idx[ok]])) < 2) {
      write_csv(tibble(note="insufficient_groups_or_n"), out_csv); return()
    }
    md2 <- md[idx[ok], , drop=FALSE]
    D2  <- as.dist(as.matrix(D)[ok, ok, drop=FALSE])
    g   <- factor(ifelse(md2$disease==1,"Crohn","Healthy"), levels=c("Healthy","Crohn"))

    bd  <- betadisper(D2, g)
    a   <- anova(bd)
    pt  <- permutest(bd, permutations=999)

    out <- tibble(
      term = c("betadisper_anova","betadisper_permutest"),
      Df   = c(a$Df[1], pt$tab$Df[1]),
      F    = c(a$`F value`[1], pt$tab$F[1]),
      p    = c(a$`Pr(>F)`[1], pt$tab$`Pr(>F)`[1])
    )
    write_csv(out, out_csv)
  }

  pcoa_plot <- function(D, png_file, ttl){
    if (is.null(D)) {
      g <- ggplot() + theme_void() + annotate("text", x=0, y=0, label="distance is NULL", size=5)
      ggsave(png_file, g, width=5.8, height=4.6, dpi=300); return()
    }
    idx <- match(labels(D), md$Sample_ID)
    ok  <- which(!is.na(idx))
    if (length(ok) < 3) {
      g <- ggplot() + theme_void() + annotate("text", x=0, y=0, label="not enough samples", size=5)
      ggsave(png_file, g, width=5.8, height=4.6, dpi=300); return()
    }
    D2  <- as.dist(as.matrix(D)[ok, ok, drop=FALSE])
    md2 <- md[idx[ok], , drop=FALSE]
    pcs <- stats::cmdscale(D2, k=2, eig=TRUE)
    df  <- tibble(
      PC1 = pcs$points[,1],
      PC2 = pcs$points[,2],
      disease = factor(ifelse(md2$disease==1,"Crohn","Healthy"), levels=c("Healthy","Crohn"))
    )
    gg <- ggplot(df, aes(PC1, PC2, color=disease)) +
      geom_point(size=2.2, alpha=0.85) +
      { if (length(unique(df$disease))>1) stat_ellipse(type="norm", level=0.95, linewidth=0.6) else NULL } +
      scale_color_manual(values = pal_site) +
      theme_bw(base_size=12) +
      labs(title=ttl, color="Group")
    ggsave(png_file, gg, width=5.8, height=4.6, dpi=300)
  }

  # Run
  run_permanova(D_bray, f_permanova_bray)
  run_permanova(D_ait,  f_permanova_ait)
  run_permdisp(D_bray,  f_permdisp_bray)
  pcoa_plot(D_bray, f_pcoa_bray, paste("PCoA (Bray) —", tools::toTitleCase(site_name)))
  pcoa_plot(D_ait,  f_pcoa_ait,  paste("PCoA (Aitchison) —", tools::toTitleCase(site_name)))
}

# ======================== Execute ==============================
run_all("oral",  oral)
run_all("fecal", fecal)

message("[beta_covariates] Done → ", outdir)
