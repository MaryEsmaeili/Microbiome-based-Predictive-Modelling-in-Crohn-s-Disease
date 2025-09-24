#!/usr/bin/env Rscript
# =============================================================================
# Beta Diversity — PCoA, PERMANOVA(+covariates & interaction), PERMDISP,
# Paired oral–fecal distances (Crohn), and group distance table.
#
# Robust to:
# - color YAML schemas (legacy colors: {}, or group/synonyms)
# - low/imbalanced sample sizes (writes placeholders/notes)
# - flexible covariate table columns
# - sample ID normalization (drops leading zeros for numeric-only IDs, drops .1 suffix)
#
# EXACT outputs (must match Snakemake rule `beta`):
#   pcoa_bray_allgroups.png
#   pcoa_jaccard_allgroups.png
#   pcoa_aitchison_allgroups.png
#   pcoa_bray_oral_CH.png
#   pcoa_bray_fecal_CH.png
#   pcoa_bray_oral_vs_fecal_paired.png
#   pairwise_oral_fecal_summary.csv
#   permanova_oral_bray_with_cov.csv
#   permanova_fecal_bray_with_cov.csv
#   permanova_interaction_bray.csv
#   permdisp_oral_bray.csv
#   permdisp_fecal_bray.csv
#   beta_group_distances.csv
#
# CLI (from Snakefile):
#   --oral-crohn    PATH
#   --fecal-crohn   PATH
#   --oral-healthy  PATH
#   --fecal-healthy PATH
#   --covariates    PATH
#   --pairs         PATH
#   --colors        PATH
#   --outdir        PATH
#   --pseudocount   NUM    (default 1e-6)
#   --permutations  INT    (default 999)
# =============================================================================

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(ggplot2)
  library(vegan); library(stringr); library(purrr); library(yaml)
})

# ---------------------------- CLI parsing -------------------------------
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) == 1 && i < length(args)) args[i + 1] else default
}

oral_crohn_path    <- get_arg("--oral-crohn")
fecal_crohn_path   <- get_arg("--fecal-crohn")
oral_healthy_path  <- get_arg("--oral-healthy")
fecal_healthy_path <- get_arg("--fecal-healthy")
covars_path        <- get_arg("--covariates")
pairs_path         <- get_arg("--pairs")
colors_path        <- get_arg("--colors")
outdir             <- get_arg("--outdir", "results/beta_models")
pseudo             <- as.numeric(get_arg("--pseudocount", "1e-6"))
nperm              <- as.integer(get_arg("--permutations", "999"))

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dbgdir <- file.path(outdir, "debug"); dir.create(dbgdir, showWarnings = FALSE)

# ---------------------------- Safe helpers ------------------------------
`%||%` <- function(a,b) if (is.null(a) || is.na(a) || (is.character(a) && !nzchar(a))) b else a

save_placeholder_png <- function(path, msg="Insufficient data") {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  g <- ggplot() + theme_void() + annotate("text", 0, 0, label = msg, size = 5)
  ggsave(path, g, width = 6, height = 5, dpi = 300, bg = "white")
}

write_note_csv <- function(path, note="no_data") {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write_csv(tibble(note = note), path)
}

# ---------------------------- Colors loader -----------------------------
load_group_colors <- function(yaml_path) {
  defaults <- c(
    "Crohn-Oral"    = "#B499E5",
    "Healthy-Oral"  = "#7FB3D5",
    "Crohn-Fecal"   = "#78688E",
    "Healthy-Fecal" = "#416388"
  )
  if (is.null(yaml_path) || !file.exists(yaml_path)) return(defaults)
  cfg <- tryCatch(yaml::read_yaml(yaml_path), error = function(e) NULL)
  if (is.null(cfg)) return(defaults)

  # Schema A: legacy colors: { Crohn-Oral: "#...", ... }
  if (!is.null(cfg$colors)) {
    cols <- cfg$colors
    get1 <- function(name) {
      v <- tryCatch(if (is.list(cols)) cols[[name]] else cols[name], error = function(e) NULL)
      v <- if (is.null(v)) NULL else as.character(v)[1]
      if (is.null(v) || is.na(v) || !nzchar(v)) NULL else v
    }
    return(c(
      "Crohn-Oral"    = get1("Crohn-Oral")    %||% defaults[["Crohn-Oral"]],
      "Healthy-Oral"  = get1("Healthy-Oral")  %||% defaults[["Healthy-Oral"]],
      "Crohn-Fecal"   = get1("Crohn-Fecal")   %||% defaults[["Crohn-Fecal"]],
      "Healthy-Fecal" = get1("Healthy-Fecal") %||% defaults[["Healthy-Fecal"]]
    ))
  }

  # Schema B: group + synonyms
  group <- cfg$group; syn <- cfg$synonyms
  safe_get <- function(container, key) {
    if (is.null(container)) return(NULL)
    if (is.list(container)) container[[key]]
    else if (is.atomic(container)) {
      nms <- names(container)
      if (!is.null(nms) && key %in% nms) container[[key]] else NULL
    } else NULL
  }
  pick <- function(primary, fallbacks) {
    v <- safe_get(group, primary); v <- if (is.null(v)) NULL else as.character(v)[1]
    if (!is.null(v) && nzchar(v)) return(v)
    syn_list <- safe_get(syn, primary); syn_list <- as.character(unlist(syn_list))
    if (length(syn_list)) for (alias in syn_list) {
      v <- safe_get(group, alias); v <- if (is.null(v)) NULL else as.character(v)[1]
      if (!is.null(v) && nzchar(v)) return(v)
    }
    for (alias in fallbacks) {
      v <- safe_get(group, alias); v <- if (is.null(v)) NULL else as.character(v)[1]
      if (!is.null(v) && nzchar(v)) return(v)
    }
    NA_character_
  }
  out <- c(
    "Crohn-Oral"    = pick("Oral_Crohn",    c("Crohn-Oral","Oral_Crohn")),
    "Healthy-Oral"  = pick("Oral_Healthy",  c("Healthy-Oral","Oral_Healthy")),
    "Crohn-Fecal"   = pick("Fecal_Crohn",   c("Crohn-Fecal","Fecal_Crohn")),
    "Healthy-Fecal" = pick("Fecal_Healthy", c("Healthy-Fecal","Fecal_Healthy"))
  )
  out[is.na(out) | out==""] <- defaults[names(out)[is.na(out) | out==""]]
  out
}
pal_named <- load_group_colors(colors_path)

# ---------------------------- ID normalization --------------------------
normalize_id <- function(x) {
  x <- gsub("\\.\\d+$", "", x)         # drop replicate suffix like .1
  if (grepl("^[0-9]+$", x)) x <- sub("^0+", "", x)  # drop leading zeros for numeric-only IDs
  toupper(x)
}

# ---------------------------- IO: abundances -----------------------------
safe_read_csv <- function(p) {
  if (is.null(p) || !file.exists(p)) return(tibble())
  suppressMessages(readr::read_csv(p, show_col_types = FALSE))
}

# Expect: first column features/taxa, remaining columns samples; values numeric (already normalized/filtered)
read_abund_matrix <- function(path, pseudo = 0) {
  df <- safe_read_csv(path)
  if (nrow(df) == 0) return(NULL)
  rn <- df[[1]]
  M  <- as.matrix(df[,-1, drop = FALSE]); storage.mode(M) <- "numeric"
  rownames(M) <- make.unique(as.character(rn))
  colnames(M) <- make.unique(vapply(colnames(df)[-1], normalize_id, character(1)))
  if (is.finite(pseudo) && pseudo > 0) M <- M + pseudo
  keepC <- colSums(M, na.rm = TRUE) > 0
  keepR <- rowSums(M, na.rm = TRUE) > 0
  M[keepR, keepC, drop = FALSE]
}

align_features <- function(lst) {
  lst <- lst[!vapply(lst, is.null, TRUE)]
  if (!length(lst)) return(lst)
  common <- Reduce(intersect, lapply(lst, rownames))
  lapply(lst, function(m) m[common, , drop = FALSE])
}

drop_bad_for_dist <- function(M, method) {
  if (is.null(M)) return(NULL)
  keep <- colSums(M, na.rm = TRUE) > 0
  M <- M[, keep, drop = FALSE]
  if (ncol(M) < 3) return(M)
  D  <- suppressWarnings(vegan::vegdist(t(M), method = method))
  dm <- as.matrix(D)
  good <- rowSums(is.na(dm)) == 0
  M[, good, drop = FALSE]
}

clr_matrix <- function(X, pseudo = 1e-6) {
  X <- as.matrix(X); X <- sweep(X, 2, colSums(X), "/")
  X[X <= 0] <- pseudo
  L  <- log(X); gm <- matrix(rowMeans(L), nrow = nrow(L), ncol = ncol(L))
  L - gm
}

mk_aitchison <- function(M, pseudo) {
  if (is.null(M) || ncol(M) < 3) return(list(D=NULL, labs=character()))
  Xclr <- clr_matrix(M, pseudo)
  list(D = stats::dist(t(Xclr), method = "euclidean"), labs = colnames(M))
}

mk_dist <- function(M, method) {
  if (is.null(M) || ncol(M) < 3) return(list(D=NULL, labs=character()))
  D  <- suppressWarnings(vegan::vegdist(t(M), method = method))
  dm <- as.matrix(D)
  good <- rowSums(is.na(dm)) == 0
  if (!any(good)) return(list(D=NULL, labs=character()))
  list(D = stats::as.dist(dm[good, good, drop = FALSE]), labs = colnames(M)[good])
}

# ---------------------------- PCoA plotting -----------------------------
pct_from_cmdscale <- function(ord) {
  if (is.null(ord$eig) || all(is.na(ord$eig))) return(c(NA, NA))
  vals <- ord$eig; keep <- which(vals > 0); if (!length(keep)) keep <- seq_along(vals)
  p <- vals / sum(abs(vals)); p[1:2] * 100
}

pcoa_plot <- function(D, lab_factor, png_file, title_txt, palette_named) {
  if (is.null(D) || length(D) == 0) { save_placeholder_png(png_file); return(invisible(NULL)) }
  ord  <- cmdscale(D, eig = TRUE, k = 2)
  labs <- labels(D)
  df <- tibble(Axis1 = ord$points[,1], Axis2 = ord$points[,2],
               Group = factor(lab_factor[labs], levels = names(palette_named)))
  pp <- pct_from_cmdscale(ord)
  lx <- ifelse(is.na(pp[1]), "PCoA 1", sprintf("PCoA 1 (%.1f%%)", pp[1]))
  ly <- ifelse(is.na(pp[2]), "PCoA 2", sprintf("PCoA 2 (%.1f%%)", pp[2]))
  pal_use <- palette_named[names(palette_named) %in% levels(df$Group)]
  g <- ggplot(df, aes(Axis1, Axis2, color = Group, fill = Group)) +
    geom_point(size = 2, alpha = 0.9) +
    { if (nrow(df) >= 6 && dplyr::n_distinct(df$Group) >= 2) stat_ellipse(type="norm", linewidth=0.6, alpha=0.12) else NULL } +
    scale_color_manual(values = pal_use, drop = FALSE) +
    scale_fill_manual(values = pal_use, drop = FALSE) +
    theme_bw(base_size = 12) + coord_equal() +
    labs(title = title_txt, x = lx, y = ly)
  ggsave(png_file, g, width = 6, height = 5, dpi = 300, bg = "white")
  invisible(NULL)
}

pcoa_plot_paired_lines <- function(D, pairs_df, png_file, title_txt, color_oral="#B499E5", color_fecal="#78688E") {
  if (is.null(D) || length(D) == 0 || nrow(pairs_df) == 0) { save_placeholder_png(png_file); return(invisible(NULL)) }
  ord  <- cmdscale(D, eig = TRUE, k = 2)
  labs <- labels(D)
  df <- tibble(Sample_ID = labs, Axis1 = ord$points[,1], Axis2 = ord$points[,2])

  # join oral/fecal positions -- (✨ اینجا قبلاً یک ) اضافه بود)
  seg_df <- pairs_df %>%
    inner_join(df, by = c("oral" = "Sample_ID"))  %>%
    rename(o1 = Axis1, o2 = Axis2) %>%
    inner_join(df, by = c("fecal" = "Sample_ID")) %>%
    rename(f1 = Axis1, f2 = Axis2)

  g <- ggplot() +
    geom_segment(data = seg_df, aes(x = o1, y = o2, xend = f1, yend = f2),
                 color = "grey60", linewidth = 0.6, alpha = 0.7) +
    geom_point(data = df %>% filter(Sample_ID %in% pairs_df$oral),
               aes(Axis1, Axis2), color = color_oral, size = 2.2) +
    geom_point(data = df %>% filter(Sample_ID %in% pairs_df$fecal),
               aes(Axis1, Axis2), color = color_fecal, size = 2.2) +
    theme_bw(base_size = 12) + coord_equal() +
    labs(title = title_txt, x = "PCoA 1", y = "PCoA 2")
  ggsave(png_file, g, width = 6, height = 5, dpi = 300, bg = "white")
  invisible(NULL)
}

# ---------------------------- Covariates & pairs ------------------------
COV_CANDS <- c("Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing")

read_covariates_robust <- function(path) {
  cv <- safe_read_csv(path)
  if (nrow(cv) == 0) return(tibble())
  # flexible renaming
  id_col  <- intersect(c("Sample_ID","SampleID","sample_id","Sample","sample","ID","id","Sample.Name","SampleName"), names(cv))[1]
  site_col<- intersect(c("site","Site","type","Type","SITE"), names(cv))[1]
  dis_col <- intersect(c("disease","Disease","status","Status","phenotype","Phenotype","group","Group","cohort","Cohort","label","Label"), names(cv))[1]

  map_site <- function(x) {
    y <- tolower(trimws(as.character(x)))
    case_when(
      str_detect(y, "oral|mouth|saliva|buccal") ~ "oral",
      str_detect(y, "fecal|faecal|stool") ~ "fecal",
      TRUE ~ NA_character_
    )
  }
  map_dis <- function(x) {
    y <- tolower(trimws(as.character(x)))
    case_when(
      y %in% c("1","yes","true","crohn","cd","case","patient","ibd","disease","crohn-oral","crohn-fecal") ~ 1L,
      y %in% c("0","no","false","healthy","control","healthy-oral","healthy-fecal") ~ 0L,
      str_detect(y, "crohn|cd|case|patient") ~ 1L,
      str_detect(y, "healthy|control") ~ 0L,
      TRUE ~ NA_integer_
    )
  }

  md <- tibble(
    Sample_ID = if (!is.na(id_col)) as.character(cv[[id_col]]) else NA_character_,
    site      = if (!is.na(site_col)) map_site(cv[[site_col]]) else NA_character_,
    disease   = if (!is.na(dis_col))  map_dis(cv[[dis_col]])  else NA_integer_
  )
  md <- bind_cols(md, cv %>% select(any_of(COV_CANDS)))
  md <- md %>% mutate(Sample_ID = vapply(Sample_ID, normalize_id, character(1))) %>% distinct()
  md
}

read_pairs <- function(path) {
  P <- safe_read_csv(path)
  if (nrow(P) == 0) return(tibble(STUDY_ID=character(), oral=character(), fecal=character()))
  nm <- names(P)
  pick <- function(cands) { cand <- intersect(cands, nm); if (length(cand)) cand[[1]] else NA_character_ }
  sid <- pick(c("STUDY_ID","Study_ID","subject","Subject","ID","id"))
  oc  <- pick(c("oral","Oral","Oral_ID","oral_id","Oral_col","Oral.col","Oral_Sample_ID"))
  fc  <- pick(c("fecal","Fecal","Fecal_ID","fecal_id","Fecal_col","Fecal.col","Fecal_Sample_ID"))
  if (is.na(oc) || is.na(fc)) return(tibble(STUDY_ID=character(), oral=character(), fecal=character()))
  tibble(
    STUDY_ID = if (!is.na(sid)) as.character(P[[sid]]) else NA_character_,
    oral     = vapply(as.character(P[[oc]]), normalize_id, character(1)),
    fecal    = vapply(as.character(P[[fc]]), normalize_id, character(1))
  ) %>% filter(!is.na(oral), !is.na(fecal))
}

# ---------------------------- PERMANOVA helpers -------------------------
permanova_with_cov <- function(M_case, M_ctrl, site_name, covars, out_csv, permutations=999) {
  if (is.null(M_case) || is.null(M_ctrl)) { write_note_csv(out_csv, "no_matrix"); return(invisible(NULL)) }
  lst <- align_features(list(M_case, M_ctrl))
  if (length(lst) < 2) { write_note_csv(out_csv, "no_common_features"); return(invisible(NULL)) }
  M_case <- lst[[1]]; M_ctrl <- lst[[2]]
  M <- cbind(M_case, M_ctrl)
  labs <- colnames(M)
  md <- covars %>% filter(tolower(site) == tolower(site_name)) %>% filter(Sample_ID %in% labs)
  if (nrow(md) < 3 || dplyr::n_distinct(md$disease) < 2) { write_note_csv(out_csv, "insufficient_md"); return(invisible(NULL)) }

  # Bray
  Mb <- drop_bad_for_dist(M[, md$Sample_ID, drop=FALSE], "bray")
  if (is.null(Mb) || ncol(Mb) < 3) { write_note_csv(out_csv, "too_few_columns"); return(invisible(NULL)) }
  Db <- mk_dist(Mb, "bray")$D
  if (is.null(Db)) { write_note_csv(out_csv, "dist_null"); return(invisible(NULL)) }

  labs_b <- labels(Db); md_b <- md[match(labs_b, md$Sample_ID), , drop = FALSE]

  # choose covariates that are present & variable
  usable <- c()
  for (v in intersect(COV_CANDS, names(md_b))) {
    vv <- md_b[[v]]
    if (sum(!is.na(vv)) >= 3 && dplyr::n_distinct(vv, na.rm=TRUE) >= 2) usable <- c(usable, v)
  }
  rhs <- paste(c("disease", usable), collapse = " + ")
  fml <- stats::as.formula(paste("Db ~", rhs))
  res <- vegan::adonis2(fml, data = md_b, permutations = permutations, by = "margin")
  out <- as.data.frame(res); out$term <- rownames(out); rownames(out) <- NULL
  out <- out %>% rename(Df=Df, SumOfSqs=SumOfSqs, R2=R2, F=`F`, p=`Pr(>F)`) %>% select(term, Df, SumOfSqs, R2, F, p)
  write_csv(out, out_csv)
}

permanova_interaction_bray <- function(M_oral, M_fecal, covars, out_csv, permutations=999) {
  if (is.null(M_oral) || is.null(M_fecal)) { write_note_csv(out_csv, "no_matrix"); return(invisible(NULL)) }
  lst <- align_features(list(M_oral, M_fecal))
  if (length(lst) < 2) { write_note_csv(out_csv, "no_common_features"); return(invisible(NULL)) }
  M_oral <- lst[[1]]; M_fecal <- lst[[2]]
  M <- cbind(M_oral, M_fecal)
  labs <- colnames(M)

  md <- covars %>% filter(Sample_ID %in% labs) %>%
    mutate(site = ifelse(tolower(site)=="oral","oral", ifelse(tolower(site)=="fecal","fecal", NA_character_))) %>%
    filter(!is.na(site))
  if (nrow(md) < 6 || dplyr::n_distinct(md$disease) < 2 || dplyr::n_distinct(md$site) < 2) {
    write_note_csv(out_csv, "insufficient_md"); return(invisible(NULL))
  }

  Mb <- drop_bad_for_dist(M[, md$Sample_ID, drop=FALSE], "bray")
  if (is.null(Mb) || ncol(Mb) < 3) { write_note_csv(out_csv, "too_few_columns"); return(invisible(NULL)) }
  Db <- mk_dist(Mb, "bray")$D
  if (is.null(Db)) { write_note_csv(out_csv, "dist_null"); return(invisible(NULL)) }

  labs_b <- labels(Db); md_b <- md[match(labs_b, md$Sample_ID), , drop = FALSE]
  # covariates
  usable <- c()
  for (v in intersect(COV_CANDS, names(md_b))) {
    vv <- md_b[[v]]
    if (sum(!is.na(vv)) >= 3 && dplyr::n_distinct(vv, na.rm=TRUE) >= 2) usable <- c(usable, v)
  }
  rhs <- paste(c("disease * site", usable), collapse = " + ")
  fml <- stats::as.formula(paste("Db ~", rhs))
  res <- vegan::adonis2(fml, data = md_b, permutations = permutations, by = "margin")
  out <- as.data.frame(res); out$term <- rownames(out); rownames(out) <- NULL
  out <- out %>% rename(Df=Df, SumOfSqs=SumOfSqs, R2=R2, F=`F`, p=`Pr(>F)`) %>% select(term, Df, SumOfSqs, R2, F, p)
  write_csv(out, out_csv)
}

permdisp_bray <- function(M_case, M_ctrl, covars, site_name, out_csv) {
  if (is.null(M_case) || is.null(M_ctrl)) { write_note_csv(out_csv, "no_matrix"); return(invisible(NULL)) }
  lst <- align_features(list(M_case, M_ctrl))
  if (length(lst) < 2) { write_note_csv(out_csv, "no_common_features"); return(invisible(NULL)) }
  M_case <- lst[[1]]; M_ctrl <- lst[[2]]
  M <- cbind(M_case, M_ctrl)
  labs <- colnames(M)
  md <- covars %>% filter(tolower(site) == tolower(site_name), Sample_ID %in% labs)
  if (nrow(md) < 3 || dplyr::n_distinct(md$disease) < 2) { write_note_csv(out_csv, "insufficient_md"); return(invisible(NULL)) }
  Mb <- drop_bad_for_dist(M[, md$Sample_ID, drop=FALSE], "bray")
  if (is.null(Mb) || ncol(Mb) < 3) { write_note_csv(out_csv, "too_few_columns"); return(invisible(NULL)) }
  Db <- mk_dist(Mb, "bray")$D
  if (is.null(Db)) { write_note_csv(out_csv, "dist_null"); return(invisible(NULL)) }
  labs_b <- labels(Db); md_b <- md[match(labs_b, md$Sample_ID), , drop=FALSE]
  grp <- factor(md_b$disease, levels=c(0,1), labels=c("Healthy","Crohn"))
  bd <- betadisper(Db, grp)
  pt <- suppressWarnings(permutest(bd, permutations = nperm))
  out <- tibble(term = "disease", F = as.numeric(pt$tab[1, "F"]), p = as.numeric(pt$tab[1, "Pr(>F)"]))
  write_csv(out, out_csv)
}

# ---------------------------- Group distance table ----------------------
write_group_distance_csv <- function(M_all, group_map_named, out_csv) {
  if (is.null(M_all) || ncol(M_all) < 4) { write_note_csv(out_csv, "too_few_samples"); return(invisible(NULL)) }
  dist_one <- function(method) {
    Mx <- drop_bad_for_dist(M_all, method); Dx <- mk_dist(Mx, method)
    if (is.null(Dx$D)) return(tibble())
    dm <- as.matrix(Dx$D); labs <- Dx$labs; g <- group_map_named[labs]; lev <- unique(na.omit(g))
    out <- list()
    for (i in seq_along(lev)) for (j in i:length(lev)) {
      gi <- lev[i]; gj <- lev[j]; idx <- which(g == gi); jdx <- which(g == gj)
      if (length(idx) > 0 && length(jdx) > 0) {
        sub <- dm[idx, jdx, drop = FALSE]; if (i == j) sub <- sub[upper.tri(sub)]
        out[[length(out) + 1]] <- tibble(group1 = gi, group2 = gj, metric = method,
                                         n_pairs = length(sub),
                                         mean_distance = ifelse(length(sub) > 0, mean(sub), NA_real_))
      }
    }
    bind_rows(out)
  }
  out <- bind_rows(dist_one("bray"), dist_one("jaccard"))
  if (!nrow(out)) write_note_csv(out_csv, "no_pairs") else write_csv(out, out_csv)
}

# ---------------------------- Main workflow ----------------------------
main <- function() {
  # Read abundances
  OC <- read_abund_matrix(oral_crohn_path, pseudo)
  FC <- read_abund_matrix(fecal_crohn_path, pseudo)
  OH <- read_abund_matrix(oral_healthy_path, pseudo)
  FH <- read_abund_matrix(fecal_healthy_path, pseudo)

  # Align features across all (intersection to keep distance definitions consistent)
  L <- align_features(list(OC, FC, OH, FH))
  if (length(L) == 4) { OC <- L[[1]]; FC <- L[[2]]; OH <- L[[3]]; FH <- L[[4]] }

  # Labels for groups
  lab_oral_c    <- setNames(rep("Crohn-Oral",    ncol(OC)), colnames(OC))
  lab_fecal_c   <- setNames(rep("Crohn-Fecal",   ncol(FC)), colnames(FC))
  lab_oral_h    <- setNames(rep("Healthy-Oral",  ncol(OH)), colnames(OH))
  lab_fecal_h   <- setNames(rep("Healthy-Fecal", ncol(FH)), colnames(FH))
  group_map_named <- c(lab_oral_c, lab_fecal_c, lab_oral_h, lab_fecal_h)

  # All-groups matrices
  M_all <- cbind(OC, FC, OH, FH)

  # --- PCoA: all groups ---
  Db_all <- mk_dist(drop_bad_for_dist(M_all, "bray"),    "bray")$D
  Dj_all <- mk_dist(drop_bad_for_dist(M_all, "jaccard"), "jaccard")$D
  Da_all <- mk_aitchison(drop_bad_for_dist(M_all, "bray"), pseudo)$D  # using same filtered columns
  pcoa_plot(Db_all, group_map_named, file.path(outdir, "pcoa_bray_allgroups.png"),
            "PCoA (Bray) — All Groups", pal_named)
  pcoa_plot(Dj_all, group_map_named, file.path(outdir, "pcoa_jaccard_allgroups.png"),
            "PCoA (Jaccard) — All Groups", pal_named)
  pcoa_plot(Da_all, group_map_named, file.path(outdir, "pcoa_aitchison_allgroups.png"),
            "PCoA (Aitchison) — All Groups", pal_named)

  # --- PCoA: Crohn vs Healthy within site (Bray only, as required) ---
  Mb_oral  <- cbind(OC, OH); Db_oral  <- mk_dist(drop_bad_for_dist(Mb_oral,  "bray"), "bray")$D
  Mb_fecal <- cbind(FC, FH); Db_fecal <- mk_dist(drop_bad_for_dist(Mb_fecal, "bray"), "bray")$D
  pcoa_plot(Db_oral,  c(lab_oral_c, lab_oral_h),  file.path(outdir, "pcoa_bray_oral_CH.png"),
            "PCoA (Bray) — Crohn Oral vs Healthy Oral", pal_named[c("Crohn-Oral","Healthy-Oral")])
  pcoa_plot(Db_fecal, c(lab_fecal_c, lab_fecal_h), file.path(outdir, "pcoa_bray_fecal_CH.png"),
            "PCoA (Bray) — Crohn Fecal vs Healthy Fecal", pal_named[c("Crohn-Fecal","Healthy-Fecal")])

  # --- Paired Crohn oral–fecal (Bray, with lines) ---
  pairs_df <- read_pairs(pairs_path)
  M_crohn_of <- cbind(OC, FC)
  Db_pairs <- mk_dist(drop_bad_for_dist(M_crohn_of, "bray"), "bray")$D
  pcoa_plot_paired_lines(Db_pairs, pairs_df,
                         file.path(outdir, "pcoa_bray_oral_vs_fecal_paired.png"),
                         "PCoA (Bray) — Crohn Oral vs Fecal (paired)",
                         color_oral = pal_named[["Crohn-Oral"]],
                         color_fecal= pal_named[["Crohn-Fecal"]])

  # --- Covariates table ---
  covars <- read_covariates_robust(covars_path)

  # --- PERMANOVA with covariates (Bray) ---
  permanova_with_cov(OC, OH, "oral",  covars, file.path(outdir, "permanova_oral_bray_with_cov.csv"),  permutations = nperm)
  permanova_with_cov(FC, FH, "fecal", covars, file.path(outdir, "permanova_fecal_bray_with_cov.csv"), permutations = nperm)

  # --- PERMANOVA interaction (disease * site) on Bray ---
  permanova_interaction_bray(cbind(OC, OH), cbind(FC, FH), covars, file.path(outdir, "permanova_interaction_bray.csv"), permutations = nperm)

  # --- PERMDISP (Bray) ---
  permdisp_bray(OC, OH, covars, "oral",  file.path(outdir, "permdisp_oral_bray.csv"))
  permdisp_bray(FC, FH, covars, "fecal", file.path(outdir, "permdisp_fecal_bray.csv"))

  # --- Group distance table (Bray & Jaccard means within/between groups) ---
  write_group_distance_csv(M_all, group_map_named, file.path(outdir, "beta_group_distances.csv"))

  # --- Paired distances summary (Crohn oral–fecal) ---
  if (nrow(pairs_df) > 0) {
    lst <- align_features(list(OC, FC)); OC2 <- lst[[1]]; FC2 <- lst[[2]]
    P <- pairs_df %>% filter(oral %in% colnames(OC2), fecal %in% colnames(FC2))
    if (nrow(P) > 0) {
      Mb <- cbind(OC2[, P$oral, drop=FALSE], FC2[, P$fecal, drop=FALSE])
      Mb <- drop_bad_for_dist(Mb, "bray"); Db <- mk_dist(Mb, "bray")$D
      if (!is.null(Db)) {
        dm <- as.matrix(Db); labs <- labels(Db)
        get_one <- function(o, f) if (o %in% labs && f %in% labs) dm[o, f] else NA_real_
        dists <- mapply(get_one, P$oral, P$fecal)
        out <- tibble(Metric="BrayCurtis", n= sum(is.finite(dists)), mean = mean(dists, na.rm=TRUE),
                      median = median(dists, na.rm=TRUE),
                      q1 = as.numeric(quantile(dists, 0.25, na.rm=TRUE)),
                      q3 = as.numeric(quantile(dists, 0.75, na.rm=TRUE)),
                      iqr = IQR(dists, na.rm=TRUE))
        write_csv(out, file.path(outdir, "pairwise_oral_fecal_summary.csv"))
      } else {
        write_note_csv(file.path(outdir, "pairwise_oral_fecal_summary.csv"), "dist_null")
      }
    } else write_note_csv(file.path(outdir, "pairwise_oral_fecal_summary.csv"), "no_pairs_overlap")
  } else write_note_csv(file.path(outdir, "pairwise_oral_fecal_summary.csv"), "no_pairs_file")

  # Debug log
  sink(file.path(dbgdir, "beta_session_info.txt")); print(sessionInfo()); sink()
  writeLines(c("OK", format(Sys.time())), con = file.path(dbgdir, "done.flag"))
}

invisible(main())
