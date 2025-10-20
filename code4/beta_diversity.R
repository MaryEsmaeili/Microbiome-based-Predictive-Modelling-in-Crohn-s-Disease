#!/usr/bin/env Rscript
# =============================================================================
# Beta Diversity (Bray–Curtis / Jaccard / Aitchison) — principled covariates
#
# What this script does
# ---------------------
# • Reads four abundance tables (oral/fecal × Crohn/Healthy) + metadata/pairs/colors.
# • Aligns features by intersection; drops zero-sum samples.
# • Computes distances:
#     - Bray–Curtis: quantitative counts; NO global pseudocount; zeros preserved.
#     - Jaccard: presence/absence via vegan::vegdist(..., binary=TRUE); NO pseudocount.
#     - Aitchison: per-sample proportions → CLR with internal pseudocount (default 1e-6)
#                  → Euclidean.
# • PCoA (cmdscale) for each distance; Axis1/Axis2 with % variance explained.
# • Inference:
#     - PERMANOVA (adonis2, 999 perms; set.seed(42)).
#     - Site-specific (oral/fecal): D ~ disease [± minimal covariates].
#     - Pooled interaction: D ~ disease * site [± minimal covariates].
#     - If STUDY_ID present: permutations are blocked (paired design respected).
#     - PERMDISP on Bray per site (dispersion check).
# • Outputs plots + CSVs + a debug flag.
#
# Key guardrails
# --------------
# 1) No global pseudocount to the raw abundance matrices (keeps Jaccard/Bray clean).
# 2) CLR/Aitchison applies an internal pseudocount ONLY inside that path.
# 3) Jaccard computed with binary=TRUE.
# 4) CLR centers by COLUMN-wise log-mean (per-sample geometric mean), not row-wise.
# 5) All inferential runs report effective N and covariates used.
# =============================================================================

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(ggplot2)
  library(vegan); library(stringr); library(purrr); library(yaml); library(jsonlite)
})

# ---------- Helpers: coercion & IDs ----------
norm_sex <- function(x) {
  y <- tolower(trimws(as.character(x)))
  ifelse(y %in% c("m","male","1"), 1L,
         ifelse(y %in% c("f","female","0"), 0L, NA_integer_))
}
to_bin01 <- function(x) {
  y <- tolower(trimws(as.character(x)))
  dplyr::case_when(
    y %in% c("1","yes","y","true","t","on","present","current","pos","+") ~ 1L,
    y %in% c("0","no","n","false","f","off","absent","none","neg","-")      ~ 0L,
    suppressWarnings(!is.na(as.numeric(y)) & as.numeric(y) %in% c(0,1))     ~ as.integer(as.numeric(y)),
    TRUE ~ NA_integer_
  )
}
to_num <- function(x) suppressWarnings(as.numeric(as.character(x)))

normalize_id <- function(x) {
  if (is.na(x)) return(NA_character_)
  s <- toupper(trimws(as.character(x)))
  s <- gsub("\\.\\d+$", "", s)      # drop .1, .2 ...
  s <- sub("^S", "", s)             # drop leading S
  s <- gsub("[^0-9]", "", s)        # keep digits only
  s <- sub("^0+", "", s)            # drop leading zeros
  ifelse(nzchar(s), s, "0")
}

`%||%` <- function(a,b) if (is.null(a)) b else if (is.atomic(a) && length(a)==1 && is.character(a) && !nzchar(a)) b else a
or_null <- function(a,b) if (is.null(a)) b else a

# ---------------------------- CLI ----------------------------
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
pseudo             <- as.numeric(get_arg("--pseudocount", "1e-6"))   # used ONLY in CLR
nperm              <- as.integer(get_arg("--permutations", "999"))   # keep default=999

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dbgdir <- file.path(outdir, "debug"); dir.create(dbgdir, showWarnings = FALSE)

# --------------------- Colors ---------------------
load_group_colors <- function(yaml_path) {
  defaults <- c(
    "Crohn-Oral"    = "#edae49",
    "Healthy-Oral"  = "#00798c",
    "Crohn-Fecal"   = "#30638e",
    "Healthy-Fecal" = "#d1495b"
  )
  if (is.null(yaml_path) || !file.exists(yaml_path)) return(defaults)
  cfg <- tryCatch(yaml::read_yaml(yaml_path), error = function(e) NULL)
  if (is.null(cfg)) return(defaults)

  if (!is.null(cfg$colors) && is.list(cfg$colors)) {
    col <- cfg$colors
    get1 <- function(name, def) {
      v <- tryCatch(col[[name]], error = function(e) NULL)
      v <- if (is.null(v)) def else as.character(v)[1]
      if (!nzchar(v)) def else v
    }
    return(c(
      "Crohn-Oral"    = get1("Crohn-Oral",    defaults[["Crohn-Oral"]]),
      "Healthy-Oral"  = get1("Healthy-Oral",  defaults[["Healthy-Oral"]]),
      "Crohn-Fecal"   = get1("Crohn-Fecal",   defaults[["Crohn-Fecal"]]),
      "Healthy-Fecal" = get1("Healthy-Fecal", defaults[["Healthy-Fecal"]])
    ))
  }
  defaults
}
pal_named <- load_group_colors(colors_path)
write_json(as.list(pal_named), file.path(outdir, "colors_used.json"), pretty = TRUE, auto_unbox = TRUE)

# ------------------ Abundance I/O ------------------
safe_read_csv <- function(p) {
  if (is.null(p) || !file.exists(p)) return(tibble())
  suppressMessages(readr::read_csv(p, show_col_types = FALSE))
}

# IMPORTANT:
# - No global pseudocount on raw matrices (preserve zeros for Jaccard; keep Bray clean).
# - Accept "..." so the function stays compatible if caller passes extra args (e.g., pseudo).
read_abund_matrix <- function(path, ...) {
  df <- safe_read_csv(path)
  if (nrow(df) == 0) return(NULL)
  rn <- df[[1]]
  M  <- as.matrix(df[,-1, drop = FALSE]); storage.mode(M) <- "numeric"
  rownames(M) <- make.unique(as.character(rn))
  colnames(M) <- make.unique(vapply(colnames(df)[-1], normalize_id, character(1)))
  # Drop all-zero rows/cols; keep real zeros.
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

# For a given method, drop samples that would yield undefined distances.
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

# CLR on per-sample proportions; centers by column log-mean (geometric mean per sample).
clr_matrix <- function(X, pseudo = 1e-6) {
  X <- as.matrix(X)
  X <- sweep(X, 2, colSums(X), "/")  # to proportions per sample
  X[X <= 0] <- pseudo                # guard zeros
  L <- log(X)
  gm <- matrix(colMeans(L), nrow = nrow(L), ncol = ncol(L), byrow = TRUE)
  L - gm
}

# Distance constructors --------------------------------------------------------

# Jaccard needs binary=TRUE; Bray uses raw counts; all others default.
mk_dist <- function(M, method) {
  if (is.null(M) || ncol(M) < 3) return(list(D = NULL, labs = character()))
  if (method == "jaccard") {
    D <- suppressWarnings(vegan::vegdist(t(M), method = "jaccard", binary = TRUE))
  } else {
    D <- suppressWarnings(vegan::vegdist(t(M), method = method))
  }
  dm <- as.matrix(D)
  good <- rowSums(is.na(dm)) == 0
  if (!any(good)) return(list(D = NULL, labs = character()))
  list(D = stats::as.dist(dm[good, good, drop = FALSE]), labs = colnames(M)[good])
}

mk_aitchison <- function(M, pseudo = 1e-6) {
  if (is.null(M) || ncol(M) < 3) return(list(D = NULL, labs = character()))
  Xclr <- clr_matrix(M, pseudo = pseudo)
  # Drop samples with any non-finite after CLR
  bad <- apply(Xclr, 2, function(v) any(!is.finite(v)))
  if (any(bad)) Xclr <- Xclr[, !bad, drop = FALSE]
  if (ncol(Xclr) < 3) return(list(D = NULL, labs = character()))
  list(D = stats::dist(t(Xclr), method = "euclidean"), labs = colnames(Xclr))
}

# --------------- Covariates & pairs (robust I/O) ------------------------------
COV_READ <- c(
  "Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use",
  "STUDY_ID","Study_ID","subject","Subject","ID","id"
)
COV_PLAN_MIN <- c("Age","Sex","BMI","Smoking")

read_covariates_robust <- function(path) {
  cv <- safe_read_csv(path)
  if (nrow(cv) == 0) return(tibble())

  pick_col <- function(nm, cands) { x <- intersect(cands, nm); if (length(x)) x[[1]] else NA_character_ }
  id_col   <- pick_col(names(cv), c("Sample_ID","SampleID","sample_id","Sample","sample","ID","id"))
  site_col <- pick_col(names(cv), c("site","Site","type","Type","SITE"))
  dis_col  <- pick_col(names(cv), c("disease","Disease","status","Status","phenotype","Phenotype","group","Group","label","Label"))

  map_site <- function(x) {
    y <- tolower(trimws(as.character(x)))
    out <- rep(NA_character_, length(y))
    v <- suppressWarnings(as.integer(y))
    out[!is.na(v) & v == 1L] <- "oral"
    out[!is.na(v) & v == 0L] <- "fecal"
    is_str <- is.na(v)
    out[is_str & stringr::str_detect(y, "\\boral\\b|mouth|saliva|buccal|\\boc\\b")] <- "oral"
    out[is_str & stringr::str_detect(y, "fecal|faecal|stool|\\bfc\\b")]               <- "fecal"
    out
  }
  map_dis <- function(x) {
    y <- tolower(trimws(as.character(x)))
    dplyr::case_when(
      y %in% c("1","yes","true","crohn","cd","case","patient","ibd","disease") ~ 1L,
      y %in% c("0","no","false","healthy","control","hc") ~ 0L,
      stringr::str_detect(y, "crohn|\\bcd\\b|case|patient|ibd") ~ 1L,
      stringr::str_detect(y, "healthy|control|\\bhc\\b") ~ 0L,
      TRUE ~ NA_integer_
    )
  }

  md <- tibble(
    Sample_ID = if (!is.na(id_col)) as.character(cv[[id_col]]) else NA_character_,
    site      = if (!is.na(site_col)) map_site(cv[[site_col]]) else NA_character_,
    disease   = if (!is.na(dis_col))  map_dis(cv[[dis_col]])  else NA_integer_
  )

  md <- dplyr::bind_cols(md, cv %>% dplyr::select(dplyr::any_of(COV_READ))) %>%
    dplyr::mutate(Sample_ID = vapply(Sample_ID, normalize_id, character(1))) %>%
    dplyr::distinct()

  # coerce types
  if ("Sex" %in% names(md)) md$Sex <- norm_sex(md$Sex)
  for (b in intersect(c("Smoking","Antibiotics_3m","PPI_use"), names(md))) md[[b]] <- to_bin01(md[[b]])
  for (v in intersect(c("Age","BMI"), names(md))) md[[v]] <- to_num(md[[v]])

  # unify STUDY_ID if present under various names
  sid_col <- intersect(c("STUDY_ID","Study_ID","subject","Subject","ID","id"), names(md))
  md$STUDY_ID <- if (length(sid_col)) as.character(md[[sid_col[1]]]) else NA_character_
  md
}

read_pairs <- function(path) {
  P <- safe_read_csv(path)
  if (nrow(P) == 0) return(tibble(STUDY_ID=character(), oral=character(), fecal=character()))
  nm <- names(P)
  pick <- function(cands) { cand <- intersect(cands, nm); if (length(cand)) cand[[1]] else NA_character_ }
  sid <- pick(c("STUDY_ID","Study_ID","subject","Subject","ID","id"))
  oc  <- pick(c("oral","Oral","Oral_ID","oral_id","Oral_Sample_ID","OC"))
  fc  <- pick(c("fecal","Fecal","Fecal_ID","fecal_id","Fecal_Sample_ID","stool","FC"))
  if (is.na(oc) || is.na(fc)) return(tibble(STUDY_ID=character(), oral=character(), fecal=character()))
  tibble(
    STUDY_ID = if (!is.na(sid)) as.character(P[[sid]]) else NA_character_,
    oral     = vapply(as.character(P[[oc]]), normalize_id, character(1)),
    fecal    = vapply(as.character(P[[fc]]), normalize_id, character(1))
  ) %>% filter(!is.na(oral), !is.na(fecal))
}

# ------------------------------ Plotting ------------------------------
pct_from_cmdscale <- function(ord) {
  if (is.null(ord$eig) || all(is.na(ord$eig))) return(c(NA, NA))
  vals <- ord$eig; keep <- which(vals > 0); if (!length(keep)) keep <- seq_along(vals)
  p <- vals / sum(abs(vals)); p[1:2] * 100
}

# Legend labels with counts e.g. "Crohn-Oral (n=41)"
legend_labels_with_n <- function(group_vec, palette_named) {
  lv <- unique(na.omit(group_vec))
  lab <- setNames(character(length(lv)), lv)
  for (g in lv) lab[[g]] <- sprintf("%s (n=%d)", g, sum(group_vec == g, na.rm = TRUE))
  list(levels=lv, labels=lab[lv], palette=palette_named[names(palette_named) %in% lv])
}

pcoa_plot <- function(D, group_map_named, png_file, title_txt, palette_named) {
  if (is.null(D) || length(D) == 0) {
    g <- ggplot() + theme_void() + annotate("text", 0, 0, label = "No distance (too few samples)", size = 5)
    ggsave(png_file, g, width = 6.4, height = 5.2, dpi = 300, bg = "white"); return(invisible(NULL))
  }
  ord  <- cmdscale(D, eig = TRUE, k = 2)
  labs <- labels(D)
  grp  <- factor(unname(group_map_named[labs]))
  labinfo <- legend_labels_with_n(grp, palette_named)

  df <- tibble(Axis1 = ord$points[,1], Axis2 = ord$points[,2], Group = factor(grp, levels = labinfo$levels))
  pp <- pct_from_cmdscale(ord)
  lx <- ifelse(is.na(pp[1]), "PCoA 1", sprintf("PCoA 1 (%.1f%%)", pp[1]))
  ly <- ifelse(is.na(pp[2]), "PCoA 2", sprintf("PCoA 2 (%.1f%%)", pp[2]))

  g <- ggplot(df, aes(Axis1, Axis2, color = Group, fill = Group)) +
    geom_point(size = 2.3, alpha = 0.9, shape = 21, stroke = 0.2) +
    { if (nrow(df) >= 6 && dplyr::n_distinct(df$Group) >= 2) stat_ellipse(type="norm", linewidth=0.6, alpha=0.12) else NULL } +
    scale_color_manual(values = labinfo$palette, breaks = labinfo$levels, labels = unname(unlist(labinfo$labels)), drop = FALSE) +
    scale_fill_manual(values  = labinfo$palette, breaks = labinfo$levels, labels = unname(unlist(labinfo$labels)), drop = FALSE) +
    theme_bw(base_size = 12) + coord_equal() +
    labs(title = title_txt, x = lx, y = ly)
  ggsave(png_file, g, width = 6.4, height = 5.2, dpi = 300, bg = "white")
  invisible(NULL)
}

pcoa_plot_paired_lines <- function(D, pairs_df, png_file, title_txt,
                                   color_oral="#B499E5", color_fecal="#78688E") {
  if (is.null(D) || length(D) == 0 || nrow(pairs_df) == 0) {
    g <- ggplot() + theme_void() + annotate("text", 0, 0, label = "No paired data", size = 5)
    ggsave(png_file, g, width = 6.4, height = 5.2, dpi = 300, bg = "white"); return(invisible(NULL))
  }
  ord  <- cmdscale(D, eig = TRUE, k = 2)
  labs <- labels(D)
  df <- tibble(Sample_ID = labs, Axis1 = ord$points[,1], Axis2 = ord$points[,2])
  seg_df <- pairs_df %>%
    inner_join(df, by = c("oral" = "Sample_ID"))  %>%
    rename(o1 = Axis1, o2 = Axis2) %>%
    inner_join(df, by = c("fecal" = "Sample_ID")) %>%
    rename(f1 = Axis1, f2 = Axis2)
  g <- ggplot() +
    geom_segment(data = seg_df, aes(x = o1, y = o2, xend = f1, yend = f2),
                 color = "grey60", linewidth = 0.6, alpha = 0.7) +
    geom_point(data = df %>% filter(Sample_ID %in% pairs_df$oral),
               aes(Axis1, Axis2), color = color_oral, size = 2.3) +
    geom_point(data = df %>% filter(Sample_ID %in% pairs_df$fecal),
               aes(Axis1, Axis2), color = color_fecal, size = 2.3) +
    theme_bw(base_size = 12) + coord_equal() +
    labs(title = title_txt, x = "PCoA 1", y = "PCoA 2")
  ggsave(png_file, g, width = 6.4, height = 5.2, dpi = 300, bg = "white")
  invisible(NULL)
}

# ---------------------------- Model utilities ----------------------------
write_note_csv   <- function(path, note="no_data")   { dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE); write_csv(tibble(note = note), path) }
write_counts_csv <- function(path, site, n_total, n_h, n_c, note="ok") { write_csv(tibble(site=site, n_total=n_total, n_healthy=n_h, n_crohn=n_c, note=note), path) }
count_hc <- function(vec) { c(h=sum(vec==0,na.rm=TRUE), c=sum(vec==1,na.rm=TRUE)) }

build_fallback_md <- function(lst_mats, labs, site_name=NULL) {
  k1 <- if (!is.null(lst_mats[[1]])) ncol(lst_mats[[1]]) else 0L  # Crohn first
  k2 <- if (!is.null(lst_mats[[2]])) ncol(lst_mats[[2]]) else 0L  # Healthy next
  tibble(Sample_ID = labs, site = site_name %||% NA_character_,
         disease   = c(rep(1L, k1), rep(0L, k2)))
}

select_minimal_covars <- function(md_df) {
  cand <- intersect(COV_PLAN_MIN, names(md_df))
  usable <- c()
  for (v in cand) {
    vv <- md_df[[v]]
    if (sum(!is.na(vv)) >= 3 && dplyr::n_distinct(vv, na.rm=TRUE) >= 2) usable <- c(usable, v)
  }
  usable
}

# Scale only continuous covariates that are actually used
scale_continuous_if_used <- function(md_df, usable_names) {
  out <- md_df
  for (v in intersect(c("Age","BMI"), usable_names)) {
    if (v %in% names(out)) {
      vv <- suppressWarnings(as.numeric(out[[v]]))
      if (sum(is.finite(vv)) >= 3 && stats::sd(vv, na.rm=TRUE) > 0) out[[v]] <- as.numeric(scale(vv))
    }
  }
  out
}

# adonis2 helper with optional strata/blocks (paired permutations)
adonis2_with_optional_strata <- function(D, formula, data, nperm, strata_vec=NULL) {
  set.seed(42)
  Dx <- D  # ensure symbol used in formula exists in frame
  if (!is.null(strata_vec) && all(!is.na(strata_vec))) {
    ctrl <- vegan::how(blocks = strata_vec)
    vegan::adonis2(formula, data = data, permutations = ctrl, by = "margin")
  } else {
    vegan::adonis2(formula, data = data, permutations = nperm, by = "margin")
  }
}

# ---------------------------- Main ----------------------------
main <- function() {
  # Read abundances (compatible with either one-arg or two-arg calls)
  OC <- read_abund_matrix(oral_crohn_path,  pseudo)
  FC <- read_abund_matrix(fecal_crohn_path, pseudo)
  OH <- read_abund_matrix(oral_healthy_path,  pseudo)
  FH <- read_abund_matrix(fecal_healthy_path, pseudo)

  # Align features (intersection)
  L <- align_features(list(OC, FC, OH, FH))
  if (length(L) == 4) { OC <- L[[1]]; FC <- L[[2]]; OH <- L[[3]]; FH <- L[[4]] }

  # Group labels
  labs_from <- function(M, label) {
    if (is.null(M) || ncol(M) == 0) setNames(character(0), character(0))
    else setNames(rep(label, ncol(M)), colnames(M))
  }
  lab_oral_c   <- labs_from(OC, "Crohn-Oral")
  lab_fecal_c  <- labs_from(FC, "Crohn-Fecal")
  lab_oral_h   <- labs_from(OH, "Healthy-Oral")
  lab_fecal_h  <- labs_from(FH, "Healthy-Fecal")
  group_map_named <- c(lab_oral_c, lab_fecal_c, lab_oral_h, lab_fecal_h)

  # Merge all groups
  M_all <- do.call(cbind, Filter(Negate(is.null), list(OC, FC, OH, FH)))

  # Covariates & pairs
  covars   <- read_covariates_robust(covars_path)
  pairs_df <- read_pairs(pairs_path)

  # ---------------- PCoA (all groups) ----------------
  Db_all <- mk_dist(drop_bad_for_dist(M_all, "bray"),    "bray")$D
  Dj_all <- mk_dist(drop_bad_for_dist(M_all, "jaccard"), "jaccard")$D
  Da_all <- mk_aitchison(M_all, pseudo)$D

  pcoa_plot(Db_all, group_map_named, file.path(outdir, "pcoa_bray_allgroups.png"),
            "PCoA — All Groups (Bray–Curtis)", pal_named)
  pcoa_plot(Dj_all, group_map_named, file.path(outdir, "pcoa_jaccard_allgroups.png"),
            "PCoA — All Groups (Jaccard)",     pal_named)
  pcoa_plot(Da_all, group_map_named, file.path(outdir, "pcoa_aitchison_allgroups.png"),
            "PCoA — All Groups (Aitchison: CLR + Euclidean)", pal_named)

  # ----- PCoA: Crohn vs Healthy within site (Bray) -----
  Mb_oral  <- do.call(cbind, Filter(Negate(is.null), list(OC, OH)))
  Mb_fecal <- do.call(cbind, Filter(Negate(is.null), list(FC, FH)))
  Db_oral  <- mk_dist(drop_bad_for_dist(Mb_oral,  "bray"), "bray")$D
  Db_fecal <- mk_dist(drop_bad_for_dist(Mb_fecal, "bray"), "bray")$D

  pcoa_plot(Db_oral,  c(lab_oral_c,  lab_oral_h),
            file.path(outdir, "pcoa_bray_oral_CH.png"),
            "PCoA — Oral: Crohn vs Healthy (Bray–Curtis)", pal_named[c("Crohn-Oral","Healthy-Oral")])
  pcoa_plot(Db_fecal, c(lab_fecal_c, lab_fecal_h),
            file.path(outdir, "pcoa_bray_fecal_CH.png"),
            "PCoA — Fecal: Crohn vs Healthy (Bray–Curtis)", pal_named[c("Crohn-Fecal","Healthy-Fecal")])

  # ----- Paired Crohn oral–fecal (Bray) -----
  M_crohn_of <- do.call(cbind, Filter(Negate(is.null), list(OC, FC)))
  Db_pairs <- mk_dist(drop_bad_for_dist(M_crohn_of, "bray"), "bray")$D
  pcoa_plot_paired_lines(Db_pairs, pairs_df,
                         file.path(outdir, "pcoa_bray_oral_vs_fecal_paired.png"),
                         "PCoA — Crohn Oral vs Fecal (paired; Bray–Curtis)",
                         color_oral = pal_named[["Crohn-Oral"]],
                         color_fecal= pal_named[["Crohn-Fecal"]])

  # -------------- PERMANOVA (Bray/Aitchison) --------------
  run_permanova_site <- function(M_case, M_ctrl, site_name, with_cov = FALSE, method = c("bray","aitchison"),
                                 out_csv, out_counts_csv) {
    method <- match.arg(method)
    if (is.null(M_case) || is.null(M_ctrl)) { write_note_csv(out_csv, "no_matrix"); write_counts_csv(out_counts_csv, site_name, 0,0,0,"no_matrix"); return(invisible(NULL)) }
    lst <- align_features(list(M_case, M_ctrl))
    if (length(lst) < 2) { write_note_csv(out_csv, "no_common_features"); write_counts_csv(out_counts_csv, site_name,0,0,0,"no_common_features"); return(invisible(NULL)) }
    M <- cbind(lst[[1]], lst[[2]]); labs <- colnames(M)

    md <- covars %>% filter(tolower(site) == tolower(site_name), Sample_ID %in% labs)
    if (nrow(md) < 3 || dplyr::n_distinct(md$disease) < 2) md <- build_fallback_md(lst, labs, site_name)
    if (nrow(md) < 3 || dplyr::n_distinct(md$disease) < 2) {
      write_note_csv(out_csv, "insufficient_md"); hc <- count_hc(md$disease)
      write_counts_csv(out_counts_csv, site_name, nrow(md), hc["h"], hc["c"], "insufficient_md"); return(invisible(NULL))
    }

    # distance
    if (method == "bray") {
      Mb <- drop_bad_for_dist(M[, md$Sample_ID, drop=FALSE], "bray")
      Dx <- mk_dist(Mb, "bray")$D
    } else {
      Mb <- M[, md$Sample_ID, drop=FALSE]
      Dx <- mk_aitchison(Mb, pseudo)$D
    }
    if (is.null(Dx)) { write_note_csv(out_csv,"dist_null"); write_counts_csv(out_counts_csv, site_name, nrow(md), NA, NA, "dist_null"); return(invisible(NULL)) }

    labs_b <- labels(Dx); md_b <- md[match(labs_b, md$Sample_ID), , drop = FALSE]

    if (with_cov) {
      usable <- select_minimal_covars(md_b)
      md_b   <- scale_continuous_if_used(md_b, usable)
      rhs    <- paste(c("disease", usable), collapse = " + ")
      fml    <- stats::as.formula(paste("Dx ~", rhs))
      res    <- adonis2_with_optional_strata(Dx, fml, md_b, nperm,
                                             strata_vec = if ("STUDY_ID" %in% names(md_b)) md_b$STUDY_ID else NULL)
      out <- as.data.frame(res); out$term <- rownames(out); rownames(out) <- NULL
      out <- out %>% rename(Df=Df, SumOfSqs=SumOfSqs, R2=R2, F=`F`, p=`Pr(>F)`) %>% select(term, Df, SumOfSqs, R2, F, p)
      out$covars_used <- paste(usable, collapse = ", ")
      write_csv(out, out_csv)
      hc <- count_hc(md_b$disease)
      write_counts_csv(out_counts_csv, site_name, nrow(md_b), hc["h"], hc["c"],
                       if (length(usable)) paste0("covars: ", paste(usable, collapse=", ")) else "covars: none")
    } else {
      res <- adonis2_with_optional_strata(Dx, Dx ~ disease, md_b, nperm, strata_vec = NULL)
      out <- as.data.frame(res); out$term <- rownames(out); rownames(out) <- NULL
      out <- out %>% rename(Df=Df, SumOfSqs=SumOfSqs, R2=R2, F=`F`, p=`Pr(>F)`) %>% select(term, Df, SumOfSqs, R2, F, p)
      out$covars_used <- ""
      write_csv(out, out_csv)
      hc <- count_hc(md_b$disease)
      write_counts_csv(out_counts_csv, site_name, nrow(md_b), hc["h"], hc["c"], "no_covariates")
    }
  }

  # Bray — with_cov & nocov
  run_permanova_site(OC, OH, "oral",  with_cov = FALSE, method = "bray",
                     out_csv = file.path(outdir, "permanova_oral_bray_nocov.csv"),
                     out_counts_csv = file.path(outdir, "permanova_oral_bray_nocov_counts.csv"))
  run_permanova_site(FC, FH, "fecal", with_cov = FALSE, method = "bray",
                     out_csv = file.path(outdir, "permanova_fecal_bray_nocov.csv"),
                     out_counts_csv = file.path(outdir, "permanova_fecal_bray_nocov_counts.csv"))
  run_permanova_site(OC, OH, "oral",  with_cov = TRUE,  method = "bray",
                     out_csv = file.path(outdir, "permanova_oral_bray_with_cov.csv"),
                     out_counts_csv = file.path(outdir, "permanova_oral_bray_with_cov_counts.csv"))
  run_permanova_site(FC, FH, "fecal", with_cov = TRUE,  method = "bray",
                     out_csv = file.path(outdir, "permanova_fecal_bray_with_cov.csv"),
                     out_counts_csv = file.path(outdir, "permanova_fecal_bray_with_cov_counts.csv"))

  # Aitchison — with_cov & nocov (sensitivity)
  run_permanova_site(OC, OH, "oral",  with_cov = FALSE, method = "aitchison",
                     out_csv = file.path(outdir, "permanova_oral_aitchison_nocov.csv"),
                     out_counts_csv = file.path(outdir, "permanova_oral_aitchison_nocov_counts.csv"))
  run_permanova_site(FC, FH, "fecal", with_cov = FALSE, method = "aitchison",
                     out_csv = file.path(outdir, "permanova_fecal_aitchison_nocov.csv"),
                     out_counts_csv = file.path(outdir, "permanova_fecal_aitchison_nocov_counts.csv"))
  run_permanova_site(OC, OH, "oral",  with_cov = TRUE,  method = "aitchison",
                     out_csv = file.path(outdir, "permanova_oral_aitchison_with_cov.csv"),
                     out_counts_csv = file.path(outdir, "permanova_oral_aitchison_with_cov_counts.csv"))
  run_permanova_site(FC, FH, "fecal", with_cov = TRUE,  method = "aitchison",
                     out_csv = file.path(outdir, "permanova_fecal_aitchison_with_cov.csv"),
                     out_counts_csv = file.path(outdir, "permanova_fecal_aitchison_with_cov_counts.csv"))

  # -------- Disease * site interaction (Bray & Aitchison) with minimal covariates --------
  permanova_interaction <- function(M_oral, M_fecal, method = c("bray","aitchison"),
                                    covars, out_csv, out_counts_csv) {
    method <- match.arg(method)
    if (is.null(M_oral) || is.null(M_fecal)) { write_note_csv(out_csv, "no_matrix"); write_counts_csv(out_counts_csv,"both",0,0,0,"no_matrix"); return(invisible(NULL)) }
    lst <- align_features(list(M_oral, M_fecal))
    if (length(lst) < 2) { write_note_csv(out_csv, "no_common_features"); write_counts_csv(out_counts_csv,"both",0,0,0,"no_common_features"); return(invisible(NULL)) }
    M <- cbind(lst[[1]], lst[[2]]); labs <- colnames(M)

    md <- covars %>% filter(Sample_ID %in% labs) %>%
      mutate(site = ifelse(tolower(site)=="oral","oral",
                    ifelse(tolower(site)=="fecal","fecal", NA_character_))) %>%
      filter(!is.na(site))

    if (nrow(md) < 6 || dplyr::n_distinct(md$disease) < 2 || dplyr::n_distinct(md$site) < 2) {
      write_note_csv(out_csv, "insufficient_md")
      hc <- count_hc(md$disease); write_counts_csv(out_counts_csv,"both", nrow(md), hc["h"], hc["c"], "insufficient_md")
      return(invisible(NULL))
    }

    if (method == "bray") {
      Mb <- drop_bad_for_dist(M[, md$Sample_ID, drop=FALSE], "bray")
      Dx <- mk_dist(Mb, "bray")$D
    } else {
      Mb <- M[, md$Sample_ID, drop=FALSE]
      Dx <- mk_aitchison(Mb, pseudo)$D
    }
    if (is.null(Dx)) { write_note_csv(out_csv,"dist_null"); write_counts_csv(out_counts_csv,"both", nrow(md), NA, NA, "dist_null"); return(invisible(NULL)) }

    labs_b <- labels(Dx); md_b <- md[match(labs_b, md$Sample_ID), , drop = FALSE]
    usable <- select_minimal_covars(md_b)
    md_b   <- scale_continuous_if_used(md_b, usable)
    rhs    <- paste(c("disease * site", usable), collapse = " + ")
    fml    <- stats::as.formula(paste("Dx ~", rhs))

    res <- adonis2_with_optional_strata(Dx, fml, md_b, nperm,
                                        strata_vec = if ("STUDY_ID" %in% names(md_b)) md_b$STUDY_ID else NULL)

    out <- as.data.frame(res); out$term <- rownames(out); rownames(out) <- NULL
    out <- out %>% rename(Df=Df, SumOfSqs=SumOfSqs, R2=R2, F=`F`, p=`Pr(>F)`) %>% select(term, Df, SumOfSqs, R2, F, p)
    out$covars_used <- paste(usable, collapse = ", ")
    write_csv(out, out_csv)

    hc <- count_hc(md_b$disease)
    write_counts_csv(out_counts_csv,"both", nrow(md_b), hc["h"], hc["c"],
                     if (length(usable)) paste0("covars: ", paste(usable, collapse=", ")) else "covars: none")
  }

  permanova_interaction(cbind(OC, OH), cbind(FC, FH), method = "bray",
                        covars, file.path(outdir, "permanova_interaction_bray.csv"),
                        file.path(outdir, "permanova_interaction_bray_counts.csv"))
  permanova_interaction(cbind(OC, OH), cbind(FC, FH), method = "aitchison",
                        covars, file.path(outdir, "permanova_interaction_aitchison.csv"),
                        file.path(outdir, "permanova_interaction_aitchison_counts.csv"))

  # ------------------------------ PERMDISP (Bray) ------------------------------
  permdisp_bray <- function(M_case, M_ctrl, covars, site_name, out_csv, out_counts_csv) {
    if (is.null(M_case) || is.null(M_ctrl)) { write_note_csv(out_csv, "no_matrix"); write_counts_csv(out_counts_csv, site_name,0,0,0,"no_matrix"); return(invisible(NULL)) }
    lst <- align_features(list(M_case, M_ctrl))
    if (length(lst) < 2) { write_note_csv(out_csv, "no_common_features"); write_counts_csv(out_counts_csv, site_name,0,0,0,"no_common_features"); return(invisible(NULL)) }
    M <- cbind(lst[[1]], lst[[2]]); labs <- colnames(M)

    md <- covars %>% filter(tolower(site) == tolower(site_name), Sample_ID %in% labs)
    if (nrow(md) < 3 || dplyr::n_distinct(md$disease) < 2) md <- build_fallback_md(lst, labs, site_name)
    if (nrow(md) < 3 || dplyr::n_distinct(md$disease) < 2) {
      write_note_csv(out_csv, "insufficient_md"); hc <- count_hc(md$disease)
      write_counts_csv(out_counts_csv, site_name, nrow(md), hc["h"], hc["c"], "insufficient_md"); return(invisible(NULL))
    }

    Mb <- drop_bad_for_dist(M[, md$Sample_ID, drop=FALSE], "bray")
    Db <- mk_dist(Mb, "bray")$D
    if (is.null(Db)) { write_note_csv(out_csv,"dist_null"); write_counts_csv(out_counts_csv, site_name, nrow(md), NA, NA, "dist_null"); return(invisible(NULL)) }

    labs_b <- labels(Db); md_b <- md[match(labs_b, md$Sample_ID), , drop=FALSE]
    grp <- factor(md_b$disease, levels=c(0,1), labels=c("Healthy","Crohn"))
    bd <- betadisper(Db, grp)
    pt <- suppressWarnings(permutest(bd, permutations = nperm))
    out <- tibble(term = "disease", F = as.numeric(pt$tab[1, "F"]), p = as.numeric(pt$tab[1, "Pr(>F)"]))
    write_csv(out, out_csv)
    hc <- count_hc(md_b$disease); write_counts_csv(out_counts_csv, site_name, nrow(md_b), hc["h"], hc["c"], "ok")
  }

  permdisp_bray(OC, OH, covars, "oral",
                file.path(outdir, "permdisp_oral_bray.csv"),
                file.path(outdir, "permdisp_oral_bray_counts.csv"))
  permdisp_bray(FC, FH, covars, "fecal",
                file.path(outdir, "permdisp_fecal_bray.csv"),
                file.path(outdir, "permdisp_fecal_bray_counts.csv"))

  # --------------- Group distances (Bray & Jaccard) ---------------
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
  write_group_distance_csv(M_all, group_map_named, file.path(outdir, "beta_group_distances.csv"))

  # -------- Paired distances summary (Crohn oral–fecal, Bray) --------
  if (!is.null(pairs_df) && nrow(pairs_df) > 0) {
    lst <- align_features(list(OC, FC)); if (length(lst) == 2) {
      OC2 <- lst[[1]]; FC2 <- lst[[2]]
      P <- pairs_df %>% filter(oral %in% colnames(OC2), fecal %in% colnames(FC2))
      if (nrow(P) > 0) {
        Mb <- cbind(OC2[, P$oral, drop=FALSE], FC2[, P$fecal, drop=FALSE])
        Mb <- drop_bad_for_dist(Mb, "bray"); Db <- mk_dist(Mb, "bray")$D
        if (!is.null(Db)) {
          dm <- as.matrix(Db); labs <- labels(Db)
          get_one <- function(o, f) if (o %in% labs && f %in% labs) dm[o, f] else NA_real_
          dists <- mapply(get_one, P$oral, P$fecal)
          out <- tibble(Metric="BrayCurtis",
                        n= sum(is.finite(dists)),
                        mean = mean(dists, na.rm=TRUE),
                        median = median(dists, na.rm=TRUE),
                        q1 = as.numeric(quantile(dists, 0.25, na.rm=TRUE)),
                        q3 = as.numeric(quantile(dists, 0.75, na.rm=TRUE)),
                        iqr = IQR(dists, na.rm=TRUE))
          write_csv(out, file.path(outdir, "pairwise_oral_fecal_summary.csv"))
        } else write_note_csv(file.path(outdir, "pairwise_oral_fecal_summary.csv"), "dist_null")
      } else write_note_csv(file.path(outdir, "pairwise_oral_fecal_summary.csv"), "no_pairs_overlap")
    } else write_note_csv(file.path(outdir, "pairwise_oral_fecal_summary.csv"), "no_common_features")
  } else write_note_csv(file.path(outdir, "pairwise_oral_fecal_summary.csv"), "no_pairs_file")

  # Debug dumps
  if (!is.null(OC)) writeLines(colnames(OC),  file.path(dbgdir, "ids_oral_crohn.txt"))
  if (!is.null(FC)) writeLines(colnames(FC),  file.path(dbgdir, "ids_fecal_crohn.txt"))
  if (!is.null(OH)) writeLines(colnames(OH),  file.path(dbgdir, "ids_oral_healthy.txt"))
  if (!is.null(FH)) writeLines(colnames(FH),  file.path(dbgdir, "ids_fecal_healthy.txt"))
  sink(file.path(dbgdir, "beta_session_info.txt")); print(sessionInfo()); sink()
  writeLines(c("OK", format(Sys.time())), con = file.path(dbgdir, "done.flag"))
}

invisible(main())
