#!/usr/bin/env Rscript
# =============================================================================
# Beta Diversity (Bray/Jaccard/Aitchison) — with & without covariates
#
# What this script does
#   • Reads four abundance tables (Oral/Fecal × Crohn/Healthy) — already filtered/normalized.
#   • Normalizes Sample_IDs to pooled-meta convention:
#       drop ".<rep>" suffix, drop leading "S", keep digits only, strip leading zeros.
#   • Aligns features to intersection across matrices (fair distance comparisons).
#   • Computes Bray-Curtis, Jaccard, and Aitchison (CLR+Euclidean) distances.
#   • PCoA plots:
#       - All four groups together
#       - Crohn vs Healthy within oral, within fecal
#       - Crohn paired oral–fecal with connecting segments
#   • PERMANOVA per site (with covariates and without), interaction (disease*site),
#     and PERMDISP per site (Bray). Writes companion *_counts.csv with n’s & notes.
#   • Summaries: group distances (Bray/Jaccard), paired oral–fecal distances (Crohn).
#   • Saves colors actually used as JSON for reproducibility.
#
# CLI:
#   --oral-crohn    PATH
#   --fecal-crohn   PATH
#   --oral-healthy  PATH
#   --fecal-healthy PATH
#   --covariates    PATH   (optional)
#   --pairs         PATH   (optional)
#   --colors        PATH   (optional YAML palette; supports legacy and group/synonyms)
#   --outdir        PATH   (Snakemake provides results/beta_models/{level})
#   --pseudocount   NUM    (default 1e-6; for Aitchison CLR)
#   --permutations  INT    (default 999)
# =============================================================================

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(ggplot2)
  library(vegan); library(stringr); library(purrr); library(yaml); library(jsonlite)
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

# Safe OR helpers
`%||%`    <- function(a,b) if (is.null(a)) b else if (is.atomic(a) && length(a)==1 && is.character(a) && !nzchar(a)) b else a
or_null   <- function(a,b) if (is.null(a)) b else a   # never calls is.na() (safe for lists)

# --------------------- palette (legacy + new schema) --------------------
load_group_colors <- function(yaml_path) {
  defaults <- c(
    "Fecal_Crohn" =   "#30638e",
    "Oral_Crohn" =    "#edae49",
    "Fecal_Healthy" = "#d1495b",
    "Oral_Healthy" =  "#00798c"
  )
  if (is.null(yaml_path) || !file.exists(yaml_path)) return(defaults)
  cfg <- tryCatch(yaml::read_yaml(yaml_path), error = function(e) NULL)
  if (is.null(cfg)) return(defaults)

  # Legacy: flat "colors" mapping
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

  # New schema: group + synonyms (like alpha). Use or_null to avoid is.na on lists.
  group <- or_null(cfg$group,    list())
  syn   <- or_null(cfg$synonyms, list())

  pick <- function(key, def) {
    if (!is.null(group[[key]]) && nzchar(as.character(group[[key]])[1])) {
      return(as.character(group[[key]])[1])
    }
    al <- syn[[key]]
    if (!is.null(al) && length(al)) {
      for (a in al) {
        if (!is.null(group[[a]]) && nzchar(as.character(group[[a]])[1])) {
          return(as.character(group[[a]])[1])
        }
      }
    }
    def
  }

  co <- pick("Oral_Crohn",    defaults[["Crohn-Oral"]])
  ho <- pick("Oral_Healthy",  defaults[["Healthy-Oral"]])
  cf <- pick("Fecal_Crohn",   defaults[["Crohn-Fecal"]])
  hf <- pick("Fecal_Healthy", defaults[["Healthy-Fecal"]])

  c("Crohn-Oral" = co, "Healthy-Oral" = ho,
    "Crohn-Fecal" = cf, "Healthy-Fecal" = hf)
}

pal_named <- load_group_colors(colors_path)
write_json(as.list(pal_named), file.path(outdir, "colors_used.json"), pretty = TRUE, auto_unbox = TRUE)

# --------------------- ID normalization (pooled-meta rules) -------------
normalize_id <- function(x) {
  if (is.na(x)) return(NA_character_)
  s <- toupper(trimws(as.character(x)))
  s <- gsub("\\.\\d+$", "", s)
  s <- sub("^S", "", s)
  s <- gsub("[^0-9]", "", s)
  s <- sub("^0+", "", s)
  ifelse(nzchar(s), s, "0")
}

# ------------------ abundance IO & transforms ---------------------------
safe_read_csv <- function(p) {
  if (is.null(p) || !file.exists(p)) return(tibble())
  suppressMessages(readr::read_csv(p, show_col_types = FALSE))
}
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

# ----------------------- covariates & pairs (robust) --------------------
COV_CANDS <- c("Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing")

read_covariates_robust <- function(path) {
  cv <- safe_read_csv(path)
  if (nrow(cv) == 0) return(tibble())  # gracefully empty

  pick_col <- function(nm, cands) { x <- intersect(cands, nm); if (length(x)) x[[1]] else NA_character_ }

  id_col   <- pick_col(names(cv), c("Sample_ID","SampleID","sample_id","Sample","sample","ID","id"))
  site_col <- pick_col(names(cv), c("site","Site","type","Type","SITE","site_bin","site01","site_numeric"))
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

  md <- dplyr::bind_cols(md, cv %>% dplyr::select(dplyr::any_of(COV_CANDS))) %>%
    dplyr::mutate(Sample_ID = vapply(Sample_ID, normalize_id, character(1))) %>%
    dplyr::distinct()
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

# ------------------------------ plotting --------------------------------
save_placeholder_png <- function(path, msg="Insufficient data") {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  g <- ggplot() + theme_void() + annotate("text", 0, 0, label = msg, size = 5)
  ggsave(path, g, width = 6, height = 5, dpi = 300, bg = "white")
}
write_note_csv <- function(path, note="no_data") {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write_csv(tibble(note = note), path)
}
write_counts_csv <- function(path, site, n_total, n_h, n_c, note="ok") {
  write_csv(tibble(site=site, n_total=n_total, n_healthy=n_h, n_crohn=n_c, note=note), path)
}
count_hc <- function(vec) { c(h=sum(vec==0,na.rm=TRUE), c=sum(vec==1,na.rm=TRUE)) }

pct_from_cmdscale <- function(ord) {
  if (is.null(ord$eig) || all(is.na(ord$eig))) return(c(NA, NA))
  vals <- ord$eig; keep <- which(vals > 0); if (!length(keep)) keep <- seq_along(vals)
  p <- vals / sum(abs(vals)); p[1:2] * 100
}

pcoa_plot <- function(D, lab_factor, png_file, title_txt, palette_named, subtitle_counts=NULL) {
  if (is.null(D) || length(D) == 0) { save_placeholder_png(png_file, "No distance (too few samples)"); return(invisible(NULL)) }
  ord  <- cmdscale(D, eig = TRUE, k = 2)
  labs <- labels(D)
  df <- tibble(Axis1 = ord$points[,1], Axis2 = ord$points[,2],
               Group = factor(lab_factor[labs], levels = names(palette_named)))
  pp <- pct_from_cmdscale(ord)
  lx <- ifelse(is.na(pp[1]), "PCoA 1", sprintf("PCoA 1 (%.1f%%)", pp[1]))
  ly <- ifelse(is.na(pp[2]), "PCoA 2", sprintf("PCoA 2 (%.1f%%)", pp[2]))
  pal_use <- palette_named[names(palette_named) %in% levels(df$Group)]
  g <- ggplot(df, aes(Axis1, Axis2, color = Group, fill = Group)) +
    geom_point(size = 2.2, alpha = 0.9, shape = 21, stroke = 0.2) +
    { if (nrow(df) >= 6 && dplyr::n_distinct(df$Group) >= 2) stat_ellipse(type="norm", linewidth=0.6, alpha=0.12) else NULL } +
    scale_color_manual(values = pal_use, drop = FALSE) +
    scale_fill_manual(values = pal_use, drop = FALSE) +
    theme_bw(base_size = 12) + coord_equal() +
    labs(
      title = title_txt,
      subtitle = subtitle_counts %||% NULL,
      x = lx, y = ly
    )
  ggsave(png_file, g, width = 6.4, height = 5.2, dpi = 300, bg = "white")
  invisible(NULL)
}

pcoa_plot_paired_lines <- function(D, pairs_df, png_file, title_txt,
                                   color_oral="#B499E5", color_fecal="#78688E", subtitle_counts=NULL) {
  if (is.null(D) || length(D) == 0 || nrow(pairs_df) == 0) { save_placeholder_png(png_file, "No paired data"); return(invisible(NULL)) }
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
               aes(Axis1, Axis2), color = color_oral, size = 2.2) +
    geom_point(data = df %>% filter(Sample_ID %in% pairs_df$fecal),
               aes(Axis1, Axis2), color = color_fecal, size = 2.2) +
    theme_bw(base_size = 12) + coord_equal() +
    labs(title = title_txt,
         subtitle = subtitle_counts %||% NULL,
         x = "PCoA 1", y = "PCoA 2")
  ggsave(png_file, g, width = 6.4, height = 5.2, dpi = 300, bg = "white")
  invisible(NULL)
}

# ---------------------------- Main workflow -----------------------------
main <- function() {
  # Read abundances
  OC <- read_abund_matrix(oral_crohn_path, pseudo)
  FC <- read_abund_matrix(fecal_crohn_path, pseudo)
  OH <- read_abund_matrix(oral_healthy_path, pseudo)
  FH <- read_abund_matrix(fecal_healthy_path, pseudo)

  # Align features (intersection)
  L <- align_features(list(OC, FC, OH, FH))
  if (length(L) == 4) { OC <- L[[1]]; FC <- L[[2]]; OH <- L[[3]]; FH <- L[[4]] }

  # Group labels for plotting
  labs_from <- function(M, label) {
    if (is.null(M) || ncol(M) == 0) setNames(character(0), character(0))
    else setNames(rep(label, ncol(M)), colnames(M))
  }
  lab_oral_c   <- labs_from(OC, "Crohn-Oral")
  lab_fecal_c  <- labs_from(FC, "Crohn-Fecal")
  lab_oral_h   <- labs_from(OH, "Healthy-Oral")
  lab_fecal_h  <- labs_from(FH, "Healthy-Fecal")
  group_map_named <- c(lab_oral_c, lab_fecal_c, lab_oral_h, lab_fecal_h)

  # All groups merged
  M_all <- do.call(cbind, Filter(Negate(is.null), list(OC, FC, OH, FH)))

  # Covariates & pairs
  covars <- read_covariates_robust(covars_path)
  pairs_df <- read_pairs(pairs_path)

  # ---- Debug: overlap between matrix IDs and covariates ----
  labs_all <- colnames(M_all)
  if (length(labs_all)) {
    readr::write_csv(
      tibble(Sample_ID = labs_all, in_covars = labs_all %in% covars$Sample_ID),
      file.path(dbgdir, "debug_id_overlap.csv")
    )
  }

  # ------------------------ PCoA (all groups) ---------------------------
  ncounts <- table(factor(unname(group_map_named), levels=names(pal_named)))
  subtitle_all <- if (length(ncounts)) paste(sprintf("%s=%d", names(ncounts), as.integer(ncounts)), collapse=" | ") else NULL

  Db_all <- mk_dist(drop_bad_for_dist(M_all, "bray"),    "bray")$D
  Dj_all <- mk_dist(drop_bad_for_dist(M_all, "jaccard"), "jaccard")$D
  # Aitchison uses same filtered columns as Bray for fair comparison
  Da_all <- mk_aitchison(drop_bad_for_dist(M_all, "bray"), pseudo)$D

  pcoa_plot(Db_all, group_map_named, file.path(outdir, "pcoa_bray_allgroups.png"),
            "PCoA — All Groups (Bray-Curtis distance)", pal_named, subtitle_all)
  pcoa_plot(Dj_all, group_map_named, file.path(outdir, "pcoa_jaccard_allgroups.png"),
            "PCoA — All Groups (Jaccard distance)",     pal_named, subtitle_all)
  pcoa_plot(Da_all, group_map_named, file.path(outdir, "pcoa_aitchison_allgroups.png"),
            "PCoA — All Groups (Aitchison: CLR + Euclidean)", pal_named, subtitle_all)

  # ---------------- PCoA: Crohn vs Healthy within site -----------------
  Mb_oral  <- do.call(cbind, Filter(Negate(is.null), list(OC, OH)))
  Mb_fecal <- do.call(cbind, Filter(Negate(is.null), list(FC, FH)))
  Db_oral  <- mk_dist(drop_bad_for_dist(Mb_oral,  "bray"), "bray")$D
  Db_fecal <- mk_dist(drop_bad_for_dist(Mb_fecal, "bray"), "bray")$D

  sub_oral_counts  <- paste("Healthy-Oral =", ncol(OH) %||% 0, "| Crohn-Oral =", ncol(OC) %||% 0)
  sub_fecal_counts <- paste("Healthy-Fecal=", ncol(FH) %||% 0, "| Crohn-Fecal=", ncol(FC) %||% 0)

  pcoa_plot(Db_oral,  c(lab_oral_c,  lab_oral_h),
            file.path(outdir, "pcoa_bray_oral_CH.png"),
            "PCoA — Oral: Crohn vs Healthy (Bray-Curtis)", pal_named[c("Crohn-Oral","Healthy-Oral")],
            sub_oral_counts)
  pcoa_plot(Db_fecal, c(lab_fecal_c, lab_fecal_h),
            file.path(outdir, "pcoa_bray_fecal_CH.png"),
            "PCoA — Fecal: Crohn vs Healthy (Bray-Curtis)", pal_named[c("Crohn-Fecal","Healthy-Fecal")],
            sub_fecal_counts)

  # ---------------- Paired Crohn oral–fecal (Bray) ----------------------
  M_crohn_of <- do.call(cbind, Filter(Negate(is.null), list(OC, FC)))
  Db_pairs <- mk_dist(drop_bad_for_dist(M_crohn_of, "bray"), "bray")$D
  pcoa_plot_paired_lines(Db_pairs, pairs_df,
                         file.path(outdir, "pcoa_bray_oral_vs_fecal_paired.png"),
                         "PCoA — Crohn Oral vs Fecal (paired; Bray-Curtis)",
                         color_oral = pal_named[["Crohn-Oral"]],
                         color_fecal= pal_named[["Crohn-Fecal"]],
                         subtitle_counts = paste("n pairs input =", nrow(pairs_df)))

  # -------------- PERMANOVA helpers (Bray) ------------------------------
  build_fallback_md <- function(lst_mats, labs, site_name=NULL) {
    k1 <- if (!is.null(lst_mats[[1]])) ncol(lst_mats[[1]]) else 0L  # Crohn first
    k2 <- if (!is.null(lst_mats[[2]])) ncol(lst_mats[[2]]) else 0L  # Healthy next
    tibble(
      Sample_ID = labs,
      site      = site_name %||% NA_character_,
      disease   = c(rep(1L, k1), rep(0L, k2))
    )
  }

  permanova_with_cov <- function(M_case, M_ctrl, site_name, covars, out_csv, out_counts_csv, permutations=999) {
    if (is.null(M_case) || is.null(M_ctrl)) { write_note_csv(out_csv, "no_matrix"); write_counts_csv(out_counts_csv, site_name, 0,0,0,"no_matrix"); return(invisible(NULL)) }
    lst <- align_features(list(M_case, M_ctrl))
    if (length(lst) < 2) { write_note_csv(out_csv, "no_common_features"); write_counts_csv(out_counts_csv, site_name,0,0,0,"no_common_features"); return(invisible(NULL)) }
    M <- cbind(lst[[1]], lst[[2]]); labs <- colnames(M)

    md <- covars %>% filter(tolower(site) == tolower(site_name), Sample_ID %in% labs)

    if (nrow(md) < 3 || dplyr::n_distinct(md$disease) < 2) {
      md <- build_fallback_md(lst, labs, site_name)
    }
    if (nrow(md) < 3 || dplyr::n_distinct(md$disease) < 2) {
      write_note_csv(out_csv, "insufficient_md")
      hc <- count_hc(md$disease); write_counts_csv(out_counts_csv, site_name, nrow(md), hc["h"], hc["c"], "insufficient_md")
      return(invisible(NULL))
    }

    Mb <- drop_bad_for_dist(M[, md$Sample_ID, drop=FALSE], "bray")
    Db <- mk_dist(Mb, "bray")$D
    if (is.null(Db)) { write_note_csv(out_csv,"dist_null"); write_counts_csv(out_counts_csv, site_name, nrow(md), NA, NA, "dist_null"); return(invisible(NULL)) }

    labs_b <- labels(Db); md_b <- md[match(labs_b, md$Sample_ID), , drop = FALSE]

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
    hc <- count_hc(md_b$disease); write_counts_csv(out_counts_csv, site_name, nrow(md_b), hc["h"], hc["c"], paste0("covars: ", paste(usable, collapse=", ")))
  }

  permanova_nocov <- function(M_case, M_ctrl, site_name, covars, out_csv, out_counts_csv, permutations=999) {
    if (is.null(M_case) || is.null(M_ctrl)) { write_note_csv(out_csv, "no_matrix"); write_counts_csv(out_counts_csv, site_name, 0,0,0,"no_matrix"); return(invisible(NULL)) }
    lst <- align_features(list(M_case, M_ctrl))
    if (length(lst) < 2) { write_note_csv(out_csv, "no_common_features"); write_counts_csv(out_counts_csv, site_name,0,0,0,"no_common_features"); return(invisible(NULL)) }

    M <- cbind(lst[[1]], lst[[2]]); labs <- colnames(M)
    md <- covars %>% dplyr::filter(tolower(site) == tolower(site_name), Sample_ID %in% labs)

    if (nrow(md) < 3 || dplyr::n_distinct(md$disease) < 2) {
      md <- build_fallback_md(lst, labs, site_name)
    }
    if (nrow(md) < 3 || dplyr::n_distinct(md$disease) < 2) {
      write_note_csv(out_csv, "insufficient_md")
      hc <- count_hc(md$disease); write_counts_csv(out_counts_csv, site_name, nrow(md), hc["h"], hc["c"], "insufficient_md")
      return(invisible(NULL))
    }

    Mb <- drop_bad_for_dist(M[, md$Sample_ID, drop=FALSE], "bray")
    Db <- mk_dist(Mb, "bray")$D
    if (is.null(Db)) { write_note_csv(out_csv,"dist_null"); write_counts_csv(out_counts_csv, site_name, nrow(md), NA, NA, "dist_null"); return(invisible(NULL)) }

    labs_b <- labels(Db); md_b <- md[match(labs_b, md$Sample_ID), , drop = FALSE]
    res <- vegan::adonis2(Db ~ disease, data = md_b, permutations = permutations, by = "margin")
    out <- as.data.frame(res); out$term <- rownames(out); rownames(out) <- NULL
    out <- out %>% dplyr::rename(Df=Df, SumOfSqs=SumOfSqs, R2=R2, F=`F`, p=`Pr(>F)`) %>% dplyr::select(term, Df, SumOfSqs, R2, F, p)
    readr::write_csv(out, out_csv)
    hc <- count_hc(md_b$disease); write_counts_csv(out_counts_csv, site_name, nrow(md_b), hc["h"], hc["c"], "no_covariates")
  }

  # with covariates
  permanova_with_cov(OC, OH, "oral",  covars,
                     file.path(outdir, "permanova_oral_bray_with_cov.csv"),
                     file.path(outdir, "permanova_oral_bray_with_cov_counts.csv"),
                     permutations = nperm)
  permanova_with_cov(FC, FH, "fecal", covars,
                     file.path(outdir, "permanova_fecal_bray_with_cov.csv"),
                     file.path(outdir, "permanova_fecal_bray_with_cov_counts.csv"),
                     permutations = nperm)
  # no covariates
  permanova_nocov(OC, OH, "oral",  covars,
                  file.path(outdir, "permanova_oral_bray_nocov.csv"),
                  file.path(outdir, "permanova_oral_bray_nocov_counts.csv"),
                  permutations = nperm)
  permanova_nocov(FC, FH, "fecal", covars,
                  file.path(outdir, "permanova_fecal_bray_nocov.csv"),
                  file.path(outdir, "permanova_fecal_bray_nocov_counts.csv"),
                  permutations = nperm)

  # -------- PERMANOVA interaction (disease * site) WITH covariates ------
  permanova_interaction_bray <- function(M_oral, M_fecal, covars, out_csv, out_counts_csv, permutations=999) {
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
    Mb <- drop_bad_for_dist(M[, md$Sample_ID, drop=FALSE], "bray")
    Db <- mk_dist(Mb, "bray")$D
    if (is.null(Db)) { write_note_csv(out_csv,"dist_null"); write_counts_csv(out_counts_csv,"both", nrow(md), NA, NA, "dist_null"); return(invisible(NULL)) }

    labs_b <- labels(Db); md_b <- md[match(labs_b, md$Sample_ID), , drop = FALSE]

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
    hc <- count_hc(md_b$disease); write_counts_csv(out_counts_csv,"both", nrow(md_b), hc["h"], hc["c"], paste0("covars: ", paste(usable, collapse=", ")))
  }

  permanova_interaction_bray(cbind(OC, OH), cbind(FC, FH), covars,
                             file.path(outdir, "permanova_interaction_bray.csv"),
                             file.path(outdir, "permanova_interaction_bray_counts.csv"),
                             permutations = nperm)

  # ------------------------------ PERMDISP -------------------------------
  permdisp_bray <- function(M_case, M_ctrl, covars, site_name, out_csv, out_counts_csv) {
    if (is.null(M_case) || is.null(M_ctrl)) { write_note_csv(out_csv, "no_matrix"); write_counts_csv(out_counts_csv, site_name,0,0,0,"no_matrix"); return(invisible(NULL)) }
    lst <- align_features(list(M_case, M_ctrl))
    if (length(lst) < 2) { write_note_csv(out_csv, "no_common_features"); write_counts_csv(out_counts_csv, site_name,0,0,0,"no_common_features"); return(invisible(NULL)) }
    M <- cbind(lst[[1]], lst[[2]]); labs <- colnames(M)

    md <- covars %>% filter(tolower(site) == tolower(site_name), Sample_ID %in% labs)
    if (nrow(md) < 3 || dplyr::n_distinct(md$disease) < 2) {
      md <- build_fallback_md(lst, labs, site_name)
    }
    if (nrow(md) < 3 || dplyr::n_distinct(md$disease) < 2) {
      write_note_csv(out_csv, "insufficient_md")
      hc <- count_hc(md$disease); write_counts_csv(out_counts_csv, site_name, nrow(md), hc["h"], hc["c"], "insufficient_md")
      return(invisible(NULL))
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

  # --------------- Group distance table (Bray & Jaccard) ----------------
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

  # ----------- Paired distances summary (Crohn oral–fecal) --------------
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
