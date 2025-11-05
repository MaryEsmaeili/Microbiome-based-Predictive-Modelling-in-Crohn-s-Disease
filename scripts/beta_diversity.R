#!/usr/bin/env Rscript
# =============================================================================
# beta_diversity.R
#
# Unified script for beta-diversity analyses and summaries.
# Tasks:
#   1) --task main
#        - PCoA plots WITHOUT covariates and WITH covariate-adjustment
#        - Site-wise CH (Crohn vs Healthy) plots: Oral and Fecal
#        - Crohn paired Oral↔Fecal plots (with and without covariates)
#        - PERMANOVA (site-specific, with/without covariates) + interaction tests
#        - PERMDISP (Bray) and group-distance summaries
#   2) --task summarize
#        - Summarize PERMANOVA/PERMDISP outputs into CSV + overview plots
#   3) --task oral_prevnew
#        - PCoA for Oral (Crohn vs Healthy-Previous vs Healthy-New) and
#          the same with Crohn-Fecal added (Bray)
#
# Notes:
#   - This file merges and harmonizes three previous scripts into one CLI.
#   - All comments are in English.
#   - Robust to missing files; writes small "note" CSVs when data are absent.
# =============================================================================

suppressPackageStartupMessages({
  # Core data + IO
  library(readr); library(dplyr); library(tidyr); library(stringr); library(purrr)
  library(jsonlite); library(yaml)
  # Stats + ordination
  library(vegan)
  # Plotting
  library(ggplot2)
  # Optional for oral_prevnew task
  suppressWarnings(requireNamespace("optparse", quietly = TRUE))
  suppressWarnings(requireNamespace("ape", quietly = TRUE))
})

# ---------------------------- Small CLI helpers ----------------------------
args_all <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) { i <- which(args_all == flag); if (length(i)==1 && i<length(args_all)) args_all[i+1] else default }
`%||%` <- function(a,b) if (is.null(a)) b else a

# ---------------------------- Global defaults ----------------------------
FIXED_COVARS <- c("Age","Sex","BMI","Smoking","Antibiotics_3m","Immuno_ongoing","Steroids_ongoing","PPI_use")

# ---------------------------- Utilities (shared) ----------------------------
norm_sex <- function(x){
  y <- tolower(trimws(as.character(x)))
  ifelse(y %in% c("m","male","1"), 1L, ifelse(y %in% c("f","female","0"), 0L, NA_integer_))
}
to_bin01 <- function(x){
  y <- tolower(trimws(as.character(x)))
  dplyr::case_when(
    y %in% c("1","yes","y","true","t","on","present","current","pos","+") ~ 1L,
    y %in% c("0","no","n","false","f","off","absent","none","neg","-")     ~ 0L,
    suppressWarnings(!is.na(as.numeric(y)) & as.numeric(y) %in% c(0,1))    ~ as.integer(as.numeric(y)),
    TRUE ~ NA_integer_
  )
}
to_num <- function(x) suppressWarnings(as.numeric(as.character(x)))

normalize_id <- function(x){
  if (is.na(x)) return(NA_character_)
  s <- toupper(trimws(as.character(x)))
  s <- gsub("\\.\\d+$","",s); s <- sub("^S","",s); s <- gsub("[^0-9]","",s); s <- sub("^0+","",s)
  ifelse(nzchar(s), s, "0")
}
safe_read_csv <- function(p){ if (is.null(p) || !file.exists(p)) return(tibble()); suppressMessages(readr::read_csv(p, show_col_types = FALSE)) }

read_abund_matrix <- function(path){
  df <- safe_read_csv(path); if (nrow(df)==0) return(NULL)
  rn <- df[[1]]; M <- as.matrix(df[,-1, drop=FALSE]); storage.mode(M) <- "numeric"
  rownames(M) <- make.unique(as.character(rn))
  colnames(M) <- make.unique(vapply(colnames(df)[-1], normalize_id, character(1)))
  keepC <- colSums(M, na.rm=TRUE) > 0; keepR <- rowSums(M, na.rm=TRUE) > 0
  M <- M[keepR, keepC, drop=FALSE]; if (nrow(M)==0 || ncol(M)==0) return(NULL); M
}

map_to_level <- function(feat, level=c("species","genus")){
  level <- match.arg(level); s <- as.character(feat)
  if (grepl("\\|", s)){
    toks <- unlist(strsplit(s,"\\|")); gtok <- toks[grepl("g__",toks)]; stok <- toks[grepl("s__",toks)]
    genus <- if (length(gtok)) sub("^.*g__","", gtok[length(gtok)]) else NA_character_
    species <- if (length(stok)) sub("^.*s__","", stok[length(stok)]) else NA_character_
    if (level=="genus") return(ifelse(is.na(genus) || genus=="" || genus=="unclassified", NA_character_, genus))
    if (!is.na(species) && species!="" && species!="unclassified") return(species)
    return(ifelse(is.na(genus) || genus=="" || genus=="unclassified", NA_character_, paste0(genus,"_sp")))
  } else {
    parts <- unlist(strsplit(s,"[ _]"))
    if (length(parts)>=2){ if (level=="genus") return(parts[1]); return(paste(parts[1], parts[2], sep="_")) }
    return(if (level=="genus") s else paste0(s,"_sp"))
  }
}
collapse_level <- function(M, level=c("species","genus")){
  if (is.null(M)) return(NULL)
  level <- match.arg(level); if (level=="species") return(M)
  tgt <- vapply(rownames(M), map_to_level, character(1), level="genus")
  keep <- which(!is.na(tgt) & nzchar(tgt)); if (!length(keep)) return(NULL)
  M2 <- M[keep,,drop=FALSE]; tgt <- tgt[keep]
  out <- rowsum(M2, group=tgt, reorder=FALSE, na.rm=TRUE)
  out <- out[rowSums(out, na.rm=TRUE)>0,,drop=FALSE]; if (nrow(out)==0) return(NULL); as.matrix(out)
}

align_features <- function(lst){
  lst <- lst[!vapply(lst, is.null, TRUE)]; if (!length(lst)) return(lst)
  common <- Reduce(intersect, lapply(lst, rownames)); lapply(lst, function(m) m[common,,drop=FALSE])
}
align_and_cbind <- function(lst){
  lst <- lst[!vapply(lst, is.null, TRUE)]; if (!length(lst)) return(NULL)
  commons <- Reduce(intersect, lapply(lst, rownames)); if (length(commons)==0) return(NULL)
  lst2 <- lapply(lst, function(m) m[commons,,drop=FALSE]); do.call(cbind, lst2)
}

drop_bad_for_dist <- function(M, method){
  if (is.null(M)) return(NULL)
  keep <- colSums(M, na.rm=TRUE) > 0; M <- M[,keep,drop=FALSE]
  if (ncol(M) < 3) return(M)
  D  <- suppressWarnings(vegan::vegdist(t(M), method = method)); dm <- as.matrix(D)
  good <- rowSums(is.na(dm))==0; M[,good,drop=FALSE]
}

clr_matrix <- function(X, pseudo=1e-6){
  X <- sweep(as.matrix(X), 2, colSums(X), "/"); X[X<=0] <- pseudo
  L <- log(X); gm <- matrix(colMeans(L), nrow=nrow(L), ncol=ncol(L), byrow=TRUE); L - gm
}
mk_dist <- function(M, method){
  if (is.null(M) || ncol(M)<3) return(list(D=NULL, labs=character()))
  D <- if (method=="jaccard") suppressWarnings(vegan::vegdist(t(M), method="jaccard", binary=TRUE))
       else suppressWarnings(vegan::vegdist(t(M), method=method))
  dm <- as.matrix(D); good <- rowSums(is.na(dm))==0
  if (!any(good)) return(list(D=NULL, labs=character()))
  list(D = stats::as.dist(dm[good,good,drop=FALSE]), labs = colnames(M)[good])
}
mk_aitchison <- function(M, pseudo=1e-6){
  if (is.null(M) || ncol(M)<3) return(list(D=NULL, labs=character()))
  Xclr <- clr_matrix(M, pseudo=pseudo); bad <- apply(Xclr,2,function(v) any(!is.finite(v)))
  if (any(bad)) Xclr <- Xclr[,!bad,drop=FALSE]; if (ncol(Xclr)<3) return(list(D=NULL, labs=character()))
  list(D = stats::dist(t(Xclr), method="euclidean"), labs = colnames(Xclr))
}

load_group_colors <- function(yaml_path){
  # Default palette (includes a distinct light purple for Healthy-New)
  defaults <- c(
    "Crohn-Oral"       = "#edae49",
    "Healthy-Previous" = "#00798c",
    "Healthy-New"      = "#c39bd3",
    "Crohn-Fecal"      = "#30638e",
    "Healthy-Fecal"    = "#d1495b",
    "Healthy-Oral"     = "#00798c"
  )
  if (is.null(yaml_path) || !file.exists(yaml_path)) return(defaults)
  cfg <- tryCatch(yaml::read_yaml(yaml_path), error=function(e) NULL); if (is.null(cfg)) return(defaults)
  out <- defaults
  if (!is.null(cfg$colors) && is.list(cfg$colors)) {
    for (k in names(out)) {
      v <- tryCatch(cfg$colors[[k]], error=function(e) NULL)
      if (!is.null(v) && nzchar(as.character(v)[1])) out[[k]] <- as.character(v)[1]
    }
  }
  out
}

read_covariates_robust <- function(path){
  cv <- safe_read_csv(path); if (nrow(cv)==0) return(tibble())
  pick <- function(nm, cands){ x <- intersect(cands, nm); if (length(x)) x[[1]] else NA_character_ }
  id_col   <- pick(names(cv), c("Sample_ID","SampleID","sample_id","Sample","sample","ID","id"))
  site_col <- pick(names(cv), c("site","Site","type","Type","SITE"))
  dis_col  <- pick(names(cv), c("disease","Disease","status","Status","phenotype","Phenotype","group","Group","label","Label"))

  map_site <- function(x){
    y <- tolower(trimws(as.character(x))); out <- rep(NA_character_, length(y)); v <- suppressWarnings(as.integer(y))
    out[!is.na(v) & v==1L] <- "oral"; out[!is.na(v) & v==0L] <- "fecal"
    is_str <- is.na(v)
    out[is_str & str_detect(y,"\\boral\\b|mouth|saliva|buccal|\\boc\\b")]  <- "oral"
    out[is_str & str_detect(y,"fecal|faecal|stool|\\bfc\\b")]              <- "fecal"; out
  }
  map_dis <- function(x){
    y <- tolower(trimws(as.character(x)))
    case_when(
      y %in% c("1","yes","true","crohn","cd","case","patient","ibd","disease") ~ 1L,
      y %in% c("0","no","false","healthy","control","hc")                      ~ 0L,
      str_detect(y,"crohn|\\bcd\\b|case|patient|ibd")                           ~ 1L,
      str_detect(y,"healthy|control|\\bhc\\b")                                  ~ 0L,
      TRUE ~ NA_integer_
    )
  }

  md <- tibble(
    Sample_ID = if (!is.na(id_col)) as.character(cv[[id_col]]) else NA_character_,
    site      = if (!is.na(site_col)) map_site(cv[[site_col]]) else NA_character_,
    disease   = if (!is.na(dis_col))  map_dis(cv[[dis_col]])  else NA_integer_
  )
  extra <- cv %>% select(any_of(setdiff(c(FIXED_COVARS,"STUDY_ID","Study_ID","subject","Subject","ID","id"), c(id_col,site_col,dis_col))))
  md <- bind_cols(md, extra) %>% mutate(Sample_ID = vapply(Sample_ID, normalize_id, character(1))) %>% distinct()
  if ("Sex" %in% names(md)) md$Sex <- norm_sex(md$Sex)
  for (b in intersect(c("Smoking","Antibiotics_3m","PPI_use","Immuno_ongoing","Steroids_ongoing"), names(md))) md[[b]] <- to_bin01(md[[b]])
  for (v in intersect(c("Age","BMI"), names(md))) md[[v]] <- to_num(md[[v]])
  sid_col <- intersect(c("STUDY_ID","Study_ID","subject","Subject","ID","id"), names(md))
  md$STUDY_ID <- if (length(sid_col)) as.character(md[[sid_col[1]]]) else NA_character_
  md
}

read_pairs <- function(path){
  P <- safe_read_csv(path); if (nrow(P)==0) return(tibble(STUDY_ID=character(), oral=character(), fecal=character()))
  nm <- names(P); pick <- function(cands){ cand <- intersect(cands, nm); if (length(cand)) cand[[1]] else NA_character_ }
  sid <- pick(c("STUDY_ID","Study_ID","subject","Subject","ID","id"))
  oc  <- pick(c("oral","Oral","Oral_ID","oral_id","Oral_Sample_ID","OC"))
  fc  <- pick(c("fecal","Fecal","Fecal_ID","fecal_id","Fecal_Sample_ID","stool","FC"))
  if (is.na(oc) || is.na(fc)) return(tibble(STUDY_ID=character(), oral=character(), fecal=character()))
  tibble(STUDY_ID = if (!is.na(sid)) as.character(P[[sid]]) else NA_character_,
         oral     = vapply(as.character(P[[oc]]), normalize_id, character(1)),
         fecal    = vapply(as.character(P[[fc]]), normalize_id, character(1))) %>%
    filter(!is.na(oral), !is.na(fecal))
}

select_usable_covars <- function(md_df, min_non_na=2){
  cand <- intersect(FIXED_COVARS, names(md_df)); usable <- c()
  for (v in cand){
    vv <- md_df[[v]]; nn <- sum(!is.na(vv))
    if (nn >= min_non_na && dplyr::n_distinct(vv, na.rm=TRUE) >= 2) usable <- c(usable, v)
  }
  usable
}
scale_continuous_if_used <- function(md_df, usable_names){
  out <- md_df
  for (v in intersect(c("Age","BMI"), usable_names)){
    if (v %in% names(out)){
      vv <- suppressWarnings(as.numeric(out[[v]]))
      if (sum(is.finite(vv))>=3 && sd(vv,na.rm=TRUE)>0) out[[v]] <- as.numeric(scale(vv))
    }
  }
  out
}

legend_labels_with_n <- function(group_vec, pal){
  lv <- unique(na.omit(group_vec)); lab <- setNames(character(length(lv)), lv)
  for (g in lv) lab[[g]] <- sprintf("%s (n=%d)", g, sum(group_vec==g, na.rm=TRUE))
  list(levels=lv, labels=lab[lv], palette=pal[names(pal) %in% lv])
}
pcoa_plot <- function(D, group_map_named, png_file, title_txt, pal){
  if (is.null(D) || length(D)==0){
    g <- ggplot() + theme_void() + annotate("text",0,0,label="No distance (too few samples)", size=5)
    ggsave(png_file, g, width=6.4, height=5.2, dpi=300, bg="white"); return(invisible(NULL))
  }
  ord  <- cmdscale(D, eig=TRUE, k=2); labs <- labels(D); grp <- factor(unname(group_map_named[labs]))
  labinfo <- legend_labels_with_n(grp, pal)
  df <- tibble(Axis1=ord$points[,1], Axis2=ord$points[,2], Group=factor(grp, levels=labinfo$levels))
  g <- ggplot(df, aes(Axis1, Axis2, color=Group, fill=Group)) +
    geom_point(size=2.3, alpha=.9, shape=21, stroke=.2) +
    { if (nrow(df)>=6 && dplyr::n_distinct(df$Group)>=2) stat_ellipse(type="norm", linewidth=.6, alpha=.12) else NULL } +
    scale_color_manual(values=labinfo$palette, breaks=labinfo$levels, labels=unname(unlist(labinfo$labels)), drop=FALSE) +
    scale_fill_manual(values =labinfo$palette, breaks=labinfo$levels, labels=unname(unlist(labinfo$labels)), drop=FALSE) +
    theme_bw(base_size=12) + coord_equal() + labs(title=title_txt, x="PCoA 1", y="PCoA 2")
  ggsave(png_file, g, width=6.4, height=5.2, dpi=300, bg="white")
}
pcoa_plot_paired_lines <- function(D, pairs_df, png_file, title_txt, color_oral="#B499E5", color_fecal="#78688E"){
  if (is.null(D) || length(D)==0 || nrow(pairs_df)==0){
    g <- ggplot() + theme_void() + annotate("text",0,0,label="No paired data", size=5)
    ggsave(png_file, g, width=6.4, height=5.2, dpi=300, bg="white"); return(invisible(NULL))
  }
  ord <- cmdscale(D, eig=TRUE, k=2); labs <- labels(D)
  df  <- tibble(Sample_ID=labs, Axis1=ord$points[,1], Axis2=ord$points[,2])
  seg <- pairs_df %>% inner_join(df, by=c("oral"="Sample_ID")) %>% rename(o1=Axis1,o2=Axis2) %>%
         inner_join(df, by=c("fecal"="Sample_ID")) %>% rename(f1=Axis1,f2=Axis2)
  g <- ggplot() +
    geom_segment(data=seg, aes(x=o1,y=o2,xend=f1,yend=f2), color="grey60", linewidth=.6, alpha=.7) +
    geom_point(data=df %>% filter(Sample_ID %in% pairs_df$oral),  aes(Axis1,Axis2), color=color_oral, size=2.3) +
    geom_point(data=df %>% filter(Sample_ID %in% pairs_df$fecal), aes(Axis1,Axis2), color=color_fecal, size=2.3) +
    theme_bw(base_size=12) + coord_equal() + labs(title=title_txt, x="PCoA 1", y="PCoA 2")
  ggsave(png_file, g, width=6.4, height=5.2, dpi=300, bg="white")
}
pcoa_plot_from_scores <- function(scores_df, group_map_named, png_file, title_txt, pal){
  if (is.null(scores_df) || nrow(scores_df)<3){
    g <- ggplot() + theme_void() + annotate("text",0,0,label="No adjusted ordination", size=5)
    ggsave(png_file, g, width=6.4, height=5.2, dpi=300, bg="white"); return(invisible(NULL))
  }
  labs <- scores_df$Sample_ID; grp <- factor(unname(group_map_named[labs]))
  labinfo <- legend_labels_with_n(grp, pal)
  df <- tibble(Axis1=scores_df[[1]], Axis2=scores_df[[2]], Group=factor(grp, levels=labinfo$levels))
  g <- ggplot(df, aes(Axis1, Axis2, color=Group, fill=Group)) +
    geom_point(size=2.3, alpha=.9, shape=21, stroke=.2) +
    { if (nrow(df)>=6 && dplyr::n_distinct(df$Group)>=2) stat_ellipse(type="norm", linewidth=.6, alpha=.12) else NULL } +
    scale_color_manual(values=labinfo$palette, breaks=labinfo$levels, labels=unname(unlist(labinfo$labels)), drop=FALSE) +
    scale_fill_manual(values =labinfo$palette, breaks=labinfo$levels, labels=unname(unlist(labinfo$labels)), drop=FALSE) +
    theme_bw(base_size=12) + coord_equal() + labs(title=title_txt, x="Adjusted axis 1", y="Adjusted axis 2")
  ggsave(png_file, g, width=6.4, height=5.2, dpi=300, bg="white")
}
pcoa_plot_paired_from_scores <- function(scores_df, pairs_df, png_file, title_txt,
                                         color_oral="#B499E5", color_fecal="#78688E"){
  if (is.null(scores_df) || nrow(scores_df)<3 || nrow(pairs_df)==0){
    g <- ggplot() + theme_void() + annotate("text",0,0,label="No adjusted paired data", size=5)
    ggsave(png_file, g, width=6.4, height=5.2, dpi=300, bg="white"); return(invisible(NULL))
  }
  df <- tibble(Sample_ID=scores_df$Sample_ID, Axis1=scores_df[[1]], Axis2=scores_df[[2]])
  seg <- pairs_df %>% inner_join(df, by=c("oral"="Sample_ID")) %>% rename(o1=Axis1,o2=Axis2) %>%
         inner_join(df, by=c("fecal"="Sample_ID")) %>% rename(f1=Axis1,f2=Axis2)
  g <- ggplot() +
    geom_segment(data=seg, aes(x=o1,y=o2,xend=f1,yend=f2), color="grey60", linewidth=.6, alpha=.7) +
    geom_point(data=df %>% filter(Sample_ID %in% pairs_df$oral),  aes(Axis1,Axis2), color=color_oral, size=2.3) +
    geom_point(data=df %>% filter(Sample_ID %in% pairs_df$fecal), aes(Axis1,Axis2), color=color_fecal, size=2.3) +
    theme_bw(base_size=12) + coord_equal() + labs(title=title_txt, x="Adjusted axis 1", y="Adjusted axis 2")
  ggsave(png_file, g, width=6.4, height=5.2, dpi=300, bg="white")
}

capscale_residual_scores <- function(D, md, covars_use){
  if (is.null(D) || length(D)==0) return(NULL)
  if (!length(covars_use)) return(NULL)

  labs <- labels(D)
  md2 <- md %>% filter(Sample_ID %in% labs) %>% arrange(Sample_ID) %>%
    group_by(Sample_ID) %>%
    summarise(across(everything(), ~{x <- .; if (all(is.na(x))) NA else x[which(!is.na(x))[1]]}), .groups="drop") %>%
    as.data.frame(stringsAsFactors = FALSE)
  if (nrow(md2) < 3) return(NULL)
  rownames(md2) <- md2$Sample_ID

  cov_ok <- intersect(covars_use, names(md2))
  cov_ok <- cov_ok[vapply(md2[cov_ok], function(v) sum(!is.na(v))>=3 && dplyr::n_distinct(v,na.rm=TRUE)>=2, logical(1))]
  if (!length(cov_ok)) return(NULL)

  keep <- stats::complete.cases(md2[, cov_ok, drop=FALSE]); if (sum(keep)<3) return(NULL)
  md3 <- md2[keep,,drop=FALSE]; labs3 <- rownames(md3)

  Dm <- as.matrix(D)
  labs3 <- labs3[labs3 %in% rownames(Dm) & labs3 %in% colnames(Dm)]
  if (length(labs3) < 3) return(NULL)
  D3  <- stats::as.dist(Dm[labs3, labs3, drop=FALSE])

  cond_txt <- paste(cov_ok, collapse=" + ")
  fml <- stats::as.formula(paste("D3 ~ 1 + Condition(", cond_txt, ")"))
  ord <- tryCatch(vegan::capscale(fml, data=md3, add=TRUE, na.action=stats::na.exclude), error=function(e) NULL)
  if (is.null(ord)) return(NULL)

  sc <- suppressWarnings(vegan::scores(ord, display="sites"))
  if (is.null(sc) || ncol(sc)<2) return(NULL)
  pts <- as.data.frame(sc[,1:2,drop=FALSE]); pts$Sample_ID <- rownames(sc); rownames(pts) <- NULL; pts
}

write_note_csv   <- function(path, note="no_data"){ dir.create(dirname(path), recursive=TRUE, showWarnings=FALSE); write_csv(tibble(note=note), path) }
debug_cov_report <- function(md, covs_try, labs, out_csv, title=""){
  dir.create(dirname(out_csv), recursive=TRUE, showWarnings=FALSE)
  md2 <- md %>% filter(Sample_ID %in% labs); rownames(md2) <- md2$Sample_ID; md2 <- md2[labs,,drop=FALSE]
  covs <- intersect(covs_try, names(md2))
  if (!length(covs)){ write_csv(tibble(note="no_covariate_candidates", n_samples=nrow(md2)), out_csv); return(invisible(NULL)) }
  summ <- map_dfr(covs, function(v){
    x <- md2[[v]]
    tibble(covariate=v, non_na=sum(!is.na(x)), n_levels=n_distinct(x, na.rm=TRUE),
           sd_if_numeric=suppressWarnings(if (is.numeric(x)) sd(x,na.rm=TRUE) else NA_real_))
  }) %>% mutate(title=title, n_samples=nrow(md2))
  write_csv(summ, out_csv)
}
debug_labels_map <- function(D, group_map_named, out_csv){
  if (is.null(D) || length(D)==0){ write_csv(tibble(note="no_distance"), out_csv); return(invisible(NULL)) }
  labs <- labels(D); write_csv(tibble(Sample_ID=labs, group=unname(group_map_named[labs])), out_csv)
}

# ---------------------------- TASK: main (full pipeline) ----------------------------
task_main <- function(){
  # Inputs (file paths)
  oral_crohn_path     <- get_arg("--oral-crohn")
  fecal_crohn_path    <- get_arg("--fecal-crohn")
  oral_healthy_merged <- get_arg("--oral-healthy-merged") %||% get_arg("--oral-healthy") # raw healthy oral (merged)
  oral_healthy_prev   <- get_arg("--oral-healthy-prev")                                  # optional alt for WITH-COV
  fecal_healthy_path  <- get_arg("--fecal-healthy")
  covars_path         <- get_arg("--covariates")
  pairs_path          <- get_arg("--pairs")
  colors_path         <- get_arg("--colors")
  outdir              <- get_arg("--outdir", "results/beta")
  level_req           <- tolower(get_arg("--level", "species"))  # species | genus
  pseudo              <- as.numeric(get_arg("--pseudocount", "1e-6"))
  nperm               <- as.integer(get_arg("--permutations", "999"))

  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  dbgdir <- file.path(outdir, "debug"); dir.create(dbgdir, recursive = TRUE, showWarnings = FALSE)

  pal <- load_group_colors(colors_path)
  write_json(as.list(pal), file.path(outdir, "colors_used.json"), pretty=TRUE, auto_unbox=TRUE)

  # Read matrices
  OC  <- read_abund_matrix(oral_crohn_path)
  FC  <- read_abund_matrix(fecal_crohn_path)
  OHm <- read_abund_matrix(oral_healthy_merged)
  OHp <- if (!is.null(oral_healthy_prev)) read_abund_matrix(oral_healthy_prev) else NULL
  FH  <- read_abund_matrix(fecal_healthy_path)

  # Collapse to requested level
  OC  <- collapse_level(OC,  level_req)
  FC  <- collapse_level(FC,  level_req)
  OHm <- collapse_level(OHm, level_req)
  OHp <- collapse_level(OHp, level_req)
  FH  <- collapse_level(FH,  level_req)

  # Group maps
  labs_from <- function(M, label){ if (is.null(M)||ncol(M)==0) setNames(character(0), character(0)) else setNames(rep(label, ncol(M)), colnames(M)) }
  lab_oral_c   <- labs_from(OC,  "Crohn-Oral")
  lab_fecal_c  <- labs_from(FC,  "Crohn-Fecal")
  lab_oral_h_m <- labs_from(OHm, "Healthy-Oral")
  lab_fecal_h  <- labs_from(FH,  "Healthy-Fecal")

  # Build RAW and WITH-COV matrices
  M_all_RAW <- align_and_cbind(list(OC, FC, OHm, FH))
  group_map_RAW <- if (!is.null(M_all_RAW)) c(lab_oral_c, lab_fecal_c, lab_oral_h_m, lab_fecal_h)[colnames(M_all_RAW)] else character(0)

  M_all_COV <- align_and_cbind(list(OC, FC, OHp %||% OHm, FH))
  group_map_COV <- if (!is.null(M_all_COV)) c(lab_oral_c, lab_fecal_c, (if (!is.null(OHp)) labs_from(OHp,"Healthy-Oral") else lab_oral_h_m), lab_fecal_h)[colnames(M_all_COV)] else character(0)

  # Metadata and pairs
  md_all  <- read_covariates_robust(covars_path)
  pairs_df<- read_pairs(pairs_path)

  # ---------- PCoA: All groups (NO-COV) ----------
  if (!is.null(M_all_RAW)){
    Db <- mk_dist(drop_bad_for_dist(M_all_RAW,"bray"),    "bray")$D
    Dj <- mk_dist(drop_bad_for_dist(M_all_RAW,"jaccard"), "jaccard")$D
    Da <- mk_aitchison(M_all_RAW, pseudo)$D
    pcoa_plot(Db, group_map_RAW, file.path(outdir,"pcoa_bray_allgroups.png"),      sprintf("PCoA — All Groups (Bray, %s)", level_req), pal)
    pcoa_plot(Dj, group_map_RAW, file.path(outdir,"pcoa_jaccard_allgroups.png"),   sprintf("PCoA — All Groups (Jaccard, %s)", level_req), pal)
    pcoa_plot(Da, group_map_RAW, file.path(outdir,"pcoa_aitchison_allgroups.png"), sprintf("PCoA — All Groups (Aitchison, %s)", level_req), pal)
  }

  # ---------- PCoA: All groups (WITH-COV) ----------
  if (!is.null(M_all_RAW)){
    covs_try <- select_usable_covars(md_all, 2)

    # Bray
    Mb <- drop_bad_for_dist(M_all_RAW,"bray"); Db <- mk_dist(Mb,"bray")$D
    if (!is.null(Db)){
      labs_b <- labels(Db); md_b <- md_all %>% filter(Sample_ID %in% labs_b)
      covs_b <- select_usable_covars(md_b, 2)
      debug_cov_report(md_b, covs_try, labs_b, file.path(outdir,"debug/cov_report_allgroups_bray.csv"), "allgroups_bray")
      debug_labels_map(Db, group_map_RAW, file.path(outdir,"debug/labels_allgroups_bray.csv"))
      sc_b <- capscale_residual_scores(Db, md_b, covs_b)
      pcoa_plot_from_scores(sc_b, group_map_RAW, file.path(outdir,"pcoa_bray_allgroups_WITHCOV.png"),
                            sprintf("PCoA — All Groups (Bray, WITH covariates, %s)", level_req), pal)
    }
    # Jaccard
    Mj <- drop_bad_for_dist(M_all_RAW,"jaccard"); Dj <- mk_dist(Mj,"jaccard")$D
    if (!is.null(Dj)){
      labs_j <- labels(Dj); md_j <- md_all %>% filter(Sample_ID %in% labs_j)
      covs_j <- select_usable_covars(md_j, 2)
      debug_cov_report(md_j, covs_try, labs_j, file.path(outdir,"debug/cov_report_allgroups_jaccard.csv"), "allgroups_jaccard")
      debug_labels_map(Dj, group_map_RAW, file.path(outdir,"debug/labels_allgroups_jaccard.csv"))
      sc_j <- capscale_residual_scores(Dj, md_j, covs_j)
      pcoa_plot_from_scores(sc_j, group_map_RAW, file.path(outdir,"pcoa_jaccard_allgroups_WITHCOV.png"),
                            sprintf("PCoA — All Groups (Jaccard, WITH covariates, %s)", level_req), pal)
    }
    # Aitchison
    Da <- mk_aitchison(M_all_RAW, pseudo)$D
    if (!is.null(Da)){
      labs_a <- labels(Da); md_a <- md_all %>% filter(Sample_ID %in% labs_a)
      covs_a <- select_usable_covars(md_a, 2)
      debug_cov_report(md_a, covs_try, labs_a, file.path(outdir,"debug/cov_report_allgroups_aitchison.csv"), "allgroups_aitchison")
      debug_labels_map(Da, group_map_RAW, file.path(outdir,"debug/labels_allgroups_aitchison.csv"))
      sc_a <- capscale_residual_scores(Da, md_a, covs_a)
      pcoa_plot_from_scores(sc_a, group_map_RAW, file.path(outdir,"pcoa_aitchison_allgroups_WITHCOV.png"),
                            sprintf("PCoA — All Groups (Aitchison, WITH covariates, %s)", level_req), pal)
    }
  }

  # ---------- PCoA: CH within site (Bray) ----------
  Mb_oral  <- align_and_cbind(list(OC, OHm))
  Mb_fecal <- align_and_cbind(list(FC, FH))

  if (!is.null(Mb_oral)){
    Db_oral <- mk_dist(drop_bad_for_dist(Mb_oral,"bray"), "bray")$D
    pcoa_plot(Db_oral, c(lab_oral_c, lab_oral_h_m), file.path(outdir,"pcoa_bray_oral_CH.png"),
              sprintf("PCoA — Oral: Crohn vs Healthy (Bray, %s)", level_req), pal[c("Crohn-Oral","Healthy-Oral")])
    if (!is.null(Db_oral)){
      labs_b <- labels(Db_oral); md_b <- md_all %>% filter(Sample_ID %in% labs_b, tolower(site)=="oral")
      covs_b <- select_usable_covars(md_b, 2)
      debug_cov_report(md_b, covs_b, labs_b, file.path(outdir,"debug/cov_report_oral_CH_bray.csv"), "oral_CH_bray")
      debug_labels_map(Db_oral, c(lab_oral_c, lab_oral_h_m), file.path(outdir,"debug/labels_oral_CH_bray.csv"))
      sc_b <- capscale_residual_scores(Db_oral, md_b, covs_b)
      pcoa_plot_from_scores(sc_b, c(lab_oral_c, lab_oral_h_m), file.path(outdir,"pcoa_bray_oral_CH_WITHCOV.png"),
                            sprintf("PCoA — Oral: Crohn vs Healthy (Bray, WITH covariates, %s)", level_req),
                            pal[c("Crohn-Oral","Healthy-Oral")])
    }
  }
  if (!is.null(Mb_fecal)){
    Db_fecal <- mk_dist(drop_bad_for_dist(Mb_fecal,"bray"), "bray")$D
    pcoa_plot(Db_fecal, c(lab_fecal_c, lab_fecal_h), file.path(outdir,"pcoa_bray_fecal_CH.png"),
              sprintf("PCoA — Fecal: Crohn vs Healthy (Bray, %s)", level_req), pal[c("Crohn-Fecal","Healthy-Fecal")])
    if (!is.null(Db_fecal)){
      labs_b <- labels(Db_fecal); md_b <- md_all %>% filter(Sample_ID %in% labs_b, tolower(site)=="fecal")
      covs_b <- select_usable_covars(md_b, 2)
      debug_cov_report(md_b, covs_b, labs_b, file.path(outdir,"debug/cov_report_fecal_CH_bray.csv"), "fecal_CH_bray")
      debug_labels_map(Db_fecal, c(lab_fecal_c, lab_fecal_h), file.path(outdir,"debug/labels_fecal_CH_bray.csv"))
      sc_b <- capscale_residual_scores(Db_fecal, md_b, covs_b)
      pcoa_plot_from_scores(sc_b, c(lab_fecal_c, lab_fecal_h), file.path(outdir,"pcoa_bray_fecal_CH_WITHCOV.png"),
                            sprintf("PCoA — Fecal: Crohn vs Healthy (Bray, WITH covariates, %s)", level_req),
                            pal[c("Crohn-Fecal","Healthy-Fecal")])
    }
  }

  # ---------- Crohn Oral↔Fecal paired (Bray) ----------
  if (!is.null(OC) && !is.null(FC)){
    lst <- align_features(list(OC, FC)); if (length(lst)==2){
      M_crohn <- cbind(lst[[1]], lst[[2]])
      Db_pairs <- mk_dist(drop_bad_for_dist(M_crohn,"bray"), "bray")$D
      pcoa_plot_paired_lines(Db_pairs, pairs_df, file.path(outdir,"pcoa_bray_oral_vs_fecal_paired.png"),
                             sprintf("PCoA — Crohn Oral vs Fecal (paired; Bray, %s)", level_req),
                             color_oral=pal[["Crohn-Oral"]], color_fecal=pal[["Crohn-Fecal"]])
      if (!is.null(Db_pairs)){
        labs_p <- labels(Db_pairs); md_p <- md_all %>% filter(Sample_ID %in% labs_p)
        covs_p <- select_usable_covars(md_p, 2)
        sc_p <- capscale_residual_scores(Db_pairs, md_p, covs_p)
        pcoa_plot_paired_from_scores(sc_p, pairs_df, file.path(outdir,"pcoa_bray_oral_vs_fecal_paired_WITHCOV.png"),
                                     sprintf("PCoA — Crohn Oral vs Fecal (paired; Bray, WITH covariates, %s)", level_req),
                                     color_oral=pal[["Crohn-Oral"]], color_fecal=pal[["Crohn-Fecal"]])
      }
    }
  }

  # ---------- PERMANOVA (site-specific) + interaction ----------
  write_permanova_table <- function(res, covars_used){
    out <- as.data.frame(res)
    out$term <- rownames(out)
    rownames(out) <- NULL
    out <- out %>%
      dplyr::rename(
        Df       = Df,
        SumOfSqs = SumOfSqs,
        R2       = R2,
        F        = `F`,
        p        = `Pr(>F)`
      ) %>%
      dplyr::select(term, Df, SumOfSqs, R2, F, p)
    out$covars_used <- covars_used
    out
  }

  adonis2_opt <- function(D, formula, data, nperm, strata_vec = NULL){
    Dx <- D
    f_env <- environment(formula)
    if (is.null(f_env)) f_env <- parent.frame()
    eval_env <- new.env(parent = f_env)
    assign("Dx", Dx, envir = eval_env)
    environment(formula) <- eval_env

    set.seed(42)
    if (!is.null(strata_vec) && all(!is.na(strata_vec))) {
      ctrl <- vegan::how(blocks = strata_vec)
      vegan::adonis2(
        formula      = formula,
        data         = data,
        permutations = ctrl,
        by           = "margin"
      )
    } else {
      vegan::adonis2(
        formula      = formula,
        data         = data,
        permutations = nperm,
        by           = "margin"
      )
    }
  }

  # PERMANOVA: disease effect (with/without covariates), site-specific
  run_permanova_site <- function(M_case, M_ctrl, site_name,
                                 with_cov = FALSE,
                                 method = c("bray","aitchison"),
                                 out_csv){
    method <- match.arg(method)

    if (is.null(M_case) || is.null(M_ctrl)) {
      write_note_csv(out_csv, "no_matrix")
      return(invisible(NULL))
    }

    lst <- align_features(list(M_case, M_ctrl))
    if (length(lst) < 2) {
      write_note_csv(out_csv, "no_common_features")
      return(invisible(NULL))
    }

    M <- cbind(lst[[1]], lst[[2]])
    labs <- colnames(M)

    md_all_site <- md_all %>%
      dplyr::filter(tolower(site) == tolower(site_name),
                    Sample_ID %in% labs)

    if (nrow(md_all_site) < 3 || dplyr::n_distinct(md_all_site$disease) < 2) {
      write_note_csv(out_csv, "insufficient_md")
      return(invisible(NULL))
    }

    M_use <- M[, md_all_site$Sample_ID, drop = FALSE]

    Dx <- if (method == "bray") {
      mk_dist(drop_bad_for_dist(M_use, "bray"), "bray")$D
    } else {
      mk_aitchison(M_use, pseudo)$D
    }

    if (is.null(Dx)) {
      write_note_csv(out_csv, "dist_null")
      return(invisible(NULL))
    }

    labs_b <- labels(Dx)
    md_b <- md_all_site[match(labs_b, md_all_site$Sample_ID), , drop = FALSE]

    if (with_cov) {
      usable <- select_usable_covars(md_b)
      md_b <- scale_continuous_if_used(md_b, usable)
      rhs <- paste(c("disease", usable), collapse = " + ")
      fml <- as.formula(paste("Dx ~", rhs))
      res <- adonis2_opt(
        D         = Dx,
        formula   = fml,
        data      = md_b,
        nperm     = nperm,
        strata_vec = if ("STUDY_ID" %in% names(md_b)) md_b$STUDY_ID else NULL
      )
      readr::write_csv(
        write_permanova_table(res, paste(usable, collapse = ", ")),
        out_csv
      )
    } else {
      res <- adonis2_opt(
        D         = Dx,
        formula   = Dx ~ disease,
        data      = md_b,
        nperm     = nperm,
        strata_vec = NULL
      )
      readr::write_csv(write_permanova_table(res, ""), out_csv)
    }
  }

  # Site-specific PERMANOVA: disease (with/without covariates), Bray + Aitchison
  run_permanova_site(OC, OHm, "oral",  FALSE, "bray",      file.path(outdir,"permanova_oral_bray_nocov.csv"))
  run_permanova_site(FC, FH,  "fecal", FALSE, "bray",      file.path(outdir,"permanova_fecal_bray_nocov.csv"))
  run_permanova_site(OC, OHm, "oral",  FALSE, "aitchison", file.path(outdir,"permanova_oral_aitchison_nocov.csv"))
  run_permanova_site(FC, FH,  "fecal", FALSE, "aitchison", file.path(outdir,"permanova_fecal_aitchison_nocov.csv"))

  run_permanova_site(OC, OHp %||% OHm, "oral",  TRUE, "bray",      file.path(outdir,"permanova_oral_bray_with_cov.csv"))
  run_permanova_site(FC, FH,          "fecal", TRUE, "bray",      file.path(outdir,"permanova_fecal_bray_with_cov.csv"))
  run_permanova_site(OC, OHp %||% OHm, "oral",  TRUE, "aitchison", file.path(outdir,"permanova_oral_aitchison_with_cov.csv"))
  run_permanova_site(FC, FH,          "fecal", TRUE, "aitchison", file.path(outdir,"permanova_fecal_aitchison_with_cov.csv"))

  # NEW: PERMANOVA for individual covariates (Bray, site-specific)
  run_permanova_covariates_site <- function(M_case, M_ctrl, site_name,
                                            method = c("bray","aitchison"),
                                            covariates = FIXED_COVARS,
                                            out_csv) {
    method <- match.arg(method)

    if (is.null(M_case) || is.null(M_ctrl)) {
      write_note_csv(out_csv, "no_matrix")
      return(invisible(NULL))
    }

    # Align features between case and control
    lst <- align_features(list(M_case, M_ctrl))
    if (length(lst) < 2) {
      write_note_csv(out_csv, "no_common_features")
      return(invisible(NULL))
    }

    M <- cbind(lst[[1]], lst[[2]])
    labs <- colnames(M)

    # Filter metadata to this site and these samples
    md_site <- md_all %>%
      dplyr::filter(tolower(site) == tolower(site_name),
                    Sample_ID %in% labs)

    if (nrow(md_site) < 6) {
      write_note_csv(out_csv, "insufficient_md")
      return(invisible(NULL))
    }

    # Reorder matrix according to metadata
    M_use <- M[, md_site$Sample_ID, drop = FALSE]

    # Distance matrix
    Dx <- if (method == "bray") {
      mk_dist(drop_bad_for_dist(M_use, "bray"), "bray")$D
    } else {
      mk_aitchison(M_use, pseudo)$D
    }
    if (is.null(Dx)) {
      write_note_csv(out_csv, "dist_null")
      return(invisible(NULL))
    }

    labs_d <- labels(Dx)
    md_d <- md_site[match(labs_d, md_site$Sample_ID), , drop = FALSE]

    out_list <- list()

    for (cv in covariates) {
      if (!cv %in% colnames(md_d)) next

      x <- md_d[[cv]]
      # Require enough non-missing values and at least two levels
      if (sum(!is.na(x)) < 5 || dplyr::n_distinct(x, na.rm = TRUE) < 2) next

      md_cv <- md_d
      # Scale continuous covariates
      if (cv %in% c("Age","BMI")) {
        md_cv[[cv]] <- as.numeric(scale(as.numeric(md_cv[[cv]])))
      }

      # Formula: disease + this covariate
      fml <- as.formula(paste0("Dx ~ disease + ", cv))

      res <- adonis2_opt(
        D         = Dx,
        formula   = fml,
        data      = md_cv,
        nperm     = nperm,
        strata_vec = if ("STUDY_ID" %in% names(md_cv)) md_cv$STUDY_ID else NULL
      )

      tbl <- as.data.frame(res)
      tbl$term <- rownames(tbl)
      rownames(tbl) <- NULL

      tbl <- tbl %>%
        dplyr::filter(term %in% c("disease", cv)) %>%
        dplyr::rename(
          Df       = Df,
          SumOfSqs = SumOfSqs,
          R2       = R2,
          F        = `F`,
          p        = `Pr(>F)`
        ) %>%
        dplyr::mutate(
          site     = site_name,
          metric   = method,
          covariate = cv
        ) %>%
        dplyr::select(site, metric, covariate, term, Df, SumOfSqs, R2, F, p)

      out_list[[length(out_list) + 1]] <- tbl
    }

    if (!length(out_list)) {
      write_note_csv(out_csv, "no_usable_covariates")
    } else {
      out_all <- dplyr::bind_rows(out_list)
      readr::write_csv(out_all, out_csv)
    }
  }

  # Call covariate-specific PERMANOVA for Bray (this is what we need for the manuscript)
  run_permanova_covariates_site(
    M_case     = OC,
    M_ctrl     = OHp %||% OHm,
    site_name  = "oral",
    method     = "bray",
    covariates = FIXED_COVARS,
    out_csv    = file.path(outdir, "permanova_covariates_oral_bray.csv")
  )

  run_permanova_covariates_site(
    M_case     = FC,
    M_ctrl     = FH,
    site_name  = "fecal",
    method     = "bray",
    covariates = FIXED_COVARS,
    out_csv    = file.path(outdir, "permanova_covariates_fecal_bray.csv")
  )

  # PERMANOVA with disease × site interaction (oral vs fecal)
  permanova_interaction <- function(M_oral, M_fecal,
                                    method = c("bray","aitchison"),
                                    covars,
                                    with_cov,
                                    mode_label){
    method <- match.arg(method)
    if (is.null(M_oral) || is.null(M_fecal)) return(list(tbl = NULL))

    lst <- align_features(list(M_oral, M_fecal))
    if (length(lst) < 2) return(list(tbl = NULL))

    M <- cbind(lst[[1]], lst[[2]])
    labs <- colnames(M)

    md <- covars %>%
      dplyr::filter(Sample_ID %in% labs) %>%
      dplyr::mutate(
        site = dplyr::case_when(
          tolower(site) == "oral"  ~ "oral",
          tolower(site) == "fecal" ~ "fecal",
          TRUE                     ~ NA_character_
        )
      ) %>%
      dplyr::filter(!is.na(site))

    if (nrow(md) < 6 ||
        dplyr::n_distinct(md$disease) < 2 ||
        dplyr::n_distinct(md$site) < 2) {
      return(list(tbl = NULL))
    }

    usable <- if (with_cov) select_usable_covars(md) else character(0)
    if (with_cov) {
      md <- scale_continuous_if_used(md, usable)
    }

    M_use <- M[, md$Sample_ID, drop = FALSE]

    Dx <- if (method == "bray") {
      mk_dist(drop_bad_for_dist(M_use, "bray"), "bray")$D
    } else {
      mk_aitchison(M_use, pseudo)$D
    }
    if (is.null(Dx)) return(list(tbl = NULL))

    labs_b <- labels(Dx)
    md_b <- md[match(labs_b, md$Sample_ID), , drop = FALSE]

    rhs <- paste(c("disease * site", usable), collapse = " + ")
    fml <- as.formula(paste("Dx ~", rhs))

    res <- adonis2_opt(
      D         = Dx,
      formula   = fml,
      data      = md_b,
      nperm     = nperm,
      strata_vec = if ("STUDY_ID" %in% names(md_b)) md_b$STUDY_ID else NULL
    )

    out <- as.data.frame(res)
    out$term <- rownames(out)
    rownames(out) <- NULL

    out <- out %>%
      dplyr::rename(
        Df       = Df,
        SumOfSqs = SumOfSqs,
        R2       = R2,
        F        = `F`,
        p        = `Pr(>F)`
      ) %>%
      dplyr::select(term, Df, SumOfSqs, R2, F, p)

    out$covars_used <- paste(usable, collapse = ", ")
    out$mode <- mode_label

    list(tbl = out)
  }

  inter_bray_nocov <- permanova_interaction(
    align_and_cbind(list(OC, OHm)),
    align_and_cbind(list(FC, FH)),
    "bray",
    md_all,
    with_cov   = FALSE,
    mode_label = "nocov"
  )
  inter_bray_cov <- permanova_interaction(
    align_and_cbind(list(OC, OHp %||% OHm)),
    align_and_cbind(list(FC, FH)),
    "bray",
    md_all,
    with_cov   = TRUE,
    mode_label = "with_cov"
  )
  inter_ait_nocov <- permanova_interaction(
    align_and_cbind(list(OC, OHm)),
    align_and_cbind(list(FC, FH)),
    "aitchison",
    md_all,
    with_cov   = FALSE,
    mode_label = "nocov"
  )
  inter_ait_cov <- permanova_interaction(
    align_and_cbind(list(OC, OHp %||% OHm)),
    align_and_cbind(list(FC, FH)),
    "aitchison",
    md_all,
    with_cov   = TRUE,
    mode_label = "with_cov"
  )

  write_inter <- function(x_nocov, x_cov, path_csv){
    T1 <- x_nocov$tbl
    T2 <- x_cov$tbl

    if (is.null(T1) && is.null(T2)) {
      write_note_csv(path_csv, "no_data")
      return(invisible(NULL))
    }

    out <- dplyr::bind_rows(T1 %||% tibble(), T2 %||% tibble())
    if (nrow(out) == 0) {
      write_note_csv(path_csv, "no_data")
      return(invisible(NULL))
    }

    out$mode <- factor(out$mode, levels = c("nocov","with_cov"))
    readr::write_csv(out, path_csv)
  }

  write_inter(inter_bray_nocov, inter_bray_cov, file.path(outdir, "permanova_interaction_bray.csv"))
  write_inter(inter_ait_nocov,  inter_ait_cov,  file.path(outdir, "permanova_interaction_aitchison.csv"))

  # ---------- PERMDISP (Bray) ----------
  permdisp_bray <- function(M_case, M_ctrl, covars, site_name, out_csv){
    if (is.null(M_case) || is.null(M_ctrl)){ write_note_csv(out_csv,"no_matrix"); return(invisible(NULL)) }
    lst <- align_features(list(M_case, M_ctrl)); if (length(lst)<2){ write_note_csv(out_csv,"no_common_features"); return(invisible(NULL)) }
    M <- cbind(lst[[1]], lst[[2]]); labs <- colnames(M)
    md <- covars %>% filter(tolower(site)==tolower(site_name), Sample_ID %in% labs)
    if (nrow(md)<3 || n_distinct(md$disease)<2) md <- tibble(Sample_ID=labs, disease=c(rep(1L,ncol(lst[[1]])), rep(0L,ncol(lst[[2]]))))
    Mb <- drop_bad_for_dist(M[, md$Sample_ID, drop=FALSE], "bray"); Db <- mk_dist(Mb,"bray")$D
    if (is.null(Db)){ write_note_csv(out_csv,"dist_null"); return(invisible(NULL)) }
    labs_b <- labels(Db); md_b <- md[match(labs_b, md$Sample_ID),,drop=FALSE]
    grp <- factor(md_b$disease, levels=c(0,1), labels=c("Healthy","Crohn"))
    bd <- betadisper(Db, grp); pt <- suppressWarnings(permutest(bd, permutations=as.integer(get_arg("--permutations","999"))))
    write_csv(tibble(term="disease", F=as.numeric(pt$tab[1,"F"]), p=as.numeric(pt$tab[1,"Pr(>F)"])), out_csv)
  }
  permdisp_bray(OC, OHm, md_all, "oral",  file.path(outdir,"permdisp_oral_bray.csv"))
  permdisp_bray(FC, FH,  md_all, "fecal", file.path(outdir,"permdisp_fecal_bray.csv"))

  # ---------- Group-wise mean distances (RAW) ----------
  write_group_distance_csv <- function(M_all, group_map_named, out_csv){
    if (is.null(M_all) || ncol(M_all)<4){ write_note_csv(out_csv,"too_few_samples"); return(invisible(NULL)) }
    dist_one <- function(method){
      Mx <- drop_bad_for_dist(M_all, method); Dx <- mk_dist(Mx, method); if (is.null(Dx$D)) return(tibble())
      dm <- as.matrix(Dx$D); labs <- Dx$labs; g <- group_map_named[labs]; lev <- unique(na.omit(g)); out <- list()
      for (i in seq_along(lev)) for (j in i:length(lev)){
        gi <- lev[i]; gj <- lev[j]; idx <- which(g==gi); jdx <- which(g==gj)
        if (length(idx)>0 && length(jdx)>0){
          sub <- dm[idx, jdx, drop=FALSE]; if (i==j) sub <- sub[upper.tri(sub)]
          out[[length(out)+1]] <- tibble(group1=gi, group2=gj, metric=method, n_pairs=length(sub),
                                         mean_distance=ifelse(length(sub)>0, mean(sub), NA_real_))
        }
      }
      bind_rows(out)
    }
    out <- bind_rows(dist_one("bray"), dist_one("jaccard")); if (!nrow(out)) write_note_csv(out_csv,"no_pairs") else write_csv(out, out_csv)
  }
  write_group_distance_csv(M_all_RAW, group_map_RAW, file.path(outdir,"beta_group_distances.csv"))

  # ---------- Paired Crohn oral-fecal summary (Bray) ----------
  pairs_path_out <- file.path(outdir,"pairwise_oral_fecal_summary.csv")
  if (!is.null(pairs_df) && !is.null(OC) && !is.null(FC)){
    lst <- align_features(list(OC,FC)); if (length(lst)==2){
      OC2 <- lst[[1]]; FC2 <- lst[[2]]
      P <- pairs_df %>% filter(oral %in% colnames(OC2), fecal %in% colnames(FC2))
      if (nrow(P)>0){
        Mb <- cbind(OC2[,P$oral,drop=FALSE], FC2[,P$fecal,drop=FALSE])
        Mb <- drop_bad_for_dist(Mb,"bray"); Db <- mk_dist(Mb,"bray")$D
        if (!is.null(Db)){
          dm <- as.matrix(Db); labs <- labels(Db)
          get_one <- function(o,f) if (o %in% labs && f %in% labs) dm[o,f] else NA_real_
          dists <- mapply(get_one, P$oral, P$fecal)
          out <- tibble(Metric="BrayCurtis", n=sum(is.finite(dists)), mean=mean(dists,na.rm=TRUE),
                        median=median(dists,na.rm=TRUE), q1=as.numeric(quantile(dists,.25,na.rm=TRUE)),
                        q3=as.numeric(quantile(dists,.75,na.rm=TRUE)), iqr=IQR(dists,na.rm=TRUE))
          write_csv(out, pairs_path_out)
        } else write_note_csv(pairs_path_out, "dist_null")
      } else write_note_csv(pairs_path_out, "no_pairs_overlap")
    } else write_note_csv(pairs_path_out, "no_common_features")
  } else write_note_csv(pairs_path_out, "no_pairs_file")

  # Session info & done flag
  sink(file.path(outdir,"debug","beta_session_info.txt")); print(sessionInfo()); sink()
  writeLines(c("OK", format(Sys.time())), con=file.path(outdir,"debug","done.flag"))
}

# ---------------------------- TASK: summarize (PERMANOVA/PERMDISP) ----------------------------
task_summarize <- function(){
  level <- get_arg("--level", "species")
  outdir <- get_arg("--outdir", "results/beta_summary")
  indir <- file.path("results","beta", level)
  dir.create(file.path(outdir, level), showWarnings = FALSE, recursive = TRUE)

  safe_read <- function(p) if (file.exists(p)) suppressMessages(readr::read_csv(p, show_col_types = FALSE)) else NULL
  pick_term <- function(df, term_regex="^disease$|^disease:site$|^site:disease$") {
    if (is.null(df) || !all(c("term","F","R2","p") %in% names(df))) return(NULL)
    df %>% filter(str_detect(term, term_regex))
  }

  # Site-specific PERMANOVA
  oral_bray_nocov  <- pick_term(safe_read(file.path(indir, "permanova_oral_bray_nocov.csv")), "^disease$")
  fecal_bray_nocov <- pick_term(safe_read(file.path(indir, "permanova_fecal_bray_nocov.csv")), "^disease$")
  oral_bray_cov    <- pick_term(safe_read(file.path(indir, "permanova_oral_bray_with_cov.csv")), "^disease$")
  fecal_bray_cov   <- pick_term(safe_read(file.path(indir, "permanova_fecal_bray_with_cov.csv")), "^disease$")

  oral_ait_nocov   <- pick_term(safe_read(file.path(indir, "permanova_oral_aitchison_nocov.csv")), "^disease$")
  fecal_ait_nocov  <- pick_term(safe_read(file.path(indir, "permanova_fecal_aitchison_nocov.csv")), "^disease$")
  oral_ait_cov     <- pick_term(safe_read(file.path(indir, "permanova_oral_aitchison_with_cov.csv")), "^disease$")
  fecal_ait_cov    <- pick_term(safe_read(file.path(indir, "permanova_fecal_aitchison_with_cov.csv")), "^disease$")

  # Interaction
  inter_bray <- safe_read(file.path(indir, "permanova_interaction_bray.csv"))
  inter_ait  <- safe_read(file.path(indir, "permanova_interaction_aitchison.csv"))
  if (!is.null(inter_bray)) inter_bray$metric <- "Bray"
  if (!is.null(inter_ait))  inter_ait$metric  <- "Aitchison"
  pick_interaction <- function(df) {
    if (is.null(df)) return(NULL)
    df %>% filter(str_detect(term, "^disease:site$|^site:disease$")) %>%
      transmute(where="interaction(oral×fecal)", metric=metric, mode=mode, term, F, R2, p)
  }

  # PERMDISP (Bray)
  permdisp_oral  <- safe_read(file.path(indir, "permdisp_oral_bray.csv"))
  permdisp_fecal <- safe_read(file.path(indir, "permdisp_fecal_bray.csv"))
  if (!is.null(permdisp_oral))  permdisp_oral$site  <- "oral"
  if (!is.null(permdisp_fecal)) permdisp_fecal$site <- "fecal"

  rowify <- function(df, where, metric, mode) {
    if (is.null(df)) return(NULL)
    df %>% mutate(where=where, metric=metric, mode=mode) %>%
      select(where, metric, mode, term, F, R2, p)
  }

  sum_rows <- bind_rows(
    rowify(oral_bray_nocov,  "site=oral",  "Bray",      "nocov"),
    rowify(fecal_bray_nocov, "site=fecal", "Bray",      "nocov"),
    rowify(oral_bray_cov,    "site=oral",  "Bray",      "with_cov"),
    rowify(fecal_bray_cov,   "site=fecal", "Bray",      "with_cov"),
    rowify(oral_ait_nocov,   "site=oral",  "Aitchison", "nocov"),
    rowify(fecal_ait_nocov,  "site=fecal", "Aitchison", "nocov"),
    rowify(oral_ait_cov,     "site=oral",  "Aitchison", "with_cov"),
    rowify(fecal_ait_cov,    "site=fecal", "Aitchison", "with_cov"),
    pick_interaction(inter_bray),
    pick_interaction(inter_ait)
  ) %>% mutate(level=level)

  # Save CSV
  outfile_csv <- file.path(outdir, level, "permanova_permdisp_summary.csv")
  if (!is.null(sum_rows) && nrow(sum_rows) > 0) {
    readr::write_csv(sum_rows, outfile_csv)
  } else {
    readr::write_csv(tibble(note="no_data"), outfile_csv)
  }

  # Plot: PERMANOVA (R2) by site/metric/mode (only disease or interaction terms)
  if (!is.null(sum_rows) && nrow(sum_rows) > 0) {
    sum_rows2 <- sum_rows %>%
      mutate(label = paste0(metric, " · ", where, " · ", mode)) %>%
      mutate(sig = ifelse(p < 0.05, "*", ""))

    p1 <- ggplot(sum_rows2 %>% filter(str_detect(term, "^disease$|^disease:site$|^site:disease$")), 
                 aes(x=reorder(label, R2, na.rm=TRUE), y=R2, fill=metric)) +
      geom_col() +
      geom_text(aes(label=sig), vjust=-0.2) +
      coord_flip() +
      labs(title=paste("PERMANOVA —", level), x=NULL, y="R² (disease or disease:site)") +
      theme_bw(base_size = 12)

    ggsave(file.path(outdir, level, "permanova_summary.png"), p1, width=8, height=6, dpi=300, bg="white")
  }

    # --------------------------------------------------------------------------
  # PERMANOVA for individual covariates (Bray, site-specific)
  #   - expects files created by run_permanova_covariates_site():
  #       permanova_covariates_oral_bray.csv
  #       permanova_covariates_fecal_bray.csv
  #   - each file should contain: site, metric, covariate, term, Df, SumOfSqs, R2, F, p
  #   - we summarise R² and p for the covariate rows (term == covariate)
  # --------------------------------------------------------------------------
  cov_oral  <- safe_read(file.path(indir, "permanova_covariates_oral_bray.csv"))
  cov_fecal <- safe_read(file.path(indir, "permanova_covariates_fecal_bray.csv"))

  cov_list <- list(cov_oral, cov_fecal)
  cov_list <- cov_list[!vapply(cov_list, is.null, logical(1))]

  cov_outfile_csv <- file.path(outdir, level, "permanova_covariates_summary.csv")

  if (length(cov_list) == 0) {
    # No covariate PERMANOVA files found
    readr::write_csv(tibble(note = "no_covariate_permanova_data"), cov_outfile_csv)
  } else {
    cov_all <- dplyr::bind_rows(cov_list)

    # Keep only rows where "term" corresponds to the covariate itself
    # (rows for "disease" are already summarised in the main PERMANOVA summary)
    if (all(c("covariate","term","site","metric","R2","F","p") %in% names(cov_all))) {
      cov_keep <- cov_all %>%
        dplyr::filter(!is.na(covariate),
                      term == covariate)

      if (nrow(cov_keep) == 0) {
        readr::write_csv(tibble(note = "no_covariate_rows_after_filter"), cov_outfile_csv)
      } else {
        # Add level and a display label
        cov_keep <- cov_keep %>%
          dplyr::mutate(
            level = level,
            label = paste0(covariate, " · ", site),
            sig   = dplyr::if_else(p < 0.05, "*", "")
          )

        # Save summary CSV
        readr::write_csv(cov_keep, cov_outfile_csv)

        # Optional barplot: R² per covariate and site (Bray only)
        p_cov <- ggplot(
          cov_keep,
          aes(x = reorder(label, R2, na.rm = TRUE),
              y = R2,
              fill = site)
        ) +
          geom_col() +
          geom_text(aes(label = sig), vjust = -0.2) +
          coord_flip() +
          labs(
            title = paste("PERMANOVA — Covariates (Bray,", level, ")"),
            x = NULL,
            y = "R² (covariate effect)"
          ) +
          theme_bw(base_size = 12)

        ggsave(
          filename = file.path(outdir, level, "permanova_covariates_summary.png"),
          plot     = p_cov,
          width    = 8,
          height   = 6,
          dpi      = 300,
          bg       = "white"
        )
      }
    } else {
      # Columns not in expected format
      readr::write_csv(tibble(note = "covariate_permanova_invalid_format"), cov_outfile_csv)
    }
  }


  # Plot: PERMDISP (Bray) if available
  if (!is.null(permdisp_oral) || !is.null(permdisp_fecal)) {
    pd <- bind_rows(permdisp_oral, permdisp_fecal)
    if (!is.null(pd) && all(c("site","F","p") %in% names(pd))) {
      pd$level <- level
      pd$label <- paste0("Bray · ", pd$site)
      pd$sig <- ifelse(pd$p < 0.05, "*", "")
      p2 <- ggplot(pd, aes(x=reorder(label, F, na.rm=TRUE), y=F, fill=site)) +
        geom_col() + geom_text(aes(label=sig), vjust=-0.2) +
        coord_flip() +
        labs(title=paste("PERMDISP —", level), x=NULL, y="F") +
        theme_bw(base_size = 12)
      ggsave(file.path(outdir, level, "permdisp_summary.png"), p2, width=6.5, height=4.5, dpi=300, bg="white")
    }
  }
}

# ---------------------------- TASK: oral_prevnew (special PCoA) ----------------------------
task_oral_prevnew <- function(){
  # Required args
  oc  <- get_arg("--oral-crohn")   %||% stop("--oral-crohn is required")
  ohc <- get_arg("--oral-healthy-combined") %||% get_arg("--oral-healthy-merged") %||% stop("--oral-healthy-combined/--oral-healthy-merged is required")
  fc  <- get_arg("--fecal-crohn")  %||% stop("--fecal-crohn is required")
  fh  <- get_arg("--fecal-healthy")
  outdir <- get_arg("--outdir", "results/beta_oral_prevnew")
  metric <- tolower(get_arg("--metric","bray"))
  level  <- tolower(get_arg("--level","species"))

  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

  # Helpers for this task
  ensure_taxa_by_samples <- function(df) {
    row_tax <- mean(grepl("^(s__|g__)", rownames(df))) > 0.5
    col_tax <- mean(grepl("^(s__|g__)", colnames(df))) > 0.5
    if (!row_tax && col_tax) return(t(df))
    return(df)
  }
  pcoa_from_pct <- function(pct_mat) {
    if (!requireNamespace("ape", quietly = TRUE)) stop("Package 'ape' required for PCoA")
    pct <- as.matrix(pct_mat); pct[is.na(pct)] <- 0
    X <- t(pct)
    D <- vegan::vegdist(X, method = "bray")
    pc <- ape::pcoa(D)
    coords <- as.data.frame(pc$vectors[, 1:2, drop=FALSE])
    colnames(coords) <- c("PC1","PC2")
    var1 <- round(100 * pc$values$Relative_eig[1], 1)
    var2 <- round(100 * pc$values$Relative_eig[2], 1)
    list(coords=coords, var1=var1, var2=var2)
  }
  plot_pcoa <- function(df, var1, var2, title, out_png) {
    p <- ggplot(df, aes(PC1, PC2, color=Group)) +
      geom_point(size=2.4, alpha=0.90) +
      theme_minimal(base_size = 14) +
      theme(legend.title = element_text(size=13, face="bold"),
            legend.position = "right") +
      xlab(sprintf("PC1 (%.1f%%)", var1)) +
      ylab(sprintf("PC2 (%.1f%%)", var2)) +
      ggtitle(title)
    ggsave(out_png, p, width=10, height=7.5, dpi=160, bg="white")
  }
  to_mat <- function(tbl) { rn <- tbl[[1]]; m <- as.data.frame(tbl[,-1, drop=FALSE]); rownames(m) <- rn; m }

  # Load data
  oc_tbl <- read_csv(oc, show_col_types = FALSE)
  oh_tbl <- read_csv(ohc, show_col_types = FALSE)
  fc_tbl <- read_csv(fc, show_col_types = FALSE)
  fh_tbl <- if (!is.null(fh) && nzchar(fh) && file.exists(fh)) read_csv(fh, show_col_types = FALSE) else NULL


  oc_m <- ensure_taxa_by_samples(to_mat(oc_tbl))
  oh_m <- ensure_taxa_by_samples(to_mat(oh_tbl))
  fc_m <- ensure_taxa_by_samples(to_mat(fc_tbl))
  fh_m <- if (!is.null(fh_tbl)) ensure_taxa_by_samples(to_mat(fh_tbl)) else NULL


  # Optionally collapse to genus/species-like labels (no taxonomy parser here; assume already percent abundances)
  # Align taxa union for oral comparison
  tax_oral <- union(rownames(oc_m), rownames(oh_m))
  oc_o <- oc_m[match(tax_oral, rownames(oc_m)), , drop=FALSE]; rownames(oc_o) <- tax_oral; oc_o[is.na(oc_o)] <- 0
  oh_o <- oh_m[match(tax_oral, rownames(oh_m)), , drop=FALSE]; rownames(oh_o) <- tax_oral; oh_o[is.na(oh_o)] <- 0

  # Align for adding fecal Crohn
  tax_all <- Reduce(
  union,
  c(list(rownames(oc_m), rownames(oh_m), rownames(fc_m)),
      if (!is.null(fh_m)) list(rownames(fh_m)) else list())
  )

  oc_all <- oc_m[match(tax_all, rownames(oc_m)), , drop=FALSE]; rownames(oc_all) <- tax_all; oc_all[is.na(oc_all)] <- 0
  oh_all <- oh_m[match(tax_all, rownames(oh_m)), , drop=FALSE]; rownames(oh_all) <- tax_all; oh_all[is.na(oh_all)] <- 0
  fc_all <- fc_m[match(tax_all, rownames(fc_m)), , drop=FALSE]; rownames(fc_all) <- tax_all; fc_all[is.na(fc_all)] <- 0
  fh_all <- if (!is.null(fh_m)) { x <- fh_m[match(tax_all, rownames(fh_m)), , drop=FALSE]; rownames(x) <- tax_all; x[is.na(x)] <- 0; x } else NULL

  # Tag Healthy-Previous vs Healthy-New by column names
  oh_cols <- colnames(oh_m)
  is_new      <- grepl("^SRR", oh_cols)
  is_previous <- grepl("^[0-9]{6}$", oh_cols)
  is_other    <- !(is_new | is_previous)

  # (1) Oral only
  mat_oral <- cbind(oc_o, oh_o)
  pc_oral  <- pcoa_from_pct(mat_oral)
  df_oral  <- pc_oral$coords; df_oral$Sample <- rownames(df_oral)
  labs_oral <- c(
    rep("Crohn-Oral", ncol(oc_o)),
    if (any(is_previous)) rep("Healthy-Previous", sum(is_previous)) else NULL,
    if (any(is_new))      rep("Healthy-New",      sum(is_new))      else NULL,
    if (any(is_other))    rep("Healthy-Other",    sum(is_other))    else NULL
  )
  df_oral$Group <- factor(labs_oral, levels=c("Crohn-Oral","Healthy-Previous","Healthy-New","Healthy-Other"))
  plot_pcoa(df_oral, pc_oral$var1, pc_oral$var2,
            "PCoA (BRAY) — Oral: Crohn vs Healthy-Previous vs Healthy-New",
            file.path(outdir, sprintf("pcoa_oral_prev_new_crohn_%s.png", level)))

  # (2) Oral groups + Crohn-Fecal
  mat_all <- if (is.null(fh_all)) cbind(oc_all, oh_all, fc_all) else cbind(oc_all, oh_all, fc_all, fh_all)
  pc_all  <- pcoa_from_pct(mat_all)
  df_all  <- pc_all$coords; df_all$Sample <- rownames(df_all)
  labs_all <- c(
    rep("Crohn-Oral", ncol(oc_all)),
    if (any(is_previous)) rep("Healthy-Previous", sum(is_previous)) else NULL,
    if (any(is_new))      rep("Healthy-New",      sum(is_new))      else NULL,
    if (any(is_other))    rep("Healthy-Other",    sum(is_other))    else NULL,
    rep("Crohn-Fecal", ncol(fc_all)),
    if (!is.null(fh_all)) rep("Healthy-Fecal", ncol(fh_all)) else NULL
  )
  df_all$Group <- factor(
    labs_all,
    levels = c("Crohn-Oral","Healthy-Previous","Healthy-New","Healthy-Other","Crohn-Fecal","Healthy-Fecal")
  )
  plot_pcoa(df_all, pc_all$var1, pc_all$var2,
            "PCoA (BRAY) — Oral groups + Fecal",
            file.path(outdir, sprintf("pcoa_oral_prev_new_crohn_plus_fecal_%s.png", level)))

  cat("[OK] Wrote PCoA (oral prev/new) figures to:", outdir, "\n")
}

# ---------------------------- Main dispatcher ----------------------------
main_dispatch <- function(){
  task <- tolower(get_arg("--task", "main"))
  if (task == "main") {
    task_main()
  } else if (task == "summarize") {
    task_summarize()
  } else if (task == "oral_prevnew") {
    task_oral_prevnew()
  } else {
    stop("Unknown --task. Use one of: main | summarize | oral_prevnew")
  }
}

# ---------------------------- Run ----------------------------
invisible(main_dispatch())
