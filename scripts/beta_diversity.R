# beta_diversity.R
# PCoA + PERMANOVA(+files) + PERMDISP (site) + paired distances + summaries.
# Uses group colors strictly from config.yaml.

suppressPackageStartupMessages({
  library(vegan)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(yaml)
})

set.seed(42)

# ---- Load color config from YAML ----
cfg <- yaml.load_file("config.yaml")
pal_cfg <- unlist(cfg$colors)

get_group_colors <- function(order) {
  # returns a named vector of hex colors for the requested group order
  cols <- pal_cfg[order]
  if (any(is.na(cols))) stop("Missing color(s) in config.yaml for: ",
                             paste(order[is.na(cols)], collapse=", "))
  names(cols) <- order
  cols
}

# ---- Parse arguments from Snakemake ----
args <- commandArgs(trailingOnly=TRUE)
oral_file     <- args[1]
fecal_file    <- args[2]
healthy_file  <- args[3]
matched_file  <- args[4]
outdir        <- args[5]
dir.create(outdir, showWarnings=FALSE, recursive=TRUE)

# ---- Load input data ----
oral    <- read.csv(oral_file, row.names=1, check.names=FALSE)
fecal   <- read.csv(fecal_file, row.names=1, check.names=FALSE)
healthy <- read.csv(healthy_file, row.names=1, check.names=FALSE)
matched <- read.csv(matched_file)

stopifnot(all(matched$Oral_col  %in% colnames(oral)))
stopifnot(all(matched$Fecal_col %in% colnames(fecal)))

# ---- Align features across tables ----
align_features <- function(df_list) {
  common <- Reduce(intersect, lapply(df_list, rownames))
  lapply(df_list, function(df) df[common, , drop=FALSE])
}

# ---- PCoA helpers ----
axis_labels <- function(ord) {
  # percentage of variance on axes (cmdscale with eig)
  eig <- ord$eig
  pcv <- eig / sum(eig)
  c(
    paste0("PCoA 1 (", sprintf("%.1f", pcv[1]*100), "%)"),
    paste0("PCoA 2 (", sprintf("%.1f", pcv[2]*100), "%)")
  )
}

pcoa_and_plot <- function(df, sample_labels, method, filename, title, color_order=NULL, highlight=NULL) {
  d   <- t(df)
  dis <- vegan::vegdist(d, method=method)   # Jaccard here is quantitative (Ružička)
  ord <- cmdscale(dis, eig=TRUE, k=2)
  labs <- axis_labels(ord)

  groups <- unique(sample_labels)
  pcoa_df <- data.frame(
    Sample = rownames(d),
    Axis1  = ord$points[,1],
    Axis2  = ord$points[,2],
    Group  = sample_labels
  )

  use_order <- if (!is.null(color_order)) color_order else groups
  pal <- get_group_colors(use_order)

  alpha_map <- setNames(rep(0.65, length(use_order)), use_order)
  if (!is.null(highlight)) alpha_map[highlight] <- 1

  p <- ggplot(pcoa_df, aes(x=Axis1, y=Axis2, color=Group, fill=Group, alpha=Group)) +
    geom_point(size=3) +
    stat_ellipse(type="norm", linetype=2, linewidth=1, alpha=0.15, aes(group=Group)) +
    labs(title=title, x=labs[1], y=labs[2]) +
    theme_minimal(base_size=15) +
    theme(panel.background = element_rect(fill='white', color='white'),
          plot.background  = element_rect(fill='white', color='white'),
          legend.title     = element_blank()) +
    scale_color_manual(values=pal) +
    scale_fill_manual(values=pal) +
    scale_alpha_manual(values=alpha_map)
  ggsave(filename=file.path(outdir, filename), plot=p, width=6, height=5, dpi=300, bg="white")
  list(dist=dis, pcoa_df=pcoa_df)
}

pcoa_and_plot_lines <- function(df, sample_labels, method, filename, title, color_order=NULL, highlight=NULL, pair_indices=NULL) {
  d   <- t(df)
  dis <- vegan::vegdist(d, method=method)
  ord <- cmdscale(dis, eig=TRUE, k=2)
  labs <- axis_labels(ord)

  groups <- unique(sample_labels)
  pcoa_df <- data.frame(
    Sample = rownames(d),
    Axis1  = ord$points[,1],
    Axis2  = ord$points[,2],
    Group  = sample_labels,
    Pair   = pair_indices
  )

  use_order <- if (!is.null(color_order)) color_order else groups
  pal <- get_group_colors(use_order)

  alpha_map <- setNames(rep(0.65, length(use_order)), use_order)
  if (!is.null(highlight)) alpha_map[highlight] <- 1

  p <- ggplot(pcoa_df, aes(x=Axis1, y=Axis2, color=Group, fill=Group, alpha=Group)) +
    geom_line(aes(group=Pair), color="grey60", linewidth=0.65, alpha=0.45, show.legend=FALSE) +
    geom_point(size=3) +
    stat_ellipse(type="norm", linetype=2, linewidth=1, alpha=0.15, aes(group=Group)) +
    labs(title=title, x=labs[1], y=labs[2]) +
    theme_minimal(base_size=15) +
    theme(panel.background = element_rect(fill='white', color='white'),
          plot.background  = element_rect(fill='white', color='white'),
          legend.title     = element_blank()) +
    scale_color_manual(values=pal) +
    scale_fill_manual(values=pal) +
    scale_alpha_manual(values=alpha_map)
  ggsave(filename=file.path(outdir, filename), plot=p, width=6, height=5, dpi=300, bg="white")
  list(dist=dis, pcoa_df=pcoa_df)
}

# =========================
#  PCoA COMPARISONS
# =========================

# 1) Crohn Oral vs Healthy Oral (NO lines)
list1 <- align_features(list(oral, healthy))
oral1    <- list1[[1]]
healthy1 <- list1[[2]]
oral_all <- cbind(oral1, healthy1)
group_labels <- c(rep("Crohn-Oral", ncol(oral1)), rep("Healthy-Oral", ncol(healthy1)))

bray1 <- pcoa_and_plot(oral_all, group_labels, "bray",
                       "pcoa_bray_oral_vs_healthy.png",
                       "PCoA (Bray) Crohn Oral vs Healthy Oral",
                       c("Crohn-Oral","Healthy-Oral"),
                       highlight="Crohn-Oral")
jacc1 <- pcoa_and_plot(oral_all, group_labels, "jaccard",
                       "pcoa_jaccard_oral_vs_healthy.png",
                       "PCoA (Jaccard) Crohn Oral vs Healthy Oral",
                       c("Crohn-Oral","Healthy-Oral"),
                       highlight="Crohn-Oral")

# 2) Crohn Oral vs Crohn Fecal (paired, WITH lines)
list2 <- align_features(list(oral, fecal))
oral2_matched  <- list2[[1]][, matched$Oral_col,  drop=FALSE]
fecal2_matched <- list2[[2]][, matched$Fecal_col, drop=FALSE]
oral_fecal_all <- cbind(oral2_matched, fecal2_matched)
of_labels <- c(rep("Crohn-Oral", ncol(oral2_matched)), rep("Crohn-Fecal", ncol(fecal2_matched)))
pair_idx <- rep(seq_len(ncol(oral2_matched)), 2)

bray2 <- pcoa_and_plot_lines(oral_fecal_all, of_labels, "bray",
                             "pcoa_bray_oral_vs_fecal.png",
                             "PCoA (Bray) Crohn Oral vs Fecal (paired)",
                             c("Crohn-Oral","Crohn-Fecal"),
                             highlight="Crohn-Fecal",
                             pair_indices=pair_idx)
jacc2 <- pcoa_and_plot_lines(oral_fecal_all, of_labels, "jaccard",
                             "pcoa_jaccard_oral_vs_fecal.png",
                             "PCoA (Jaccard) Crohn Oral vs Fecal (paired)",
                             c("Crohn-Oral","Crohn-Fecal"),
                             highlight="Crohn-Fecal",
                             pair_indices=pair_idx)

# 3) All groups (NO lines)
list3 <- align_features(list(oral, fecal, healthy))
oral3    <- list3[[1]]
fecal3   <- list3[[2]]
healthy3 <- list3[[3]]
all_df <- cbind(oral3, fecal3, healthy3)
all_labels <- c(rep("Crohn-Oral", ncol(oral3)),
                rep("Crohn-Fecal", ncol(fecal3)),
                rep("Healthy-Oral", ncol(healthy3)))

bray3 <- pcoa_and_plot(all_df, all_labels, "bray",
                       "pcoa_bray_allgroups.png",
                       "PCoA (Bray) All Groups",
                       c("Crohn-Oral","Crohn-Fecal","Healthy-Oral"),
                       highlight="Crohn-Oral")
jacc3 <- pcoa_and_plot(all_df, all_labels, "jaccard",
                       "pcoa_jaccard_allgroups.png",
                       "PCoA (Jaccard) All Groups",
                       c("Crohn-Oral","Crohn-Fecal","Healthy-Oral"),
                       highlight="Crohn-Oral")

# =========================
#  PERMANOVA + PERMDISP
# =========================

write_permanova <- function(dist, groups, filepath, title){
  md <- data.frame(group=groups)
  res <- adonis2(dist ~ group, data=md, permutations=999)
  # Pull the first row (group effect)
  line <- capture.output({
    cat(title, "\n")
    print(res)
    cat("\nKey: F=", round(res$F[1], 3), " R2=", round(res$R2[1], 3),
        " p=", signif(res$`Pr(>F)`[1], 4), "\n", sep="")
  })
  writeLines(line, filepath)
}

write_permdisp <- function(dist, groups, filepath, title){
  md <- data.frame(group=groups)
  bd <- betadisper(dist, md$group)
  perm <- permutest(bd, permutations=999)
  line <- capture.output({
    cat(title, "\n")
    print(perm)
  })
  writeLines(line, filepath)
}

# Crohn Oral vs Healthy Oral
write_permanova(bray1$dist, group_labels,
                file.path(outdir,"permanova_oral_CH_bray.txt"),
                "PERMANOVA (Bray) Crohn-Oral vs Healthy-Oral")
write_permanova(jacc1$dist, group_labels,
                file.path(outdir,"permanova_oral_CH_jaccard.txt"),
                "PERMANOVA (Jaccard) Crohn-Oral vs Healthy-Oral")

# Crohn Oral vs Crohn Fecal (SITE effect within Crohn)
site_labels <- factor(c(rep("Oral",  ncol(oral2_matched)),
                        rep("Fecal", ncol(fecal2_matched))),
                      levels=c("Oral","Fecal"))

write_permanova(bray2$dist, site_labels,
                file.path(outdir,"permanova_site_bray.txt"),
                "PERMANOVA (Bray) Site effect (Crohn Oral vs Fecal)")
write_permanova(jacc2$dist, site_labels,
                file.path(outdir,"permanova_site_jaccard.txt"),
                "PERMANOVA (Jaccard) Site effect (Crohn Oral vs Fecal)")

# PERMDISP (homogeneity of dispersion) for SITE
write_permdisp(bray2$dist, site_labels,
               file.path(outdir,"permdisp_site_bray.txt"),
               "PERMDISP (Bray) Site effect (Crohn Oral vs Fecal)")
write_permdisp(jacc2$dist, site_labels,
               file.path(outdir,"permdisp_site_jaccard.txt"),
               "PERMDISP (Jaccard) Site effect (Crohn Oral vs Fecal)")

# Also keep your combined summary as before
sink(file.path(outdir, "beta_stats.txt"))
cat("## Beta Diversity Statistical Tests\n\n")
cat("Crohn Oral vs Healthy Oral (Bray/Jaccard)\n")
print(adonis2(bray1$dist ~ group_labels, permutations=999))
print(adonis2(jacc1$dist ~ group_labels, permutations=999))
cat("\nCrohn Oral vs Crohn Fecal (Bray/Jaccard)\n")
print(adonis2(bray2$dist ~ site_labels, permutations=999))
print(adonis2(jacc2$dist ~ site_labels, permutations=999))
cat("\nAll Groups (Bray/Jaccard)\n")
print(adonis2(bray3$dist ~ all_labels, permutations=999))
print(adonis2(jacc3$dist ~ all_labels, permutations=999))
sink()

# Save all-groups Bray distance matrix
write.csv(as.matrix(bray3$dist), file.path(outdir, "beta_group_distances.csv"))
# =========================
#  PAIRED ORAL–FECAL DISTANCES (Crohn)
# =========================

pairwise_bray    <- numeric(ncol(oral2_matched))
pairwise_jaccard <- numeric(ncol(oral2_matched))
for (i in seq_len(ncol(oral2_matched))) {
  o <- as.numeric(oral2_matched[, i])
  f <- as.numeric(fecal2_matched[, i])
  pairwise_bray[i]    <- vegan::vegdist(rbind(o, f), method = "bray")[1]
  pairwise_jaccard[i] <- vegan::vegdist(rbind(o, f), method = "jaccard")[1]
}

pairwise_df <- data.frame(
  STUDY_ID   = matched$STUDY_ID,
  BrayCurtis = pairwise_bray,
  Jaccard    = pairwise_jaccard
)
write.csv(pairwise_df,
          file.path(outdir, "pairwise_oral_fecal_distances.csv"),
          row.names = FALSE)

# Long format (create before use)
pairwise_long <- pairwise_df |>
  tidyr::pivot_longer(cols = c("BrayCurtis","Jaccard"),
                      names_to = "Metric", values_to = "Distance")
pairwise_long$Metric <- factor(pairwise_long$Metric,
                               levels = c("BrayCurtis","Jaccard"))

# Summary CSV (Snakemake expects this)
pairwise_summary <- pairwise_long |>
  dplyr::group_by(Metric) |>
  dplyr::summarise(
    n      = dplyr::n(),
    mean   = mean(Distance, na.rm = TRUE),
    median = median(Distance, na.rm = TRUE),
    q1     = as.numeric(quantile(Distance, 0.25, na.rm = TRUE)),
    q3     = as.numeric(quantile(Distance, 0.75, na.rm = TRUE)),
    iqr    = IQR(Distance, na.rm = TRUE)
  )
readr::write_csv(pairwise_summary,
                 file.path(outdir, "pairwise_oral_fecal_summary.csv"))

# Violin plot
p <- ggplot(pairwise_long, aes(x=Metric, y=Distance, fill=Metric)) +
  geom_violin(trim=FALSE, alpha=0.7, width=0.7) +
  geom_jitter(width=0.15, alpha=0.5, color="black") +
  stat_summary(fun=median, geom="point", size=3, color="black") +
  labs(title="Paired Oral–Fecal Beta Diversity (Crohn)", y="Distance", x="") +
  theme_bw(base_size=14) +
  scale_fill_manual(values=c("BrayCurtis"="#1B4F72","Jaccard"="#7FB3D5"))
ggsave(file.path(outdir, "pairwise_oral_fecal_violinplot.png"),
       p, width=6, height=5, dpi=300, bg="white")

message("[INFO] All beta-diversity outputs saved to: ", outdir)
