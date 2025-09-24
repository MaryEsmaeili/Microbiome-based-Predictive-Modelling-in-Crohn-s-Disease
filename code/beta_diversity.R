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
  cols <- pal_cfg[order]
  if (any(is.na(cols))) stop("Missing color(s) in config.yaml for: ",
                             paste(order[is.na(cols)], collapse=", "))
  names(cols) <- order
  cols
}

# ---- Parse arguments from Snakemake ----
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 6) {
  stop("Expected 6 args: oral_file fecal_file healthy_oral_file healthy_fecal_file matched_file outdir")
}
oral_file        <- args[1]
fecal_file       <- args[2]
healthy_oral_file<- args[3]
healthy_fecal_file<-args[4]
matched_file     <- args[5]
outdir           <- args[6]
dir.create(outdir, showWarnings=FALSE, recursive=TRUE)

# ---- Load input data ----
oral          <- read.csv(oral_file, row.names=1, check.names=FALSE)
fecal         <- read.csv(fecal_file, row.names=1, check.names=FALSE)
healthy_oral  <- read.csv(healthy_oral_file, row.names=1, check.names=FALSE)
healthy_fecal <- read.csv(healthy_fecal_file, row.names=1, check.names=FALSE)
matched       <- read.csv(matched_file)

stopifnot(all(matched$Oral_col  %in% colnames(oral)))
stopifnot(all(matched$Fecal_col %in% colnames(fecal)))

# ---- Align features across tables ----
align_features <- function(df_list) {
  common <- Reduce(intersect, lapply(df_list, rownames))
  lapply(df_list, function(df) df[common, , drop=FALSE])
}

# ---- PCoA helpers ----
axis_labels <- function(ord) {
  eig <- ord$eig
  pcv <- eig / sum(eig)
  c(
    paste0("PCoA 1 (", sprintf("%.1f", pcv[1]*100), "%)"),
    paste0("PCoA 2 (", sprintf("%.1f", pcv[2]*100), "%)")
  )
}

pcoa_and_plot <- function(df, sample_labels, method, filename, title, color_order=NULL, highlight=NULL) {
  d   <- t(df)
  dis <- vegan::vegdist(d, method=method)
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

# 1) Crohn Oral vs Healthy Oral
list_oh <- align_features(list(oral, healthy_oral))
oral_oh    <- list_oh[[1]]
healthy_oh <- list_oh[[2]]
mat_oh <- cbind(oral_oh, healthy_oh)
labels_oh <- c(rep("Crohn-Oral", ncol(oral_oh)), rep("Healthy-Oral", ncol(healthy_oh)))

bray_oh <- pcoa_and_plot(mat_oh, labels_oh, "bray",
                         "pcoa_bray_oral_vs_healthy.png",
                         "PCoA (Bray) Crohn Oral vs Healthy Oral",
                         c("Crohn-Oral","Healthy-Oral"),
                         highlight="Crohn-Oral")
jacc_oh <- pcoa_and_plot(mat_oh, labels_oh, "jaccard",
                         "pcoa_jaccard_oral_vs_healthy.png",
                         "PCoA (Jaccard) Crohn Oral vs Healthy Oral",
                         c("Crohn-Oral","Healthy-Oral"),
                         highlight="Crohn-Oral")

# 2) Crohn Fecal vs Healthy Fecal
list_fh <- align_features(list(fecal, healthy_fecal))
fecal_fh    <- list_fh[[1]]
healthy_fh  <- list_fh[[2]]
mat_fh <- cbind(fecal_fh, healthy_fh)
labels_fh <- c(rep("Crohn-Fecal", ncol(fecal_fh)), rep("Healthy-Fecal", ncol(healthy_fh)))

bray_fh <- pcoa_and_plot(mat_fh, labels_fh, "bray",
                         "pcoa_bray_fecal_vs_healthyfecal.png",
                         "PCoA (Bray) Crohn Fecal vs Healthy Fecal",
                         c("Crohn-Fecal","Healthy-Fecal"),
                         highlight="Crohn-Fecal")
jacc_fh <- pcoa_and_plot(mat_fh, labels_fh, "jaccard",
                         "pcoa_jaccard_fecal_vs_healthyfecal.png",
                         "PCoA (Jaccard) Crohn Fecal vs Healthy Fecal",
                         c("Crohn-Fecal","Healthy-Fecal"),
                         highlight="Crohn-Fecal")

# 3) Crohn Oral vs Crohn Fecal (paired, WITH lines)
list_of <- align_features(list(oral, fecal))
oral_of_mat  <- list_of[[1]][, matched$Oral_col,  drop=FALSE]
fecal_of_mat <- list_of[[2]][, matched$Fecal_col, drop=FALSE]
mat_of <- cbind(oral_of_mat, fecal_of_mat)
labels_of <- c(rep("Crohn-Oral", ncol(oral_of_mat)), rep("Crohn-Fecal", ncol(fecal_of_mat)))
pair_idx <- rep(seq_len(ncol(oral_of_mat)), 2)

bray_of <- pcoa_and_plot_lines(mat_of, labels_of, "bray",
                               "pcoa_bray_oral_vs_fecal.png",
                               "PCoA (Bray) Crohn Oral vs Fecal (paired)",
                               c("Crohn-Oral","Crohn-Fecal"),
                               highlight="Crohn-Fecal",
                               pair_indices=pair_idx)
jacc_of <- pcoa_and_plot_lines(mat_of, labels_of, "jaccard",
                               "pcoa_jaccard_oral_vs_fecal.png",
                               "PCoA (Jaccard) Crohn Oral vs Fecal (paired)",
                               c("Crohn-Oral","Crohn-Fecal"),
                               highlight="Crohn-Fecal",
                               pair_indices=pair_idx)

# 4) All groups (NO lines) – 4 groups now
list_all <- align_features(list(oral, fecal, healthy_oral, healthy_fecal))
oral_all    <- list_all[[1]]
fecal_all   <- list_all[[2]]
healthy_all <- list_all[[3]]
healthy_f_all <- list_all[[4]]
all_mat <- cbind(oral_all, fecal_all, healthy_all, healthy_f_all)
all_labels <- c(rep("Crohn-Oral",    ncol(oral_all)),
                rep("Crohn-Fecal",   ncol(fecal_all)),
                rep("Healthy-Oral",  ncol(healthy_all)),
                rep("Healthy-Fecal", ncol(healthy_f_all)))

bray_all <- pcoa_and_plot(all_mat, all_labels, "bray",
                          "pcoa_bray_allgroups.png",
                          "PCoA (Bray) All Groups",
                          c("Crohn-Oral","Crohn-Fecal","Healthy-Oral","Healthy-Fecal"),
                          highlight="Crohn-Oral")
jacc_all <- pcoa_and_plot(all_mat, all_labels, "jaccard",
                          "pcoa_jaccard_allgroups.png",
                          "PCoA (Jaccard) All Groups",
                          c("Crohn-Oral","Crohn-Fecal","Healthy-Oral","Healthy-Fecal"),
                          highlight="Crohn-Oral")

# =========================
#  PERMANOVA + PERMDISP
# =========================

write_permanova <- function(dist, groups, filepath, title){
  md <- data.frame(group=groups)
  res <- adonis2(dist ~ group, data=md, permutations=999)
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
write_permanova(bray_oh$dist, labels_oh,
                file.path(outdir,"permanova_oral_CH_bray.txt"),
                "PERMANOVA (Bray) Crohn-Oral vs Healthy-Oral")
write_permanova(jacc_oh$dist, labels_oh,
                file.path(outdir,"permanova_oral_CH_jaccard.txt"),
                "PERMANOVA (Jaccard) Crohn-Oral vs Healthy-Oral")

# Crohn Fecal vs Healthy Fecal
write_permanova(bray_fh$dist, labels_fh,
                file.path(outdir,"permanova_fecal_CH_bray.txt"),
                "PERMANOVA (Bray) Crohn-Fecal vs Healthy-Fecal")
write_permanova(jacc_fh$dist, labels_fh,
                file.path(outdir,"permanova_fecal_CH_jaccard.txt"),
                "PERMANOVA (Jaccard) Crohn-Fecal vs Healthy-Fecal")

# Crohn Oral vs Crohn Fecal (SITE effect within Crohn)
site_labels <- factor(c(rep("Oral",  ncol(oral_of_mat)),
                        rep("Fecal", ncol(fecal_of_mat))),
                      levels=c("Oral","Fecal"))

write_permanova(bray_of$dist, site_labels,
                file.path(outdir,"permanova_site_bray.txt"),
                "PERMANOVA (Bray) Site effect (Crohn Oral vs Fecal)")
write_permanova(jacc_of$dist, site_labels,
                file.path(outdir,"permanova_site_jaccard.txt"),
                "PERMANOVA (Jaccard) Site effect (Crohn Oral vs Fecal)")

# PERMDISP for SITE (Crohn pairs)
write_permdisp(bray_of$dist, site_labels,
               file.path(outdir,"permdisp_site_bray.txt"),
               "PERMDISP (Bray) Site effect (Crohn Oral vs Fecal)")
write_permdisp(jacc_of$dist, site_labels,
               file.path(outdir,"permdisp_site_jaccard.txt"),
               "PERMDISP (Jaccard) Site effect (Crohn Oral vs Fecal)")

# Summary file
sink(file.path(outdir, "beta_stats.txt"))
cat("## Beta Diversity Statistical Tests\n\n")
cat("Crohn Oral vs Healthy Oral (Bray/Jaccard)\n")
print(adonis2(bray_oh$dist ~ labels_oh, permutations=999))
print(adonis2(jacc_oh$dist ~ labels_oh, permutations=999))

cat("\nCrohn Fecal vs Healthy Fecal (Bray/Jaccard)\n")
print(adonis2(bray_fh$dist ~ labels_fh, permutations=999))
print(adonis2(jacc_fh$dist ~ labels_fh, permutations=999))

cat("\nCrohn Oral vs Crohn Fecal (Bray/Jaccard)\n")
print(adonis2(bray_of$dist ~ site_labels, permutations=999))
print(adonis2(jacc_of$dist ~ site_labels, permutations=999))

cat("\nAll Groups (Bray/Jaccard)\n")
print(adonis2(bray_all$dist ~ all_labels, permutations=999))
print(adonis2(jacc_all$dist ~ all_labels, permutations=999))
sink()

# Save all-groups Bray distance matrix (now 4 groups)
write.csv(as.matrix(bray_all$dist), file.path(outdir, "beta_group_distances.csv"))

# =========================
#  PAIRED ORAL–FECAL DISTANCES (Crohn)
# =========================

pairwise_bray    <- numeric(ncol(oral_of_mat))
pairwise_jaccard <- numeric(ncol(oral_of_mat))
for (i in seq_len(ncol(oral_of_mat))) {
  o <- as.numeric(oral_of_mat[, i])
  f <- as.numeric(fecal_of_mat[, i])
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

pairwise_long <- pairwise_df |>
  tidyr::pivot_longer(cols = c("BrayCurtis","Jaccard"),
                      names_to = "Metric", values_to = "Distance")
pairwise_long$Metric <- factor(pairwise_long$Metric,
                               levels = c("BrayCurtis","Jaccard"))

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
