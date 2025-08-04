# beta_diversity.R
# Computes PCoA, PERMANOVA, and paired distances for Crohn/Healthy oral/fecal groups.
# Plots with ellipses, saves each plot as PNG (not transparent).

# ---- Load libraries ----
library(vegan)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# ---- Parse arguments from Snakemake ----
args <- commandArgs(trailingOnly=TRUE)
oral_file     <- args[1]
fecal_file    <- args[2]
healthy_file  <- args[3]
matched_file  <- args[4]
outdir        <- args[5]

dir.create(outdir, showWarnings=FALSE, recursive=TRUE)

# ---- Load input data ----
oral      <- read.csv(oral_file, row.names=1, check.names=FALSE)
fecal     <- read.csv(fecal_file, row.names=1, check.names=FALSE)
healthy   <- read.csv(healthy_file, row.names=1, check.names=FALSE)
matched   <- read.csv(matched_file)

# ---- Helper: intersect features and order identically for all tables ----
align_features <- function(df_list) {
  # Find common features/taxa (rownames)
  common <- Reduce(intersect, lapply(df_list, rownames))
  # Subset and order identically
  lapply(df_list, function(df) df[common, , drop=FALSE])
}

# ---- Utility: function for PCoA and plotting with ellipses ----
pcoa_and_plot <- function(df, sample_labels, dist_method, filename, title) {
  # Transpose so samples are rows
  d <- t(df)
  # Calculate distance matrix
  dist <- vegan::vegdist(d, method=dist_method)
  # Perform PCoA
  ord <- cmdscale(dist, eig=TRUE, k=2)
  pcoa_df <- data.frame(
    Sample = rownames(d),
    Axis1 = ord$points[,1],
    Axis2 = ord$points[,2],
    Group = sample_labels
  )
  # Plot: points + ellipse, white bg, no transparency
  p <- ggplot(pcoa_df, aes(x=Axis1, y=Axis2, color=Group, fill=Group)) +
    geom_point(size=3, alpha=1) +
    stat_ellipse(type="norm", linetype=2, size=1, alpha=0.18, aes(group=Group)) + # group ellipse
    labs(title=title, x="PCoA 1", y="PCoA 2") +
    theme_minimal(base_size=15) +
    theme(
      panel.background = element_rect(fill='white', color='white'),
      plot.background = element_rect(fill='white', color='white'),
      legend.title=element_blank()
    )
  ggsave(filename=file.path(outdir, filename), plot=p, width=6, height=5, dpi=300, bg="white")
  return(list(dist=dist, pcoa_df=pcoa_df))
}

# ---- PCoA comparisons ----

# 1. Crohn Oral vs Healthy Oral
list_aligned <- align_features(list(oral, healthy))
oral_aligned <- list_aligned[[1]]
healthy_aligned <- list_aligned[[2]]
oral_all <- cbind(oral_aligned, healthy_aligned)
group_labels <- c(rep("Crohn-Oral", ncol(oral_aligned)), rep("Healthy-Oral", ncol(healthy_aligned)))
bray1 <- pcoa_and_plot(oral_all, group_labels, "bray", "pcoa_bray_oral_vs_healthy.png", "PCoA (Bray) Crohn Oral vs Healthy Oral")
jacc1 <- pcoa_and_plot(oral_all, group_labels, "jaccard", "pcoa_jaccard_oral_vs_healthy.png", "PCoA (Jaccard) Crohn Oral vs Healthy Oral")

# 2. Crohn Oral vs Crohn Fecal (matched)
list_aligned2 <- align_features(list(oral, fecal))
oral_matched_aligned <- list_aligned2[[1]][, matched$Oral_col, drop=FALSE]
fecal_matched_aligned <- list_aligned2[[2]][, matched$Fecal_col, drop=FALSE]
oral_fecal_all <- cbind(oral_matched_aligned, fecal_matched_aligned)
of_labels <- c(rep("Oral", ncol(oral_matched_aligned)), rep("Fecal", ncol(fecal_matched_aligned)))
bray2 <- pcoa_and_plot(oral_fecal_all, of_labels, "bray", "pcoa_bray_oral_vs_fecal.png", "PCoA (Bray) Crohn Oral vs Fecal")
jacc2 <- pcoa_and_plot(oral_fecal_all, of_labels, "jaccard", "pcoa_jaccard_oral_vs_fecal.png", "PCoA (Jaccard) Crohn Oral vs Fecal")

# 3. All groups: Crohn Oral, Crohn Fecal, Healthy Oral
list_aligned3 <- align_features(list(oral, fecal, healthy))
oral3 <- list_aligned3[[1]]
fecal3 <- list_aligned3[[2]]
healthy3 <- list_aligned3[[3]]
all_df <- cbind(oral3, fecal3, healthy3)
all_labels <- c(
  rep("Crohn-Oral", ncol(oral3)),
  rep("Crohn-Fecal", ncol(fecal3)),
  rep("Healthy-Oral", ncol(healthy3))
)
bray3 <- pcoa_and_plot(all_df, all_labels, "bray", "pcoa_bray_allgroups.png", "PCoA (Bray) All Groups")
jacc3 <- pcoa_and_plot(all_df, all_labels, "jaccard", "pcoa_jaccard_allgroups.png", "PCoA (Jaccard) All Groups")

# ---- PERMANOVA tests ----
sink(file.path(outdir, "beta_stats.txt"))
cat("## Beta Diversity Statistical Tests\n\n")

cat("PERMANOVA (Bray) Crohn Oral vs Healthy Oral:\n")
print(adonis2(t(oral_all) ~ group_labels, permutations=999, method="bray"))

cat("\nPERMANOVA (Jaccard) Crohn Oral vs Healthy Oral:\n")
print(adonis2(t(oral_all) ~ group_labels, permutations=999, method="jaccard"))

cat("\nPERMANOVA (Bray) Crohn Oral vs Crohn Fecal (matched):\n")
print(adonis2(t(oral_fecal_all) ~ of_labels, permutations=999, method="bray"))

cat("\nPERMANOVA (Jaccard) Crohn Oral vs Crohn Fecal (matched):\n")
print(adonis2(t(oral_fecal_all) ~ of_labels, permutations=999, method="jaccard"))

cat("\nPERMANOVA (Bray) All Groups:\n")
print(adonis2(t(all_df) ~ all_labels, permutations=999, method="bray"))

cat("\nPERMANOVA (Jaccard) All Groups:\n")
print(adonis2(t(all_df) ~ all_labels, permutations=999, method="jaccard"))

sink()

# ---- Save distance matrix for "all groups" as CSV ----
write.csv(as.matrix(bray3$dist), file.path(outdir, "beta_group_distances.csv"))
# ---- PAIRWISE ORAL-FECAL (Crohn) DISTANCES (within subject) ----
# Calculate Bray-Curtis and Jaccard distance between oral and fecal sample for each Crohn subject (paired)
pairwise_bray <- numeric(ncol(oral_matched_aligned))
pairwise_jaccard <- numeric(ncol(oral_matched_aligned))

for (i in seq_len(ncol(oral_matched_aligned))) {
  o <- as.numeric(oral_matched_aligned[, i])
  f <- as.numeric(fecal_matched_aligned[, i])
  pairwise_bray[i] <- vegan::vegdist(rbind(o, f), method="bray")[1]
  pairwise_jaccard[i] <- vegan::vegdist(rbind(o, f), method="jaccard")[1]
}

pairwise_df <- data.frame(
  STUDY_ID = matched$STUDY_ID,
  BrayCurtis = pairwise_bray,
  Jaccard = pairwise_jaccard
)
write.csv(pairwise_df, file.path(outdir, "pairwise_oral_fecal_distances.csv"), row.names=FALSE)

# Pivot data for ggplot
library(tidyr)
pairwise_long <- pairwise_df %>%
  tidyr::pivot_longer(cols = c("BrayCurtis", "Jaccard"),
                      names_to = "Metric", values_to = "Distance")

# Violin plot for both metrics together
library(ggplot2)
p <- ggplot(pairwise_long, aes(x=Metric, y=Distance, fill=Metric)) +
  geom_violin(trim=FALSE, alpha=0.7, width=0.7) +
  geom_jitter(width=0.15, alpha=0.5, color="black") +
  labs(title="Paired Oral-Fecal Beta Diversity (Crohn)", y="Distance", x="Metric") +
  theme_bw(base_size=14) +
  scale_fill_manual(values=c("BrayCurtis"="#56B4E9", "Jaccard"="#D55E00"))

ggsave(file.path(outdir, "pairwise_oral_fecal_violinplot.png"), plot=p, width=6, height=5, dpi=300)
