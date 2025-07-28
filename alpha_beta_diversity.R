### Load libraries
library(data.table)
library(vegan)
library(ggplot2)
library(dplyr)
library(gridExtra) # For multi-panel plots

### Set Output Directory
outdir <- "/Users/maryamesmaeili/Documents/Hanze/Internship/outputs"
if(!dir.exists(outdir)) dir.create(outdir, recursive=TRUE)

### Define Colors
my_colors <- c("Crohn-Oral"="#1f77b4", "Crohn-Fecal"="#ff7f0e", "Healthy-Oral"="#2ca02c")

### Load Data
oral_abund_crohn   <- as.data.frame(fread("/Users/maryamesmaeili/Documents/Hanze/Internship/data/oral_abund_crohn.csv"))
fecal_abund_crohn  <- as.data.frame(fread("/Users/maryamesmaeili/Documents/Hanze/Internship/data/fecal_abund_crohn.csv"))
oral_abund_healthy <- as.data.frame(fread("/Users/maryamesmaeili/Documents/Hanze/Internship/data/oral_abund_healthy.csv"))

### Set taxa names as rowname
set_rowname <- function(df) {
  rownames(df) <- df[[1]]
  df <- df[,-1]
  return(df)
}

oral_abund_crohn   <- set_rowname(oral_abund_crohn)
fecal_abund_crohn  <- set_rowname(fecal_abund_crohn)
oral_abund_healthy <- set_rowname(oral_abund_healthy)


### Only keep common taxa
common_taxa <- Reduce(intersect, list(rownames(oral_abund_crohn),
                                      rownames(fecal_abund_crohn),
                                      rownames(oral_abund_healthy)))
oral_abund_crohn   <- oral_abund_crohn[common_taxa, ]
fecal_abund_crohn  <- fecal_abund_crohn[common_taxa, ]
oral_abund_healthy <- oral_abund_healthy[common_taxa, ]

### Transpose for vegan (samples as rows)
oral_crohn_t   <- t(oral_abund_crohn)
fecal_crohn_t  <- t(fecal_abund_crohn)
oral_healthy_t <- t(oral_abund_healthy)

### Calculate alpha diversity indices
alpha_df <- bind_rows(
  data.frame(Group = "Crohn-Oral",   Shannon = diversity(oral_crohn_t,   index="shannon"), 
             Simpson = diversity(oral_crohn_t, index="simpson"), 
             Richness = specnumber(oral_crohn_t)),
  data.frame(Group = "Crohn-Fecal",  Shannon = diversity(fecal_crohn_t,  index="shannon"), 
             Simpson = diversity(fecal_crohn_t, index="simpson"), 
             Richness = specnumber(fecal_crohn_t)),
  data.frame(Group = "Healthy-Oral", Shannon = diversity(oral_healthy_t, index="shannon"), 
             Simpson = diversity(oral_healthy_t, index="simpson"), 
             Richness = specnumber(oral_healthy_t))
)
alpha_df$Group <- factor(alpha_df$Group, levels = names(my_colors))

### Boxplots for alpha diversity
shannon_plot <- ggplot(alpha_df, aes(x = Group, y = Shannon, fill = Group)) +
  geom_boxplot() + theme_bw() + ggtitle("Alpha Diversity (Shannon)") +
  scale_fill_manual(values=my_colors)

simpson_plot <- ggplot(alpha_df, aes(x = Group, y = Simpson, fill = Group)) +
  geom_boxplot() + theme_bw() + ggtitle("Alpha Diversity (Simpson)") +
  scale_fill_manual(values=my_colors)

richness_plot <- ggplot(alpha_df, aes(x = Group, y = Richness, fill = Group)) +
  geom_boxplot() + theme_bw() + ggtitle("Observed Richness") +
  scale_fill_manual(values=my_colors)

# Save one by one
ggsave(filename=file.path(outdir, "alpha_shannon_boxplot.png"), shannon_plot, width=8, height=6, dpi=150)
ggsave(filename=file.path(outdir, "alpha_simpson_boxplot.png"), simpson_plot, width=8, height=6, dpi=150)
ggsave(filename=file.path(outdir, "alpha_richness_boxplot.png"), richness_plot, width=8, height=6, dpi=150)

# Save all in one
png(file.path(outdir, "boxplots_alpha_all.png"), width=1600, height=600, res=130)
grid.arrange(shannon_plot, simpson_plot, richness_plot, nrow=1)
dev.off()

# Save alpha_df as CSV
write.csv(alpha_df, file.path(outdir, "alpha_diversity_summary.csv"), row.names=FALSE)

# Statistical tests for alpha diversity
sink(file.path(outdir, "alpha_diversity_stats.txt"))
cat("Wilcoxon (Shannon: Crohn-Oral vs Crohn-Fecal):\n")
print(wilcox.test(alpha_df$Shannon[alpha_df$Group=="Crohn-Oral"], alpha_df$Shannon[alpha_df$Group=="Crohn-Fecal"]))
cat("\nWilcoxon (Shannon: Crohn-Oral vs Healthy-Oral):\n")
print(wilcox.test(alpha_df$Shannon[alpha_df$Group=="Crohn-Oral"], alpha_df$Shannon[alpha_df$Group=="Healthy-Oral"]))
cat("\nWilcoxon (Shannon: Crohn-Fecal vs Healthy-Oral):\n")
print(wilcox.test(alpha_df$Shannon[alpha_df$Group=="Crohn-Fecal"], alpha_df$Shannon[alpha_df$Group=="Healthy-Oral"]))
sink()

### Beta diversity (Bray-Curtis) and PCoA
all_samples <- rbind(oral_crohn_t, fecal_crohn_t, oral_healthy_t)
group_labels <- c(rep("Crohn-Oral", nrow(oral_crohn_t)),
                  rep("Crohn-Fecal", nrow(fecal_crohn_t)),
                  rep("Healthy-Oral", nrow(oral_healthy_t)))
bc_dm <- vegdist(all_samples, method="bray")
pcoa_res <- cmdscale(bc_dm, k=2)
pcoa_df <- data.frame(PC1=pcoa_res[,1], PC2=pcoa_res[,2], Group=group_labels)

# Save PCoA plot
pcoa_plot <- ggplot(pcoa_df, aes(x=PC1, y=PC2, color=Group)) +
  geom_point(size=3) + theme_bw() + ggtitle("PCoA (Bray-Curtis) - All Groups") +
  scale_color_manual(values=my_colors)

ggsave(filename=file.path(outdir, "pcoa_braycurtis_all_groups.png"), pcoa_plot, width=9, height=8, dpi=150)
write.csv(pcoa_df, file.path(outdir, "pcoa_coordinates.csv"), row.names=FALSE)

# Separate PCoA plots per group 
library(vegan)

plot_group_pcoa <- function(abund_mat, group_name, color, outdir) {
  if(nrow(abund_mat) < 2) return() # needs at least 2 samples
  bc_dm <- vegdist(abund_mat, method="bray")
  pcoa_res <- cmdscale(bc_dm, k=2)
  df <- data.frame(PC1=pcoa_res[,1], PC2=pcoa_res[,2], Sample=rownames(abund_mat))
  p <- ggplot(df, aes(x=PC1, y=PC2)) +
    geom_point(size=3, color=color) +
    theme_bw() +
    ggtitle(paste("PCoA (Bray-Curtis):", group_name))
  ggsave(filename=file.path(outdir, paste0("pcoa_braycurtis_", gsub("-", "_", group_name), ".png")), p, width=7, height=6, dpi=150)
}

# Call for each group
plot_group_pcoa(oral_crohn_t, "Crohn-Oral", my_colors["Crohn-Oral"], outdir)
plot_group_pcoa(fecal_crohn_t, "Crohn-Fecal", my_colors["Crohn-Fecal"], outdir)
plot_group_pcoa(oral_healthy_t, "Healthy-Oral", my_colors["Healthy-Oral"], outdir)
###########################
# Read matched sample IDs
matched_df <- read.csv("/Users/maryamesmaeili/Documents/Hanze/Internship/data/matched_sample_ids.csv", stringsAsFactors=FALSE)
oral_ids  <- matched_df$Oral_col
fecal_ids <- matched_df$Fecal_col

# Check if these columns exist in your abundance data
cat("Missing oral:", setdiff(oral_ids, colnames(oral_abund_crohn)), "\n")
cat("Missing fecal:", setdiff(fecal_ids, colnames(fecal_abund_crohn)), "\n")

# Subset the abundance tables (columns must match)
oral_matched  <- oral_abund_crohn[, oral_ids]
fecal_matched <- fecal_abund_crohn[, fecal_ids]

# Double check the shape
stopifnot(ncol(oral_matched) == ncol(fecal_matched))
stopifnot(all(colnames(oral_matched) == oral_ids))
stopifnot(all(colnames(fecal_matched) == fecal_ids))

# Transpose for Procrustes (samples x taxa -> samples as rows)
oral_matched_t  <- t(oral_matched)
fecal_matched_t <- t(fecal_matched)

# Run Procrustes analysis

procrustes_res <- procrustes(oral_matched_t, fecal_matched_t, symmetric=TRUE)
protest_res <- protest(oral_matched_t, fecal_matched_t)

# Plot Procrustes
png(file.path(outdir, "procrustes_oral_fecal_crohn.png"), width=800, height=600, res=120)
plot(procrustes_res, kind=1, main=paste("Procrustes Analysis (Oral vs. Fecal Crohn)\np-value:", round(protest_res$signif,4)))
dev.off()

# Save statistical results
sink(file.path(outdir, "procrustes_oral_fecal_crohn_stats.txt"))
print(protest_res)
sink()

##########################
### PERMANOVA test for group difference
sink(file.path(outdir, "permanova_braycurtis.txt"))
group_factor <- factor(group_labels, levels = names(my_colors))
print(adonis2(bc_dm ~ group_factor))
sink()

cat("All outputs are saved to: ", outdir, "\n")
