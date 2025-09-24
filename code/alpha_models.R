#!/usr/bin/env Rscript

# Alpha diversity models (oral & fecal) with covariates
# - Inputs: alpha CSVs (Crohn oral/fecal, Healthy oral/fecal) + pooled covariates
# - Outputs (under --outdir, default results/alpha_models):
#     alpha_models_fecal.csv
#     alpha_models_oral.csv
#     boxplot_fecal_[Shannon|Richness|Evenness]_with_covariates.png
#     boxplot_oral_[Shannon|Richness|Evenness]_with_covariates.png

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(stringr); library(purrr)
  library(ggplot2)
  library(broom)
  library(sandwich); library(lmtest)
  library(effsize)
  library(yaml)
  library(rlang)
})

# Silence R CMD check / LSP false-positives for NSE columns
utils::globalVariables(c(
  "Sample_ID","site","disease","Shannon","Richness","Evenness",
  "metric","value","metric_l","metric_std",
  "p_wilcox","cliffs_delta","cliffs_ci_low","cliffs_ci_high"
))

# ---------------- colors from config.yaml ----------------
load_colors <- function(path = "config.yaml") {
  cfg  <- yaml::read_yaml(path)
  cols <- cfg$colors
  req  <- c("Crohn-Oral","Crohn-Fecal","Healthy-Oral","Healthy-Fecal")
  miss <- setdiff(req, names(cols))
  if (length(miss)) stop("Missing color keys in config.yaml: ", paste(miss, collapse=", "))
  list(
    oh = list( # Crohn-Oral vs Healthy-Oral
      order   = c("Crohn-Oral","Healthy-Oral"),
      palette = unname(c(cols[["Crohn-Oral"]], cols[["Healthy-Oral"]])))
    ,
    fh = list( # Crohn-Fecal vs Healthy-Fecal
      order   = c("Crohn-Fecal","Healthy-Fecal"),
      palette = unname(c(cols[["Crohn-Fecal"]], cols[["Healthy-Fecal"]])))
  )
}

# ---------------- args ----------------
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(f, d=NULL){ i <- which(args==f); if(length(i)==0||i==length(args)) d else args[i+1] }
alpha_oral_path    <- get_arg("--alpha-oral")
alpha_fecal_path   <- get_arg("--alpha-fecal")
alpha_healthy_path <- get_arg("--alpha-healthy")
alpha_hfec_path    <- get_arg("--alpha-healthy-fecal")
covars_path        <- get_arg("--covariates")
outdir             <- get_arg("--outdir","results/alpha_models")
dir.create(outdir, showWarnings=FALSE, recursive=TRUE)

safe_read <- function(p){
  if(is.null(p) || is.na(p) || !file.exists(p)) return(tibble())
  readr::read_csv(p, show_col_types = FALSE)
}

# ---------------- helpers ----------------
pick_col <- function(df, candidates){
  if(nrow(df)==0) return(NA_character_)
  nm <- names(df); low <- tolower(nm); cand_low <- tolower(candidates)
  hit <- match(cand_low, low, nomatch=0); hit <- hit[hit>0]
  if(length(hit)) return(nm[hit[1]])
  for(p in cand_low){
    idx <- which(str_detect(low, fixed(p)))
    if(length(idx)) return(nm[idx[1]])
  }
  NA_character_
}

# Normalize any alpha CSV to (Sample_ID, site, disease, Shannon, Richness, Evenness)
normalize_alpha <- function(df, site_label, disease_label){
  if(nrow(df)==0){
    return(tibble(Sample_ID=character(), site=character(), disease=double(),
                  Shannon=double(), Richness=double(), Evenness=double()))
  }

  # If long format (metric/value), pivot wider first
  maybe_metric <- pick_col(df, c("metric","measure","index"))
  maybe_value  <- pick_col(df, c("value","val","score"))
  if(!is.na(maybe_metric) && !is.na(maybe_value)){
    idcol <- pick_col(df, c("Sample_ID","sample_id","sample","id"))
    df <- df %>%
      rename(Sample_ID = all_of(idcol),
             metric    = all_of(maybe_metric),
             value     = all_of(maybe_value)) %>%
      mutate(Sample_ID = as.character(.data$Sample_ID),
             metric_l  = tolower(.data$metric),
             metric_std = case_when(
               str_detect(.data$metric_l,"shan")           ~ "Shannon",
               str_detect(.data$metric_l,"pielou|even")    ~ "Evenness",
               str_detect(.data$metric_l,"rich|observ")    ~ "Richness",
               TRUE ~ NA_character_
             )) %>%
      filter(!is.na(.data$metric_std)) %>%
      select(.data$Sample_ID, .data$metric_std, .data$value) %>%
      tidyr::pivot_wider(names_from=.data$metric_std, values_from=.data$value)
  }

  # Find columns in wide table
  id_col <- pick_col(df, c("Sample_ID","sample_id","sample","id"))
  sh_col <- pick_col(df, c("Shannon","shannon","shannon_index","alpha_shannon"))
  r_col  <- pick_col(df, c("Richness","Observed","observed","observed_features","observed_otus","sobs"))
  e_col  <- pick_col(df, c("Evenness","Pielou","evenness","pielou"))

  out <- df %>%
    rename(Sample_ID = all_of(id_col)) %>%
    mutate(Sample_ID = as.character(.data$Sample_ID),
           site      = site_label,
           disease   = as.numeric(disease_label))

  if(!is.na(sh_col)) out <- rename(out, Shannon  = all_of(sh_col))
  if(!is.na(r_col))  out <- rename(out, Richness = all_of(r_col))
  if(!is.na(e_col))  out <- rename(out, Evenness = all_of(e_col))

  # Ensure three alpha columns exist
  if (!"Shannon"  %in% names(out))  out$Shannon  <- NA_real_
  if (!"Richness" %in% names(out))  out$Richness <- NA_real_
  if (!"Evenness" %in% names(out))  out$Evenness <- NA_real_

  out <- out %>%
    transmute(
      Sample_ID = .data$Sample_ID,
      site      = .data$site,
      disease   = .data$disease,
      Shannon   = suppressWarnings(as.numeric(.data$Shannon)),
      Richness  = suppressWarnings(as.numeric(.data$Richness)),
      Evenness  = suppressWarnings(as.numeric(.data$Evenness))
    )

  # Compute Pielou (Evenness) if feasible
  need_even <- is.na(out$Evenness) & !is.na(out$Shannon) & !is.na(out$Richness) & out$Richness > 1
  out$Evenness[need_even] <- out$Shannon[need_even] / log(out$Richness[need_even])

  out
}

blank_model_df <- function(){
  tibble(metric=character(), term=character(), estimate=double(),
         std.error=double(), statistic=double(), p.value=double(),
         q.value=double(), q_disease_across_metrics=double())
}

placeholder_plot <- function(path, title_txt="No data to display"){
  g <- ggplot() + theme_void() +
    annotate("text", x=0, y=0, label=title_txt, size=5)
  ggsave(path, g, width=5.5, height=4.0, dpi=300)
}

ucfirst <- function(s) paste0(toupper(substr(s,1,1)), substr(s,2,nchar(s)))

# ---------------- read alpha tables ----------------
alpha_oral    <- normalize_alpha(safe_read(alpha_oral_path),  "oral",  1)
alpha_fecal   <- normalize_alpha(safe_read(alpha_fecal_path), "fecal", 1)
alpha_h       <- normalize_alpha(safe_read(alpha_healthy_path),    "oral",  0)
alpha_h_fecal <- normalize_alpha(safe_read(alpha_hfec_path),       "fecal", 0)

alpha_all <- bind_rows(alpha_oral, alpha_fecal, alpha_h, alpha_h_fecal) %>%
  distinct(Sample_ID, site, .keep_all = TRUE)
alpha_raw <- alpha_all

pal <- load_colors("config.yaml")

# ---------------- covariates ----------------
covars <- safe_read(covars_path) %>%
  mutate(
    Sample_ID = as.character(.data$Sample_ID),
    site      = as.character(.data$site),
    disease   = as.numeric(.data$disease)
  )

# Inner join = only samples present in both alpha & covariates
df <- inner_join(alpha_all, covars, by=c("Sample_ID","site","disease"))

# Numeric-ify candidate covariates (ignore if missing)
wish_covars <- c("Age","Sex","BMI","Smoking","Antibiotics_3m","PPI_use","Steroids_ongoing","Immuno_ongoing")
for(v in intersect(wish_covars, names(df))) df[[v]] <- suppressWarnings(as.numeric(df[[v]]))
missing_cov <- setdiff(wish_covars, names(df))
if(length(missing_cov)) message("[alpha_models] Skipping missing covariates: ", paste(missing_cov, collapse=", "))

metrics <- c("Shannon","Richness","Evenness")

# ---------------- site-wise analysis ----------------
run_site <- function(site_name) {
  d_cov <- df       %>% filter(.data$site == site_name) %>% tidyr::drop_na(.data$disease)
  d_raw <- alpha_raw %>% filter(.data$site == site_name) %>% tidyr::drop_na(.data$disease)

  # --- Wilcoxon + Cliff's delta on raw (Crohn vs Healthy) ---
  wilx <- map_df(metrics, function(m) {
    if (!(m %in% names(d_raw)))
      return(tibble(metric=m, p_wilcox=NA_real_, cliffs_delta=NA_real_, cliffs_ci_low=NA_real_, cliffs_ci_high=NA_real_))
    dd <- dplyr::select(d_raw, .data$disease, !!sym(m)) %>% tidyr::drop_na()
    if (dplyr::n_distinct(dd$disease) < 2) {
      tibble(metric=m, p_wilcox=NA_real_, cliffs_delta=NA_real_, cliffs_ci_low=NA_real_, cliffs_ci_high=NA_real_)
    } else {
      wt <- wilcox.test(reformulate("disease", response = m), data = dd, exact = FALSE)
      cd <- tryCatch(effsize::cliff.delta(reformulate("disease", response = m), data = dd, conf.level = 0.95),
                     error=function(e) NULL)
      tibble(
        metric=m,
        p_wilcox      = wt$p.value,
        cliffs_delta  = if (!is.null(cd)) unname(cd$estimate) else NA_real_,
        cliffs_ci_low = if (!is.null(cd)) unname(cd$conf.int[1]) else NA_real_,
        cliffs_ci_high= if (!is.null(cd)) unname(cd$conf.int[2]) else NA_real_
      )
    }
  }) %>% mutate(q_wilcox = p.adjust(.data$p_wilcox, method = "BH"))

  # --- Linear model with HC3 on joined data (if two classes exist) ---
  covar_pool  <- intersect(wish_covars, names(d_cov))
  lm_rows <- list()

  for (m in metrics) {
    if (!(m %in% names(d_cov))) next
    dd <- dplyr::select(d_cov, tidyselect::all_of(c(m, "disease", covar_pool))) %>% tidyr::drop_na(any_of(c(m, "disease")))
    if (nrow(dd) < 5 || dplyr::n_distinct(dd$disease) < 2) {
      lm_rows[[m]] <- list(table = tibble(metric=m, term="(skipped_one_class_or_small_n)",
                                          estimate=NA_real_, std.error=NA_real_, statistic=NA_real_,
                                          p.value=NA_real_, q.value=NA_real_), disease_p = NA_real_)
      next
    }
    rhs <- paste(c("disease", covar_pool), collapse = " + ")
    fml <- as.formula(paste0(m, " ~ ", rhs))
    fit <- tryCatch(stats::lm(fml, data = dd), error=function(e) NULL)
    if (is.null(fit)) next
    ct  <- lmtest::coeftest(fit, vcov = sandwich::vcovHC(fit, type = "HC3"))
    tb  <- broom::tidy(ct) %>% mutate(metric = m) %>%
      group_by(.data$metric) %>% mutate(q.value = p.adjust(.data$p.value, "BH")) %>% ungroup()
    disease_p <- tb %>% filter(.data$term == "disease") %>% pull(.data$p.value) %>% {if(length(.)==0) NA_real_ else .[1]}
    lm_rows[[m]] <- list(table = tb, disease_p = disease_p)
  }

  model_tbl <- bind_rows(map(lm_rows, "table"))
  disease_ps <- map_dbl(lm_rows, "disease_p")
  disease_qs <- if (length(disease_ps)==0 || all(is.na(disease_ps))) rep(NA_real_, length(disease_ps)) else p.adjust(disease_ps, "BH")
  if (length(disease_qs)) {
    model_tbl <- model_tbl %>%
      left_join(tibble(metric = names(lm_rows), q_disease_across_metrics = as.numeric(disease_qs)), by="metric")
  }

  # --- Boxplots (raw) with YAML colors ---
  make_box <- function(metric, label_df) {
    dd <- dplyr::select(d_raw, .data$disease, !!sym(metric)) %>% tidyr::drop_na()
    if (nrow(dd) == 0) return(NULL)
    names(dd)[2] <- "value"
    lab <- label_df %>% filter(.data$metric == !!metric)
    p_lab <- ifelse(is.na(lab$q_wilcox), "q = NA", paste0("FDR q = ", formatC(lab$q_wilcox, format="e", digits=2)))
    cd   <- ifelse(is.na(lab$cliffs_delta), "δ = NA", paste0("Cliff’s δ = ", round(lab$cliffs_delta, 2)))
    ggplot(dd, aes(x = factor(.data$disease, levels=c(0,1), labels=c("Healthy","Crohn")),
                   y = .data$value,
                   fill = factor(.data$disease, levels=c(0,1), labels=c("Healthy","Crohn")))) +
      geom_boxplot(outlier.shape = NA) +
      geom_jitter(width = 0.15, alpha = 0.6, size = 1.7, color="black") +
      scale_fill_manual(values = if (site_name=="oral") pal$oh$palette else pal$fh$palette) +
      labs(x = "", y = metric, title = paste0(ucfirst(site_name), " — ", metric)) +
      annotate("text", x = 1.5, y = max(dd$value, na.rm=TRUE),
               label = paste(p_lab, cd, sep=" | "), vjust=-0.5, size=3.3) +
      theme_bw(base_size = 12) +
      theme(plot.title = element_text(face="bold"), legend.position="none")
  }

  plots <- list()
  for (m in metrics) {
    fn <- file.path(outdir, paste0("boxplot_", site_name, "_", m, "_with_covariates.png"))
    g  <- make_box(m, wilx)
    if (is.null(g)) {
      placeholder_plot(fn, paste0(ucfirst(site_name)," — ", m, "\n(No data for both classes)"))
    } else {
      ggsave(fn, g, width = 5.5, height = 4.3, dpi = 300)
    }
    plots[[m]] <- fn
  }

  list(models = model_tbl, wilx = wilx, plots = unlist(plots))
}

# Run per site
fec <- run_site("fecal")
orl <- run_site("oral")

# Write model tables (never leave empty files missing)
if (is.null(fec$models) || nrow(fec$models)==0) {
  readr::write_csv(blank_model_df(), file.path(outdir, "alpha_models_fecal.csv"))
} else {
  readr::write_csv(fec$models, file.path(outdir, "alpha_models_fecal.csv"))
}
if (is.null(orl$models) || nrow(orl$models)==0) {
  readr::write_csv(blank_model_df(), file.path(outdir, "alpha_models_oral.csv"))
} else {
  readr::write_csv(orl$models, file.path(outdir, "alpha_models_oral.csv"))
}
