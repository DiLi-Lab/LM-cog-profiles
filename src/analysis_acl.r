#!/usr/bin/env Rscript
library(ggplot2)
library(ggcorrplot)
library(dplyr)
library(viridis)
library(tidyr)
library(broom)
library(mgcv)
library(tidygam)
library(gsubfn)
library(tidyverse)
library(boot)
library(broman)
library(rsample)
library(plotrix)
library(ggrepel)
library(JuliaCall)

set.seed(123)
theme_set(theme_light())
options(digits = 8)
options(dplyr.summarise.inform = TRUE)

library(jglmm)
options(JULIA_HOME = "/Applications/Julia-1.10.app/Contents/Resources/julia/bin")
julia_setup(JULIA_HOME = "/Applications/Julia-1.10.app/Contents/Resources/julia/bin")
jglmm_setup()

LOG_TF <- TRUE
CONT_RESP_VARIABLES <- c("FFD", "SFD", "FD", "FPRT", "FRT", "TFT", "RRT", "word_rt")
SCORES_RENAMED <- c(
    "Memory updating", "MWT", "Operation span", "RIAS total",
    "RIAS non-verbal", "RIAS verbal", "Simon", "SLRT pseudo- words", "SLRT words",
    "Sentence span", "Spatial short-term memory", "Stroop", "FAIR"
)

split_groups <- function(df, score) {
    df_low <- df[df[, score] == "low", ]
    df_high <- df[df[, score] == "high", ]
    return(list(df_low, df_high))
}

z_score <- function(x) {
    return((x - mean(x)) / sd(x))
}

remove_outlier <- function(df, reading_measure) {
    reading_times <- as.numeric(df[[reading_measure]])
    z_score <- z_score(reading_times)
    abs_z_score <- abs(z_score)
    df$outlier <- abs_z_score > 3
    # print number of outliers / total number of reading times
    print(paste(sum(df$outlier), "/", length(df$outlier)))
    # remove outliers
    df <- df[df$outlier == FALSE, ]
    return(df)
}

model_cross_val <- function(form, df_in, predicted_var, mixed_effects, num_folds = 10, shuffle_folds = FALSE, remove_outliers = FALSE, log_transform = FALSE, merge_with_scores = TRUE) {
    df <- df_in
    if (predicted_var %in% CONT_RESP_VARIABLES) {
        if (log_transform == TRUE) {
            # remove 0s
            df <- df[df[[predicted_var]] != 0, ]
            df[[predicted_var]] <- log(df[[predicted_var]])
            if (remove_outliers == TRUE) {
                df <- remove_outlier(df, predicted_var)
            }
        }
    }

    folds <- cut(seq(1, nrow(df)), breaks = num_folds, labels = FALSE)
    if (shuffle_folds == TRUE) {
        folds <- sample(folds)
    }
    estimates <- c()
    for (i in 1:num_folds) {
        test_indices <- which(folds == i, arr.ind = TRUE)
        test_data <- df[test_indices, ]
        train_data <- df[-test_indices, ]
        if (mixed_effects) {
            model <- jglmm(as.formula(form), data = train_data)
        } else {
            model <- lm(as.formula(form), data = train_data)
        }
        stdev <- sigma(model)
        densities <- dnorm(
            test_data[[predicted_var]],
            mean = predict(model, newdata = test_data, allow.new.levels = TRUE),
            sd = stdev,
            log = TRUE
        )
        # get all scores and subj_id from test_data (SCORE_NAMES)
        test_data_scores <- test_data[, c("subj_id", SCORE_NAMES)]
        # create data frame with densities and scores
        # if scores should be available in the output
        if (merge_with_scores == TRUE) {
            densities <- data.frame(loglik = densities, test_data_scores)
            estimates <- rbind(estimates, densities)
        }
        # for the group analyses, don't merge with scores
        else {
            estimates <- c(estimates, densities)
        }
    }
    return(estimates)
}

get_effect_sizes <- function(form, df_in, predicted_var, model_name, score_name, log_transform = FALSE, remove_outliers = FALSE) {
    df <- df_in
    if (predicted_var %in% CONT_RESP_VARIABLES) {
        if (log_transform == TRUE) {
            #  remove 0s
            df <- df[df[[predicted_var]] != 0, ]
            df[[predicted_var]] <- log(df[[predicted_var]])
            if (remove_outliers == TRUE) {
                df <- remove_outlier(df, predicted_var)
            }
        }
    }

    model <- jglmm(as.formula(form), data = df)
    mod_summary <- tidy(model)
    #  extract effect sizes
    # fixed_score <- testsum[testsum$term == test_score, c("estimate", "std.error", "p.value")]
    interaction_term <- mod_summary[mod_summary$term == paste0(model_name, " & ", score_name), c("estimate", "std.error", "p.value")]
    return(interaction_term)
}

delta_ll_separate_baseline <- function(df, models, psychometrics) {
    dll_df <- data.frame()
    for (model in models) {
        model_prev <- paste0(model, "_prev_word")
        model_prevprev <- paste0(model, "_prevprev_word")
        pred_effect <- ifelse(grepl("surprisal", model), "surprisal", "entropy")
        df_eval <- df %>%
            drop_na() %>%
            distinct()
        for (psychometric in psychometrics) {
            print(paste0("Fitting model for ", model, " reading measure ", psychometric))
            print(" --- ")
            regression_forms <- c(
                paste0(
                    psychometric, " ~  1 + (1|subj_id) + word_length + log_lex_freq + "
                    , model
                    # , " + ", model_prev, " + ", model_prevprev
                ),
                paste0(psychometric, " ~ 1 + (1|subj_id) + word_length + log_lex_freq")
            )
            baseline_loglik <- model_cross_val(
                form = regression_forms[2], df_in = df_eval, predicted_var = psychometric, mixed_effects = TRUE,
                remove_outliers = FALSE, log_transform = LOG_TF)
            target_loglik <- model_cross_val(
                form = regression_forms[1], df_in = df_eval, predicted_var = psychometric, mixed_effects = TRUE,
                remove_outliers = FALSE, log_transform = LOG_TF)
            loglik_df <- data.frame(
                delta_loglik = target_loglik$loglik - baseline_loglik$loglik,
                model = model,
                psychometric = psychometric,
                pred_effect = pred_effect
            )
            # merge score and subj_id info from baseline df (is the same as target df) with ll_df
            loglik_df <- cbind(loglik_df, baseline_loglik)
            dll_df <- rbind(dll_df, loglik_df)
        }
    }
    return(dll_df)
}

delta_ll_combined_baseline_spillover <- function(df, entropies, surps, psychometrics) {
    dll_df <- data.frame()
    for (i in 1:length(entropies)) {
        entropy <- entropies[i]
        entropy_name <- strsplit(entropy, "__")[[1]][1]
        entropy_prev_word <- paste0(entropy, "_prev_word")
        entropy_prev_prev_word <- paste0(entropy, "_prevprev_word")
        surp <- surps[i]
        surp_name <- strsplit(surp, "__")[[1]][1]
        surp_prev_word <- paste0(surp, "_prev_word")
        surp_prev_prev_word <- paste0(surp, "_prevprev_word")
        if (entropy_name == surp_name) {
            model <- entropy_name
        } else {
            return(-1)
        }
        df_eval <- df %>%
            drop_na() %>%
            distinct()
        for (psychometric in psychometrics) {
            print(paste0("Fitting model for ", model, " measure ", psychometric))
            print(" --- ")
            regression_forms <- c(
                paste0(
                    psychometric, " ~  1 + (1|subj_id) + word_length + log_lex_freq + ",
                    entropy, " + ", entropy_prev_word, " + ", entropy_prev_prev_word, " + ",
                    surp, " + ", surp_prev_word, " + ", surp_prev_prev_word
                ),
                paste0(psychometric, " ~ 1 + (1|subj_id) + word_length + log_lex_freq")
            )
            baseline_loglik <- model_cross_val(
                form = regression_forms[2], df_in = df_eval, predicted_var = psychometric, mixed_effects = TRUE,
                remove_outliers = FALSE, log_transform = LOG_TF)
            target_loglik <- model_cross_val(
                form = regression_forms[1], df_in = df_eval, predicted_var = psychometric, mixed_effects = TRUE,
                remove_outliers = FALSE, log_transform = LOG_TF)
            loglik_df <- data.frame(
                delta_loglik = target_loglik$loglik - baseline_loglik$loglik,
                model = model,
                psychometric = psychometric,
                pred_effect = "combined"
            )
            loglik_df <- cbind(loglik_df, baseline_loglik)
            dll_df <- rbind(dll_df, loglik_df)
        }
    }
    return(dll_df)
}

delta_ll_combined_baseline <- function(df, entropies, surps, target_variables) {
    dll_df <- data.frame()
    for (i in 1:length(entropies)) {
        entropy <- entropies[i]
        entropy_name <- strsplit(entropy, "__")[[1]][1]
        surp <- surps[i]
        surp_name <- strsplit(surp, "__")[[1]][1]
        if (entropy_name == surp_name) {
            model <- entropy_name
        } else {
            return(-1)
        }
        df_eval <- df %>%
            drop_na() %>%
            distinct()
        for (target_variable in target_variables) {
            print(paste0("Fitting model for ", model, " measure ", target_variable))
            print(" --- ")
            regression_forms <- c(
                paste0(
                    target_variable, " ~  1 + (1|subj_id) + word_length + log_lex_freq + ",
                    entropy, " + ", surp
                ),
                paste0(target_variable, " ~ 1 + (1|subj_id) + word_length + log_lex_freq")
            )
            baseline_loglik <- model_cross_val(
                form = regression_forms[2], df_in = df_eval, predicted_var = target_variable, mixed_effects = TRUE,
                remove_outliers = FALSE, log_transform = LOG_TF)
            target_loglik <- model_cross_val(
                form = regression_forms[1], df_in = df_eval, predicted_var = target_variable, mixed_effects = TRUE,
                remove_outliers = FALSE, log_transform = LOG_TF)
            loglik_df <- data.frame(
                delta_loglik = target_loglik$loglik - baseline_loglik$loglik,
                model = model,
                psychometric = target_variable,
                pred_effect = "combined"
            )
            loglik_df <- cbind(loglik_df, baseline_loglik)
            dll_df <- rbind(dll_df, loglik_df)
        }
    }
    return(dll_df)
}

delta_ll_score_interactions <- function(df, models_entropy, models_surp, scores, psychometrics) {
    dll_xscore_df <- data.frame()
    for (i in 1:length(models_entropy)) {
        entropy <- models_entropy[i]
        surp <- models_surp[i]
        #  remove _added_entropy and _multiplied_prob from model names
        entropy_name <- gsub("__added_entropy", "", entropy)
        surp_name <- gsub("__accumulated_surprisal", "", surp)
        if (entropy_name == surp_name) {
            model <- entropy_name
        } else {
            return(-1)
        }
        for (score in scores) {
            print(paste0("Fitting model for ", model, " score ", score))
            print(" --- ")
            df_eval <- df %>%
                drop_na() %>%
                distinct()
            for (psychometric in psychometrics) {
                regression_forms_surp <- c(
                    paste0(psychometric, " ~  1 + (1|subj_id) + word_length + log_lex_freq +", entropy, "+ ", surp, "*", score),
                    paste0(psychometric, " ~ 1 + (1|subj_id) + word_length + log_lex_freq + ", score, "+ ", entropy, "+ ", surp)
                )
                regression_forms_ent <- c(
                    paste0(psychometric, " ~  1 + (1|subj_id) + word_length + log_lex_freq +", surp, "+ ", entropy, "*", score),
                    paste0(psychometric, " ~ 1 + (1|subj_id) + word_length + log_lex_freq + ", score, "+ ", surp, "+ ", entropy)
                )
                baseline_loglik_surp <- model_cross_val(
                    form = regression_forms_surp[2], df_in = df_eval, predicted_var = psychometric, mixed_effects = TRUE,
                    remove_outliers = FALSE, log_transform = LOG_TF)
                target_loglik_surp <- model_cross_val(
                    form = regression_forms_surp[1], df_in = df_eval, predicted_var = psychometric, mixed_effects = TRUE,
                    remove_outliers = FALSE, log_transform = LOG_TF)
                loglik_df_surp <- data.frame(
                    delta_loglik = target_loglik_surp$loglik - baseline_loglik_surp$loglik,
                    model = surp,
                    score = score,
                    psychometric = psychometric
                )
                dll_xscore_df <- rbind(dll_xscore_df, loglik_df_surp)
                baseline_loglik_ent <- model_cross_val(
                    form = regression_forms_ent[2], df_in = df_eval, predicted_var = psychometric, mixed_effects = TRUE,
                    remove_outliers = FALSE, log_transform = LOG_TF)
                target_loglik_ent <- model_cross_val(
                    form = regression_forms_ent[1], df_in = df_eval, predicted_var = psychometric, mixed_effects = TRUE,
                    remove_outliers = FALSE, log_transform = LOG_TF)
                loglik_df_ent <- data.frame(
                    delta_loglik = target_loglik_ent$loglik - baseline_loglik_ent$loglik,
                    model = entropy,
                    score = score,
                    psychometric = psychometric
                )
                dll_xscore_df <- rbind(dll_xscore_df, loglik_df_ent)
            }
        }
    }
    return(dll_xscore_df)
}

get_group_dlls <- function(split_df, lm, rm, score, log_transform = FALSE, remove_outliers = FALSE) {
    df_low <- split_df[[1]]
    df_high <- split_df[[2]]

    if (log_transform == TRUE) {
        df_low <- df_low[df_low[[rm]] != 0, ]
        df_low[[rm]] <- log(df_low[[rm]])
        if (remove_outliers == TRUE) {
            df_low <- remove_outlier(df_low, rm)
        }
        df_high <- df_high[df_high[[rm]] != 0, ]
        df_high[[rm]] <- log(df_high[[rm]])
        if (remove_outliers == TRUE) {
            df_high <- remove_outlier(df_high, rm)
        }
    }
    #  ensure same number of data points in each group: downsample the larger one
    diff_group <- nrow(df_low) - nrow(df_high)
    if (diff_group > 0) {
        df_low <- df_low[sample(nrow(df_low), nrow(df_high)), ]
    } else {
        df_high <- df_high[sample(nrow(df_high), nrow(df_low)), ]
    }
    formulas <- c(
        paste0(rm, " ~ 1 + (1|subj_id) + word_length + log_lex_freq "),
        paste0(rm, " ~ 1 + (1|subj_id) + word_length + log_lex_freq + ", lm)
    )
    # here, log_tf ALWAYS has to be false since data has to be log transformed *before*
    low_baseline <- model_cross_val(formulas[1], df_low, rm, mixed_effects = TRUE, log_transform = FALSE, merge_with_scores = FALSE)
    low_target <- model_cross_val(formulas[2], df_low, rm, mixed_effects = TRUE, log_transform = FALSE, merge_with_scores = FALSE)
    delta_low <- low_target - low_baseline
    high_baseline <- model_cross_val(formulas[1], df_high, rm, mixed_effects = TRUE, log_transform = FALSE, merge_with_scores = FALSE)
    high_target <- model_cross_val(formulas[2], df_high, rm, mixed_effects = TRUE, log_transform = FALSE, merge_with_scores = FALSE)
    delta_high <- high_target - high_baseline
    delta_power_gain <- delta_high - delta_low
    loglik_df <- data.frame(
        delta_low = delta_low,
        delta_high = delta_high,
        delta_loglik = delta_power_gain,
        model = lm,
        score = score,
        psychometric = rm
    )
    return(loglik_df)
}

analyze_group_dlls <- function(language_models, reading_measures, scores, et_group_df, log_transform = FALSE, remove_outliers = FALSE) {
    results <- data.frame()
    for (lm in language_models) {
        for (rm in reading_measures) {
            for (score in scores) {
                print(paste0("Analyzing ", lm, " and ", rm, " for ", score))
                print("-------------------")
                split_df <- split_groups(et_group_df, score)
                loglik_df <- get_group_dlls(split_df, lm, rm, score, log_transform = log_transform, remove_outliers = remove_outliers)
                results <- rbind(results, loglik_df)
            }
        }
    }
    return(results)
}


remove_outlier <- function(df, reading_measure) {
    reading_times <- as.numeric(df[[reading_measure]])
    #  log transform all reading times that are not 0
    # reading_times[reading_times != 0] <- log(reading_times[reading_times != 0])
    z_score <- z_score(reading_times)
    abs_z_score <- abs(z_score)
    df$outlier <- abs_z_score > 3
    #  print number of outliers / total number of reading times
    print(paste(sum(df$outlier), "/", length(df$outlier)))
    #  remove outliers
    df <- df[df$outlier == FALSE, ]
    return(df)
}

preprocess <- function(df, predictors_to_normalize) {
    # first, copy df in order to not overwrite original
    df_copy <- df
    df_copy$subj_id <- as.factor(df_copy$subj_id)
    #  convert to log lex freq
    df_copy$lex_freq <- as.numeric(df_copy$lex_freq)
    # find lowest log lex freq that is not 0
    min_lex_freq <- min(df_copy$lex_freq[df_copy$lex_freq > 0])
    #  set lex freq to min where 0
    df_copy$lex_freq[df_copy$lex_freq <= 0] <- min_lex_freq
    df_copy$log_lex_freq <- log(df_copy$lex_freq)

    # normalize baseline predictors
    df_copy$log_lex_freq <- scale(df_copy$log_lex_freq)
    df_copy$word_length <- scale(df_copy$word_length)

    # normalize surprisal/entropy predictors
    for (predictor in predictors_to_normalize) {
        df_copy[[predictor]] <- as.numeric(df_copy[[predictor]])
        df_copy[[predictor]] <- scale(df_copy[[predictor]])
    }

    # first, flip group associations of the StrRTEffect and SimRTEffect
    # this is because the RTEffect was computed as incongruent - congruent
    # -> higher values mean worse performance, therefore, flip it
    # where "high" -> "low" and vice versa
    high_indices <- df_copy$StrRTEffect == "high"
    low_indices <- df_copy$StrRTEffect == "low"
    df_copy$StrRTEffect[high_indices] <- "low"
    df_copy$StrRTEffect[low_indices] <- "high"
    colnames(df_copy) <- gsub("\\.", "_", colnames(df_copy))
    return(df_copy)
}

extract_predictor_var_names <- function(df) {
    surprisal_predictors <- grep("_accumulated_surprisal$", colnames(df), value = TRUE)
    surprisal_predictors_m1 <- grep("_accumulated_surprisal_prev_", colnames(df), value = TRUE)
    surprisal_predictors_m2 <- grep("_accumulated_surprisal_prevprev", colnames(df), value = TRUE)
    entropy_predictors <- grep("_added_entropy$", colnames(et_groups), value = TRUE)
    entropy_predictors_m1 <- grep("_added_entropy_prev_", colnames(et_groups), value = TRUE)
    entropy_predictors_m2 <- grep("_added_entropy_prevprev", colnames(et_groups), value = TRUE)
    all_predictors <- c(
        surprisal_predictors, surprisal_predictors_m1, surprisal_predictors_m2,
        entropy_predictors, entropy_predictors_m1, entropy_predictors_m2
    )
    return(all_predictors)
}

preprocess_scores <- function(df) {
    # remove all rows with NA values
    df_copy <- df
    df_copy <- df_copy[complete.cases(df_copy), ]
    # make all rows in SCORE_NAMES numeric
    for (score in SCORE_NAMES) {
        df_copy[[score]] <- as.numeric(df_copy[[score]])
    }
    # first, flip scores of the StrRTEffect and SimRTEffect
    # this is because the RTEffect was computed as incongruent - congruent
    # -> higher values mean worse performance, therefore, flip it
    df_copy$StrRTEffect <- -df_copy$StrRTEffect
    df_copy$SimRTEffect <- -df_copy$SimRTEffect
    # normalize all scores
    for (score in SCORE_NAMES) {
        df_copy[[score]] <- scale(df_copy[[score]])
    }
    return(df_copy)
}

scores_raw <- read.csv("data/indico/psychometric_scores.csv")
SCORE_NAMES <- colnames(scores_raw)[2:ncol(scores_raw)]
scores_df <- preprocess_scores(scores_raw)


et_data <- read.csv("data/indico/indico_ET_with_groups.csv")
# merge psychometric scores with et_groups on subj_id
et_groups <- merge(et_data, scores_df, by = "subj_id")

# get names of surprisal/entropy predictors
pred_names <- extract_predictor_var_names(et_groups)
# exclude finetuned models
pred_names <- pred_names[!grepl("finetuned", pred_names)]
et_groups_preprocessed <- preprocess(et_groups, predictors_to_normalize = pred_names)

reading_measures <- c("FPRT")
lm_surp_names <- grep("surprisal$", pred_names, value = TRUE)
lm_entropy_names <- grep("_added_entropy$", pred_names, value = TRUE)
models <- c(lm_surp_names, lm_entropy_names)

et_preproc_shuffled <- et_groups_preprocessed[sample(1:nrow(et_groups_preprocessed)), ]

# Experiment 1: DLL between models with and without surprisal term
dll_baseline_combined <- delta_ll_combined_baseline(et_preproc_shuffled, lm_entropy_names, lm_surp_names, reading_measures)
# write.csv(dll_baseline_combined, "data/indico/dll_baseline_combined.csv")
dll_baseline_combined <- read.csv("data/indico/dll_baseline_combined.csv")

dll_baseline_separate <- delta_ll_separate_baseline(et_preproc_shuffled, models, reading_measures)
# write.csv(dll_baseline_separate, "data/indico/dll_baseline_separate.csv")
dll_baseline_separate <- read.csv("data/indico/dll_baseline_separate.csv")

if (all(colnames(dll_baseline_combined) == colnames(dll_baseline_separate))) {
    dll_baseline <- rbind(dll_baseline_combined, dll_baseline_separate)
}
dll_baseline_summarized <- dll_baseline %>%
    group_by(model, pred_effect, psychometric) %>%
    summarise(
        m = mean(delta_loglik), se = std.error(delta_loglik),
        upper = m + 1.96 * se, lower = m - 1.96 * se
    ) %>%
    ungroup()

dll_baseline_summarized$model <- as.factor(dll_baseline_summarized$model)
# remove _accumulated_surprisal and _added_entropy from model names
dll_baseline_summarized$model <- gsub("__accumulated_surprisal", "", dll_baseline_summarized$model)
dll_baseline_summarized$model <- gsub("__added_entropy", "", dll_baseline_summarized$model)
# refactor and drop unused levels
dll_baseline_summarized$model <- factor(dll_baseline_summarized$model, levels = c("gpt2_base", "gpt2_large", "llama_7b", "llama_13b", "mixtral"))
# rename
dll_baseline_summarized$model <- factor(dll_baseline_summarized$model, labels = c("GPT-2 base", "GPT-2 large", "Llama 7B", "Llama 13B", "Mixtral"))

# reorder predictor levels
dll_baseline_summarized$pred_effect <- factor(dll_baseline_summarized$pred_effect, levels = c("entropy", "surprisal", "combined"))
# rename
dll_baseline_summarized$pred_effect <- factor(dll_baseline_summarized$pred_effect, labels = c("Entropy", "Surprisal", "Combined"))
colnames(dll_baseline_summarized) <- c("Model", "pred_effect", "psychometric", "m", "se", "upper", "lower")

ggplot(data = dll_baseline_summarized, aes(x = pred_effect, y = m, colour = Model)) +
    geom_point(
        position = position_dodge(width = .5), size = 2
    ) +
    geom_errorbar(aes(ymin = lower, ymax = upper),
        width = .25, position = position_dodge(width = .5), linewidth = 0.4
    ) +
    scale_y_continuous(labels = function(x) format(x, scientific = TRUE)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    ylab("Delta log-likelihood") +
    xlab("") +
    scale_colour_viridis(discrete = TRUE) +
    # facet_wrap(. ~ psychometric, nrow = 2, scales = "free") +
    theme(text = element_text(family = "sans")) +
    theme(legend.position = "bottom") + 
    theme(axis.title = element_blank())
ggsave("plots/indico_deltaloglik_baseline.pdf", width = 8, height = 8, dpi = 200)


# Experiment 2a: DLL between models with and without interaction term
# define array of scores that we want to analyze
et_preproc_shuffled_complete <- et_preproc_shuffled[complete.cases(et_preproc_shuffled[c("MUmean_median_group", "OSmean_median_group", "SSmean_median_group", "SSTMRelScore_median_group")]), ]

scores <- c(
    "SLRTWord", "SLRTPseudo", "MWTPR",
    "RIASVixPR", "RIASNixPR", "RIASGixPR",
    "MUmean", "OSmean", "SSmean", "SSTMRelScore",
    "StrRTEffect", "SimRTEffect", "FAIRKPR"
)

dll_xscore_df <- delta_ll_score_interactions(et_preproc_shuffled_complete, lm_entropy_names, lm_surp_names, scores, reading_measures)
write.csv(dll_xscore_df, "data/indico/dll_xscore_df_baseline_vs_interactions.csv")

# read csv without index
dll_xscore_df <- read.csv("data/indico/dll_xscore_df_baseline_vs_interactions.csv", row.names = 1)

##### summarize and plot
dll_xscore_df$score <- as.factor(dll_xscore_df$score)
# drop unused levels
dll_xscore_df$score <- droplevels(dll_xscore_df$score)
dll_xscore_df$model <- as.factor(dll_xscore_df$model)


permt <- dll_xscore_df_fair %>%
    group_by(model, score) %>%
    do(tidy((paired.perm.test(.$delta_loglik, n.perm = 500, pval = TRUE))))
colnames(permt)[3] <- "p.value"

dll_xscore_summarized <- dll_xscore_df %>%
    group_by(model, score) %>%
    summarise(
        m = mean(delta_loglik), se = std.error(delta_loglik),
        upper = m + 1.96 * se, lower = m - 1.96 * se
    ) %>%
    ungroup()

# merge tt and dll_xscore_summarized on model and score and keep all columns
dll_xscore_summarized <- merge(dll_xscore_summarized, permt, by = c("model", "score"), all = TRUE)

dll_xscore_summarized <- mutate(dll_xscore_summarized,
    target = case_when(
        (score == "StrAccuracyEffect" | score == "StrRTEffect" | score == "SimAccuracyEffect" | score == "SimRTEffect" | score == "FAIRKPR") ~ "Cognitive control",
        (score == "SSmean" | score == "OSmean" | score == "MUmean" | score == "SSTMRelScore") ~ "Working memory",
        (score == "RIASVixPR" | score == "RIASNixPR" | score == "RIASGixPR" | score == "MWTPR") ~ "Intelligence",
        (score == "SLRTWord" | score == "SLRTPseudo") ~ "Reading fluency"
    )
)

dll_xscore_summarized$target <- as.factor(dll_xscore_summarized$target)
# rename scores
dll_xscore_summarized$score <- factor(dll_xscore_summarized$score, labels = SCORES_RENAMED)
#  new variable: if "_prob in model name, then "surprisal", if "_entropy" then "entropy"
dll_xscore_summarized <- mutate(dll_xscore_summarized,
    metric = case_when(
        grepl("__accumulated_surprisal", model) ~ "Surprisal",
        grepl("__added_entropy", model) ~ "Entropy"
    )
)

# remove _added_entropy and _multiplied_prob from model names
dll_xscore_summarized$model <- gsub("__accumulated_surprisal", "", dll_xscore_summarized$model)
dll_xscore_summarized$model <- gsub("__added_entropy", "", dll_xscore_summarized$model)
# refactor model and drop unused levels
dll_xscore_summarized$model <- as.factor(dll_xscore_summarized$model)

# reorder levels of model
dll_xscore_summarized$model <- factor(dll_xscore_summarized$model, levels = c("gpt2_base", "gpt2_large", "llama_7b", "llama_13b", "mixtral"))
# rename
dll_xscore_summarized$model <- factor(dll_xscore_summarized$model, labels = c("GPT-2 base", "GPT-2 large", "Llama 7B", "Llama 13B", "Mixtral"))

# new variable significance
dll_xscore_summarized$significance <- ifelse(dll_xscore_summarized$p.value < 0.05, "sig.", "not sig.")
dll_xscore_summarized$significance <- as.factor(dll_xscore_summarized$significance)
colnames(dll_xscore_summarized)[1] <- "Model"
colnames(dll_xscore_summarized)[10] <- "Significance"

ggplot(data = dll_xscore_summarized, aes(x = stringr::str_wrap(score, 9), y = m, colour = Model, shape = Significance)) +
  geom_point(
        position = position_dodge(width = .5), size = 1.5
        ) +
        geom_errorbar(aes(ymin = lower, ymax = upper),
            width = .25, position = position_dodge(width = .5), linewidth = 0.4
        ) +
    facet_grid(metric ~ target, scales = "free") +
    scale_x_discrete(guide = guide_axis(angle = 45)) +
    scale_y_continuous(labels = function(x) format(x, scientific = TRUE)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    ylab("Delta log likelihood (average over words)") +
    xlab("Score") +
    scale_colour_viridis(discrete = TRUE) +
    scale_shape_manual(values = c(1, 19)) +
    theme(text = element_text(family = "sans")) +
    theme(axis.title = element_blank()) +
    theme(legend.position = "bottom")
ggsave("plots/indico_deltaloglik_target-baseline-sig.pdf", width = 11, height = 8, dpi = 200)


# Experiment 2b: Quantification of interaction term

scores <- c("FAIRKPR")

# surprisal
xscore_effect_df_surp <- data.frame()
for (i in 1:length(lm_surp_names)) {
    entropy <- lm_entropy_names[i]
    surp <- lm_surp_names[i]
    model_name <- gsub("__accumulated_surprisal", "", surp)
    for (score in scores) {
        print(paste0("Fitting model for ", model_name, " score", score))
        print(" --- ")
        df_eval <- et_preproc_shuffled_complete %>%
            drop_na() %>%
            distinct()
        for (psychometric in reading_measures) {
            regression_form <- paste0(psychometric, " ~  1 + (1|subj_id) + word_length + log_lex_freq +", entropy, " +", surp, "*", score)
            effect_sizes <- get_effect_sizes(regression_form, df_eval, psychometric, surp, score, log_transform = LOG_TF)
            effect_sizes$model <- model_name
            effect_sizes$score <- score
            effect_sizes$psychometric <- psychometric
            xscore_effect_df_surp <- rbind(xscore_effect_df_surp, effect_sizes)
        }
    }
}
write.csv(xscore_effect_df_surp, "data/indico/xscore_effect_df_surp.csv")
xscore_effect_df_surp <- read.csv("data/indico/xscore_effect_df_surp.csv")

xscore_wide <- pivot_wider(
    data = xscore_effect_df_surp,
    id_cols = score,
    names_from = model,
    values_from = c("estimate", "std.error")
)

xscore_wide$score <- as.factor(xscore_wide$score)
xscore_wide <- mutate(xscore_wide,
    target = case_when(
        (score == "StrAccuracyEffect" | score == "StrRTEffect" | score == "SimAccuracyEffect" | score == "SimRTEffect" | score == "FAIRKPR") ~ "Cognitive control",
        (score == "SSmean" | score == "OSmean" | score == "MUmean" | score == "SSTMRelScore") ~ "Working memory",
        (score == "RIASVixPR" | score == "RIASNixPR" | score == "RIASGixPR" | score == "MWTPR") ~ "Intelligence",
        (score == "SLRTWord" | score == "SLRTPseudo") ~ "Reading fluency"
    )
)
#  sort rows of df by target
xscore_wide <- xscore_wide[order(xscore_wide$target), ]
xscore_wide$score <- factor(xscore_wide$score, levels = xscore_wide$score)

SCORES_RENAMED <- c(
    "Stroop", "Simon", "FAIR",
    "MWT", "RIAS verbal", "RIAS non-verbal", "RIAS total",
    "SLRT words", "SLRT pseudo-words",
    "Memory updating", "Operation span", "Sentence span", "Spatial short-term memory"
)
xscore_wide$score <- factor(xscore_wide$score, labels = SCORES_RENAMED)

# create output for latex table
sink("tables/indico_effect_sizes_interactions_suprisal.txt")
for (i in 1:nrow(xscore_wide)) {
    print(paste0(
        xscore_wide$target[i], " & ",
        xscore_wide$score[i], " & $",
        round(xscore_wide$estimate_gpt2_base[i], digits = 3), " \\db{", round(xscore_wide$std.error_gpt2_base[i], digits = 3), "}$ & $",
        round(xscore_wide$estimate_gpt2_large[i], digits = 3), " \\db{", round(xscore_wide$std.error_gpt2_large[i], digits = 3), "}$ & $",
        round(xscore_wide$estimate_llama_7b[i], digits = 3), " \\db{", round(xscore_wide$std.error_llama_7b[i], digits = 3), "}$ & $",
        round(xscore_wide$estimate_llama_13b[i], digits = 3), " \\db{", round(xscore_wide$std.error_llama_13b[i], digits = 3), "}$ & $",
        round(xscore_wide$estimate_mixtral[i], digits = 3), " \\db{", round(xscore_wide$std.error_mixtral[i], digits = 3), "}$ \\"
    ))
}
sink()

xscore_effect_df_ent_old <- xscore_effect_df_ent
xscore_effect_df_ent <- data.frame()
for (i in 1:length(lm_surp_names)) {
    entropy <- lm_entropy_names[i]
    surp <- lm_surp_names[i]
    model_name <- gsub("__accumulated_surprisal", "", surp)
    for (score in scores) {
        print(paste0("Fitting model for ", model_name, " score", score))
        print(" --- ")
        df_eval <- et_preproc_shuffled_complete %>%
            drop_na() %>%
            distinct()
        for (psychometric in reading_measures) {
            regression_form <- paste0(psychometric, " ~  1 + (1|subj_id) + word_length + log_lex_freq +", surp, " +", entropy, "*", score)
            effect_sizes <- get_effect_sizes(regression_form, df_eval, psychometric, entropy, score, log_transform = LOG_TF)
            effect_sizes$model <- model_name
            effect_sizes$score <- score
            effect_sizes$psychometric <- psychometric
            xscore_effect_df_ent <- rbind(xscore_effect_df_ent, effect_sizes)
        }
    }
}

write.csv(xscore_effect_df_ent, "data/indico/xscore_effect_df_ent.csv")
xscore_effect_df_ent <- read.csv("data/indico/xscore_effect_df_ent.csv")

xscore_wide <- pivot_wider(
    data = xscore_effect_df_ent,
    id_cols = score,
    names_from = model,
    values_from = c("estimate", "std.error")
)

xscore_wide$score <- as.factor(xscore_wide$score)
xscore_wide <- mutate(xscore_wide,
    target = case_when(
        (score == "StrAccuracyEffect" | score == "StrRTEffect" | score == "SimAccuracyEffect" | score == "SimRTEffect" | score == "FAIRKPR") ~ "Cognitive control",
        (score == "SSmean" | score == "OSmean" | score == "MUmean" | score == "SSTMRelScore") ~ "Working memory",
        (score == "RIASVixPR" | score == "RIASNixPR" | score == "RIASGixPR" | score == "MWTPR") ~ "Intelligence",
        (score == "SLRTWord" | score == "SLRTPseudo") ~ "Reading fluency"
    )
)
#  sort rows of df by target
xscore_wide <- xscore_wide[order(xscore_wide$target), ]
xscore_wide$score <- factor(xscore_wide$score, levels = xscore_wide$score)

SCORES_RENAMED <- c(
    "Stroop effect", "Simon effect", "FAIR",
    "MWT", "RIAS verbal", "RIAS non-verbal", "RIAS total",
    "SLRT words", "SLRT pseudo-words",
    "Memory updating", "Operation span", "Sentence span", "Spatial short-term memory"
)
xscore_wide$score <- factor(xscore_wide$score, labels = SCORES_RENAMED)

sink("tables/indico_effect_sizes_interactions_entropy.txt")
for (i in 1:nrow(xscore_wide)) {
    print(paste0(
        xscore_wide$target[i], " & ",
        xscore_wide$score[i], " & $",
        round(xscore_wide$estimate_gpt2_base[i], digits = 3), " \\db{", round(xscore_wide$std.error_gpt2_base[i], digits = 3), "}$ & $",
        round(xscore_wide$estimate_gpt2_large[i], digits = 3), " \\db{", round(xscore_wide$std.error_gpt2_large[i], digits = 3), "}$ & $",
        round(xscore_wide$estimate_llama_7b[i], digits = 3), " \\db{", round(xscore_wide$std.error_llama_7b[i], digits = 3), "}$ & $",
        round(xscore_wide$estimate_llama_13b[i], digits = 3), " \\db{", round(xscore_wide$std.error_llama_13b[i], digits = 3), "}$ & $",
        round(xscore_wide$estimate_mixtral[i], digits = 3), " \\db{", round(xscore_wide$std.error_mixtral[i], digits = 3), "}$ \\"
    ))
}
sink()

# Experiment 3: Compare DLL between groups

scores_grouped <- c(
    "SLRTWord_median_group", "SLRTPseudo_median_group", "MWTPR_median_group",
    "RIASVixPR_median_group", "RIASNixPR_median_group", "RIASGixPR_median_group",
    "FAIRKPR_median_group",
    "MUmean_median_group", "OSmean_median_group", "SSmean_median_group", "SSTMRelScore_median_group",
    "StrRTEffect_median_group", "SimRTEffect_median_group"
)

results <- analyze_group_dlls(models, reading_measures, scores_grouped, et_preproc_shuffled_complete, log_transform = LOG_TF)
# remove all results from results for "StrRTEffect_median_group", "SimRTEffect_median_group"
results <- results[!grepl("StrRTEffect_median_group", results$score) & !grepl("SimRTEffect_median_group", results$score), ]
# add results_new to results
results <- rbind(results, results_new)

write.csv(results, "data/indico/dll_groups_ppg_results.csv")
results <- read.csv("data/indico/dll_groups_ppg_results.csv")

permt <- results %>%
    group_by(model, score) %>%
    do(tidy((paired.perm.test(.$delta_loglik, n.perm = 500, pval = TRUE))))
colnames(permt)[3] <- "p.value"


dll_xscore_groups_summarized <- results %>%
    group_by(model, score) %>%
    summarise(
        m = mean(delta_loglik), se = std.error(delta_loglik),
        upper = m + 1.96 * se, lower = m - 1.96 * se
    ) %>%
    ungroup()

# merge tt and dll_xscore_summarized on model and score and keep all columns
dll_xscore_groups_summarized <- merge(dll_xscore_groups_summarized, permt, by = c("model", "score"), all = TRUE)

dll_xscore_groups_summarized$score <- gsub("_median_group", "", dll_xscore_groups_summarized$score)
dll_xscore_groups_summarized$score <- as.factor(dll_xscore_groups_summarized$score)

dll_xscore_groups_summarized <- mutate(dll_xscore_groups_summarized,
    target = case_when(
        (score == "StrAccuracyEffect" | score == "StrRTEffect" | score == "SimAccuracyEffect" | score == "SimRTEffect" | score == "FAIRKPR") ~ "Cognitive control",
        (score == "SSmean" | score == "OSmean" | score == "MUmean" | score == "SSTMRelScore") ~ "Working memory",
        (score == "RIASVixPR" | score == "RIASNixPR" | score == "RIASGixPR" | score == "MWTPR") ~ "Intelligence",
        (score == "SLRTWord" | score == "SLRTPseudo") ~ "Reading fluency"
    )
)

SCORES_RENAMED <- c(
    "FAIR", "Memory updating", "MWT", "Operation span", "RIAS total",
    "RIAS non-verbal", "RIAS verbal", "Simon", "SLRT pseudo- words", "SLRT words",
    "Sentence span", "Spatial short-term memory", "Stroop"
)
# rename scores
dll_xscore_groups_summarized$score <- factor(dll_xscore_groups_summarized$score, labels = SCORES_RENAMED)

#  new variable: if "_prob in model name, then "surprisal", if "_entropy" then "entropy"
dll_xscore_groups_summarized <- mutate(dll_xscore_groups_summarized,
    metric = case_when(
        grepl("__accumulated_surprisal", model) ~ "Surprisal",
        grepl("__added_entropy", model) ~ "Entropy"
    )
)

# remove _added_entropy and _multiplied_prob from model names
dll_xscore_groups_summarized$model <- gsub("__accumulated_surprisal", "", dll_xscore_groups_summarized$model)
dll_xscore_groups_summarized$model <- gsub("__added_entropy", "", dll_xscore_groups_summarized$model)
# refactor model and drop unused levels
dll_xscore_groups_summarized$model <- as.factor(dll_xscore_groups_summarized$model)

# reorder model levels
dll_xscore_groups_summarized$model <- factor(dll_xscore_groups_summarized$model, levels = c("gpt2_base", "gpt2_large", "llama_7b", "llama_13b", "mixtral"))
dll_xscore_groups_summarized$model <- factor(dll_xscore_groups_summarized$model, labels = c("GPT-2 base", "GPT-2 large", "Llama 7B", "Llama 13B", "Mixtral"))

dll_xscore_groups_summarized$significance <- ifelse(dll_xscore_groups_summarized$p.value < 0.05, "sig.", "not sig.")
dll_xscore_groups_summarized$significance <- as.factor(dll_xscore_groups_summarized$significance)

colnames(dll_xscore_groups_summarized)[1]  <- "Model"
colnames(dll_xscore_groups_summarized)[10] <- "Significance"

ggplot(data = dll_xscore_groups_summarized, aes(x = stringr::str_wrap(score, 9), y = m, colour = Model, shape = Significance)) +
    geom_point(
        position = position_dodge(width = .5), size = 1.5
    ) +
    geom_errorbar(aes(ymin = lower, ymax = upper),
        width = .25, position = position_dodge(width = .5), linewidth = 0.4
    ) +
    facet_grid(metric ~ target, scales = "free") +
    scale_x_discrete(guide = guide_axis(angle = 45)) +
    scale_y_continuous(labels = function(x) format(x, scientific = TRUE)) +
    scale_shape_manual(values = c(1, 19)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    ylab("Predictive power gain (high group = reference)") +
    xlab("Psychometric score") +
    scale_colour_viridis(discrete = T) +
    theme(text = element_text(family = "sans")) +
    theme(legend.position = "bottom") +
    theme(axis.title = element_blank())

ggsave("plots/indico_deltaloglik_groups-pg_small.pdf", width = 11, height = 5, dpi = 200)



# Appendix

scores_subset <- scores_df[, scores]
scores_new_names <- c(
    "SLRT words", "SLRT pseudo-words", "MWT",
    "RIAS verbal", "RIAS non-verbal", "RIAS total",
    "Memory updating", "Operation span", "Sentence span", "Spatial short-term memory",
    "Stroop", "Simon", "FAIR"
)
colnames(scores_subset) <- scores_new_names
# compute correlations between scores
corr <- round(cor(scores_subset), 2)
corr_p <- cor_pmat(scores_subset)

# Example usage
modified_ggcorrplot(corr,
                    p.mat = corr_p,
                    hc.order = FALSE,
                    lab = TRUE,
                    insig = "blank",
                    type = 'lower',
                    digits = 2,
                    ggtheme = theme_light,
                    outline.color = "white",
                    colors = c("#6D9EC1", "white", "#E46726"),) +
  labs(x = NULL, y = NULL)

ggsave("plots/correlation_matrix.pdf", width = 10, height = 10, dpi = 200)
 
# First need to define some helper functions ----
# Helper function to remove the diagonal elements of a matrix
.remove_diag <- function(mat) {
  diag(mat) <- NA
  return(mat)
}

# Helper function to order the correlation matrix using hierarchical clustering
.hc_cormat_order <- function(corr, hc.method = "complete") {
  dist <- as.dist((1 - corr) / 2)
  hc <- hclust(dist, method = hc.method)
  return(hc$order)
}

# Helper function to get the lower triangular part of the matrix
.get_lower_tri <- function(mat, show.diag = FALSE) {
  mat[upper.tri(mat, diag = !show.diag)] <- NA
  return(mat)
}

# Helper function to get the upper triangular part of the matrix
.get_upper_tri <- function(mat, show.diag = FALSE) {
  mat[lower.tri(mat, diag = !show.diag)] <- NA
  return(mat)
}

# Second, modify the ggcorrplot function ----
# Modified ggcorrplot function
modified_ggcorrplot <- function (corr, method = c("square", "circle"), type = c("full", 
                                                                                "lower", "upper"), ggtheme = ggplot2::theme_minimal, title = "", 
                                 show.legend = TRUE, legend.title = "Corr", show.diag = NULL, 
                                 colors = c("blue", "white", "red"), outline.color = "gray", 
                                 hc.order = FALSE, hc.method = "complete", lab = FALSE, lab_col = "black", 
                                 lab_size = 4, p.mat = NULL, sig.level = 0.05, insig = c("pch", 
                                                                                         "blank"), pch = 4, pch.col = "black", pch.cex = 5, tl.cex = 12, 
                                 tl.col = "black", tl.srt = 45, digits = 2, as.is = FALSE) 
{
  type <- match.arg(type)
  method <- match.arg(method)
  insig <- match.arg(insig)
  if (is.null(show.diag)) {
    if (type == "full") {
      show.diag <- TRUE
    }
    else {
      show.diag <- FALSE
    }
  }
  if (inherits(corr, "cor_mat")) {
    cor.mat <- corr
    corr <- .tibble_to_matrix(cor.mat)
    p.mat <- .tibble_to_matrix(attr(cor.mat, "pvalue"))
  }
  if (!is.matrix(corr) & !is.data.frame(corr)) {
    stop("Need a matrix or data frame!")
  }
  corr <- as.matrix(corr)
  corr <- base::round(x = corr, digits = digits)
  if (hc.order) {
    ord <- .hc_cormat_order(corr, hc.method = hc.method)
    corr <- corr[ord, ord]
    if (!is.null(p.mat)) {
      p.mat <- p.mat[ord, ord]
      p.mat <- base::round(x = p.mat, digits = digits)
    }
  }
  if (!show.diag) {
    corr <- .remove_diag(corr)
    p.mat <- .remove_diag(p.mat)
  }
  if (type == "lower") {
    corr <- .get_lower_tri(corr, show.diag)
    p.mat <- .get_lower_tri(p.mat, show.diag)
  }
  else if (type == "upper") {
    corr <- .get_upper_tri(corr, show.diag)
    p.mat <- .get_upper_tri(p.mat, show.diag)
  }
  corr <- reshape2::melt(corr, na.rm = TRUE, as.is = as.is)
  colnames(corr) <- c("Var1", "Var2", "value")
  corr$pvalue <- rep(NA, nrow(corr))
  corr$signif <- rep(NA, nrow(corr))
  if (!is.null(p.mat)) {
    p.mat <- reshape2::melt(p.mat, na.rm = TRUE)
    corr$coef <- corr$value
    corr$pvalue <- p.mat$value
    corr$signif <- as.numeric(p.mat$value <= sig.level)
    p.mat <- subset(p.mat, p.mat$value > sig.level)
  }
  corr$abs_corr <- abs(corr$value) * 10
  p <- ggplot2::ggplot(data = corr, mapping = ggplot2::aes_string(x = "Var1", 
                                                                  y = "Var2", fill = "value"))
  if (method == "square") {
    p <- p + ggplot2::geom_tile(color = outline.color)
  }
  else if (method == "circle") {
    p <- p + ggplot2::geom_point(color = outline.color, 
                                 shape = 21, ggplot2::aes_string(size = "abs_corr")) + 
      ggplot2::scale_size(range = c(4, 10)) + ggplot2::guides(size = "none")
  }
  p <- p + ggplot2::scale_fill_gradient2(low = colors[1], 
                                         high = colors[3], mid = colors[2], midpoint = 0, limit = c(-1, 
                                                                                                    1), space = "Lab", name = legend.title)
  if (class(ggtheme)[[1]] == "function") {
    p <- p + ggtheme()
  }
  else if (class(ggtheme)[[1]] == "theme") {
    p <- p + ggtheme
  }
  p <- p + ggplot2::theme(axis.text.x = ggplot2::element_text(angle = tl.srt, 
                                                              vjust = 1, size = tl.cex, hjust = 1), axis.text.y = ggplot2::element_text(size = tl.cex)) + 
    ggplot2::coord_fixed()
  label <- round(x = corr[, "value"], digits = digits)
  if (!is.null(p.mat) & insig == "blank") {
    ns <- corr$pvalue > sig.level
    if (sum(ns) > 0) 
      label[ns] <- ""
  }
  if (lab) {
    p <- p + ggplot2::geom_text(mapping = ggplot2::aes_string(x = "Var1", 
                                                              y = "Var2"), label = label, color = lab_col, size = lab_size)
  }
  if (!is.null(p.mat) & insig == "pch") {
    p <- p + ggplot2::geom_point(data = p.mat, mapping = ggplot2::aes_string(x = "Var1", 
                                                                             y = "Var2"), shape = pch, size = pch.cex, color = pch.col)
  }
  if (title != "") {
    p <- p + ggplot2::ggtitle(title)
  }
  if (!show.legend) {
    p <- p + ggplot2::theme(legend.position = "none")
  }
  p
}

