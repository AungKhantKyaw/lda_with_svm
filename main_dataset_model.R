# Install necessary packages (if not already installed)
packages <- c("caret", "ggplot2", "e1071", "pROC", "MASS", "bigstatsr", 
              "irlba", "MLmetrics", "dplyr", "arm", "RSpectra", "PRROC")

install.packages("corrplot")
install.packages("pheatmap")

installed <- rownames(installed.packages())
for (pkg in packages) {
  if (!(pkg %in% installed)) install.packages(pkg)
}
# Load libraries
library(caret)
library(ggplot2)
library(reshape2)
library(e1071)
library(pROC)
library(MASS)
library(bigstatsr)
library(RSpectra)
library(Matrix)
library(irlba)
library(microbenchmark) 
library(MLmetrics)
library(dplyr)
library(kernlab)
library(arm)
library(PRROC)

# ===================================================
# ----- LOAD DATASET -----
# ===================================================
df <- read.csv("main_dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
#df <- df[sample(nrow(df), 5000), ]

# --- Check for missing values ---
sum(is.na(df))  # Count total missing values

# --- Remove rows with missing values ---
df <- na.omit(df)

# --- Check for duplicates and remove them ---
df <- df[!duplicated(df), ]

# --- Convert categorical variables (characters) to numeric ---
df[] <- lapply(df, function(col) {
  if (is.character(col)) as.numeric(factor(col)) else col
})


# --- Separate features and labels ---
X <- df[, !colnames(df) %in% c("label")]
y <- df$label

# --- Feature scaling ---
X_scaled <- as.data.frame(scale(X))

n_features <- ncol(X_scaled)
cat("Number of features:", n_features, "\n")
print(colnames(X_scaled))

ggplot(df, aes(x = factor(label))) +
  geom_bar(fill = c("lightblue", "salmon")) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  labs(title = "Distribution of Phishing vs. Non-Phishing URLs",
       x = "Label (0: Phishing, 1: Non-Phishing)",
       y = "Number of URLs") +
  theme_minimal()

# Find highly correlated feature pairs
cor_matrix <- cor(X_scaled)
high_corr <- findCorrelation(cor_matrix, cutoff = 0.85)  # from caret package

# Remove them
X_reduced <- X_scaled[, -high_corr]

# Redraw heatmap
library(corrplot)
corrplot(cor(X_reduced), method = "color", tl.cex = 0.7,
         title = "Reduced Feature Correlation Heatmap", mar=c(0,0,1,0))

# ===================================================
# ----- SPLIT DATA INTO TRAIN (70%) / TEST (30%) -----
# ===================================================
set.seed(123)
sample_size <- floor(0.7 * nrow(X_scaled))
train_indices <- sample(seq_len(nrow(X_scaled)), size = sample_size)

# Train set (70%)
X_train <- X_scaled[train_indices, ]
y_train <- y[train_indices]

# Test set (30%)
X_test <- X_scaled[-train_indices, ]
y_test <- y[-train_indices]

cat("Training set size (70%):", nrow(X_train), "\n")
cat("Test set size (30%):", nrow(X_test), "\n")

# ===================================================
# ----- SPLIT TRAIN SET INTO TRAIN (80%) / VALIDATION (20%) -----
# ===================================================
set.seed(123)
train_size <- floor(0.7 * nrow(X_train))
train_indices_from_train <- sample(seq_len(nrow(X_train)), size = train_size)

# 80% of train set (56% of the original data)
X_train_final <- X_train[train_indices_from_train, ]
y_train_final <- y_train[train_indices_from_train]

# Validation set (20% of train set, 14% of original)
X_valid <- X_train[-train_indices_from_train, ]
y_valid <- y_train[-train_indices_from_train]

cat("Final Train set size (70% of 70%):", nrow(X_train_final), "\n")
cat("Validation set size (30% of 70%):", nrow(X_valid), "\n")


# --- Perform SVM parameter tuning ---
tune_result <- tune(svm,
                    train.x = X_train_final,
                    train.y = as.factor(y_train_final),
                    kernel = "radial",
                    ranges = list(
                      cost = 2^(-5:5),
                      gamma = 2^(-5:5)
                    ),
                    tunecontrol = tune.control(sampling = "cross", cross = 5)
)

# --- Best model and parameters ---
best_model <- tune_result$best.model
print(best_model)

# --- Show the full result table ---
results <- tune_result$performances
results$Accuracy <- 1 - results$error   # Compute Accuracy manually
print(results)

# --- Show the best Accuracy ---
best_accuracy <- max(results$Accuracy)
cat("\nBest Cross-Validated Accuracy:", round(best_accuracy * 100, 2), "%\n")

# --- Plotting tuning results ---
plot(tune_result)

best_model <- tune_result$best.model
summary(best_model)

########### TUNE ##########################
# tuned <- tune.svm(x = X_train_final, y = as.factor(y_train_final),
#                   kernel = "radial",
#                   gamma = 2^(-5:2), cost = 2^(-1:4))
# 
# best_model <- tuned$best.model
# 
# best_gamma <- best_model$gamma
# best_cost <- best_model$cost
# 
# cat("Best Gamma:", best_gamma, "\n")
# cat("Best Cost:", best_cost, "\n")
# best_gamma = 0.015625 #old
# best_cost  = 16 #old

best_gamma = 0.03125
best_cost = 0.0625

# ===================================================
# ----- SVM WITHOUT LDA - WHOLE FEATURE -----
# ===================================================

# Initialize results container
results_no_lda <- data.frame(Fold = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(), 
                             Accuracy = numeric(), AUC = numeric(), Precision = numeric(), 
                             Recall = numeric(), F1_Score = numeric(), Training_Time = numeric())

# 10-fold cross-validation
num_folds <- 10
folds <- createFolds(y_train_final, k = num_folds, list = TRUE)

for (i in seq_along(folds)) {
  cat("Processing Fold", i, "\n")
  
  # Train/Test Split
  train_idx <- unlist(folds[-i])  
  test_idx <- unlist(folds[i])
  X_train_fold <- X_train_final[train_idx, ]
  y_train_fold <- y_train_final[train_idx]
  X_test_fold <- X_train_final[test_idx, ]
  y_test_fold <- y_train_final[test_idx]
  
  start_time <- Sys.time()
  svm_model_no_lda <- svm(X_train_fold, as.factor(y_train_fold), kernel = "radial",                          
                          gamma = best_gamma,
                          cost = best_cost, probability = TRUE)
  end_time <- Sys.time()
  training_time_no_lda <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  pred_no_lda <- predict(svm_model_no_lda, X_test_fold)
  pred_proba_no_lda <- attr(predict(svm_model_no_lda, X_test_fold, probability = TRUE), "probabilities")[, 2]
  
  conf_matrix <- confusionMatrix(pred_no_lda, as.factor(y_test_fold))
  
  # Store results
  results_no_lda <- rbind(results_no_lda, data.frame(
    Fold = i, 
    TP = conf_matrix$table["1", "1"], 
    TN = conf_matrix$table["0", "0"], 
    FP = conf_matrix$table["1", "0"], 
    FN = conf_matrix$table["0", "1"], 
    Accuracy = conf_matrix$overall["Accuracy"],
    AUC = auc(y_test_fold, pred_proba_no_lda),
    Precision = conf_matrix$byClass["Precision"],
    Recall = conf_matrix$byClass["Recall"],
    F1_Score = conf_matrix$byClass["F1"],
    Training_Time = training_time_no_lda
  ))
}

print(results_no_lda)
summary(results_no_lda)

# ===================================================
# ----- SVM WITH LDA -----
# ===================================================
num_folds <- 10
folds <- createFolds(y_train_final, k = num_folds, list = TRUE)

results_lda <- data.frame(Fold = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(), 
                          Accuracy = numeric(), AUC = numeric(), Precision = numeric(), 
                          Recall = numeric(), F1_Score = numeric(), Training_Time = numeric())

for (i in seq_along(folds)) {
  cat("Processing Fold", i, "\n")
  
  # Train/Test Split
  train_idx <- unlist(folds[-i])  
  test_idx <- unlist(folds[i])
  X_train_fold <- X_train_final[train_idx, ]
  y_train_fold <- y_train_final[train_idx]
  X_test_fold <- X_train_final[test_idx, ]
  y_test_fold <- y_train_final[test_idx]
  
  # Apply LDA
  lda_model <- lda(y_train_fold ~ ., data = data.frame(y_train_fold, X_train_fold))
  X_train_lda <- as.matrix(X_train_fold) %*% lda_model$scaling
  X_test_lda <- as.matrix(X_test_fold) %*% lda_model$scaling
  
  start_time <- Sys.time()
  svm_model_lda <- svm(X_train_lda,
                       as.factor(y_train_fold),
                       kernel = "radial",
                       gamma = best_gamma,
                       cost = best_cost,
                       probability = TRUE)
  end_time <- Sys.time()
  training_time_lda <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  pred_lda <- predict(svm_model_lda, X_test_lda)
  pred_proba_lda <- attr(predict(svm_model_lda, X_test_lda, probability = TRUE), "probabilities")[, 2]
  
  conf_matrix_lda <- confusionMatrix(pred_lda, as.factor(y_test_fold))
  
  # Store results
  results_lda <- rbind(results_lda, data.frame(
    Fold = i, 
    TP = conf_matrix_lda$table["1", "1"], 
    TN = conf_matrix_lda$table["0", "0"], 
    FP = conf_matrix_lda$table["1", "0"], 
    FN = conf_matrix_lda$table["0", "1"], 
    Accuracy = conf_matrix_lda$overall["Accuracy"],
    AUC = auc(y_test_fold, pred_proba_lda),
    Precision = conf_matrix_lda$byClass["Precision"],
    Recall = conf_matrix_lda$byClass["Recall"],
    F1_Score = conf_matrix_lda$byClass["F1"],
    Training_Time = training_time_lda
  ))
}

print(results_lda)
summary(results_lda)


results_long <- reshape2::melt(results_lda[, c("Fold", "Accuracy", "AUC", "Precision", "Recall", "F1_Score")], id.vars = "Fold")

ggplot(results_long, aes(x = factor(Fold), y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~variable, scales = "free_y") +
  labs(title = "SVM with LDA - Cross-Validation Metrics", x = "Fold", y = "Metric Value") +
  theme_minimal()

# ===================================================
# ----- FEATURE IMPORTANCE FROM LDA -----
# ===================================================
lda_scaling <- lda_model$scaling
lda_importance <- data.frame(
  Feature = rownames(lda_scaling),
  Coefficient = lda_scaling[, 1],
  Abs_Coefficient = abs(lda_scaling[, 1])
)
lda_importance <- lda_importance[order(-lda_importance$Abs_Coefficient), ]

cat("=== Top 21 Important Features from LDA ===\n")
print(head(lda_importance, 21))

ggplot(head(lda_importance, 21), aes(x = reorder(Feature, Abs_Coefficient), y = Abs_Coefficient)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 21 Important Features (LDA)", x = "Feature", y = "Absolute Coefficient")



# ------------ 21 features with LDA -------------------------------
W <- lda_model$scaling
W_vec <- as.vector(W)

X_train_weighted <- sweep(X_train_final, 2, W_vec, `*`)
X_valid_weighted  <- sweep(X_valid, 2, W_vec, `*`)

cat("Dimensions of weighted training data:", dim(X_train_weighted), "\n")
cat("Dimensions of weighted validation data:", dim(X_valid_weighted), "\n")

X_weighted_data <- as.matrix(X_train_weighted)
y_weighted_data <- y_train_final

set.seed(123)

num_folds <- 10
num_features_to_select <- 21

folds <- createFolds(y_weighted_data, k = num_folds, list = TRUE, returnTrain = FALSE)

# Initialize results container
results_21cols <- data.frame(Fold = integer(), TP = integer(), TN = integer(), FP = integer(),
                             FN = integer(), Accuracy = numeric(), AUC = numeric(), Precision = numeric(),
                             Recall = numeric(), F1_Score = numeric(), Training_Time = numeric(), CPU_Time = numeric())

# Initialize lists for storing ROC/PR-related data
all_preds_proba <- list()
all_true_labels <- list()
all_conf_matrices <- list()

for (i in seq_along(folds)) {
  cat("Processing Fold", i, "\n")
  
  test_idx <- folds[[i]]
  train_idx <- setdiff(seq_along(y_weighted_data), test_idx)
  
  X_train_fold <- X_weighted_data[train_idx, ]
  y_train_fold <- y_weighted_data[train_idx]
  
  X_test_fold <- X_weighted_data[test_idx, ]
  y_test_fold <- y_weighted_data[test_idx]
  
  # Randomly select 21 features
  selected_features <- sample(1:ncol(X_weighted_data), num_features_to_select)
  
  X_train_selected <- X_train_fold[, selected_features]
  X_test_selected <- X_test_fold[, selected_features]
  
  # Track CPU and wall-clock time
  cpu_start <- proc.time()
  start_time <- Sys.time()
  
  svm_model <- svm(X_train_selected, as.factor(y_train_fold), kernel = "radial", probability = TRUE)
  
  end_time <- Sys.time()
  cpu_end <- proc.time()
  
  training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  cpu_time <- (cpu_end - cpu_start)[["user.self"]] + (cpu_end - cpu_start)[["sys.self"]]
  
  preds <- predict(svm_model, X_test_selected)
  preds_proba <- attr(predict(svm_model, X_test_selected, probability = TRUE), "probabilities")[, 2]
  
  conf_matrix <- confusionMatrix(preds, as.factor(y_test_fold), positive = "1")
  
  results_21cols <- rbind(results_21cols, data.frame(
    Fold = i,
    TP = conf_matrix$table["1", "1"],
    TN = conf_matrix$table["0", "0"],
    FP = conf_matrix$table["1", "0"],
    FN = conf_matrix$table["0", "1"],
    Accuracy = conf_matrix$overall["Accuracy"],
    AUC = ifelse(length(unique(y_test_fold)) > 1, auc(y_test_fold, preds_proba), NA),
    Precision = conf_matrix$byClass["Precision"],
    Recall = conf_matrix$byClass["Recall"],
    F1_Score = conf_matrix$byClass["F1"],
    Training_Time = training_time,
    CPU_Time = cpu_time
  ))
  
  if (length(unique(y_test_fold)) > 1) {
    all_preds_proba[[i]] <- preds_proba
    all_true_labels[[i]] <- as.numeric(as.character(y_test_fold))
  }
  all_conf_matrices[[i]] <- conf_matrix$table
}

print(results_21cols)
summary(results_21cols)

# Save the results
write.csv(results_21cols, file = "results_21_random_features_may_8.csv", row.names = FALSE)


# ===================================================
# ----- SVM WITH LDA WEIGHTED FEATURES -----
# ===================================================
# Fit LDA on full training set to get feature importances
lda_model <- lda(y_train_final ~ ., data = data.frame(y_train_final, X_train_final))

W <- lda_model$scaling
W_vec <- as.vector(W)

X_train_weighted <- sweep(X_train_final, 2, W_vec, `*`)
X_valid_weighted <- sweep(X_valid, 2, W_vec, `*`)

cat("Dimensions of weighted training data:", dim(X_train_weighted), "\n")
cat("Dimensions of weighted validation data:", dim(X_valid_weighted), "\n")

# Get top 21 LDA features
top_features <- order(abs(W_vec), decreasing = TRUE)[1:21]
X_weighted_data <- as.matrix(X_train_weighted)
y_weighted_data <- as.factor(y_train_final)

set.seed(123)

# Split into 10 batches
n <- nrow(X_weighted_data)
batch_size <- floor(n / 10)
indices <- split(1:n, ceiling(seq_along(1:n) / batch_size))

# Make sure exactly 10 batches (merge remainder to last)
if (length(indices) > 10) {
  indices[[10]] <- c(indices[[10]], unlist(indices[11:length(indices)]))
  indices <- indices[1:10]
}

# Initialize results container
results_top21 <- data.frame(Fold = integer(), TP = integer(), TN = integer(), FP = integer(),
                            FN = integer(), Accuracy = numeric(), AUC = numeric(), Precision = numeric(),
                            Recall = numeric(), F1_Score = numeric(), Training_Time = numeric(), CPU_Time = numeric())

# Initialize lists to store predictions and true labels
all_preds_proba <- list()
all_true_labels <- list()
all_conf_matrices <- list()

for (i in seq_along(indices)) {
  cat("Processing Batch", i, "\n")
  
  test_idx <- indices[[i]]
  train_idx <- setdiff(1:n, test_idx)
  
  X_train <- X_weighted_data[train_idx, ]
  y_train <- y_weighted_data[train_idx]
  X_test <- X_weighted_data[test_idx, ]
  y_test <- y_weighted_data[test_idx]
  
  # Select top 21 LDA features
  X_train_selected <- X_train[, top_features]
  X_test_selected <- X_test[, top_features]
  
  # Measure CPU and wall-clock time
  cpu_start <- proc.time()
  start_time <- Sys.time()
  
  svm_model_top21 <- svm(X_train_selected, as.factor(y_train), kernel = "radial",
                         gamma = best_gamma,
                         cost = best_cost,
                         probability = TRUE)
  
  end_time <- Sys.time()
  cpu_end <- proc.time()
  
  training_time_top21 <- as.numeric(difftime(end_time, start_time, units = "secs"))
  cpu_time_top21 <- (cpu_end - cpu_start)[["user.self"]] + (cpu_end - cpu_start)[["sys.self"]]
  
  pred_top21 <- predict(svm_model_top21, X_test_selected)
  pred_proba_top21 <- attr(predict(svm_model_top21, X_test_selected, probability = TRUE), "probabilities")[, 2]
  
  conf_matrix_top21 <- confusionMatrix(pred_top21, as.factor(y_test))
  
  results_top21 <- rbind(results_top21, data.frame(
    Fold = i,
    TP = conf_matrix_top21$table["1", "1"],
    TN = conf_matrix_top21$table["0", "0"],
    FP = conf_matrix_top21$table["1", "0"],
    FN = conf_matrix_top21$table["0", "1"],
    Accuracy = conf_matrix_top21$overall["Accuracy"],
    AUC = ifelse(length(unique(y_test)) > 1, auc(y_test, pred_proba_top21), NA),
    Precision = conf_matrix_top21$byClass["Precision"],
    Recall = conf_matrix_top21$byClass["Recall"],
    F1_Score = conf_matrix_top21$byClass["F1"],
    Training_Time = training_time_top21,
    CPU_Time = cpu_time_top21
  ))
  
  # Store predictions and true labels
  if (length(unique(y_test)) > 1) {
    all_preds_proba[[i]] <- pred_proba_top21
    all_true_labels[[i]] <- as.numeric(as.character(y_test))
  }
  all_conf_matrices[[i]] <- conf_matrix_top21$table
}

print(results_top21)
summary(results_top21)

# Save to CSV
write.csv(results_top21, "results_top21_batch_may_8.csv", row.names = FALSE)


# ----- Plotting Average Confusion Matrix -----
if (require(ggplot2)) {
  # Average the confusion matrices
  averaged_cm <- Reduce(`+`, all_conf_matrices) / num_folds
  cm_df_avg <- as.data.frame(averaged_cm)
  plt_cm_avg <- ggplot(data = cm_df_avg, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = sprintf("%.2f", Freq)), vjust = 0.5) + # Display average counts
    labs(title = "Average Confusion Matrix (Top 21 Features)", fill = "Average Frequency") +
    theme_minimal()
  print(plt_cm_avg)
} else {
  cat("Please install the 'ggplot2' package to plot the average confusion matrix.\n")
}

# ----- Plotting Average ROC Curve -----
if (require(ggplot2) && require(pROC)) {
  if (length(all_true_labels) > 0) {
    # Combine true labels and predictions
    combined_labels <- unlist(all_true_labels)
    combined_preds_proba <- unlist(all_preds_proba)
    
    if (length(unique(combined_labels)) > 1) {
      roc_obj_avg <- roc(combined_labels, combined_preds_proba)
      roc_df_avg <- data.frame(FPR = 1 - roc_obj_avg$specificities, TPR = roc_obj_avg$sensitivities)
      
      plt_roc_avg <- ggplot(roc_df_avg, aes(x = FPR, y = TPR)) +
        geom_line(color = "blue") +
        geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
        labs(title = paste("Average ROC Curve (Top 21 Features)", "\nAUC =", round(auc(roc_obj_avg), 3)),
             x = "False Positive Rate", y = "True Positive Rate") +
        theme_minimal()
      print(plt_roc_avg)
    } else {
      cat("Warning: Cannot plot average ROC curve as all test sets had only one class.\n")
    }
  } else {
    cat("Warning: No ROC data to plot.\n")
  }
} else {
  cat("Please install 'ggplot2' and 'pROC' packages to plot the average ROC curve.\n")
}

# ----- Plotting Average Precision-Recall Curve -----
if (require(ggplot2) && require(PRROC)) {
  if (length(all_true_labels) > 0) {
    combined_labels <- unlist(all_true_labels)
    combined_preds_proba <- unlist(all_preds_proba)
    
    if (length(unique(combined_labels)) > 1) {
      pr_obj_avg <- pr.curve(combined_labels, combined_preds_proba, curve = TRUE)
      pr_df_avg <- data.frame(Recall = pr_obj_avg$curve[, 1], Precision = pr_obj_avg$curve[, 2], Type = "Model")
      baseline_df <- data.frame(Recall = c(0, 1), Precision = rep(mean(combined_labels), 2), Type = "Baseline")
      plot_df_pr <- rbind(pr_df_avg, baseline_df)
      
      plt_pr_avg <- ggplot(plot_df_pr, aes(x = Recall, y = Precision, color = Type)) +
        geom_line() +
        labs(title = paste("Average Precision-Recall Curve (Top 21 Features)"),
             x = "Recall", y = "Precision", color = "Curve Type") +
        scale_color_manual(values = c("Model" = "blue", "Baseline" = "red")) +
        theme_minimal()
      print(plt_pr_avg)
    } else {
      cat("Warning: Cannot plot average Precision-Recall curve as all test sets had only one class.\n")
    }
  } else {
    cat("Warning: No PR data to plot.\n")
  }
} else {
  cat("Please install 'ggplot2' and 'PRROC' packages to plot the average Precision-Recall curve.\n")
}


# ===== Final Model Training for Test Prediction =====
cat("\nTraining Final SVM Model on All Training Data with Top 21 LDA-Weighted Features\n")

X_train_final_selected <- X_weighted_data[, top_features]
y_train_final <- y_weighted_data

svm_model_top21_final <- svm(X_train_final_selected, as.factor(y_train_final), kernel = "radial", 
                             gamma = best_gamma,
                             cost = best_cost,
                             probability = TRUE)



# ===================================================
# ----- PLOT PERFORMANCE COMPARISON -----
# ===================================================
results_no_lda$Type <- "No LDA"
results_lda$Type <- "LDA"
results_top21$Type <- "LDA Top 21"
all_results <- rbind(results_no_lda, results_lda, results_top21)

ggplot(all_results, aes(x = Type, y = Accuracy)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "SVM Accuracy Comparison", y = "Accuracy")


# Get the absolute values of LDA loadings
lda_weights <- abs(lda_model$scaling[, 1])

# Get indices of the top 21 features
top_21_indices <- order(lda_weights, decreasing = TRUE)[1:21]

# If you're using the real LDA-weighted data
X_train_weighted_top21 <- X_train_weighted[, top_21_indices]

########################### SVD INC ###################################

# Incremental W (SVD) update
incremental_svd_lda <- function(prev_cov_svd, prev_mean_svd, prev_n_svd, X_new, n_components_svd) {
  new_n_svd <- prev_n_svd + nrow(X_new)
  new_mean_svd <- (prev_mean_svd * prev_n_svd + colSums(X_new)) / new_n_svd
  X_centered_svd <- scale(X_new, center = new_mean_svd, scale = FALSE)
  cov_update_svd <- (prev_cov_svd * (prev_n_svd - 1) + crossprod(X_centered_svd)) / (new_n_svd - 1)
  k_svd <- min(ncol(cov_update_svd), n_components_svd)
  
  svd_result_svd <- tryCatch({
    svd(cov_update_svd)
  }, error = function(e) {
    warning("SVD failed.")
    return(NULL)
  })
  if (is.null(svd_result_svd)) return(NULL)
  
  list(
    projection_svd = svd_result_svd$v[, 1:k_svd, drop = FALSE],
    updated_cov_svd = cov_update_svd,
    updated_mean_svd = new_mean_svd,
    updated_n_svd = new_n_svd
  )
}

# Performance metrics
compute_metrics_svd <- function(y_true_svd, y_pred_svd, y_scores_svd) {
  if (length(unique(y_pred_svd)) < 2) {
    warning("Only one class in predictions.")
    return(data.frame(Accuracy_svd = NA, Precision_svd = NA, Recall_svd = NA, F1_Score_svd = NA,
                      AUC_svd = NA, TP_svd = NA, TN_svd = NA, FP_svd = NA, FN_svd = NA,
                      W_Update_Time = NA, CPU_User_svd = NA, CPU_System_svd = NA))
  }
  cm_svd <- caret::confusionMatrix(factor(y_pred_svd), factor(y_true_svd))
  auc_value_svd <- tryCatch({
    pROC::auc(factor(y_true_svd), y_scores_svd)
  }, error = function(e) {
    warning("AUC failed.")
    return(NA)
  })
  
  data.frame(
    Accuracy_svd = cm_svd$overall["Accuracy"],
    Precision_svd = cm_svd$byClass["Precision"],
    Recall_svd = cm_svd$byClass["Recall"],
    F1_Score_svd = cm_svd$byClass["F1"],
    AUC_svd = auc_value_svd,
    TP_svd = cm_svd$table[2, 2],
    TN_svd = cm_svd$table[1, 1],
    FP_svd = cm_svd$table[1, 2],
    FN_svd = cm_svd$table[2, 1]
  )
}

# SVM update
update_svm_svd <- function(model_svd, X_transformed_svd, y_batch_svd, X_all_svd = NULL, y_all_svd = NULL, gamma_manual_svd, cost_manual_svd) {
  data_x <- if (!is.null(X_all_svd)) X_all_svd else X_transformed_svd
  data_y <- if (!is.null(y_all_svd)) y_all_svd else y_batch_svd
  e1071::svm(data_x, as.factor(data_y), kernel = "radial", gamma = gamma_manual_svd,
             cost = cost_manual_svd, probability = TRUE, class.weights = c('0' = 1, '1' = 1))
}

# Incremental learning pipeline (SVD)
incremental_learning_pipeline_svd_eval_batch1 <- function(X_train_svd, y_train_svd, batch_size_svd, gamma_manual_svd, cost_manual_svd) {
  num_samples_svd <- nrow(X_train_svd)
  num_batches_svd <- ceiling(num_samples_svd / batch_size_svd)
  stats_svd <- list(cov_svd = NULL, mean_svd = NULL, n_svd = NULL)
  results_svd <- data.frame()
  
  # Storage for evaluation
  all_probabilities_eval_svd <- list()
  all_true_labels_eval_svd <- list()
  all_predictions_eval_svd <- list()
  
  # === Trunk 1: Base model, no result reported ===
  X_batch_svd_batch1 <- X_train_svd[1:batch_size_svd, ]
  y_batch_svd_batch1 <- y_train_svd[1:batch_size_svd]
  
  stats_svd$n_svd <- nrow(X_batch_svd_batch1)
  stats_svd$mean_svd <- colMeans(X_batch_svd_batch1)
  stats_svd$cov_svd <- cov(X_batch_svd_batch1)
  
  svd_result_svd <- svd(stats_svd$cov_svd)
  k <- min(ncol(X_batch_svd_batch1), stats_svd$n_svd - 1)
  projection_matrix_svd <- svd_result_svd$v[, 1:k, drop = FALSE]
  X_transformed_svd_batch1 <- X_batch_svd_batch1 %*% projection_matrix_svd
  
  # Initial SVM model trained only on base trunk
  svm_model_svd <- update_svm_svd(NULL, X_transformed_svd_batch1, y_batch_svd_batch1, gamma_manual_svd = gamma_manual_svd, cost_manual_svd = cost_manual_svd)
  
  # === Trunks 2 to 10: Incremental updates ===
  for (batch_idx_svd in 2:num_batches_svd) {
    start_idx <- (batch_idx_svd - 1) * batch_size_svd + 1
    end_idx <- min(batch_idx_svd * batch_size_svd, num_samples_svd)
    X_batch <- X_train_svd[start_idx:end_idx, ]
    y_batch <- y_train_svd[start_idx:end_idx]
    
    start_time_w <- Sys.time()
    cpu_start <- proc.time()
    
    update_result_svd <- incremental_svd_lda(
      stats_svd$cov_svd, stats_svd$mean_svd, stats_svd$n_svd,
      X_batch, n_components_svd = k
    )
    
    cpu_end <- proc.time()
    end_time_w <- Sys.time()
    
    if (is.null(update_result_svd)) {
      warning(paste("Batch", batch_idx_svd, "SVD failed. Skipping this batch."))
      empty_metrics <- data.frame(
        Accuracy_svd = NA, Precision_svd = NA, Recall_svd = NA, F1_Score_svd = NA,
        AUC_svd = NA, TP_svd = NA, TN_svd = NA, FP_svd = NA, FN_svd = NA,
        W_Update_Time = 0, CPU_User_svd = 0, CPU_System_svd = 0, Trunk = batch_idx_svd
      )
      results_svd <- rbind(results_svd, empty_metrics)
      next
    }
    
    projection_matrix_svd <- update_result_svd$projection_svd
    stats_svd <- update_result_svd[c("updated_cov_svd", "updated_mean_svd", "updated_n_svd")]
    names(stats_svd) <- c("cov_svd", "mean_svd", "n_svd")
    
    # Apply updated projection (W)
    X_transformed <- X_batch %*% projection_matrix_svd
    X_train_update <- rbind(X_transformed_svd_batch1, X_transformed)
    y_train_update <- c(y_batch_svd_batch1, y_batch)
    
    # Update SVM with combined data
    svm_model_svd <- update_svm_svd(svm_model_svd, X_transformed, y_batch, X_all_svd = X_train_update, y_all_svd = y_train_update, gamma_manual_svd = gamma_manual_svd, cost_manual_svd = cost_manual_svd)
    
    pred_eval <- predict(svm_model_svd, X_train_update)
    prob_scores_eval <- attr(predict(svm_model_svd, X_train_update, probability = TRUE), "probabilities")[, 2]
    metrics_row <- compute_metrics_svd(y_train_update, pred_eval, prob_scores_eval)
    
    metrics_row$W_Update_Time <- as.numeric(difftime(end_time_w, start_time_w, units = "secs"))
    metrics_row$CPU_User_svd <- (cpu_end - cpu_start)["user.self"]
    metrics_row$CPU_System_svd <- (cpu_end - cpu_start)["sys.self"]
    metrics_row$Trunk <- batch_idx_svd
    
    results_svd <- rbind(results_svd, metrics_row)
    all_probabilities_eval_svd[[batch_idx_svd]] <- prob_scores_eval
    all_true_labels_eval_svd[[batch_idx_svd]] <- as.numeric(as.character(y_train_update))
    all_predictions_eval_svd[[batch_idx_svd]] <- factor(as.character(pred_eval), levels = levels(as.factor(y_batch)))
  }
  
  return(list(
    results = results_svd,
    probabilities_eval = all_probabilities_eval_svd,
    true_labels_eval = all_true_labels_eval_svd,
    predictions_eval = all_predictions_eval_svd
  ))
}

# ======================
# Run the SVD-INC Pipeline
# ======================
set.seed(42)

X_train_weighted_svd <- X_train_weighted_top21
y_train_final_svd <- y_train_final

num_trunks_svd <- 10
trunk_size_svd <- ceiling(nrow(X_train_weighted_svd) / num_trunks_svd)

best_gamma_svd <- 0.03125
best_cost_svd <- 0.0625

batch_results_svd <- incremental_learning_pipeline_svd_eval_batch1(
  as.matrix(X_train_weighted_svd),
  as.factor(y_train_final_svd),
  batch_size_svd = trunk_size_svd,
  gamma_manual_svd = best_gamma_svd,
  cost_manual_svd = best_cost_svd
)

print(batch_results_svd$results)
write.csv(batch_results_svd$results, file = "results_svd_may_8.csv", row.names = FALSE)

# ----- Plotting -----
if (require(ggplot2) && require(caret) && require(pROC) && require(PRROC)) {
  # ----- Confusion Matrix Visualization -----
  final_predictions_eval_svd <- batch_results_svd$predictions_eval[[length(batch_results_svd$predictions_eval)]]
  final_true_labels_eval_svd <- factor(batch_results_svd$true_labels_eval[[length(batch_results_svd$true_labels_eval)]], levels = levels(final_predictions_eval_svd))
  cm_final_eval_svd <- caret::confusionMatrix(final_predictions_eval_svd, final_true_labels_eval_svd)
  cm_df_eval_svd <- as.data.frame(cm_final_eval_svd$table)
  plt_cm_eval_svd <- ggplot(data = cm_df_eval_svd, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), vjust = 0.5) +
    labs(title = "Confusion Matrix (SVD Incremental)", fill = "Frequency") +
    theme_minimal()
  print(plt_cm_eval_svd)
  
  # ----- ROC Curve -----
  all_probs_eval_svd <- unlist(batch_results_svd$probabilities_eval)
  all_labels_eval_svd <- unlist(batch_results_svd$true_labels_eval)
  
  if (length(unique(all_labels_eval_svd)) > 1) {
    roc_obj_eval_svd <- roc(all_labels_eval_svd, all_probs_eval_svd)
    plt_roc_eval_svd <- ggplot(data.frame(FPR = 1 - roc_obj_eval_svd$specificities, TPR = roc_obj_eval_svd$sensitivities),
                               aes(x = FPR, y = TPR)) +
      geom_line(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      labs(title = "ROC Curve (SVD Incremental)",
           x = "False Positive Rate", y = "True Positive Rate",
           caption = paste("AUC =", round(auc(roc_obj_eval_svd), 3))) +
      theme_minimal()
    print(plt_roc_eval_svd)
  } else {
    cat("Warning: Cannot plot ROC curve as all true labels in the evaluation set are the same.\n")
  }
  
  # ----- Precision-Recall Curve -----
  if (require(PRROC) && length(unique(all_labels_eval_svd)) > 1) {
    pr_data_eval_svd <- pr.curve(all_labels_eval_svd, all_probs_eval_svd, curve = TRUE)
    plt_pr_eval_svd <- ggplot(data.frame(Recall = pr_data_eval_svd$curve[, 1], Precision = pr_data_eval_svd$curve[, 2]),
                              aes(x = Recall, y = Precision)) +
      geom_line(color = "blue") +
      geom_hline(yintercept = mean(all_labels_eval_svd), linetype = "dashed", color = "red") + # Baseline
      labs(title = "Precision-Recall Curve (SVD Incremental)",
           x = "Recall", y = "Precision",
           caption = paste("AUC-PR =", round(pr_data_eval_svd$auc.integral, 3))) +
      theme_minimal()
    print(plt_pr_eval_svd)
  } else if (!require(PRROC)) {
    cat("Please install the 'PRROC' package to plot Precision-Recall curves.\n")
  } else {
    cat("Warning: Cannot plot Precision-Recall curve as all true labels in the evaluation set are the same.\n")
  }
  
} else {
  cat("Please install 'ggplot2', 'caret', 'pROC', and 'PRROC' packages to plot.\n")
}


######################## INC PCA ###############################

incremental_pca_lda <- function(prev_mean_pca, prev_var_pca, prev_n_pca, X_new_pca, n_components_pca) {
  new_n_pca <- prev_n_pca + nrow(X_new_pca)
  new_mean_pca <- (prev_mean_pca * prev_n_pca + colSums(X_new_pca)) / new_n_pca
  X_centered_ob <- scale(X_new_pca, center = prev_mean_pca, scale = FALSE)
  S_ob <- crossprod(X_centered_ob)
  new_var_pca <- (prev_var_pca * (prev_n_pca - 1) + S_ob) / (new_n_pca - 1)
  eig_decomp_pca <- eigen(new_var_pca, symmetric = TRUE)
  num_components <- min(n_components_pca, ncol(eig_decomp_pca$vectors))
  projection_matrix_pca <- eig_decomp_pca$vectors[, 1:num_components, drop = FALSE]
  return(list(
    projection_pca = projection_matrix_pca,
    updated_mean_pca = new_mean_pca,
    updated_var_pca = new_var_pca,
    updated_n_pca = new_n_pca
  ))
}

compute_metrics_pca <- function(y_true, y_pred, y_scores) {
  if (length(unique(y_pred)) < 2) {
    warning("y_pred contains only one class, skipping batch metrics calculation.")
    return(data.frame(Accuracy_pca = NA, Precision_pca = NA, Recall_pca = NA, F1_Score_pca = NA, AUC_pca = NA, TP_pca = NA, TN_pca = NA, FP_pca = NA, FN_pca = NA))
  }
  cm <- caret::confusionMatrix(factor(y_pred, levels = levels(factor(y_true))), factor(y_true))
  auc_value <- tryCatch({
    pROC::auc(pROC::roc(y_true, y_scores))
  }, error = function(e) {
    warning("AUC calculation failed.")
    return(NA)
  })
  return(data.frame(
    Accuracy_pca = cm$overall["Accuracy"],
    Precision_pca = cm$byClass["Precision"],
    Recall_pca = cm$byClass["Recall"],
    F1_Score_pca = cm$byClass["F1"],
    AUC_pca = auc_value,
    TP_pca = cm$table[2, 2],
    TN_pca = cm$table[1, 1],
    FP_pca = cm$table[1, 2],
    FN_pca = cm$table[2, 1]
  ))
}

incremental_learning_pipeline_pca_eval_batch1 <- function(X_train, y_train, batch_size, gamma_manual, cost_manual) {
  num_samples <- nrow(X_train)
  num_batches <- ceiling(num_samples / batch_size)
  stats <- list(mean = NULL, var = NULL, n = NULL)
  results <- data.frame()
  all_probabilities_eval_pca <- list()
  all_true_labels_eval_pca <- list()
  all_predictions_eval_pca <- list()
  
  # Batch 1 — initialize but do NOT evaluate
  start_idx <- 1
  end_idx <- min(batch_size, num_samples)
  X_batch1 <- X_train[start_idx:end_idx, ]
  y_batch1 <- y_train[start_idx:end_idx]
  
  stats$n <- nrow(X_batch1)
  stats$mean <- colMeans(X_batch1)
  stats$var <- cov(X_batch1)
  
  eig_decomp <- eigen(stats$var, symmetric = TRUE)
  projection_matrix <- eig_decomp$vectors[, 1:min(ncol(X_batch1), stats$n - 1), drop = FALSE]
  X_transformed_batch1 <- X_batch1 %*% projection_matrix
  
  svm_model <- e1071::svm(X_transformed_batch1, as.factor(y_batch1),
                          kernel = "radial", gamma = gamma_manual, cost = cost_manual,
                          probability = TRUE, class.weights = c('0' = 1, '1' = 1))
  
  # From batch 2 onward
  # Modify the loop inside the pipeline
  for (batch_idx in 2:num_batches) {
    start_idx <- (batch_idx - 1) * batch_size + 1
    end_idx <- min(batch_idx * batch_size, num_samples)
    X_batch <- X_train[start_idx:end_idx, ]
    y_batch <- y_train[start_idx:end_idx]
    
    # Time tracking (wall-clock + CPU)
    cpu_start <- proc.time()
    start_time <- Sys.time()
    
    update_result <- incremental_pca_lda(stats$mean, stats$var, stats$n, X_batch, n_components_pca = min(ncol(X_batch), stats$n - 1))
    
    end_time <- Sys.time()
    cpu_end <- proc.time()
    
    training_time_w <- as.numeric(difftime(end_time, start_time, units = "secs"))
    cpu_time <- (cpu_end - cpu_start)[["user.self"]] + (cpu_end - cpu_start)[["sys.self"]]
    
    projection_matrix <- update_result$projection_pca
    stats$n <- update_result$updated_n_pca
    stats$mean <- update_result$updated_mean_pca
    stats$var <- update_result$updated_var_pca
    
    X_transformed <- X_batch %*% projection_matrix
    X_train_update <- rbind(X_transformed_batch1, X_transformed)
    y_train_update <- c(y_batch1, y_batch)
    
    svm_model <- e1071::svm(X_train_update, as.factor(y_train_update),
                            kernel = "radial", gamma = gamma_manual, cost = cost_manual,
                            probability = TRUE, class.weights = c('0' = 1, '1' = 1))
    
    pred_eval <- predict(svm_model, X_train_update)
    prob_scores_eval <- attr(predict(svm_model, X_train_update, probability = TRUE), "probabilities")[, 2]
    metrics_row <- compute_metrics_pca(y_train_update, pred_eval, prob_scores_eval)
    
    metrics_row$Training_Time_pca <- training_time_w
    metrics_row$CPU_Time_pca <- cpu_time
    metrics_row$Batch <- batch_idx
    
    results <- rbind(results, metrics_row)
    
    all_probabilities_eval_pca[[batch_idx]] <- prob_scores_eval
    all_true_labels_eval_pca[[batch_idx]] <- as.numeric(as.character(y_train_update))
    all_predictions_eval_pca[[batch_idx]] <- factor(as.character(pred_eval), levels = levels(as.factor(y_train_update)))
  }
  
  return(list(
    results = results,
    final_model = svm_model,
    final_projection = projection_matrix,
    final_stats = stats,
    probabilities_eval = all_probabilities_eval_pca,
    true_labels_eval = all_true_labels_eval_pca,
    predictions_eval = all_predictions_eval_pca
  ))
}

# ========================
# Run the OB-INC pipeline
# ========================
set.seed(42)

X_train_weighted_pca <- X_train_weighted_top21
y_train_final_pca <- y_train_final

num_batches_pca <- 10
batch_size_pca <- ceiling(nrow(X_train_weighted_pca) / num_batches_pca)

batch_results_pca <- incremental_learning_pipeline_pca_eval_batch1(
  as.matrix(X_train_weighted_pca),
  as.factor(y_train_final_pca),
  batch_size = batch_size_pca,
  gamma_manual = best_gamma,
  cost_manual = best_cost
)

print(batch_results_pca$results)
write.csv(batch_results_pca$results, file = "results_pca_may_8.csv", row.names = FALSE)

# ----- Plotting -----
if (require(ggplot2) && require(caret) && require(pROC) && require(PRROC)) {
  # ----- Confusion Matrix Visualization -----
  final_predictions_eval_pca <- batch_results_pca$predictions_eval[[length(batch_results_pca$predictions_eval)]]
  final_true_labels_eval_pca <- factor(batch_results_pca$true_labels_eval[[length(batch_results_pca$true_labels_eval)]], levels = levels(final_predictions_eval_pca))
  cm_final_eval_pca <- caret::confusionMatrix(final_predictions_eval_pca, final_true_labels_eval_pca)
  cm_df_eval_pca <- as.data.frame(cm_final_eval_pca$table)
  plt_cm_eval_pca <- ggplot(data = cm_df_eval_pca, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), vjust = 0.5) +
    labs(title = "Confusion Matrix (PCA Incremental)", fill = "Frequency") +
    theme_minimal()
  print(plt_cm_eval_pca)
  
  # ----- ROC Curve -----
  all_probs_eval_pca <- unlist(batch_results_pca$probabilities_eval)
  all_labels_eval_pca <- unlist(batch_results_pca$true_labels_eval)
  
  if (length(unique(all_labels_eval_pca)) > 1) {
    roc_obj_eval_pca <- roc(all_labels_eval_pca, all_probs_eval_pca)
    plt_roc_eval_pca <- ggplot(data.frame(FPR = 1 - roc_obj_eval_pca$specificities, TPR = roc_obj_eval_pca$sensitivities),
                              aes(x = FPR, y = TPR)) +
      geom_line(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      labs(title = "ROC Curve (PCA Incremental)",
           x = "False Positive Rate", y = "True Positive Rate",
           caption = paste("AUC =", round(auc(roc_obj_eval_pca), 3))) +
      theme_minimal()
    print(plt_roc_eval_pca)
  } else {
    cat("Warning: Cannot plot ROC curve as all true labels in the evaluation set are the same.\n")
  }
  
  # ----- Precision-Recall Curve -----
  if (require(PRROC) && length(unique(all_labels_eval_pca)) > 1) {
    pr_data_eval_pca <- pr.curve(all_labels_eval_pca, all_probs_eval_pca, curve = TRUE)
    plt_pr_eval_pca <- ggplot(data.frame(Recall = pr_data_eval_pca$curve[, 1], Precision = pr_data_eval_pca$curve[, 2]),
                             aes(x = Recall, y = Precision)) +
      geom_line(color = "blue") +
      geom_hline(yintercept = mean(all_labels_eval_pca), linetype = "dashed", color = "red") + # Baseline
      labs(title = "Precision-Recall Curve (PCA Incremental)",
           x = "Recall", y = "Precision",
           caption = paste("AUC-PR =", round(pr_data_eval_pca$auc.integral, 3))) +
      theme_minimal()
    print(plt_pr_eval_pca)
  } else if (!require(PRROC)) {
    cat("Please install the 'PRROC' package to plot Precision-Recall curves.\n")
  } else {
    cat("Warning: Cannot plot Precision-Recall curve as all true labels in the evaluation set are the same.\n")
  }
  
} else {
  cat("Please install 'ggplot2', 'caret', 'pROC', and 'PRROC' packages to plot.\n")
}

######################## QR INC ######################################

incremental_qr_lda <- function(prev_cov_qr, prev_mean_qr, prev_n_qr, X_new_qr, n_components_qr) {
  new_n_qr <- prev_n_qr + nrow(X_new_qr)
  new_mean_qr <- (prev_mean_qr * prev_n_qr + colSums(X_new_qr)) / new_n_qr
  X_centered_qr <- scale(X_new_qr, center = new_mean_qr, scale = FALSE)
  cov_update_qr <- (prev_cov_qr * (prev_n_qr - 1) + crossprod(X_centered_qr)) / (new_n_qr - 1)
  
  qr_result_qr <- qr(cov_update_qr)
  projection_matrix_qr <- qr.Q(qr_result_qr)[, 1:n_components_qr, drop = FALSE]
  
  return(list(
    projection_qr = projection_matrix_qr,
    updated_cov_qr = cov_update_qr,
    updated_mean_qr = new_mean_qr,
    updated_n_qr = new_n_qr
  ))
}

compute_metrics_qr <- function(y_true, y_pred, y_scores) {
  if (length(unique(y_pred)) < 2) {
    warning("y_pred contains only one class, skipping batch metrics calculation.")
    return(data.frame(Accuracy_qr = NA, Precision_qr = NA, Recall_qr = NA,
                      F1_Score_qr = NA, AUC_qr = NA, TP_qr = NA, TN_qr = NA,
                      FP_qr = NA, FN_qr = NA, Training_Time_qr = NA))
  }
  
  cm <- caret::confusionMatrix(factor(y_pred, levels = levels(factor(y_true))), factor(y_true))
  auc_value <- tryCatch({
    pROC::auc(roc(y_true, y_scores))
  }, error = function(e) {
    warning("AUC calculation failed.")
    return(NA)
  })
  
  return(data.frame(
    Accuracy_qr = cm$overall["Accuracy"],
    Precision_qr = cm$byClass["Precision"],
    Recall_qr = cm$byClass["Recall"],
    F1_Score_qr = cm$byClass["F1"],
    AUC_qr = auc_value,
    TP_qr = cm$table[2, 2],
    TN_qr = cm$table[1, 1],
    FP_qr = cm$table[1, 2],
    FN_qr = cm$table[2, 1]
  ))
}

update_svm_qr <- function(model_qr, X_transformed_qr, y_batch_qr, X_all_qr = NULL, y_all_qr = NULL, gamma_manual_qr, cost_manual_qr) {
  if (is.null(model_qr)) {
    return(e1071::svm(X_transformed_qr, as.factor(y_batch_qr),
                      kernel = "radial", gamma = gamma_manual_qr, cost = cost_manual_qr,
                      probability = TRUE, class.weights = c('0' = 1, '1' = 1)))
  } else {
    return(e1071::svm(X_all_qr, as.factor(y_all_qr),
                      kernel = "radial", gamma = gamma_manual_qr, cost = cost_manual_qr,
                      probability = TRUE, class.weights = c('0' = 1, '1' = 1)))
  }
}

incremental_learning_pipeline_qr_eval_batch1 <- function(X_train, y_train, batch_size, gamma_manual, cost_manual) {
  num_samples <- nrow(X_train)
  num_batches <- ceiling(num_samples / batch_size)
  
  stats <- list(cov = NULL, mean = NULL, n = NULL)
  svm_model <- NULL
  results <- data.frame(
    Accuracy_qr = numeric(0),
    Precision_qr = numeric(0),
    Recall_qr = numeric(0),
    F1_Score_qr = numeric(0),
    AUC_qr = numeric(0),
    TP_qr = numeric(0),
    TN_qr = numeric(0),
    FP_qr = numeric(0),
    FN_qr = numeric(0),
    QR_Update_Time = numeric(0),
    CPU_Time_QR_Update = numeric(0),
    SVM_Training_Time = numeric(0),
    CPU_Time_SVM_Training = numeric(0),
    Total_Training_Time_qr = numeric(0),
    Batch = integer(0)
  )
  
  all_probabilities_eval_batch1 <- list()
  all_true_labels_eval_batch1 <- list()
  all_predictions_eval_batch1 <- list()
  
  # === Batch 1 Initialization ONLY ===
  start_idx_batch1 <- 1
  end_idx_batch1 <- min(batch_size, num_samples)
  X_batch1 <- X_train[start_idx_batch1:end_idx_batch1, ]
  y_batch1 <- y_train[start_idx_batch1:end_idx_batch1]
  
  stats$n <- nrow(X_batch1)
  stats$mean <- colMeans(X_batch1)
  stats$cov <- cov(X_batch1)
  
  qr_result <- qr(stats$cov)
  projection_matrix <- qr.Q(qr_result)[, 1:min(ncol(X_batch1), stats$n - 1), drop = FALSE]
  X_transformed_batch1 <- X_batch1 %*% projection_matrix
  
  svm_model <- e1071::svm(X_transformed_batch1, as.factor(y_batch1), kernel = "radial",
                          gamma = gamma_manual, cost = cost_manual,
                          probability = TRUE, class.weights = c('0' = 1, '1' = 1))
  
  # === Loop Through Remaining Batches for Incremental Updates ===
  for (batch_idx in 2:num_batches) {
    start_idx <- (batch_idx - 1) * batch_size + 1
    end_idx <- min(batch_idx * batch_size, num_samples)
    X_batch <- X_train[start_idx:end_idx, ]
    y_batch <- y_train[start_idx:end_idx]
    
    # === QR Update CPU and Wall-clock timing ===
    cpu_start_qr <- proc.time()
    start_qr_time <- Sys.time()
    
    update_result <- incremental_qr_lda(stats$cov, stats$mean, stats$n, X_batch,
                                        n_components_qr = min(ncol(X_batch), stats$n - 1))
    
    end_qr_time <- Sys.time()
    cpu_end_qr <- proc.time()
    
    qr_update_time <- as.numeric(difftime(end_qr_time, start_qr_time, units = "secs"))
    cpu_time_qr <- (cpu_end_qr - cpu_start_qr)[["user.self"]] + (cpu_end_qr - cpu_start_qr)[["sys.self"]]
    
    projection_matrix <- update_result$projection_qr
    stats$n <- update_result$updated_n_qr
    stats$mean <- update_result$updated_mean_qr
    stats$cov <- update_result$updated_cov_qr
    
    # === SVM Training CPU and Wall-clock timing ===
    cpu_start_svm <- proc.time()
    start_svm_time <- Sys.time()
    
    X_transformed <- X_batch %*% projection_matrix
    X_train_update <- rbind(X_transformed_batch1, X_transformed)
    y_train_update <- c(y_batch1, y_batch)
    
    svm_model <- e1071::svm(X_train_update, as.factor(y_train_update), kernel = "radial",
                            gamma = gamma_manual, cost = cost_manual,
                            probability = TRUE, class.weights = c('0' = 1, '1' = 1))
    
    end_svm_time <- Sys.time()
    cpu_end_svm <- proc.time()
    
    svm_train_time <- as.numeric(difftime(end_svm_time, start_svm_time, units = "secs"))
    cpu_time_svm <- (cpu_end_svm - cpu_start_svm)[["user.self"]] + (cpu_end_svm - cpu_start_svm)[["sys.self"]]
    
    # === Evaluation ===
    pred_eval <- predict(svm_model, X_transformed)
    prob_scores_eval <- attr(predict(svm_model, X_transformed, probability = TRUE), "probabilities")[, 2]
    metrics_row <- compute_metrics_qr(y_batch, pred_eval, prob_scores_eval)
    
    metrics_row$QR_Update_Time <- qr_update_time
    metrics_row$CPU_Time_QR_Update <- cpu_time_qr
    metrics_row$SVM_Training_Time <- svm_train_time
    metrics_row$CPU_Time_SVM_Training <- cpu_time_svm
    metrics_row$Total_Training_Time_qr <- qr_update_time + svm_train_time
    metrics_row$Batch <- batch_idx
    
    metrics_row <- metrics_row[, colnames(results)]
    results <- rbind(results, metrics_row)
    
    all_probabilities_eval_batch1[[batch_idx - 1]] <- prob_scores_eval
    all_true_labels_eval_batch1[[batch_idx - 1]] <- as.numeric(as.character(y_batch))
    all_predictions_eval_batch1[[batch_idx - 1]] <- factor(as.character(pred_eval), levels = levels(as.factor(y_batch)))
  }
  
  return(list(
    results = results,
    final_model = svm_model,
    final_projection = projection_matrix,
    final_stats = stats,
    probabilities_eval = all_probabilities_eval_batch1,
    true_labels_eval = all_true_labels_eval_batch1,
    predictions_eval = all_predictions_eval_batch1
  ))
}

# ==== RUNNING QR INC with best parameters ====

set.seed(42)
X_train_weighted_qr <- X_train_weighted_top21
y_train_final_qr <- y_train_final

num_batches_qr <- 10
batch_size_qr <- ceiling(nrow(X_train_weighted_qr) / num_batches_qr)

batch_results_qr <- incremental_learning_pipeline_qr_eval_batch1(
  as.matrix(X_train_weighted_qr),
  as.factor(y_train_final_qr),
  batch_size = batch_size_qr,
  gamma_manual = best_gamma,
  cost_manual = best_cost
)

print(batch_results_qr$results)
write.csv(batch_results_qr$results, file = "results_qr_may_8.csv", row.names = FALSE)

# ----- Plotting -----
if (require(ggplot2) && require(caret) && require(pROC) && require(PRROC)) {
  # ----- Confusion Matrix Visualization -----
  final_predictions_eval <- batch_results_qr$predictions_eval[[length(batch_results_qr$predictions_eval)]]
  final_true_labels_eval <- factor(batch_results_qr$true_labels_eval[[length(batch_results_qr$true_labels_eval)]], levels = levels(final_predictions_eval))
  cm_final_eval <- caret::confusionMatrix(final_predictions_eval, final_true_labels_eval)
  cm_df_eval <- as.data.frame(cm_final_eval$table)
  plt_cm_eval <- ggplot(data = cm_df_eval, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), vjust = 0.5) +
    labs(title = "Confusion Matrix (QR Incremental)", fill = "Frequency") +
    theme_minimal()
  print(plt_cm_eval)
  
  # ----- ROC Curve -----
  all_probs_eval <- unlist(batch_results_qr$probabilities_eval)
  all_labels_eval <- unlist(batch_results_qr$true_labels_eval)
  
  if (length(unique(all_labels_eval)) > 1) {
    roc_obj_eval <- roc(all_labels_eval, all_probs_eval)
    plt_roc_eval <- ggplot(data.frame(FPR = 1 - roc_obj_eval$specificities, TPR = roc_obj_eval$sensitivities),
                           aes(x = FPR, y = TPR)) +
      geom_line(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      labs(title = "ROC Curve (QR Incremental)",
           x = "False Positive Rate", y = "True Positive Rate",
           caption = paste("AUC =", round(auc(roc_obj_eval), 3))) +
      theme_minimal()
    print(plt_roc_eval)
  } else {
    cat("Warning: Cannot plot ROC curve as all true labels in the evaluation set are the same.\n")
  }
  
  # ----- Precision-Recall Curve -----
  if (require(PRROC) && length(unique(all_labels_eval)) > 1) {
    pr_data_eval <- pr.curve(all_labels_eval, all_probs_eval, curve = TRUE)
    plt_pr_eval <- ggplot(data.frame(Recall = pr_data_eval$curve[, 1], Precision = pr_data_eval$curve[, 2]),
                          aes(x = Recall, y = Precision)) +
      geom_line(color = "blue") +
      geom_hline(yintercept = mean(all_labels_eval), linetype = "dashed", color = "red") + # Baseline
      labs(title = "Precision-Recall Curve (QR Incremental)",
           x = "Recall", y = "Precision",
           caption = paste("AUC-PR =", round(pr_data_eval$auc.integral, 3))) +
      theme_minimal()
    print(plt_pr_eval)
  } else if (!require(PRROC)) {
    cat("Please install the 'PRROC' package to plot Precision-Recall curves.\n")
  } else {
    cat("Warning: Cannot plot Precision-Recall curve as all true labels in the evaluation set are the same.\n")
  }
  
} else {
  cat("Please install 'ggplot2', 'caret', 'pROC', and 'PRROC' packages to plot.\n")
}



final_model_qr <- batch_results_qr$final_model
final_projection_qr <- batch_results_qr$final_projection

### ---------------- TEST DATA ---------------------------- ###

# ========== Helper: Update QR Stats ==========
update_qr_stats <- function(stats, X_batch, n_components_qr) {
  update_result <- incremental_qr_lda(
    stats$cov, stats$mean, stats$n,
    X_batch,
    n_components_qr = min(ncol(X_batch), stats$n - 1)
  )
  
  list(
    projection = update_result$projection_qr,
    stats = list(
      cov = update_result$updated_cov_qr,
      mean = update_result$updated_mean_qr,
      n = update_result$updated_n_qr
    )
  )
}

# ========== Helper: Train SVM ==========
train_svm_model <- function(X, y, gamma, cost) {
  e1071::svm(
    X, as.factor(y),
    kernel = "radial",
    gamma = gamma,
    cost = cost,
    probability = TRUE,
    class.weights = c('0' = 1, '1' = 1)
  )
}

# ========== Helper: Evaluate Model ==========
evaluate_model_qr <- function(model, X_eval, y_eval, projection_matrix) {
  X_proj <- X_eval %*% projection_matrix
  pred <- predict(model, X_proj)
  prob <- attr(predict(model, X_proj, probability = TRUE), "probabilities")[, 2]
  list(metrics = compute_metrics_qr(y_eval, pred, prob), predictions = pred, probabilities = prob)
}

# ========== Main Function: Incremental QR on Test ==========
incremental_qr_on_test <- function(X_test_full, y_test_full,
                                   initial_model, initial_proj, initial_stats,
                                   gamma_manual, cost_manual,
                                   verbose = TRUE) {
  
  # Define batches
  total_samples <- nrow(X_test_full)
  batch_size <- floor(0.03 * total_samples)
  num_batches <- 10
  
  if (verbose) {
    cat("Total test samples:", total_samples, "\n")
    cat("Batch size (3%):", batch_size, "\n")
    cat("Number of batches:", num_batches, "\n")
  }
  
  # Fixed evaluation set (first 3%)
  X_eval <- X_test_full[1:batch_size, ]
  y_eval <- y_test_full[1:batch_size]
  
  # Initialize state
  results <- data.frame()
  svm_model <- initial_model
  projection_matrix <- initial_proj
  stats <- initial_stats
  
  # Incremental loop
  for (batch_idx in 1:num_batches) {
    if (verbose) cat("Batch", batch_idx, "...\n")
    
    # Determine batch indices
    start_idx <- (batch_idx - 1) * batch_size + 1
    end_idx <- min(batch_idx * batch_size, total_samples)
    
    if (start_idx > total_samples) break
    
    X_batch <- X_test_full[start_idx:end_idx, ]
    y_batch <- y_test_full[start_idx:end_idx]
    
    # === Start timing ===
    cpu_start <- proc.time()
    start_time <- Sys.time()
    
    # Update stats & projection
    qr_update <- update_qr_stats(stats, X_batch, n_components_qr = ncol(X_batch))
    projection_matrix <- qr_update$projection
    stats <- qr_update$stats
    
    # Combine eval + current batch, re-project
    X_train_new <- rbind(X_eval, X_batch)
    y_train_new <- c(y_eval, y_batch)
    X_train_transformed <- X_train_new %*% projection_matrix
    
    # Retrain SVM
    svm_model <- train_svm_model(X_train_transformed, y_train_new, gamma_manual, cost_manual)
    
    # Evaluate on fixed eval set
    eval_results <- evaluate_model_qr(svm_model, X_eval, y_eval, projection_matrix)
    metrics <- eval_results$metrics
    
    end_time <- Sys.time()
    cpu_end <- proc.time()
    
    cpu_time_qr <- (cpu_end - cpu_start)[["user.self"]] + (cpu_end - cpu_start)[["sys.self"]]
    
    # === Store results ===
    metrics_row <- data.frame(
      Batch = batch_idx,
      Training_Time_qr = as.numeric(difftime(end_time, start_time, units = "secs")),
      CPU_Time_qr = cpu_time_qr,
      Accuracy = metrics$Accuracy,
      Precision = metrics$Precision,
      Recall = metrics$Recall,
      F1_Score = metrics$F1_Score,
      AUC = metrics$AUC,
      TP = metrics$TP,
      TN = metrics$TN,
      FP = metrics$FP,
      FN = metrics$FN
    )
    
    results <- rbind(results, metrics_row)
  }
  
  # Final evaluation on the fixed evaluation set
  final_eval <- evaluate_model_qr(svm_model, X_eval, y_eval, projection_matrix)
  
  return(list(
    results = results,
    final_evaluation = final_eval,
    y_eval = y_eval
  ))
}


# ----- Apply feature weighting -----
X_test_weighted_qr <- sweep(X_test, 2, W_vec, `*`)

# ----- Select top features -----
X_test_selected_qr <- X_test_weighted_qr[, top_features]

# ----- Convert to matrix -----
X_test_matrix <- as.matrix(X_test_selected_qr)

incremental_results_test_qr_output <- incremental_qr_on_test(
  X_test_matrix, as.factor(y_test),
  initial_model = final_model_qr,
  initial_proj = final_projection_qr,
  initial_stats = batch_results_qr$final_stats,
  gamma_manual = best_gamma,
  cost_manual = best_cost
)

print(incremental_results_test_qr_output$results)
final_eval_results <- incremental_results_test_qr_output$final_evaluation$metrics
print("Final Evaluation Metrics on Fixed Set:")
print(final_eval_results)
write.csv(incremental_results_test_qr_output$results, file = "results_test_qr_may_8.csv", row.names = FALSE)

# ----- Plotting -----
if (require(ggplot2) && require(caret) && require(pROC) && require(PRROC)) {
  # ----- Confusion Matrix Visualization -----
  cm_final_eval_table <- caret::confusionMatrix(incremental_results_test_qr_output$final_evaluation$predictions,
                                                incremental_results_test_qr_output$y_eval) # Use outputted y_eval
  cm_df_eval <- as.data.frame(cm_final_eval_table$table)
  plt_cm_eval <- ggplot(data = cm_df_eval, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), vjust = 0.5) +
    labs(title = "Confusion Matrix on Test Data (Incremental QR)", fill = "Frequency") +
    theme_minimal()
  print(plt_cm_eval)
  
  # ----- ROC Curve -----
  roc_obj_eval <- roc(as.numeric(as.character(incremental_results_test_qr_output$y_eval)), # Use outputted y_eval
                      incremental_results_test_qr_output$final_evaluation$probabilities)
  plt_roc_eval <- ggplot(data.frame(FPR = 1 - roc_obj_eval$specificities, TPR = roc_obj_eval$sensitivities),
                         aes(x = FPR, y = TPR)) +
    geom_line(color = "blue") +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
    labs(title = "ROC Curve onTest Data (Incremental QR)",
         x = "False Positive Rate", y = "True Positive Rate",
         caption = paste("AUC =", round(auc(roc_obj_eval), 3))) +
    theme_minimal()
  print(plt_roc_eval)
  
  # ----- Precision-Recall Curve -----
  pr_data_eval <- pr.curve(as.numeric(as.character(incremental_results_test_qr_output$y_eval)), # Use outputted y_eval
                           incremental_results_test_qr_output$final_evaluation$probabilities,
                           curve = TRUE)
  plt_pr_eval <- ggplot(data.frame(Recall = pr_data_eval$curve[, 1], Precision = pr_data_eval$curve[, 2]),
                        aes(x = Recall, y = Precision)) +
    geom_line(color = "blue") +
    geom_hline(yintercept = mean(as.numeric(as.character(incremental_results_test_qr_output$y_eval))),
               linetype = "dashed", color = "red") + # Baseline
    labs(title = "Precision-Recall Curve on Test Data (Incremental QR)",
         x = "Recall", y = "Precision",
         caption = paste("AUC-PR =", round(pr_data_eval$auc.integral, 3))) +
    theme_minimal()
  print(plt_pr_eval)
  
} else {
  cat("Please install 'ggplot2', 'caret', 'pROC', and 'PRROC' packages to plot.\n")
}



# Ensure test data is selected with top features
X_test_selected <- X_test[, top_features]
y_test <- as.factor(y_test)

# Split test set into 10 batches
batches <- split(1:nrow(X_test_selected), cut(1:nrow(X_test_selected), breaks = 10, labels = FALSE))

# Initialize result storage with 13 columns
test_results_top21 <- data.frame(
  Batch = integer(),
  TP = integer(),
  TN = integer(),
  FP = integer(),
  FN = integer(),
  Accuracy = numeric(),
  AUC = numeric(),
  Precision = numeric(),
  Recall = numeric(),
  F1_Score = numeric(),
  Training_CPU_Time = numeric(),
  Inference_Time = numeric(),
  Inference_CPU_Time = numeric()
)

all_probabilities_test <- numeric()
all_true_labels_test <- numeric()
all_predictions_test <- factor(levels = levels(y_test))

# Start with initial train set
X_current_train <- X_train_final_selected
y_current_train <- as.factor(y_train_final)  # Ensure factor with consistent levels

# Loop through each batch
for (i in seq_along(batches)) {
  cat("Processing Test Batch", i, "\n")
  
  idx <- batches[[i]]
  X_batch <- X_test_selected[idx, ]
  y_batch <- y_test[idx]
  
  # Retrain SVM with updated training set
  cpu_start_train <- proc.time()
  svm_model_top21_final <- svm(X_current_train, y_current_train,
                               kernel = "radial",
                               gamma = best_gamma,
                               cost = best_cost,
                               probability = TRUE)
  cpu_end_train <- proc.time()
  training_cpu_time <- (cpu_end_train - cpu_start_train)[["user.self"]] + 
    (cpu_end_train - cpu_start_train)[["sys.self"]]
  
  # Inference
  start_time <- Sys.time()
  cpu_start_infer <- proc.time()
  preds <- predict(svm_model_top21_final, X_batch)
  preds_proba <- attr(predict(svm_model_top21_final, X_batch, probability = TRUE), "probabilities")[, 2]
  cpu_end_infer <- proc.time()
  end_time <- Sys.time()
  
  inference_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  inference_cpu_time <- (cpu_end_infer - cpu_start_infer)[["user.self"]] + 
    (cpu_end_infer - cpu_start_infer)[["sys.self"]]
  
  # Ensure consistent factor levels for predictions
  preds <- factor(preds, levels = levels(y_test))
  
  # Confusion Matrix
  cm <- table(Predicted = preds, Actual = y_batch)
  TP <- ifelse("1" %in% rownames(cm) & "1" %in% colnames(cm), cm["1", "1"], 0)
  TN <- ifelse("0" %in% rownames(cm) & "0" %in% colnames(cm), cm["0", "0"], 0)
  FP <- ifelse("1" %in% rownames(cm) & "0" %in% colnames(cm), cm["1", "0"], 0)
  FN <- ifelse("0" %in% rownames(cm) & "1" %in% colnames(cm), cm["0", "1"], 0)
  
  # Calculate accuracy with error handling
  accuracy <- tryCatch({
    mean(preds == y_batch)
  }, error = function(e) {
    warning(paste("Accuracy calculation failed for batch", i, ":", e$message))
    NA
  })
  
  auc_val <- tryCatch({
    auc(as.numeric(as.character(y_batch)), preds_proba)
  }, error = function(e) {
    warning(paste("AUC calculation failed for batch", i, ":", e$message))
    NA
  })
  
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  recall <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  # Create new row for results
  new_row <- data.frame(
    Batch = i,
    TP = TP,
    TN = TN,
    FP = FP,
    FN = FN,
    Accuracy = accuracy,
    AUC = auc_val,
    Precision = precision,
    Recall = recall,
    F1_Score = f1,
    Training_CPU_Time = training_cpu_time,
    Inference_Time = inference_time,
    Inference_CPU_Time = inference_cpu_time
  )
  
  # Debug: Print column names to check for mismatch
  cat("Columns in new_row:", paste(colnames(new_row), collapse = ", "), "\n")
  cat("Columns in test_results_top21:", paste(colnames(test_results_top21), collapse = ", "), "\n")
  
  # Store results
  test_results_top21 <- rbind(test_results_top21, new_row)
  
  # Collect prediction details
  all_probabilities_test <- c(all_probabilities_test, preds_proba)
  all_true_labels_test <- c(all_true_labels_test, as.numeric(as.character(y_batch)))
  all_predictions_test <- factor(c(as.character(all_predictions_test), as.character(preds)), 
                                 levels = levels(y_test))
  
  # Add batch to training set for next loop
  X_current_train <- rbind(X_current_train, X_batch)
  y_current_train <- factor(c(as.character(y_current_train), as.character(y_batch)), 
                            levels = levels(y_test))  # Ensure factor levels
}

# Review results
print(test_results_top21)
summary(test_results_top21)

write.csv(test_results_top21, file = "results_test_top_21_may_8.csv", row.names = FALSE)

# === Plotting ===
if (require(ggplot2) && require(caret) && require(pROC)) {
  # ----- Confusion Matrix Visualization -----
  cm_final <- caret::confusionMatrix(all_predictions_test, y_test)
  cm_df <- as.data.frame(cm_final$table)
  plt_cm <- ggplot(data = cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), vjust = 0.5) +
    labs(title = "Confusion Matrix on Test Data (Top 21 Features)", fill = "Frequency") +
    theme_minimal()
  print(plt_cm)
  
  # ----- Precision-Recall Curve -----
  pr_data <- pr.curve(all_true_labels_test, all_probabilities_test, curve = TRUE)
  plt_pr <- ggplot(data.frame(Recall = pr_data$curve[, 1], Precision = pr_data$curve[, 2]),
                   aes(x = Recall, y = Precision)) +
    geom_line() +
    geom_hline(yintercept = mean(all_true_labels_test), linetype = "dashed", color = "red") + # Baseline
    labs(title = "Precision-Recall Curve on Test Data (Top 21 Features)",
         x = "Recall", y = "Precision",
         caption = paste("AUC-PR =", round(pr_data$auc.integral, 3))) +
    theme_minimal()
  print(plt_pr)
  
  # ----- ROC Curve (as before) -----
  if (length(unique(all_true_labels_test)) > 1) {
    roc_obj_test <- roc(all_true_labels_test, all_probabilities_test)
    plt_roc <- ggplot(data.frame(FPR = 1 - roc_obj_test$specificities, TPR = roc_obj_test$sensitivities),
                      aes(x = FPR, y = TPR)) +
      geom_line(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "blue") +
      labs(title = "ROC Curve on Test Data (Top 21 Features)",
           x = "False Positive Rate", y = "True Positive Rate",
           caption = paste("AUC =", round(auc(roc_obj_test), 3))) +
      theme_minimal()
    print(plt_roc)
  } else {
    cat("Warning: Cannot plot ROC curve as all true labels in the test set are the same.\n")
  }
  
} else {
  cat("Please install 'ggplot2', 'caret', and 'pROC' packages to plot.\n")
}
