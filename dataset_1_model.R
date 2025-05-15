# Install necessary packages (if not already installed)
packages <- c("caret", "ggplot2", "e1071", "pROC", "MASS", "bigstatsr", 
              "irlba", "MLmetrics", "dplyr", "arm", "RSpectra", "PRROC")

installed <- rownames(installed.packages())
for (pkg in packages) {
  if (!(pkg %in% installed)) install.packages(pkg)
}
install.packages("ps")        # For accurate CPU tracking
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
library(doParallel)
library(foreach)
library(PRROC)

# ===================================================
# ----- LOAD DATASET -----
# ===================================================
df <- read.csv("dataset_full.csv")

# ===================================================
# ----- Separate Features and Target Variable -----
# ===================================================
X <- df[, !colnames(df) %in% c("phishing")]
y <- df$phishing

ggplot(df, aes(x = factor(phishing))) +
  geom_bar(fill = c("lightblue", "salmon")) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  labs(title = "Distribution of Phishing vs. Non-Phishing URLs",
       x = "Label (1: Phishing, 0: Non-Phishing)",
       y = "Number of URLs") +
  theme_minimal()

# ===================================================
# ----- Feature Scaling -----
# ===================================================

# Check for numeric columns
numeric_cols <- sapply(X, is.numeric)
X_numeric <- X[, numeric_cols, drop = FALSE]

# Identify and remove constant columns
non_constant_cols <- sapply(X_numeric, function(col) var(col, na.rm = TRUE) != 0)
X_numeric_non_constant <- X_numeric[, non_constant_cols, drop = FALSE]

# Scale the non-constant numeric columns
X_scaled <- as.data.frame(scale(X_numeric_non_constant))

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

# ===================================================
# ----- SVM WITHOUT LDA - WHOLE FEATURE -----
# ===================================================
set.seed(123)
folds <- createFolds(y_train_final, k = 10)

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
  
  # Remove constant features from this fold
  non_constant_cols <- sapply(X_train_fold, function(col) var(col, na.rm = TRUE) != 0)
  X_train_fold <- X_train_fold[, non_constant_cols, drop = FALSE]
  X_test_fold <- X_test_fold[, non_constant_cols, drop = FALSE]  # Match columns
  
  start_time <- Sys.time()
  svm_model_no_lda <- svm(X_train_fold, as.factor(y_train_fold), kernel = "radial", probability = TRUE)
  end_time <- Sys.time()
  training_time_no_lda <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  pred_no_lda <- predict(svm_model_no_lda, X_test_fold)
  pred_proba_no_lda <- attr(predict(svm_model_no_lda, X_test_fold, probability = TRUE), "probabilities")[, 2]
  
  conf_matrix <- confusionMatrix(pred_no_lda, as.factor(y_test_fold))
  
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
set.seed(123)
folds <- createFolds(y_train_final, k = 10, returnTrain = FALSE)

results_lda <- data.frame(Fold = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(), 
                          Accuracy = numeric(), AUC = numeric(), Precision = numeric(), 
                          Recall = numeric(), F1_Score = numeric(), Training_Time = numeric())

for (i in seq_along(folds)) {
  cat("Processing Fold", i, "\n")
  
  train_idx <- unlist(folds[-i])
  test_idx <- unlist(folds[i])
  
  X_train_fold <- X_train_final[train_idx, ]
  y_train_fold <- y_train_final[train_idx]
  X_test_fold <- X_train_final[test_idx, ]
  y_test_fold <- y_train_final[test_idx]
  
  # Remove constant features in this fold
  non_constant_cols <- sapply(X_train_fold, function(col) var(col, na.rm = TRUE) != 0)
  X_train_fold <- X_train_fold[, non_constant_cols, drop = FALSE]
  X_test_fold <- X_test_fold[, non_constant_cols, drop = FALSE]
  
  # Remove highly correlated features to avoid multicollinearity
  cor_matrix <- cor(X_train_fold)
  high_corr <- findCorrelation(cor_matrix, cutoff = 0.99)  # You can lower this threshold if needed
  if (length(high_corr) > 0) {
    X_train_fold <- X_train_fold[, -high_corr, drop = FALSE]
    X_test_fold <- X_test_fold[, -high_corr, drop = FALSE]
  }
  
  # Fit LDA
  lda_model <- lda(y_train_fold ~ ., data = data.frame(y_train_fold, X_train_fold))
  X_train_lda <- as.matrix(X_train_fold) %*% lda_model$scaling
  X_test_lda <- as.matrix(X_test_fold) %*% lda_model$scaling
  
  # Train SVM on LDA-transformed data
  start_time <- Sys.time()
  svm_model_lda <- svm(X_train_lda, as.factor(y_train_fold), kernel = "radial", probability = TRUE)
  end_time <- Sys.time()
  training_time_lda <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  pred_lda <- predict(svm_model_lda, X_test_lda)
  pred_proba_lda <- attr(predict(svm_model_lda, X_test_lda, probability = TRUE), "probabilities")[, 2]
  
  conf_matrix_lda <- confusionMatrix(pred_lda, as.factor(y_test_fold))
  
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

# Save to CSV
write.csv(results_lda, "results_lda.csv", row.names = FALSE)

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

cat("=== Top 20 Important Features from LDA ===\n")
print(head(lda_importance, 20))

ggplot(head(lda_importance, 20), aes(x = reorder(Feature, Abs_Coefficient), y = Abs_Coefficient)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 Important Features (LDA)", x = "Feature", y = "Absolute Coefficient")

# Fit LDA on full training set to get feature importances
lda_model <- lda(y_train_final ~ ., data = data.frame(y_train_final, X_train_final))

# Get LDA coefficients (feature weights)
W <- lda_model$scaling
W_vec <- as.vector(W)

# Weight original features using LDA coefficients
X_train_weighted <- sweep(X_train_final, 2, W_vec, `*`)
X_valid_weighted <- sweep(X_valid, 2, W_vec, `*`)

###### FIND THE BEST FEATURE TO SELECT ################

# Setup parallel backend
cores <- parallel::detectCores() - 1
cl <- makeCluster(cores)
registerDoParallel(cl)

set.seed(123)
num_folds <- 10
folds <- createFolds(y_train_final, k = num_folds)

X_weighted_data <- as.matrix(X_train_weighted)
y_weighted_data <- y_train_final

feature_counts <- c(95, 98)
all_results <- list()

for (n_features in feature_counts) {
  
  max_features <- length(W_vec)
  actual_n_features <- min(n_features, max_features)
  top_n_features <- order(abs(W_vec), decreasing = TRUE)[1:actual_n_features]
  cat("\n===== Running CV for Top", actual_n_features, "Features =====\n")
  
  fold_results <- foreach(i = 1:num_folds, .combine = rbind, .packages = c("e1071", "pROC", "caret")) %dopar% {
    tryCatch({
      valid_idx <- folds[[i]]
      X_train_fold <- X_weighted_data[-valid_idx, top_n_features]
      y_train_fold <- y_weighted_data[-valid_idx]
      X_valid_fold <- X_weighted_data[valid_idx, top_n_features]
      y_valid_fold <- y_weighted_data[valid_idx]
      
      # Skip if missing values
      if (any(is.na(X_train_fold)) || any(is.na(X_valid_fold))) return(NULL)
      
      # Train SVM
      start_time <- Sys.time()
      model <- svm(X_train_fold, as.factor(y_train_fold), kernel = "radial", probability = TRUE)
      end_time <- Sys.time()
      training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      pred <- predict(model, X_valid_fold, probability = TRUE)
      pred_class <- factor(pred, levels = c("0", "1"))
      pred_prob <- attr(pred, "probabilities")[, 2]
      y_valid_fold <- factor(y_valid_fold, levels = c("0", "1"))
      
      conf <- confusionMatrix(pred_class, y_valid_fold)
      auc_val <- tryCatch(auc(as.numeric(as.character(y_valid_fold)), pred_prob), error = function(e) NA)
      
      TP <- conf$table["1", "1"]
      TN <- conf$table["0", "0"]
      FP <- conf$table["1", "0"]
      FN <- conf$table["0", "1"]
      
      data.frame(
        Fold = i,
        TP = TP,
        TN = TN,
        FP = FP,
        FN = FN,
        Accuracy = conf$overall["Accuracy"],
        AUC = auc_val,
        Precision = conf$byClass["Precision"],
        Recall = conf$byClass["Recall"],
        F1_Score = conf$byClass["F1"],
        Training_Time = training_time
      )
    }, error = function(e) {
      return(NULL)  # avoid printing inside foreach
    })
  }
  
  # Skip if no valid folds
  if (is.null(fold_results) || nrow(fold_results) == 0) {
    cat(sprintf("No valid folds for Top %d features — skipping.\n", n_features))
    next
  }
  
  fold_results <- fold_results[complete.cases(fold_results), ]
  all_results[[paste0("Top_", n_features)]] <- fold_results
  
  # Summary
  cat(sprintf("Top %d Features: Mean Accuracy = %.4f ± %.4f | Mean AUC = %.4f ± %.4f\n",
              actual_n_features,
              mean(fold_results$Accuracy, na.rm = TRUE),
              sd(fold_results$Accuracy, na.rm = TRUE),
              mean(fold_results$AUC, na.rm = TRUE),
              sd(fold_results$AUC, na.rm = TRUE)))
}

stopCluster(cl)

###### END FIND THE BEST FEATURE TO SELECT ################

# ===================================================
# ----- SVM WITH LDA-WEIGHTED TOP 95 FEATURES SELECT -----
# ===================================================

# Get top 95 features based on absolute LDA coefficients
top_features <- order(abs(W_vec), decreasing = TRUE)[1:95]

# Convert to matrix for CV loop
X_weighted_data <- as.matrix(X_train_weighted)
y_weighted_data <- y_train_final

set.seed(123)

num_folds <- 10
folds <- createFolds(y_weighted_data, k = num_folds, list = TRUE, returnTrain = FALSE)

# Initialize results container for Top 95 Features
results_top95 <- data.frame(Fold = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(),
                            Accuracy = numeric(), AUC = numeric(), Precision = numeric(),
                            Recall = numeric(), F1_Score = numeric(), Training_Time = numeric())

# Lists to store confusion matrices and ROC/PR data for plotting
all_conf_matrices_top95 <- list()
all_preds_proba_top95 <- list()
all_true_labels_top95 <- list()

for (i in 1:num_folds) {
  
  cat("Processing Fold", i, "\n")
  
  valid_idx <- folds[[i]]
  
  X_train_fold <- X_weighted_data[-valid_idx, ]
  y_train_fold <- y_weighted_data[-valid_idx]
  
  X_valid_fold <- X_weighted_data[valid_idx, ]
  y_valid_fold <- y_weighted_data[valid_idx]
  
  # Select top 95 LDA-weighted features
  X_train_selected <- X_train_fold[, top_features]
  X_valid_selected <- X_valid_fold[, top_features]
  
  # Train SVM
  start_time <- Sys.time()
  svm_model_top95 <- svm(X_train_selected, as.factor(y_train_fold), kernel = "radial", probability = TRUE)
  end_time <- Sys.time()
  training_time_top95 <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  pred_top95 <- predict(svm_model_top95, X_valid_selected)
  pred_proba_top95 <- attr(predict(svm_model_top95, X_valid_selected, probability = TRUE), "probabilities")[, 2]
  
  # Confusion matrix
  conf_matrix_top95 <- confusionMatrix(pred_top95, as.factor(y_valid_fold))
  all_conf_matrices_top95[[i]] <- conf_matrix_top95$table
  
  # Collect metrics for each fold
  results_top95 <- rbind(results_top95, data.frame(
    Fold = i,
    TP = conf_matrix_top95$table["1", "1"],
    TN = conf_matrix_top95$table["0", "0"],
    FP = conf_matrix_top95$table["1", "0"],
    FN = conf_matrix_top95$table["0", "1"],
    Accuracy = conf_matrix_top95$overall["Accuracy"],
    AUC = ifelse(length(unique(y_valid_fold)) > 1, auc(y_valid_fold, pred_proba_top95), NA),
    Precision = conf_matrix_top95$byClass["Precision"],
    Recall = conf_matrix_top95$byClass["Recall"],
    F1_Score = conf_matrix_top95$byClass["F1"],
    Training_Time = training_time_top95
  ))
  
  # Store predictions and true labels for ROC and PR averaging
  if (length(unique(y_valid_fold)) > 1) {
    all_preds_proba_top95[[i]] <- pred_proba_top95
    all_true_labels_top95[[i]] <- as.numeric(as.character(y_valid_fold))
  }
  
  # Output results for each fold
  cat(sprintf("Fold %d -> Accuracy: %.4f | AUC: %.4f | Precision: %.4f | Recall: %.4f | F1 Score: %.4f\n",
              i,
              conf_matrix_top95$overall["Accuracy"],
              ifelse(length(unique(y_valid_fold)) > 1, auc(y_valid_fold, pred_proba_top95), NA),
              conf_matrix_top95$byClass["Precision"],
              conf_matrix_top95$byClass["Recall"],
              conf_matrix_top95$byClass["F1"]))
}

# Summary of Cross-Validation Results for Top 95 Features
cat("\n===== Cross-Validation Summary for Top 95 Features =====\n")
cat("Mean Accuracy:", mean(results_top95$Accuracy, na.rm = TRUE), "±", sd(results_top95$Accuracy, na.rm = TRUE), "\n")
cat("Mean AUC:", mean(results_top95$AUC, na.rm = TRUE), "±", sd(results_top95$AUC, na.rm = TRUE), "\n")
cat("Mean Precision:", mean(results_top95$Precision, na.rm = TRUE), "±", sd(results_top95$Precision, na.rm = TRUE), "\n")
cat("Mean Recall:", mean(results_top95$Recall, na.rm = TRUE), "±", sd(results_top95$Recall, na.rm = TRUE), "\n")
cat("Mean F1 Score:", mean(results_top95$F1_Score, na.rm = TRUE), "±", sd(results_top95$F1_Score, na.rm = TRUE), "\n")
cat("Mean Training Time:", mean(results_top95$Training_Time, na.rm = TRUE), "±", sd(results_top95$Training_Time, na.rm = TRUE), "\n")

print(results_top95)
summary(results_top95)

# Save to CSV
write.csv(results_top95, "results_top95_features_selection.csv", row.names = FALSE)

# ----- Plotting Average Confusion Matrix -----
if (require(ggplot2)) {
  # Average the confusion matrices
  averaged_cm_top95 <- Reduce(`+`, all_conf_matrices_top95) / num_folds
  cm_df_avg_top95 <- as.data.frame(averaged_cm_top95)
  plt_cm_avg_top95 <- ggplot(data = cm_df_avg_top95, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = sprintf("%.2f", Freq)), vjust = 0.5) + # Display average counts
    labs(title = "Average Confusion Matrix (Top 95 Features)", fill = "Average Frequency") +
    theme_minimal()
  print(plt_cm_avg_top95)
} else {
  cat("Please install the 'ggplot2' package to plot the average confusion matrix.\n")
}

# ----- Plotting Average ROC Curve -----
if (require(ggplot2) && require(pROC)) {
  if (length(all_true_labels_top95) > 0) {
    # Combine true labels and predictions
    combined_labels_top95 <- unlist(all_true_labels_top95)
    combined_preds_proba_top95 <- unlist(all_preds_proba_top95)
    
    if (length(unique(combined_labels_top95)) > 1) {
      roc_obj_avg_top95 <- roc(combined_labels_top95, combined_preds_proba_top95)
      roc_df_avg_top95 <- data.frame(FPR = 1 - roc_obj_avg_top95$specificities, TPR = roc_obj_avg_top95$sensitivities)
      
      plt_roc_avg_top95 <- ggplot(roc_df_avg_top95, aes(x = FPR, y = TPR)) +
        geom_line(color = "blue") +
        geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
        labs(title = paste("Average ROC Curve (Top 95 Features)"),
             x = "False Positive Rate", y = "True Positive Rate") +
        theme_minimal()
      print(plt_roc_avg_top95)
    } else {
      cat("Warning: Cannot plot average ROC curve as some test sets had only one class.\n")
    }
  } else {
    cat("Warning: No ROC data to plot.\n")
  }
} else {
  cat("Please install 'ggplot2' and 'pROC' packages to plot the average ROC curve.\n")
}

# ----- Plotting Average Precision-Recall Curve -----
if (require(ggplot2) && require(PRROC)) {
  if (length(all_true_labels_top95) > 0) {
    combined_labels_top95 <- unlist(all_true_labels_top95)
    combined_preds_proba_top95 <- unlist(all_preds_proba_top95)
    
    if (length(unique(combined_labels_top95)) > 1) {
      pr_obj_avg_top95 <- pr.curve(combined_labels_top95, combined_preds_proba_top95, curve = TRUE)
      pr_df_avg_top95 <- data.frame(Recall = pr_obj_avg_top95$curve[, 1], Precision = pr_obj_avg_top95$curve[, 2], Type = "Model")
      baseline_df_top95 <- data.frame(Recall = c(0, 1), Precision = rep(mean(combined_labels_top95), 2), Type = "Baseline")
      plot_df_pr_top95 <- rbind(pr_df_avg_top95, baseline_df_top95)
      
      plt_pr_avg_top95 <- ggplot(plot_df_pr_top95, aes(x = Recall, y = Precision, color = Type)) +
        geom_line() +
        labs(title = paste("Average Precision-Recall Curve (Top 95 Features)"),
             x = "Recall", y = "Precision", color = "Curve Type") +
        scale_color_manual(values = c("Model" = "blue", "Baseline" = "red")) +
        theme_minimal()
      print(plt_pr_avg_top95)
    } else {
      cat("Warning: Cannot plot average Precision-Recall curve as some test sets had only one class.\n")
    }
  } else {
    cat("Warning: No PR data to plot.\n")
  }
} else {
  cat("Please install 'ggplot2' and 'PRROC' packages to plot the average Precision-Recall curve.\n")
}

# ===================================================
# ----- SVM WITH LDA-WEIGHTED TOP 95 FEATURES -----
# ================ 10-BATCH VERSION ================

# Get top 95 features based on absolute LDA coefficients
top_features_batch <- order(abs(W_vec), decreasing = TRUE)[1:95]

# Convert to matrix for batching
X_weighted_data_batch <- as.matrix(X_train_weighted)
y_weighted_data_batch <- y_train_final

set.seed(123)

num_batches <- 10
n_batch <- nrow(X_weighted_data_batch)
batch_size <- floor(n_batch / num_batches)

# Create sequential batch indices
batch_indices <- split(1:n_batch, ceiling(seq_along(1:n_batch) / batch_size))

# Initialize results container
results_top95_batch <- data.frame(Batch = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(),
                                  Accuracy = numeric(), AUC = numeric(), Precision = numeric(),
                                  Recall = numeric(), F1_Score = numeric(), Training_Time = numeric())

all_conf_matrices_top95_batch <- list()
all_preds_proba_top95_batch <- list()
all_true_labels_top95_batch <- list()

for (i in 1:num_batches) {
  
  cat("Processing Batch", i, "\n")
  
  valid_idx_batch <- batch_indices[[i]]
  train_idx_batch <- setdiff(1:n_batch, valid_idx_batch)
  
  X_train_batch <- X_weighted_data_batch[train_idx_batch, ]
  y_train_batch <- y_weighted_data_batch[train_idx_batch]
  
  X_valid_batch <- X_weighted_data_batch[valid_idx_batch, ]
  y_valid_batch <- y_weighted_data_batch[valid_idx_batch]
  
  # Select top 95 LDA-weighted features
  X_train_selected_batch <- X_train_batch[, top_features_batch]
  X_valid_selected_batch <- X_valid_batch[, top_features_batch]
  
  # Train SVM
  start_time_batch <- Sys.time()
  svm_model_top95_batch <- svm(X_train_selected_batch, as.factor(y_train_batch), kernel = "radial", probability = TRUE)
  end_time_batch <- Sys.time()
  training_time_top95_batch <- as.numeric(difftime(end_time_batch, start_time_batch, units = "secs"))
  
  pred_top95_batch <- predict(svm_model_top95_batch, X_valid_selected_batch)
  pred_proba_top95_batch <- attr(predict(svm_model_top95_batch, X_valid_selected_batch, probability = TRUE), "probabilities")[, 2]
  
  conf_matrix_top95_batch <- confusionMatrix(pred_top95_batch, as.factor(y_valid_batch))
  all_conf_matrices_top95_batch[[i]] <- conf_matrix_top95_batch$table
  
  # Collect metrics for each batch
  results_top95_batch <- rbind(results_top95_batch, data.frame(
    Batch = i,
    TP = conf_matrix_top95_batch$table["1", "1"],
    TN = conf_matrix_top95_batch$table["0", "0"],
    FP = conf_matrix_top95_batch$table["1", "0"],
    FN = conf_matrix_top95_batch$table["0", "1"],
    Accuracy = conf_matrix_top95_batch$overall["Accuracy"],
    AUC = ifelse(length(unique(y_valid_batch)) > 1, auc(y_valid_batch, pred_proba_top95_batch), NA),
    Precision = conf_matrix_top95_batch$byClass["Precision"],
    Recall = conf_matrix_top95_batch$byClass["Recall"],
    F1_Score = conf_matrix_top95_batch$byClass["F1"],
    Training_Time = training_time_top95_batch
  ))
  
  if (length(unique(y_valid_batch)) > 1) {
    all_preds_proba_top95_batch[[i]] <- pred_proba_top95_batch
    all_true_labels_top95_batch[[i]] <- as.numeric(as.character(y_valid_batch))
  }
  
  cat(sprintf("Batch %d -> Accuracy: %.4f | AUC: %.4f | Precision: %.4f | Recall: %.4f | F1 Score: %.4f\n",
              i,
              conf_matrix_top95_batch$overall["Accuracy"],
              ifelse(length(unique(y_valid_batch)) > 1, auc(y_valid_batch, pred_proba_top95_batch), NA),
              conf_matrix_top95_batch$byClass["Precision"],
              conf_matrix_top95_batch$byClass["Recall"],
              conf_matrix_top95_batch$byClass["F1"]))
}

# Summary of Batch Evaluation Results
cat("\n===== Batch Evaluation Summary for Top 95 Features =====\n")
cat("Mean Accuracy:", mean(results_top95_batch$Accuracy, na.rm = TRUE), "±", sd(results_top95_batch$Accuracy, na.rm = TRUE), "\n")
cat("Mean AUC:", mean(results_top95_batch$AUC, na.rm = TRUE), "±", sd(results_top95_batch$AUC, na.rm = TRUE), "\n")
cat("Mean Precision:", mean(results_top95_batch$Precision, na.rm = TRUE), "±", sd(results_top95_batch$Precision, na.rm = TRUE), "\n")
cat("Mean Recall:", mean(results_top95_batch$Recall, na.rm = TRUE), "±", sd(results_top95_batch$Recall, na.rm = TRUE), "\n")
cat("Mean F1 Score:", mean(results_top95_batch$F1_Score, na.rm = TRUE), "±", sd(results_top95_batch$F1_Score, na.rm = TRUE), "\n")
cat("Mean Training Time:", mean(results_top95_batch$Training_Time, na.rm = TRUE), "±", sd(results_top95_batch$Training_Time, na.rm = TRUE), "\n")

print(results_top95_batch)
summary(results_top95_batch)

# Save to CSV
write.csv(results_top95_batch, "results_top95_batch.csv", row.names = FALSE)

# ===== Final Model Training for Test Prediction =====
cat("\nTraining Final SVM Model on All Training Data with Top 95 LDA-Weighted Features\n")

X_train_final_selected <- X_weighted_data[, top_features]
y_train_final <- y_weighted_data

svm_model_top95_final <- svm(X_train_final_selected, as.factor(y_train_final), kernel = "radial", probability = TRUE)


# Get the absolute values of LDA loadings
lda_weights <- abs(lda_model$scaling[, 1])

# Get indices of the top 21 features
top_95_indices <- order(lda_weights, decreasing = TRUE)[1:95]

# If you're using the real LDA-weighted data
X_train_weighted_top95 <- X_train_weighted[, top_95_indices]

######################## QR INC ######################################

library(ps)

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
                      FP_qr = NA, FN_qr = NA))
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

incremental_learning_pipeline_qr_eval_batch <- function(X_train, y_train, gamma_manual, cost_manual) {
  num_samples <- nrow(X_train)
  num_trunks <- 10
  trunk_size <- floor(0.7 * num_samples / num_trunks)
  
  stats <- list(cov = NULL, mean = NULL, n = NULL)
  svm_model <- NULL
  results <- data.frame()
  
  for (trunk_idx in 1:num_trunks) {
    start_idx <- ((trunk_idx - 1) * trunk_size) + 1
    end_idx <- min(trunk_idx * trunk_size, floor(0.7 * num_samples))
    
    X_new <- X_train[start_idx:end_idx, ]
    y_new <- y_train[start_idx:end_idx]
    
    if (trunk_idx == 1) {
      stats$n <- nrow(X_new)
      stats$mean <- colMeans(X_new)
      stats$cov <- cov(X_new)
      
      qr_result <- qr(stats$cov)
      projection_matrix <- qr.Q(qr_result)[, 1:min(ncol(X_new), stats$n - 1), drop = FALSE]
      
      X_transformed <- X_new %*% projection_matrix
      
      # Remove constant columns
      X_transformed <- X_transformed[, apply(X_transformed, 2, function(col) length(unique(col)) > 1), drop = FALSE]
      
      svm_model <- e1071::svm(X_transformed, as.factor(y_new), kernel = "radial",
                              gamma = gamma_manual, cost = cost_manual,
                              probability = TRUE, class.weights = c('0' = 1, '1' = 1))
      next
    }
    
    start_time <- Sys.time()
    cpu_start <- proc.time()
    ps_cpu_start <- ps::ps_cpu_times()
    
    update_result <- incremental_qr_lda(stats$cov, stats$mean, stats$n, X_new,
                                        n_components_qr = min(ncol(X_new), stats$n - 1))
    
    end_time <- Sys.time()
    cpu_end <- proc.time()
    ps_cpu_end <- ps::ps_cpu_times()
    
    projection_matrix <- update_result$projection_qr
    stats$n <- update_result$updated_n_qr
    stats$mean <- update_result$updated_mean_qr
    stats$cov <- update_result$updated_cov_qr
    
    X_transformed <- X_new %*% projection_matrix
    
    # Remove constant columns
    X_transformed <- X_transformed[, apply(X_transformed, 2, function(col) length(unique(col)) > 1), drop = FALSE]
    
    svm_model <- e1071::svm(X_transformed, as.factor(y_new), kernel = "radial",
                            gamma = gamma_manual, cost = cost_manual,
                            probability = TRUE, class.weights = c('0' = 1, '1' = 1))
    
    pred <- predict(svm_model, X_transformed)
    prob_scores <- attr(predict(svm_model, X_transformed, probability = TRUE), "probabilities")[, 2]
    metrics_row <- compute_metrics_qr(y_new, pred, prob_scores)
    
    time_to_update_w <- as.numeric(difftime(end_time, start_time, units = "secs"))
    cpu_time_user <- cpu_end["user.self"] - cpu_start["user.self"]
    cpu_time_sys <- cpu_end["sys.self"] - cpu_start["sys.self"]
    cpu_time_total <- cpu_time_user + cpu_time_sys
    
    # Use correct vector indexing for ps_cpu_times
    cpu_percent <- (ps_cpu_end["user"] - ps_cpu_start["user"]) +
      (ps_cpu_end["system"] - ps_cpu_start["system"])
    
    metrics_row$Training_Time_W_qr <- time_to_update_w
    metrics_row$CPU_Time_W_qr <- cpu_time_total
    metrics_row$CPU_Usage_W_qr <- cpu_percent
    metrics_row$Trunk <- trunk_idx
    
    results <- rbind(results, metrics_row)
  }
  
  return(list(
    results = results,
    final_model = svm_model,
    final_projection = projection_matrix,
    final_stats = stats
  ))
}


# ==== RUNNING QR INC with CPU tracking ====

set.seed(42)
X_train_weighted_qr <- X_train_weighted_top95
y_train_final_qr <- y_train_final

batch_results_qr <- incremental_learning_pipeline_qr_eval_batch(
  as.matrix(X_train_weighted_qr),
  as.factor(y_train_final_qr),
  gamma_manual = 0.025,
  cost_manual = 5
)

print(batch_results_qr$results)

# Save to CSV
write.csv(batch_results_qr$results, "results_qr.csv", row.names = FALSE)

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

# # Hyperparameter tuning
# set.seed(42)
# tune_idx <- sample(1:nrow(X_train_weighted_top95), size = 0.2 * nrow(X_train_weighted_top95))
# X_tune <- X_train_weighted_top95[tune_idx, ]
# y_tune <- y_train_final[tune_idx]
# 
# gamma_range <- 2^seq(-10, 1, 2)
# cost_range <- 2^seq(-2, 5, 2)
# 
# tune_result <- tune(svm, train.x = X_tune, train.y = as.factor(y_tune), kernel = "radial", ranges = list(gamma = gamma_range, cost = cost_range), probability = TRUE)
# best_gamma <- tune_result$best.parameters$gamma
# best_cost <- tune_result$best.parameters$cost
# 
# cat("Best Gamma:", best_gamma, "\n")
# cat("Best Cost:", best_cost, "\n")


################## TEST DATA ########################
best_gamma = 0.025
best_cost = 5
# best_gamma = 1
# best_cost  = 16

library(ps)

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
  compute_metrics_qr(y_eval, pred, prob)
}

# ========== Main Function: Incremental QR on Test ==========
incremental_qr_on_test <- function(X_test_full, y_test_full,
                                   initial_model, initial_proj, initial_stats,
                                   gamma_manual, cost_manual,
                                   verbose = TRUE) {
  
  # Define batches
  total_samples <- nrow(X_test_full)
  batch_size <- floor(0.03 * total_samples)  # 3% of 26595 = 797
  num_batches <- 10
  
  if (verbose) {
    cat("Total test samples:", total_samples, "\n")
    cat("Batch size (3%):", batch_size, "\n")
    cat("Number of batches:", num_batches, "\n")
  }
  
  # Fixed evaluation set (first 3%)
  X_eval <- X_test_full[1:batch_size, ]
  y_eval <- y_test_full[1:batch_size]
  X_eval_transformed <- X_eval %*% initial_proj
  
  # Initialize state
  results <- data.frame(
    Accuracy_qr = numeric(),
    Precision_qr = numeric(),
    Recall_qr = numeric(),
    F1_Score_qr = numeric(),
    AUC_qr = numeric(),
    TP_qr = numeric(),
    TN_qr = numeric(),
    FP_qr = numeric(),
    FN_qr = numeric(),
    Training_Time_qr = numeric(),
    CPU_Usage_qr = numeric(),
    Batch = integer(),
    stringsAsFactors = FALSE
  )
  svm_model <- initial_model
  projection_matrix <- initial_proj
  stats <- initial_stats
  all_probabilities_eval <- list()
  all_true_labels_eval <- list()
  all_predictions_eval <- list()
  
  # Get current process handle
  proc <- tryCatch(
    ps::ps_handle(),
    error = function(e) {
      warning("Failed to get process handle: ", e$message)
      return(NULL)
    }
  )
  
  # Incremental loop
  for (batch_idx in 1:num_batches) {
    if (verbose) cat("Batch", batch_idx, "...\n")
    
    # Determine batch indices
    start_idx <- (batch_idx - 1) * batch_size + 1
    end_idx <- min(batch_idx * batch_size, total_samples)
    
    # Skip if outside bounds
    if (start_idx > total_samples) {
      if (verbose) cat("Batch", batch_idx, "skipped: start_idx exceeds total_samples\n")
      break
    }
    
    X_batch <- X_test_full[start_idx:end_idx, ]
    y_batch <- y_test_full[start_idx:end_idx]
    
    start_time <- Sys.time()
    
    # Get CPU usage before
    cpu_usage <- NA
    if (!is.null(proc)) {
      cpu_before <- tryCatch(
        ps::ps_cpu_times(proc),
        error = function(e) {
          warning("Failed to get CPU times before batch ", batch_idx, ": ", e$message)
          return(NULL)
        }
      )
      if (!is.null(cpu_before)) {
        cpu_start_total <- sum(cpu_before[["user"]], cpu_before[["system"]], na.rm = TRUE)
      } else {
        cpu_start_total <- NA
      }
    }
    
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
    
    # Evaluate on fixed eval set (projected)
    metrics_row <- evaluate_model_qr(svm_model, X_eval, y_eval, projection_matrix)
    
    # Make predictions and store probabilities for plotting
    X_eval_proj <- X_eval %*% projection_matrix
    pred_eval <- predict(svm_model, X_eval_proj)
    prob_eval <- attr(predict(svm_model, X_eval_proj, probability = TRUE), "probabilities")[, 2]
    all_probabilities_eval[[batch_idx]] <- prob_eval
    all_true_labels_eval[[batch_idx]] <- as.numeric(as.character(y_eval))
    all_predictions_eval[[batch_idx]] <- factor(as.character(pred_eval), levels = levels(as.factor(y_eval)))
    
    end_time <- Sys.time()
    
    # Get CPU usage after
    if (!is.null(proc) && !is.na(cpu_start_total)) {
      cpu_after <- tryCatch(
        ps::ps_cpu_times(proc),
        error = function(e) {
          warning("Failed to get CPU times after batch ", batch_idx, ": ", e$message)
          return(NULL)
        }
      )
      if (!is.null(cpu_after)) {
        cpu_end_total <- sum(cpu_after[["user"]], cpu_after[["system"]], na.rm = TRUE)
        cpu_usage <- cpu_end_total - cpu_start_total
      }
    }
    
    # Add metrics to results
    metrics_row$Training_Time_qr <- as.numeric(difftime(end_time, start_time, units = "secs"))
    metrics_row$CPU_Usage_qr <- cpu_usage
    metrics_row$Batch <- batch_idx
    
    results <- rbind(results, metrics_row)
  }
  
  return(list(
    results = results,
    probabilities_eval = all_probabilities_eval,
    true_labels_eval = all_true_labels_eval,
    predictions_eval = all_predictions_eval
  ))
}

# ----- Apply feature weighting -----
X_test_weighted_qr <- sweep(X_test, 2, W_vec, `*`)

# ----- Select top features -----
X_test_selected_qr <- X_test_weighted_qr[, top_features]

# ----- Convert to matrix -----
X_test_matrix <- as.matrix(X_test_selected_qr)

# Run the pipeline
incremental_results_test_qr <- incremental_qr_on_test(
  X_test_matrix, as.factor(y_test),
  initial_model = final_model_qr,
  initial_proj = final_projection_qr,
  initial_stats = batch_results_qr$final_stats,
  gamma_manual = best_gamma,
  cost_manual = best_cost
)

# Print results
print(incremental_results_test_qr$results)

# Save to CSV
write.csv(incremental_results_test_qr$results, "results_qr_test.csv", row.names = FALSE)

# ----- Plotting -----
if (require(ggplot2) && require(caret) && require(pROC) && require(PRROC)) {
  # ----- Confusion Matrix Visualization (Final Batch) -----
  final_predictions_eval <- incremental_results_test_qr$predictions_eval[[length(incremental_results_test_qr$predictions_eval)]]
  final_true_labels_eval <- factor(incremental_results_test_qr$true_labels_eval[[length(incremental_results_test_qr$true_labels_eval)]], levels = levels(final_predictions_eval))
  cm_final_eval <- caret::confusionMatrix(final_predictions_eval, final_true_labels_eval)
  cm_df_eval <- as.data.frame(cm_final_eval$table)
  plt_cm_eval <- ggplot(data = cm_df_eval, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), vjust = 0.5) +
    labs(title = "Confusion Matrix on Test Data (Incremental QR)", fill = "Frequency") +
    theme_minimal()
  print(plt_cm_eval)
  
  # ----- ROC Curve (Aggregated over Batches) -----
  all_probs_eval <- unlist(incremental_results_test_qr$probabilities_eval)
  all_labels_eval <- unlist(incremental_results_test_qr$true_labels_eval)
  
  if (length(unique(all_labels_eval)) > 1) {
    roc_obj_eval <- roc(all_labels_eval, all_probs_eval)
    plt_roc_eval <- ggplot(data.frame(FPR = 1 - roc_obj_eval$specificities, TPR = roc_obj_eval$sensitivities),
                           aes(x = FPR, y = TPR)) +
      geom_line(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      labs(title = "ROC Curve on Test Data (Incremental QR)",
           x = "False Positive Rate", y = "True Positive Rate",
           caption = paste("AUC =", round(auc(roc_obj_eval), 3))) +
      theme_minimal()
    print(plt_roc_eval)
  } else {
    cat("Warning: Cannot plot ROC curve as all true labels in the evaluation set are the same.\n")
  }
  
  # ----- Precision-Recall Curve (Aggregated over Batches) -----
  if (require(PRROC) && length(unique(all_labels_eval)) > 1) {
    pr_data_eval <- pr.curve(all_labels_eval, all_probs_eval, curve = TRUE)
    plt_pr_eval <- ggplot(data.frame(Recall = pr_data_eval$curve[, 1], Precision = pr_data_eval$curve[, 2]),
                          aes(x = Recall, y = Precision)) +
      geom_line(color = "blue") +
      geom_hline(yintercept = mean(all_labels_eval), linetype = "dashed", color = "red") + # Baseline
      labs(title = "Precision-Recall Curve on Test Data (Incremental QR)",
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


# === Prepare Test Data with Top 95 Features ===
# === Apply LDA Weighting and Select Top 95 for Test Set ===
X_test_weighted <- sweep(X_test, 2, W_vec, `*`)
X_test_selected <- X_test_weighted[, top_95_indices]

# Ensure y_test is a factor
y_test <- as.factor(y_test)

# === Split Test Data into 10 Batches ===
batch_size <- floor(nrow(X_test_selected) / 10)
batches <- split(1:nrow(X_test_selected), cut(1:nrow(X_test_selected), breaks = 10, labels = FALSE))

# === Preallocate Results DataFrame ===
test_results_top95 <- data.frame(
  Batch = 1:10, TP = 0, TN = 0, FP = 0, FN = 0,
  Accuracy = 0, AUC = NA, Precision = 0,
  Recall = 0, F1_Score = 0, Inference_Time = 0, CPU_Usage = 0
)

# Track predictions
all_probabilities_test <- numeric()
all_true_labels_test <- numeric()
all_predictions_test <- factor(levels = levels(y_test))

# Initial training set
X_current_train <- X_train_weighted_top95
y_current_train <- y_train_final

# === Batch-wise Testing and Retraining ===
for (i in seq_along(batches)) {
  cat("Processing Test Batch", i, "\n")
  
  idx <- batches[[i]]
  X_batch <- X_test_selected[idx, ]
  y_batch <- y_test[idx]
  
  # Train model with current training set
  svm_model_top95_final <- svm(X_current_train, as.factor(y_current_train),
                               kernel = "radial", gamma = 0.025, cost = 5,
                               probability = TRUE, class.weights = c('0' = 1, '1' = 1))
  
  # Predict with timing
  start_time <- Sys.time()
  time_taken <- system.time({
    preds <- predict(svm_model_top95_final, X_batch)
    preds_proba <- attr(predict(svm_model_top95_final, X_batch, probability = TRUE), "probabilities")[, 2]
  })
  end_time <- Sys.time()
  
  inference_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  cpu_usage <- time_taken[["elapsed"]]
  
  # Confusion Matrix
  cm <- table(Predicted = preds, Actual = y_batch)
  TP <- ifelse("1" %in% rownames(cm) & "1" %in% colnames(cm), cm["1", "1"], 0)
  TN <- ifelse("0" %in% rownames(cm) & "0" %in% colnames(cm), cm["0", "0"], 0)
  FP <- ifelse("1" %in% rownames(cm) & "0" %in% colnames(cm), cm["1", "0"], 0)
  FN <- ifelse("0" %in% rownames(cm) & "1" %in% colnames(cm), cm["0", "1"], 0)
  
  accuracy <- mean(preds == y_batch)
  auc_val <- tryCatch({
    auc(as.numeric(as.character(y_batch)), preds_proba)
  }, error = function(e) NA)
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  recall <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  # Update results
  test_results_top95[i, ] <- c(i, TP, TN, FP, FN, accuracy, auc_val,
                               precision, recall, f1, inference_time, cpu_usage)
  
  # Track predictions
  all_predictions_test <- factor(c(as.character(all_predictions_test), as.character(preds)), levels = levels(y_test))
  all_probabilities_test <- c(all_probabilities_test, preds_proba)
  all_true_labels_test <- c(all_true_labels_test, as.numeric(as.character(y_batch)))
  
  # Add batch to training data
  X_current_train <- rbind(X_current_train, X_batch)
  y_current_train <- c(y_current_train, as.character(y_batch))
}

# === Results Summary ===
print(test_results_top95)
summary(test_results_top95)

# Save results
write.csv(test_results_top95, file = "results_test_top_95.csv", row.names = FALSE)

# === Plotting ===
if (require(ggplot2) && require(caret) && require(pROC) && require(PRROC)) {
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
  if (require(PRROC) && length(unique(all_true_labels_test)) > 1 && length(unique(all_probabilities_test)) > 1) {
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
  } else if (!require(PRROC)) {
    cat("Please install the 'PRROC' package for Precision-Recall curve.\n")
  } else {
    cat("Warning: Cannot plot Precision-Recall curve due to insufficient unique labels or probabilities.\n")
  }
  
  
  # ----- ROC Curve (as before) -----
  if (length(unique(all_true_labels_test)) > 1 && length(unique(all_probabilities_test)) > 1) {
    roc_obj_test <- roc(all_true_labels_test, all_probabilities_test)
    plt_roc <- ggplot(data.frame(FPR = 1 - roc_obj_test$specificities, TPR = roc_obj_test$sensitivities),
                      aes(x = FPR, y = TPR)) +
      geom_line(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "blue") +
      labs(title = "ROC Curve on Test Data (Top 95 Features)",
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