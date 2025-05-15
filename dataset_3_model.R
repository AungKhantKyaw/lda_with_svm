# Install necessary packages (if not already installed)
packages <- c("caret", "ggplot2", "e1071", "pROC", "MASS", "bigstatsr", 
              "irlba", "MLmetrics", "dplyr", "arm", "RSpectra", "PRROC", 
              "ROSE","tidyr")

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
library(doParallel)
library(foreach)
library(ROSE)
library(tm)
library(stringr)
library(tidyr)
library(PRROC)

# ======================
#     LOAD DATA
# ======================
df <- read.csv("CEAS_08.csv", stringsAsFactors = FALSE)
#df <- df[sample(nrow(df), 5000), ]

# ============================
#     SEPARATE FEATURES / TARGET
# ============================
X <- df[, !colnames(df) %in% "label"]
y <- df$label

# Bar plot for binary target
ggplot(df, aes(x = factor(label))) +
  geom_bar(fill = c("lightblue", "salmon")) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  labs(title = "Distribution of Phishing vs. Non-Phishing Email",
       x = "Label (0 = Non-Phishing, 1 = Phishing)",
       y = "Number of Messages") +
  theme_minimal()

corpus <- VCorpus(VectorSource(df$body))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)

dtm <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))
dtm <- removeSparseTerms(dtm, 0.94)

X <- as.data.frame(as.matrix(dtm))

# ======================
#     SPLIT DATA (70/30)
# ======================
set.seed(123)
train_idx <- sample(seq_len(nrow(X)), size = floor(0.7 * nrow(X)))

X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test  <- X[-train_idx, ]
y_test  <- y[-train_idx]

# ===========================
#     TRAIN/VALID SPLIT (80/20 of train)
# ===========================
set.seed(123)
train_sub_idx <- sample(seq_len(nrow(X_train)), size = floor(0.8 * nrow(X_train)))
X_train_final <- X_train[train_sub_idx, ]
y_train_final <- y_train[train_sub_idx]
X_valid <- X_train[-train_sub_idx, ]
y_valid <- y_train[-train_sub_idx]

# ============================
#     FIX DUPLICATED COLNAMES
# ============================
X_train_final <- X_train_final[, !duplicated(colnames(X_train_final))]
colnames(X_train_final) <- make.names(colnames(X_train_final), unique = TRUE)

# ============================
#     HANDLE CLASS IMBALANCE
# ============================
combined_data <- cbind(X_train_final, label = y_train_final)
balanced_data <- ovun.sample(label ~ ., data = combined_data, method = "both", 
                             N = nrow(combined_data), seed = 1)$data

X_train_final <- balanced_data[, !colnames(balanced_data) %in% "label"]
y_train_final <- balanced_data$label


best_gamma = 0.125
best_cost = 16

# ======================================
#     SVM Cross-Validation WITHOUT LDA
# ======================================
set.seed(123)
num_folds <- 10
folds <- createFolds(y_train_final, k = num_folds)

results <- data.frame(Fold = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(), 
                      Accuracy = numeric(), AUC = numeric(), Precision = numeric(), 
                      Recall = numeric(), F1_Score = numeric(), Training_Time = numeric())

for (i in seq_along(folds)) {
  cat("\nProcessing Fold", i, "\n")
  
  # Fold split
  test_idx <- folds[[i]]
  train_idx <- setdiff(seq_len(nrow(X_train_final)), test_idx)
  
  X_train_fold <- X_train_final[train_idx, , drop = FALSE]
  y_train_fold <- y_train_final[train_idx]
  X_test_fold  <- X_train_final[test_idx, , drop = FALSE]
  y_test_fold  <- y_train_final[test_idx]
  
  # Filter numeric columns only
  X_train_fold <- X_train_fold[, sapply(X_train_fold, is.numeric), drop = FALSE]
  X_test_fold  <- X_test_fold[, colnames(X_train_fold), drop = FALSE]
  
  # Preprocessing (impute, center, scale)
  preProc <- preProcess(X_train_fold, method = c("medianImpute", "center", "scale"))
  X_train_fold <- predict(preProc, X_train_fold)
  X_test_fold  <- predict(preProc, X_test_fold)
  
  cat("  -> Train samples:", nrow(X_train_fold), " | Test samples:", nrow(X_test_fold), "\n")
  cat("  -> Class balance:", table(y_train_fold), "\n")
  
  if (nrow(X_train_fold) < 10 || length(unique(y_train_fold)) < 2) {
    cat("  Skipping Fold", i, ": insufficient data or only one class.\n")
    next
  }
  
  # Train SVM
  start_time <- Sys.time()
  svm_model <- svm(X_train_fold, as.factor(y_train_fold), kernel = "radial", probability = TRUE)
  end_time <- Sys.time()
  training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  # Predict
  preds <- predict(svm_model, X_test_fold)
  pred_probs <- attr(predict(svm_model, X_test_fold, probability = TRUE), "probabilities")[, "1"]
  
  # Metrics
  conf <- confusionMatrix(preds, as.factor(y_test_fold), positive = "1")
  auc_score <- auc(y_test_fold, pred_probs)
  
  results <- rbind(results, data.frame(
    Fold = i,
    TP = conf$table["1", "1"],
    TN = conf$table["0", "0"],
    FP = conf$table["1", "0"],
    FN = conf$table["0", "1"],
    Accuracy = conf$overall["Accuracy"],
    AUC = auc_score,
    Precision = conf$byClass["Precision"],
    Recall = conf$byClass["Recall"],
    F1_Score = conf$byClass["F1"],
    Training_Time = training_time
  ))
}

# ============================
#     RESULTS
# ============================
print(results)
summary(results)

# ===================================================
# ----- SVM WITH LDA -----
# ===================================================
set.seed(123)
folds <- createFolds(y_train_final, k = 10, returnTrain = FALSE)

results_lda <- data.frame(Fold = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(),
                          Accuracy = numeric(), AUC = numeric(), Precision = numeric(),
                          Recall = numeric(), F1_Score = numeric(), Training_Time = numeric())

# Store all predictions and true labels for overall metrics
all_predictions_lda <- factor()
all_probabilities_lda <- numeric()
all_true_labels_lda <- factor()

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
  svm_model_lda <- svm(X_train_lda, as.factor(y_train_fold), kernel = "radial",
                       gamma = best_gamma,
                       cost = best_cost,
                       probability = TRUE)
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
    AUC = tryCatch(auc(y_test_fold, pred_proba_lda), error = function(e) NA),
    Precision = conf_matrix_lda$byClass["Precision"],
    Recall = conf_matrix_lda$byClass["Recall"],
    F1_Score = conf_matrix_lda$byClass["F1"],
    Training_Time = training_time_lda
  ))
  
  # Store predictions and true labels for overall metrics with explicit level management
  current_preds_factor <- factor(pred_lda)
  all_predictions_lda <- factor(c(as.character(all_predictions_lda), as.character(current_preds_factor)),
                                levels = union(levels(all_predictions_lda), levels(current_preds_factor)))
  
  all_probabilities_lda <- c(all_probabilities_lda, pred_proba_lda)
  
  current_true_factor <- factor(y_test_fold)
  all_true_labels_lda <- factor(c(as.character(all_true_labels_lda), as.character(current_true_factor)),
                                levels = union(levels(all_true_labels_lda), levels(current_true_factor)))
  
  # --- DEBUGGING PRINTS ---
  cat("Fold", i, " - Levels of Predicted:", levels(pred_lda), "\n")
  cat("Fold", i, " - Levels of True:", levels(as.factor(y_test_fold)), "\n")
  cat("Fold", i, " - First 10 Predictions:", as.character(head(pred_lda)), "\n")
  cat("Fold", i, " - First 10 True Labels:", as.character(head(y_test_fold)), "\n")
  cat("Fold", i, " - Dimensions of X_train_lda:", dim(X_train_lda), "\n")
  cat("Fold", i, " - Dimensions of X_test_lda:", dim(X_test_lda), "\n")
  # --- END DEBUGGING PRINTS ---
}

print(results_lda)
summary(results_lda)

# Save to CSV
write.csv(results_lda, "results_lda_may_9.csv", row.names = FALSE)

# === Overall Evaluation Metrics and Plots ===
if (require(caret) && require(pROC) && require(PRROC) && require(ggplot2)) {
  cat("\n=== Overall Evaluation Metrics (Across All Folds) ===\n")
  
  # Inspect levels before confusion matrix
  cat("Levels of all_predictions_lda (Overall):", levels(all_predictions_lda), "\n")
  cat("Levels of all_true_labels_lda (Overall):", levels(all_true_labels_lda), "\n")
  
  # Confusion Matrix
  cm_overall_lda <- caret::confusionMatrix(all_predictions_lda, all_true_labels_lda)
  print("Overall Confusion Matrix:")
  print(cm_overall_lda)
  
  # Plotting Overall Confusion Matrix
  cm_df_lda <- as.data.frame(cm_overall_lda$table)
  plt_cm_lda <- ggplot(data = cm_df_lda, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), vjust = 0.5) +
    labs(title = "Overall Confusion Matrix (LDA + SVM)", fill = "Frequency") +
    theme_minimal()
  print(plt_cm_lda)
  
  # ROC Curve
  if (length(unique(all_true_labels_lda)) > 1 && length(unique(all_probabilities_lda)) > 1) {
    roc_obj_lda <- roc(as.numeric(as.character(all_true_labels_lda)), all_probabilities_lda)
    auc_overall_lda <- auc(roc_obj_lda)
    cat("\nOverall AUC:", auc_overall_lda, "\n")
    
    roc_df_lda <- data.frame(FPR = 1 - roc_obj_lda$specificities, TPR = roc_obj_lda$sensitivities)
    plt_roc_lda <- ggplot(roc_df_lda, aes(x = FPR, y = TPR)) +
      geom_line(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      labs(title = "Overall ROC Curve (LDA + SVM)",
           x = "False Positive Rate", y = "True Positive Rate",
           caption = paste("AUC =", round(auc_overall_lda, 3))) +
      theme_minimal()
    print(plt_roc_lda)
  } else {
    cat("\nWarning: Cannot plot overall ROC curve as the number of unique true labels or probabilities is less than 2.\n")
  }
  
  # Precision-Recall Curve
  if (require(PRROC) && length(unique(all_true_labels_lda)) > 1 && length(unique(all_probabilities_lda)) > 1) {
    pr_obj_lda <- pr.curve(as.numeric(as.character(all_true_labels_lda)), all_probabilities_lda, curve = TRUE)
    auc_pr_overall_lda <- pr_obj_lda$auc.integral
    cat("Overall AUC-PR:", auc_pr_overall_lda, "\n")
    
    pr_df_lda <- data.frame(Recall = pr_obj_lda$curve[, 1], Precision = pr_obj_lda$curve[, 2])
    plt_pr_lda <- ggplot(pr_df_lda, aes(x = Recall, y = Precision)) +
      geom_line(color = "blue") +
      geom_hline(yintercept = mean(as.numeric(as.character(all_true_labels_lda))), linetype = "dashed", color = "red") + # Baseline
      labs(title = "Overall Precision-Recall Curve (LDA + SVM)",
           x = "Recall", y = "Precision",
           caption = paste("AUC-PR =", round(auc_pr_overall_lda, 3))) +
      theme_minimal()
    print(plt_pr_lda)
  } else {
    cat("\nWarning: Cannot plot overall Precision-Recall curve as the number of unique true labels or probabilities is less than 2, or PRROC package is not available.\n")
  }
  
} else {
  cat("\nPlease install 'caret', 'pROC', 'PRROC', and 'ggplot2' packages for overall evaluation metrics and plots.\n")
}

# === Plotting Performance Metrics vs. Fold ===
if (require(ggplot2) && require(tidyr)) {
  results_long_lda <- gather(results_lda, key = "Metric", value = "Value",
                             Accuracy, AUC, Precision, Recall, F1_Score)
  
  plt_metrics_lda <- ggplot(results_long_lda, aes(x = Fold, y = Value, color = Metric)) +
    geom_line() +
    geom_point() +
    facet_wrap(~ Metric, scales = "free_y") +
    labs(title = "Performance Metrics vs. Fold (LDA + SVM Cross-Validation)",
         x = "Fold", y = "Value", color = "Metric") +
    theme_minimal() +
    theme(legend.position = "bottom")
  print(plt_metrics_lda)
} else {
  cat("Please install 'ggplot2' and 'tidyr' packages to plot performance metrics vs. fold.\n")
}


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
cores <- max(2, parallel::detectCores() / 2)
cl <- makeCluster(cores)
registerDoParallel(cl)

# Setup data and parameters
set.seed(123)
num_folds <- 2
folds <- createFolds(y_train_final, k = num_folds)
X_weighted_data <- as.matrix(X_train_weighted)
y_weighted_data <- y_train_final
feature_counts <- c(70, 72, 74, 75)
all_results <- list()

# Ensure W_vec is defined
if (!exists("W_vec") || is.null(W_vec)) {
  stop("W_vec is not defined. Make sure it's created from a linear model (e.g., linear SVM).")
}

# Loop over different numbers of top features
for (n_features in feature_counts) {
  
  max_features <- length(W_vec)
  actual_n_features <- min(n_features, max_features)
  top_n_features <- order(abs(W_vec), decreasing = TRUE)[1:actual_n_features]
  cat("\n===== Running CV for Top", actual_n_features, "Features =====\n")
  
  # Run cross-validation in parallel
  fold_results_list <- foreach(i = 1:num_folds, .packages = c("e1071", "pROC", "caret")) %dopar% {
    tryCatch({
      valid_idx <- folds[[i]]
      train_idx <- setdiff(seq_len(nrow(X_weighted_data)), valid_idx)
      
      X_train_fold <- X_weighted_data[train_idx, top_n_features, drop = FALSE]
      y_train_fold <- y_weighted_data[train_idx]
      X_valid_fold <- X_weighted_data[valid_idx, top_n_features, drop = FALSE]
      y_valid_fold <- y_weighted_data[valid_idx]
      
      # Skip if missing values
      if (anyNA(X_train_fold) || anyNA(X_valid_fold)) return(NULL)
      
      # Train SVM model
      start_time <- Sys.time()
      model <- svm(X_train_fold, as.factor(y_train_fold), kernel = "radial",
                   gamma = best_gamma,
                   cost = best_cost,
                   probability = TRUE)
      end_time <- Sys.time()
      training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      # Predict and evaluate
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
      cat(sprintf("Error in Fold %d: %s\n", i, e$message))
      return(NULL)
    })
  }
  
  # Clean and summarize results
  fold_results_list <- fold_results_list[!sapply(fold_results_list, is.null)]
  
  if (length(fold_results_list) == 0) {
    cat(sprintf("No valid folds for Top %d features — skipping.\n", n_features))
    next
  }
  
  fold_results <- do.call(rbind, fold_results_list)
  fold_results <- fold_results[complete.cases(fold_results), ]
  
  all_results[[paste0("Top_", n_features)]] <- fold_results
  
  cat(sprintf("Top %d Features: Mean Accuracy = %.4f ± %.4f | Mean AUC = %.4f ± %.4f\n",
              actual_n_features,
              mean(fold_results$Accuracy, na.rm = TRUE),
              sd(fold_results$Accuracy, na.rm = TRUE),
              mean(fold_results$AUC, na.rm = TRUE),
              sd(fold_results$AUC, na.rm = TRUE)))
  
  gc()
}

# Stop parallel backend
stopCluster(cl)


###### END FIND THE BEST FEATURE TO SELECT ################

# ===================================================
# ----- SVM WITH LDA-WEIGHTED TOP 70 FEATURES -----
# ===================================================

# Fit LDA on full training set to get feature importances
lda_model <- lda(y_train_final ~ ., data = data.frame(y_train_final, X_train_final))

# Get LDA coefficients (feature weights)
W <- lda_model$scaling
W_vec <- as.vector(W)

# Weight original features using LDA coefficients
X_train_weighted <- sweep(X_train_final, 2, W_vec, `*`)
X_valid_weighted <- sweep(X_valid, 2, W_vec, `*`)

# Get top 70 features based on absolute LDA coefficients
top_features <- order(abs(W_vec), decreasing = TRUE)[1:70]

# Convert to matrix for CV loop
X_weighted_data <- as.matrix(X_train_weighted)
y_weighted_data <- y_train_final

set.seed(123)

# Ensure y_weighted_data is a factor with the correct levels BEFORE the loop
y_weighted_data <- as.factor(y_weighted_data)

num_folds <- 10
folds <- createFolds(y_weighted_data, k = num_folds, list = TRUE, returnTrain = FALSE)

# Initialize results container for Top 70 Features
results_top70 <- data.frame(Fold = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(),
                            Accuracy = numeric(), AUC = numeric(), Precision = numeric(),
                            Recall = numeric(), F1_Score = numeric(), Training_Time = numeric())

# === Initialize Accumulators for Plotting ===
all_preds_top70 <- factor(levels = levels(y_weighted_data))
all_proba_top70 <- numeric()
all_actual_top70 <- numeric() # Initialize as empty

for (i in 1:num_folds) {

  cat("Processing Fold", i, "\n")

  valid_idx <- folds[[i]]

  X_train_fold <- X_weighted_data[-valid_idx, ]
  y_train_fold <- y_weighted_data[-valid_idx]

  X_valid_fold <- X_weighted_data[valid_idx, ]
  y_valid_fold <- y_weighted_data[valid_idx]

  # Select top 70 LDA-weighted features
  X_train_selected <- X_train_fold[, top_features]
  X_valid_selected <- X_valid_fold[, top_features]
  X_test_selected <- X_test[, top_features]

  # Train SVM
  start_time <- Sys.time()
  svm_model_top70 <- svm(X_train_selected, as.factor(y_train_fold), kernel = "radial",
                          gamma = best_gamma,
                          cost = best_cost,
                          probability = TRUE)
  end_time <- Sys.time()
  training_time_top70 <- as.numeric(difftime(end_time, start_time, units = "secs"))

  pred_top70 <- predict(svm_model_top70, X_valid_selected)
  pred_proba_top70 <- attr(predict(svm_model_top70, X_valid_selected, probability = TRUE), "probabilities")[, 2]

  # Confusion matrix
  conf_matrix_top70 <- confusionMatrix(pred_top70, as.factor(y_valid_fold))

  # Collect metrics for each fold
  results_top70 <- rbind(results_top70, data.frame(
    Fold = i,
    TP = conf_matrix_top70$table["1", "1"],
    TN = conf_matrix_top70$table["0", "0"],
    FP = conf_matrix_top70$table["1", "0"],
    FN = conf_matrix_top70$table["0", "1"],
    Accuracy = conf_matrix_top70$overall["Accuracy"],
    AUC = auc(y_valid_fold, pred_proba_top70),
    Precision = conf_matrix_top70$byClass["Precision"],
    Recall = conf_matrix_top70$byClass["Recall"],
    F1_Score = conf_matrix_top70$byClass["F1"],
    Training_Time = training_time_top70
  ))

  # Accumulate predictions and true labels for overall plotting
  pred_top70_factor <- factor(pred_top70, levels = levels(y_weighted_data))
  all_preds_top70 <- factor(c(as.character(all_preds_top70), as.character(pred_top70_factor)), levels = levels(y_weighted_data))
  all_proba_top70 <- c(all_proba_top70, pred_proba_top70)
  all_actual_top70 <- c(all_actual_top70, as.numeric(as.character(y_valid_fold))) # Corrected line

  # Output results for each fold
  cat(sprintf("Fold %d -> Accuracy: %.4f | AUC: %.4f | Precision: %.4f | Recall: %.4f | F1 Score: %.4f\n",
              i,
              conf_matrix_top70$overall["Accuracy"],
              auc(y_valid_fold, pred_proba_top70),
              conf_matrix_top70$byClass["Precision"],
              conf_matrix_top70$byClass["Recall"],
              conf_matrix_top70$byClass["F1"]))
}

# Summary of Cross-Validation Results for Top 70 Features
cat("\n===== Cross-Validation Summary for Top 70 Features =====\n")
cat("Mean Accuracy:", mean(results_top70$Accuracy, na.rm = TRUE), "±", sd(results_top70$Accuracy, na.rm = TRUE), "\n")
cat("Mean AUC:", mean(results_top70$AUC, na.rm = TRUE), "±", sd(results_top70$AUC, na.rm = TRUE), "\n")
cat("Mean Precision:", mean(results_top70$Precision, na.rm = TRUE), "±", sd(results_top70$Precision, na.rm = TRUE), "\n")
cat("Mean Recall:", mean(results_top70$Recall, na.rm = TRUE), "±", sd(results_top70$Recall, na.rm = TRUE), "\n")
cat("Mean F1 Score:", mean(results_top70$F1_Score, na.rm = TRUE), "±", sd(results_top70$F1_Score, na.rm = TRUE), "\n")
cat("Mean Training Time:", mean(results_top70$Training_Time, na.rm = TRUE), "±", sd(results_top70$Training_Time, na.rm = TRUE), "\n")

print(results_top70)
summary(results_top70)

# Save to CSV
write.csv(results_top70, "results_top70_may_9_2.csv", row.names = FALSE)

# === Plotting ===
if (require(ggplot2) && require(caret) && require(pROC) && require(PRROC) && require(tidyr)) {
  # ----- Confusion Matrix Visualization (Overall) -----
  all_preds_top70_factor <- factor(all_preds_top70, levels = levels(y_weighted_data))
  cm_overall_top70 <- caret::confusionMatrix(all_preds_top70_factor, y_weighted_data)
  cm_df_top70 <- as.data.frame(cm_overall_top70$table)
  plt_cm_top70 <- ggplot(data = cm_df_top70, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), vjust = 0.5) +
    labs(title = "Confusion Matrix (SVM with LDA-Weighted Top 70)", fill = "Frequency") +
    theme_minimal()
  print(plt_cm_top70)

  # ----- ROC Curve (Overall) -----
  if (length(unique(all_actual_top70)) > 1) {
    roc_obj_top70 <- roc(all_actual_top70, all_proba_top70)
    plt_roc_top70 <- ggplot(data.frame(FPR = 1 - roc_obj_top70$specificities, TPR = roc_obj_top70$sensitivities),
                            aes(x = FPR, y = TPR)) +
      geom_line(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      labs(title = "ROC Curve (SVM with LDA-Weighted Top 70)",
           x = "False Positive Rate", y = "True Positive Rate",
           caption = paste("AUC =", round(auc(roc_obj_top70), 3))) +
      theme_minimal()
    print(plt_roc_top70)
  } else {
    cat("Warning: Cannot plot ROC curve as all true labels are the same.\n")
  }

  # ----- Precision-Recall Curve (Overall) -----
  if (require(PRROC) && length(unique(all_actual_top70)) > 1) {
    pr_data_top70 <- pr.curve(all_actual_top70, all_proba_top70, curve = TRUE)
    plt_pr_top70 <- ggplot(data.frame(Recall = pr_data_top70$curve[, 1], Precision = pr_data_top70$curve[, 2]),
                           aes(x = Recall, y = Precision)) +
      geom_line(color = "blue") +
      geom_hline(yintercept = mean(all_actual_top70), linetype = "dashed", color = "red") + # Baseline
      labs(title = "Precision-Recall Curve (SVM with LDA-Weighted Top 70)",
           x = "Recall", y = "Precision",
           caption = paste("AUC-PR =", round(pr_data_top70$auc.integral, 3))) +
      theme_minimal()
    print(plt_pr_top70)
  } else if (!require(PRROC)) {
    cat("Please install the 'PRROC' package to plot Precision-Recall curves.\n")
  } else {
    cat("Warning: Cannot plot Precision-Recall curve as all true labels are the same.\n")
  }

  # ----- Performance Metrics vs. Fold -----
  results_long_top70 <- gather(results_top70, key = "Metric", value = "Value",
                               Accuracy, AUC, Precision, Recall, F1_Score)

  plt_metrics_top70 <- ggplot(results_long_top70, aes(x = Fold, y = Value, color = Metric)) +
    geom_line() +
    geom_point() +
    facet_wrap(~ Metric, scales = "free_y") +
    labs(title = "Performance Metrics vs. Fold (SVM with LDA-Weighted Top 70)",
         x = "Fold", y = "Value", color = "Metric") +
    theme_minimal() +
    theme(legend.position = "bottom")
  print(plt_metrics_top70)

} else {
  cat("Please install 'ggplot2', 'caret', 'pROC', and 'PRROC' packages to plot.\n")
}

###### BATCH Top 70 Features ############################

set.seed(123)

# Get top 70 features based on absolute LDA coefficients
top_features <- order(abs(W_vec), decreasing = TRUE)[1:70]

# Convert to matrix
X_weighted_data <- as.matrix(X_train_weighted)
y_weighted_data <- as.factor(y_train_final)

# Split into 10 batches
n <- nrow(X_weighted_data)
batch_size <- floor(n / 10)
indices <- split(1:n, ceiling(seq_along(1:n) / batch_size))

# Merge any remainder into the last batch
if (length(indices) > 10) {
  indices[[10]] <- c(indices[[10]], unlist(indices[11:length(indices)]))
  indices <- indices[1:10]
}

# Initialize results container
results_top70_batch <- data.frame(Batch = integer(), TP = integer(), TN = integer(), FP = integer(),
                                  FN = integer(), Accuracy = numeric(), AUC = numeric(), Precision = numeric(),
                                  Recall = numeric(), F1_Score = numeric(), Training_Time = numeric(), CPU_Time = numeric())

# Initialize lists to store predictions and true labels
all_preds_proba_top70 <- list()
all_true_labels_top70 <- list()
all_conf_matrices_top70 <- list()

for (i in seq_along(indices)) {
  cat("Processing Batch", i, "\n")
  
  test_idx <- indices[[i]]
  train_idx <- setdiff(1:n, test_idx)
  
  X_train <- X_weighted_data[train_idx, ]
  y_train <- y_weighted_data[train_idx]
  X_test <- X_weighted_data[test_idx, ]
  y_test <- y_weighted_data[test_idx]
  
  # Select top 70 LDA-weighted features
  X_train_selected <- X_train[, top_features]
  X_test_selected <- X_test[, top_features]
  
  # Measure CPU and wall-clock time
  cpu_start <- proc.time()
  start_time <- Sys.time()
  
  svm_model_top70 <- svm(X_train_selected, as.factor(y_train), kernel = "radial",
                         gamma = best_gamma,
                         cost = best_cost,
                         probability = TRUE)
  
  end_time <- Sys.time()
  cpu_end <- proc.time()
  
  training_time_top70 <- as.numeric(difftime(end_time, start_time, units = "secs"))
  cpu_time_top70 <- (cpu_end - cpu_start)[["user.self"]] + (cpu_end - cpu_start)[["sys.self"]]
  
  pred_top70 <- predict(svm_model_top70, X_test_selected)
  pred_proba_top70 <- attr(predict(svm_model_top70, X_test_selected, probability = TRUE), "probabilities")[, 2]
  
  conf_matrix_top70 <- confusionMatrix(pred_top70, as.factor(y_test))
  
  results_top70_batch <- rbind(results_top70_batch, data.frame(
    Batch = i,
    TP = conf_matrix_top70$table["1", "1"],
    TN = conf_matrix_top70$table["0", "0"],
    FP = conf_matrix_top70$table["1", "0"],
    FN = conf_matrix_top70$table["0", "1"],
    Accuracy = conf_matrix_top70$overall["Accuracy"],
    AUC = ifelse(length(unique(y_test)) > 1, auc(y_test, pred_proba_top70), NA),
    Precision = conf_matrix_top70$byClass["Precision"],
    Recall = conf_matrix_top70$byClass["Recall"],
    F1_Score = conf_matrix_top70$byClass["F1"],
    Training_Time = training_time_top70,
    CPU_Time = cpu_time_top70
  ))
  
  # Store predictions and true labels for ROC later
  if (length(unique(y_test)) > 1) {
    all_preds_proba_top70[[i]] <- pred_proba_top70
    all_true_labels_top70[[i]] <- as.numeric(as.character(y_test))
  }
  all_conf_matrices_top70[[i]] <- conf_matrix_top70$table
}

print(results_top70_batch)
summary(results_top70_batch)

# Save to CSV
write.csv(results_top70_batch, "results_top70_batch_may_9_2.csv", row.names = FALSE)



# # ===== Final Model Training for Test Prediction =====
cat("\nTraining Final SVM Model on All Training Data with Top 70 LDA-Weighted Features\n")

X_train_final_selected <- X_weighted_data[, top_features]
y_train_final <- y_weighted_data

svm_model_top70_final <- svm(X_train_final_selected, as.factor(y_train_final), kernel = "radial",
                             gamma = best_gamma,
                             cost = best_cost, probability = TRUE)


# Get the absolute values of LDA loadings
lda_weights <- abs(lda_model$scaling[, 1])

# Get indices of the top 70 features
top_70_indices <- order(lda_weights, decreasing = TRUE)[1:70]

# If you're using the real LDA-weighted data
X_train_weighted_top70 <- X_train_weighted[, top_70_indices]


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

get_class_weights <- function(y) {
  tbl <- table(y)
  total <- sum(tbl)
  weights <- total / (length(tbl) * tbl)
  return(as.list(weights))
}

incremental_learning_pipeline_qr <- function(X_train, y_train, gamma_manual, cost_manual) {
  num_samples <- nrow(X_train)
  num_trunks <- 10
  trunk_size <- floor(num_samples / num_trunks)
  
  stats <- list(cov = NULL, mean = NULL, n = NULL)
  svm_model <- NULL
  results <- data.frame()
  
  # === Trunk 1 (Base Model Training) ===
  X_trunk1 <- X_train[1:trunk_size, ]
  y_trunk1 <- y_train[1:trunk_size]
  
  stats$n <- nrow(X_trunk1)
  stats$mean <- colMeans(X_trunk1)
  stats$cov <- cov(X_trunk1)
  
  qr_result <- qr(stats$cov)
  projection_matrix <- qr.Q(qr_result)[, 1:min(ncol(X_trunk1), stats$n - 1), drop = FALSE]
  X_transformed_trunk1 <- X_trunk1 %*% projection_matrix
  
  class_weights <- get_class_weights(y_trunk1)
  svm_model <- e1071::svm(X_transformed_trunk1, as.factor(y_trunk1),
                          kernel = "radial", gamma = gamma_manual, cost = cost_manual,
                          probability = TRUE, class.weights = class_weights)
  
  all_transformed_X <- X_transformed_trunk1
  all_y <- y_trunk1
  
  # === Trunks 2 to 10 ===
  for (trunk_idx in 2:num_trunks) {
    start_idx <- (trunk_idx - 1) * trunk_size + 1
    end_idx <- ifelse(trunk_idx == num_trunks, num_samples, trunk_idx * trunk_size)
    
    X_trunk <- X_train[start_idx:end_idx, ]
    y_trunk <- y_train[start_idx:end_idx]
    
    if (length(unique(y_trunk)) < 2) {
      warning(paste("Trunk", trunk_idx, "has only one class; skipping."))
      next
    }
    
    # CPU usage tracking
    cpu_start <- proc.time()
    update_result <- incremental_qr_lda(stats$cov, stats$mean, stats$n, X_trunk,
                                        n_components_qr = min(ncol(X_trunk), stats$n - 1))
    cpu_end <- proc.time()
    
    cpu_time <- cpu_end - cpu_start
    w_update_time <- as.numeric(cpu_time["elapsed"])
    
    projection_matrix <- update_result$projection_qr
    stats$n <- update_result$updated_n_qr
    stats$mean <- update_result$updated_mean_qr
    stats$cov <- update_result$updated_cov_qr
    
    X_transformed_trunk <- X_trunk %*% projection_matrix
    all_transformed_X <- rbind(all_transformed_X, X_transformed_trunk)
    all_y <- c(all_y, y_trunk)
    
    class_weights <- get_class_weights(all_y)
    svm_model <- e1071::svm(all_transformed_X, as.factor(all_y),
                            kernel = "radial", gamma = gamma_manual, cost = cost_manual,
                            probability = TRUE, class.weights = class_weights)
    
    pred_eval <- predict(svm_model, X_transformed_trunk1)
    prob_scores_eval <- attr(predict(svm_model, X_transformed_trunk1, probability = TRUE), "probabilities")[, 2]
    
    metrics_row <- compute_metrics_qr(y_trunk1, pred_eval, prob_scores_eval)
    metrics_row$Training_Time_qr <- w_update_time
    metrics_row$CPU_User_Time_qr <- as.numeric(cpu_time["user.self"])
    metrics_row$CPU_System_Time_qr <- as.numeric(cpu_time["sys.self"])
    metrics_row$Batch <- trunk_idx
    
    results <- rbind(results, metrics_row)
  }
  
  # Save to CSV
  write.csv(results, "qr_incremental_results.csv", row.names = FALSE)
  
  return(list(
    results = results,
    final_model = svm_model,
    final_projection = projection_matrix,
    final_stats = stats
  ))
}

set.seed(42)
shuffle_idx <- sample(nrow(X_train_weighted_top70))
X_train_weighted_qr <- X_train_weighted_top70[shuffle_idx, ]
y_train_final_qr <- y_train_final[shuffle_idx]
y_train_final_qr_factor <- as.factor(y_train_final_qr)

batch_results_qr <- incremental_learning_pipeline_qr(
  as.matrix(X_train_weighted_qr),
  y_train_final_qr_factor,
  gamma_manual = best_gamma,
  cost_manual = best_cost
)

print(batch_results_qr$results)

write.csv(batch_results_qr$results, file = "results_qr_may_9_2.csv", row.names = FALSE)

if (require(ggplot2) && require(tidyr)) {
  # Prepare data
  results_long_qr <- tidyr::gather(batch_results_qr$results,
                                   key = "Metric", value = "Value",
                                   Accuracy_qr, Precision_qr, Recall_qr,
                                   F1_Score_qr, AUC_qr)
  
  # Plot metrics over batches
  plt_qr_metrics <- ggplot(results_long_qr, aes(x = Batch, y = Value, color = Metric)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    facet_wrap(~ Metric, scales = "free_y") +
    labs(title = "Incremental QR Performance Metrics",
         x = "Batch Number", y = "Metric Value") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "bottom")
  
  print(plt_qr_metrics)
  
} else {
  cat("Please install 'ggplot2' and 'tidyr' packages to generate plots.\n")
}

if (require(ggplot2) && require(caret)) {
  y_true_cm <- y_train_final_qr_factor[1:floor(nrow(X_train_weighted_qr)/10)]
  y_pred_cm <- predict(batch_results_qr$final_model,
                       X_train_weighted_qr[1:floor(nrow(X_train_weighted_qr)/10), ])
  
  cm <- caret::confusionMatrix(factor(y_pred_cm, levels = levels(y_true_cm)), y_true_cm)
  print(cm)
  
  cm_df <- as.data.frame(cm$table)
  ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), vjust = 0.5) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = "Confusion Matrix (QR)",
         fill = "Count") +
    theme_minimal()
}

if (require(pROC)) {
  y_true_roc <- as.numeric(as.character(y_train_final_qr[1:floor(nrow(X_train_weighted_qr)/10)]))
  probs_roc <- attr(predict(batch_results_qr$final_model,
                            X_train_weighted_qr[1:floor(nrow(X_train_weighted_qr)/10), ],
                            probability = TRUE), "probabilities")[, 2]
  
  roc_obj <- pROC::roc(y_true_roc, probs_roc)
  auc_val <- auc(roc_obj)
  
  roc_df <- data.frame(
    FPR = 1 - roc_obj$specificities,
    TPR = roc_obj$sensitivities
  )
  
  ggplot(roc_df, aes(x = FPR, y = TPR)) +
    geom_line(color = "blue", size = 1.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
    labs(title = paste("ROC Curve (QR)"),
         x = "False Positive Rate", y = "True Positive Rate") +
    theme_minimal()
}


if (require(PRROC)) {
  pr_obj <- pr.curve(scores.class0 = probs_roc[y_true_roc == 1],
                     scores.class1 = probs_roc[y_true_roc == 0],
                     curve = TRUE)
  
  pr_df <- data.frame(Recall = pr_obj$curve[, 1],
                      Precision = pr_obj$curve[, 2])
  
  ggplot(pr_df, aes(x = Recall, y = Precision)) +
    geom_line(color = "blue", size = 1.2) +
    geom_hline(yintercept = mean(y_true_roc), linetype = "dashed", color = "red") +
    labs(title = paste("Precision-Recall Curve (QR)"),
         x = "Recall", y = "Precision") +
    theme_minimal()
}

ggplot(batch_results_qr$results, aes(x = Batch, y = Training_Time_qr)) +
  geom_line(color = "darkred", size = 1) +
  geom_point(color = "darkred", size = 2) +
  labs(title = "Training Time per Batch (Incremental QR-LDA + SVM)",
       x = "Batch Number", y = "Training Time (seconds)") +
  theme_minimal(base_size = 14)


metrics_long <- melt(batch_results_qr$results,
                     id.vars = "Batch",
                     measure.vars = c("Accuracy_qr", "F1_Score_qr", "AUC_qr"),
                     variable.name = "Metric", value.name = "Value")

ggplot(metrics_long, aes(x = Batch, y = Value, color = Metric)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Batch-wise Evolution of Model Metrics",
       x = "Batch Number", y = "Metric Value") +
  theme_minimal()

# Compute cumulative samples if needed
trunk_size <- floor(nrow(X_train_weighted_qr) / 10)
batch_results_qr$results$Cumulative_Size <- batch_results_qr$results$Batch * trunk_size

# Plot accuracy over samples
ggplot(batch_results_qr$results, aes(x = Cumulative_Size, y = Accuracy_qr)) +
  geom_line(color = "blue", size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Learning Curve: Trunk Size vs Accuracy",
       x = "Samples Seen", y = "Accuracy") +
  theme_minimal()

final_model_qr <- batch_results_qr$final_model
final_projection_qr <- batch_results_qr$final_projection

best_gamma_test = 0.125
best_cost_test = 16

# ========== Helper: Compute Class Weights ==========
get_class_weights <- function(y) {
  tbl <- table(y)
  total <- sum(tbl)
  weights <- total / (length(tbl) * tbl)
  weights_named <- setNames(as.list(weights), names(weights))
  return(weights_named)
}

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

# ========== Helper: Train SVM Model ==========
train_svm_model <- function(X, y, gamma, cost) {
  if (length(unique(y)) < 2) {
    warning("Only one class in training data. Skipping SVM training.")
    return(NULL)
  }
  
  class_weights <- get_class_weights(y)
  
  e1071::svm(
    X, as.factor(y),
    kernel = "radial",
    gamma = gamma,
    cost = cost,
    probability = TRUE,
    class.weights = class_weights
  )
}

# ========== Helper: Evaluate Model ==========
evaluate_model_qr <- function(model, X_eval, y_eval, projection_matrix) {
  if (ncol(X_eval) != nrow(projection_matrix)) {
    stop("Dimension mismatch: can't project X_eval with current projection matrix.")
  }
  if (is.null(model)) {
    warning("Cannot evaluate: model is NULL.")
    return(data.frame())
  }
  
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
  
  total_samples <- nrow(X_test_full)
  batch_size_process <- floor(0.03 * total_samples)
  num_batches <- 10
  
  if (verbose) {
    cat("Total test samples:", total_samples, "\n")
    cat("Batch size (3%):", batch_size_process, "\n")
    cat("Number of batches:", num_batches, "\n")
  }
  
  X_eval <- X_test_full[1:batch_size_process, , drop = FALSE]
  y_eval <- y_test_full[1:batch_size_process]
  y_eval_factor <- as.factor(y_eval)
  overall_preds <- NULL
  overall_proba <- NULL
  overall_actual <- as.numeric(as.character(y_eval_factor))
  
  results <- data.frame()
  svm_model <- initial_model
  projection_matrix <- initial_proj
  stats <- initial_stats
  
  for (batch_idx in 1:num_batches) {
    if (verbose) cat("Batch", batch_idx, "...\n")
    
    start_idx <- (batch_idx - 1) * batch_size_process + 1
    end_idx <- min(batch_idx * batch_size_process, total_samples)
    if (start_idx > total_samples) break
    
    X_batch_process <- X_test_full[start_idx:end_idx, ]
    y_batch_process <- y_test_full[start_idx:end_idx]
    
    # --- Measure QR (W) Update Time ---
    wall_start <- Sys.time()
    cpu_start <- proc.time()
    
    qr_update <- update_qr_stats(stats, X_batch_process, n_components_qr = ncol(X_batch_process))
    
    cpu_end <- proc.time()
    wall_end <- Sys.time()
    
    cpu_time <- cpu_end - cpu_start
    wall_elapsed <- as.numeric(difftime(wall_end, wall_start, units = "secs"))
    
    projection_matrix <- qr_update$projection
    stats <- qr_update$stats
    
    # --- Train SVM ---
    X_train_new <- rbind(X_eval, X_batch_process)
    y_train_new <- c(y_eval, y_batch_process)
    X_train_transformed <- X_train_new %*% projection_matrix
    
    svm_model <- train_svm_model(X_train_transformed, y_train_new, gamma_manual, cost_manual)
    
    if (!is.null(svm_model)) {
      # --- Evaluate on fixed evaluation set ---
      metrics_row <- evaluate_model_qr(svm_model, X_eval, y_eval, projection_matrix)
      
      # --- Store timing and batch info ---
      metrics_row$W_Update_Time_Sec <- wall_elapsed
      metrics_row$CPU_User_Time_qr <- as.numeric(cpu_time["user.self"])
      metrics_row$CPU_System_Time_qr <- as.numeric(cpu_time["sys.self"])
      metrics_row$Batch <- batch_idx
      
      results <- rbind(results, metrics_row)
      
      if (batch_idx == 1) {
        eval_proj <- X_eval %*% projection_matrix
        overall_preds <- predict(svm_model, eval_proj)
        overall_proba <- attr(predict(svm_model, eval_proj, probability = TRUE), "probabilities")[, 2]
      }
    } else {
      warning(paste("Skipping evaluation at batch", batch_idx, "due to invalid model."))
    }
  }
  
  return(list(
    results = results,
    overall_preds = overall_preds,
    overall_proba = overall_proba,
    overall_actual = overall_actual,
    y_eval_factor = y_eval_factor
  ))
}

# ========== Run Process on Test Data ==========

# Apply feature weighting
X_test_weighted_qr <- sweep(X_test, 2, W_vec, `*`)

# Select top features
X_test_selected_qr <- X_test_weighted_qr[, top_features]

# Convert to matrix
X_test_matrix <- as.matrix(X_test_selected_qr)

# Run incremental QR on test set
incremental_results_test_qr_list <- incremental_qr_on_test(
  X_test_matrix, as.factor(y_test),
  initial_model = final_model_qr,
  initial_proj = final_projection_qr,
  initial_stats = batch_results_qr$final_stats,
  gamma_manual = best_gamma_test,
  cost_manual = best_cost_test
)

# Extract results
incremental_results_test_qr <- incremental_results_test_qr_list$results
overall_preds_test_qr <- incremental_results_test_qr_list$overall_preds
overall_proba_test_qr <- incremental_results_test_qr_list$overall_proba
overall_actual_test_qr <- incremental_results_test_qr_list$overall_actual
y_eval_factor_test_qr <- incremental_results_test_qr_list$y_eval_factor

# View results
print(incremental_results_test_qr)


# ========== Save to CSV ==========
write.csv(incremental_results_test_qr, "incremental_qr_test_results.csv", row.names = FALSE)


# === Plotting Performance Metrics vs. Batch on Test Data ===
if (require(ggplot2) && require(tidyr) && require(caret) && require(pROC) && require(PRROC)) {
  results_long_test_qr <- gather(incremental_results_test_qr, key = "Metric", value = "Value",
                                 Accuracy_qr, AUC_qr, Precision_qr, Recall_qr, F1_Score_qr)
  
  plt_metrics_test_qr <- ggplot(results_long_test_qr, aes(x = Batch, y = Value, color = Metric)) +
    geom_line() +
    geom_point() +
    facet_wrap(~ Metric, scales = "free_y") +
    labs(title = "Performance Metrics (Incremental QR on Test Data)",
         x = "Batch", y = "Value", color = "Metric") +
    theme_minimal() +
    theme(legend.position = "bottom")
  print(plt_metrics_test_qr)
  
  # ----- Confusion Matrix Visualization (Overall Test Set) -----
  if (!is.null(overall_preds_test_qr) && length(overall_preds_test_qr) == length(y_eval_factor_test_qr)) {
    cm_overall_test_qr <- caret::confusionMatrix(overall_preds_test_qr, y_eval_factor_test_qr)
    cm_df_test_qr <- as.data.frame(cm_overall_test_qr$table)
    plt_cm_test_qr <- ggplot(data = cm_df_test_qr, aes(x = Prediction, y = Reference, fill = Freq)) +
      geom_tile() +
      scale_fill_gradient(low = "white", high = "steelblue") +
      geom_text(aes(label = Freq), vjust = 0.5) +
      labs(title = "Confusion Matrix (Test Data - Incremental QR)", fill = "Frequency") +
      theme_minimal()
    print(plt_cm_test_qr)
  } else {
    cat("Warning: Cannot plot confusion matrix as prediction and true label lengths do not match or predictions are NULL.\n")
  }
  
  # ----- ROC Curve (Overall Test Set) -----
  if (!is.null(overall_proba_test_qr) && length(unique(overall_actual_test_qr)) > 1 &&
      length(overall_proba_test_qr) == length(overall_actual_test_qr)) {
    roc_obj_test_qr <- roc(overall_actual_test_qr, overall_proba_test_qr)
    plt_roc_test_qr <- ggplot(data.frame(FPR = 1 - roc_obj_test_qr$specificities, TPR = roc_obj_test_qr$sensitivities),
                              aes(x = FPR, y = TPR)) +
      geom_line(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      labs(title = "ROC Curve (Test Data - Incremental QR)",
           x = "False Positive Rate", y = "True Positive Rate",
           caption = paste("AUC =", round(auc(roc_obj_test_qr), 3))) +
      theme_minimal()
    print(plt_roc_test_qr)
  } else {
    cat("Warning: Cannot plot ROC curve as probabilities are NULL, true labels have one unique value, or lengths do not match.\n")
  }
  
  # ----- Precision-Recall Curve (Overall Test Set) -----
  if (!is.null(overall_proba_test_qr) && require(PRROC) && length(unique(overall_actual_test_qr)) > 1 &&
      length(overall_proba_test_qr) == length(overall_actual_test_qr)) {
    pr_data_test_qr <- pr.curve(overall_actual_test_qr, overall_proba_test_qr, curve = TRUE)
    plt_pr_test_qr <- ggplot(data.frame(Recall = pr_data_test_qr$curve[, 1], Precision = pr_data_test_qr$curve[, 2]),
                             aes(x = Recall, y = Precision)) +
      geom_line(color = "blue") +
      geom_hline(yintercept = mean(overall_actual_test_qr), linetype = "dashed", color = "red") + # Baseline
      labs(title = "Precision-Recall Curve (Test Data - Incremental QR)",
           x = "Recall", y = "Precision",
           caption = paste("AUC-PR =", round(pr_data_test_qr$auc.integral, 3))) +
      theme_minimal()
    print(plt_pr_test_qr)
  } else if (!require(PRROC)) {
    cat("Please install the 'PRROC' package to plot Precision-Recall curves.\n")
  } else {
    cat("Warning: Cannot plot Precision-Recall curve as probabilities are NULL, true labels have one unique value, or lengths do not match.\n")
  }
  
} else {
  cat("Please install 'ggplot2', 'tidyr', 'caret', 'pROC', and 'PRROC' packages to plot.\n")
}



# === Prepare Test Data with Top 70 Features ===
X_test_weighted <- sweep(X_test, 2, W_vec, `*`)
X_test_selected <- X_test_weighted[, top_70_indices]
y_test <- as.factor(y_test)

# === Split Test Data into 10 Batches ===
batches <- split(1:nrow(X_test_selected), cut(1:nrow(X_test_selected), breaks = 10, labels = FALSE))

# === Initialize Results DataFrame with Detailed Metrics ===
test_results_top70 <- data.frame(
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

# === Storage for Overall Predictions and Labels ===
all_probabilities_top70 <- numeric()
all_true_labels_top70 <- numeric()
all_predictions_top70 <- factor(levels = levels(y_test))

# === Start with Initial Training Data ===
X_current_train <- X_train_final_selected
y_current_train <- factor(y_train_final, levels = levels(y_test))

# === Batch-wise Evaluation ===
for (i in seq_along(batches)) {
  cat("Processing Test Batch", i, "\n")
  
  idx <- batches[[i]]
  X_batch <- X_test_selected[idx, ]
  y_batch <- y_test[idx]
  
  # --- Train SVM ---
  cpu_start_train <- proc.time()
  svm_model_top70_final <- svm(X_current_train, y_current_train,
                               kernel = "radial",
                               gamma = best_gamma,
                               cost = best_cost,
                               probability = TRUE)
  cpu_end_train <- proc.time()
  training_cpu_time <- (cpu_end_train - cpu_start_train)[["user.self"]] +
    (cpu_end_train - cpu_start_train)[["sys.self"]]
  
  # --- Inference ---
  start_time <- Sys.time()
  cpu_start_infer <- proc.time()
  preds <- predict(svm_model_top70_final, X_batch)
  preds_proba <- attr(predict(svm_model_top70_final, X_batch, probability = TRUE), "probabilities")[, 2]
  cpu_end_infer <- proc.time()
  end_time <- Sys.time()
  
  inference_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  inference_cpu_time <- (cpu_end_infer - cpu_start_infer)[["user.self"]] +
    (cpu_end_infer - cpu_start_infer)[["sys.self"]]
  
  preds <- factor(preds, levels = levels(y_test))  # Ensure consistency
  
  # --- Confusion Matrix & Metrics ---
  cm <- table(Predicted = preds, Actual = y_batch)
  TP <- ifelse("1" %in% rownames(cm) & "1" %in% colnames(cm), cm["1", "1"], 0)
  TN <- ifelse("0" %in% rownames(cm) & "0" %in% colnames(cm), cm["0", "0"], 0)
  FP <- ifelse("1" %in% rownames(cm) & "0" %in% colnames(cm), cm["1", "0"], 0)
  FN <- ifelse("0" %in% rownames(cm) & "1" %in% colnames(cm), cm["0", "1"], 0)
  
  accuracy <- tryCatch({ mean(preds == y_batch) }, error = function(e) NA)
  auc_val <- tryCatch({ auc(as.numeric(as.character(y_batch)), preds_proba) }, error = function(e) NA)
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  recall <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  # --- Append Results ---
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
  
  test_results_top70 <- rbind(test_results_top70, new_row)
  
  # --- Store Predictions for Overall Analysis ---
  all_probabilities_top70 <- c(all_probabilities_top70, preds_proba)
  all_true_labels_top70 <- c(all_true_labels_top70, as.numeric(as.character(y_batch)))
  all_predictions_top70 <- factor(c(as.character(all_predictions_top70), as.character(preds)), 
                                  levels = levels(y_test))
  
  # --- Add Batch to Training Set for Next Iteration ---
  X_current_train <- rbind(X_current_train, X_batch)
  y_current_train <- factor(c(as.character(y_current_train), as.character(y_batch)), 
                            levels = levels(y_test))
}

# === View Results ===
print(test_results_top70)
summary(test_results_top70)

# === Save to CSV ===
write.csv(test_results_top70, file = "results_test_top_70_may_9_3.csv", row.names = FALSE)


# === Prepare Test Data with Top 70 Features ===
X_test_weighted <- sweep(X_test, 2, W_vec, `*`)
X_test_selected <- X_test_weighted[, top_70_indices]
y_test <- as.factor(y_test)

# === Split Test Data into 10 Batches ===
batches <- split(1:nrow(X_test_selected), cut(1:nrow(X_test_selected), breaks = 10, labels = FALSE))

# === Initialize Results DataFrame with Detailed Metrics ===
test_results_top70 <- data.frame(
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

# === Storage for Overall Predictions and Labels ===
all_probabilities_top70 <- numeric()
all_true_labels_top70 <- numeric()
all_predictions_top70 <- factor(levels = levels(y_test))

# === Start with Initial Training Data ===
X_current_train <- X_train_final_selected
y_current_train <- factor(y_train_final, levels = levels(y_test))

# === Batch-wise Evaluation ===
for (i in seq_along(batches)) {
  cat("Processing Test Batch", i, "\n")
  
  idx <- batches[[i]]
  X_batch <- X_test_selected[idx, ]
  y_batch <- y_test[idx]
  
  if (length(unique(y_batch)) < 2) {
    warning(paste("Skipping batch", i, "- only one class present"))
    next
  }
  
  # --- Train SVM ---
  cpu_start_train <- proc.time()
  svm_model_top70_final <- svm(X_current_train, y_current_train,
                               kernel = "radial",
                               gamma = best_gamma,
                               cost = best_cost,
                               probability = TRUE)
  cpu_end_train <- proc.time()
  training_cpu_time <- (cpu_end_train - cpu_start_train)[["user.self"]] +
    (cpu_end_train - cpu_start_train)[["sys.self"]]
  
  # --- Inference ---
  start_time <- Sys.time()
  cpu_start_infer <- proc.time()
  preds <- predict(svm_model_top70_final, X_batch)
  preds_proba <- attr(predict(svm_model_top70_final, X_batch, probability = TRUE), "probabilities")[, 2]
  cpu_end_infer <- proc.time()
  end_time <- Sys.time()
  
  inference_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  inference_cpu_time <- (cpu_end_infer - cpu_start_infer)[["user.self"]] +
    (cpu_end_infer - cpu_start_infer)[["sys.self"]]
  
  preds <- factor(preds, levels = levels(y_test))
  
  # --- Confusion Matrix & Metrics ---
  cm <- table(Predicted = preds, Actual = y_batch)
  TP <- ifelse("1" %in% rownames(cm) & "1" %in% colnames(cm), cm["1", "1"], 0)
  TN <- ifelse("0" %in% rownames(cm) & "0" %in% colnames(cm), cm["0", "0"], 0)
  FP <- ifelse("1" %in% rownames(cm) & "0" %in% colnames(cm), cm["1", "0"], 0)
  FN <- ifelse("0" %in% rownames(cm) & "1" %in% colnames(cm), cm["0", "1"], 0)
  
  accuracy <- tryCatch({ mean(preds == y_batch) }, error = function(e) NA)
  auc_val <- tryCatch({ auc(as.numeric(as.character(y_batch)), preds_proba) }, error = function(e) NA)
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  recall <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  # --- Append Results ---
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
  
  test_results_top70 <- rbind(test_results_top70, new_row)
  
  all_probabilities_top70 <- c(all_probabilities_top70, preds_proba)
  all_true_labels_top70 <- c(all_true_labels_top70, as.numeric(as.character(y_batch)))
  all_predictions_top70 <- factor(c(as.character(all_predictions_top70), as.character(preds)), 
                                  levels = levels(y_test))
  
  X_current_train <- rbind(X_current_train, X_batch)
  y_current_train <- factor(c(as.character(y_current_train), as.character(y_batch)), 
                            levels = levels(y_test))
}

# === Plotting Performance Metrics ===
if (require(ggplot2) && require(tidyr)) {
  results_long_test_top70 <- gather(test_results_top70, key = "Metric", value = "Value",
                                    Accuracy, AUC, Precision, Recall, F1_Score)
  
  plt_metrics_test_top70 <- ggplot(results_long_test_top70, aes(x = Batch, y = Value, color = Metric)) +
    geom_line() +
    geom_point() +
    facet_wrap(~ Metric, scales = "free_y") +
    labs(title = "Performance Metrics (Test Data - Top 70 Features)",
         x = "Batch", y = "Value", color = "Metric") +
    theme_minimal() +
    theme(legend.position = "bottom")
  print(plt_metrics_test_top70)
} else {
  cat("Please install 'ggplot2' and 'tidyr' packages to plot.\n")
}

# === Overall Evaluation Metrics ===
if (require(caret) && require(pROC) && require(PRROC)) {
  cat("\n=== Overall Evaluation Metrics (Across All Batches) ===\n")
  
  cm_overall <- caret::confusionMatrix(all_predictions_top70, as.factor(all_true_labels_top70))
  print("Confusion Matrix:")
  print(cm_overall)
  
  cm_df <- as.data.frame(cm_overall$table)
  plt_cm <- ggplot(data = cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), vjust = 0.5) +
    labs(title = "Confusion Matrix (Test Data)", fill = "Frequency") +
    theme_minimal()
  print(plt_cm)
  
  if (length(unique(all_true_labels_top70)) > 1 && length(unique(all_probabilities_top70)) > 1) {
    roc_obj <- roc(as.numeric(as.character(all_true_labels_top70)), all_probabilities_top70)
    auc_overall <- auc(roc_obj)
    cat("\nAUC (Test Data):", auc_overall, "\n")
    
    roc_df <- data.frame(FPR = 1 - roc_obj$specificities, TPR = roc_obj$sensitivities)
    plt_roc <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
      geom_line(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      labs(title = "ROC Curve (Test Data)",
           x = "False Positive Rate", y = "True Positive Rate",
           caption = paste("AUC =", round(auc_overall, 3))) +
      theme_minimal()
    print(plt_roc)
  }
  
  if (length(unique(all_true_labels_top70)) > 1 && length(unique(all_probabilities_top70)) > 1) {
    pr_obj <- pr.curve(scores.class0 = all_probabilities_top70, weights.class0 = all_true_labels_top70, curve = TRUE)
    auc_pr_overall <- pr_obj$auc.integral
    cat("AUC-PR (Overall):", auc_pr_overall, "\n")
    
    pr_df <- data.frame(Recall = pr_obj$curve[, 1], Precision = pr_obj$curve[, 2])
    plt_pr <- ggplot(pr_df, aes(x = Recall, y = Precision)) +
      geom_line(color = "blue") +
      geom_hline(yintercept = mean(as.numeric(as.character(all_true_labels_top70))), linetype = "dashed", color = "red") +
      labs(title = "Precision-Recall Curve (Test Data)",
           x = "Recall", y = "Precision",
           caption = paste("AUC-PR =", round(auc_pr_overall, 3))) +
      theme_minimal()
    print(plt_pr)
  }
  
} else {
  cat("\nPlease install 'caret', 'pROC', and 'PRROC' packages for overall evaluation metrics and plots.\n")
}


