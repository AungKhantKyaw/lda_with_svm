# Install necessary packages (if not already installed)
packages <- c("caret", "ggplot2", "e1071", "pROC", "MASS", "bigstatsr", 
              "irlba", "MLmetrics", "dplyr", "arm", "RSpectra", "tm", "PRROC", 
              "smotefamily", "tidyr")

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
library(tm)
library(PRROC)
library(slam)
library(smotefamily)
library(tidyr)

# ===== LOAD DATA =====
df <- read.csv("Dataset_5971.csv", stringsAsFactors = FALSE)

# Convert LABEL to binary target
df$target_binary <- ifelse(tolower(df$LABEL) == "ham", 0, 1)



# Convert URL/EMAIL/PHONE from "yes"/"no" to 1/0
convert_to_numeric <- function(x) as.numeric(tolower(x) == "yes")
df$URL <- convert_to_numeric(df$URL)
#df$EMAIL <- convert_to_numeric(df$EMAIL)
df$PHONE <- convert_to_numeric(df$PHONE)

# ===== TEXT CLEANING =====


corpus <- VCorpus(VectorSource(df$TEXT))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)

# Create TF-IDF weighted DTM
dtm <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))
dtm <- removeSparseTerms(dtm, 0.97)

# ===== REMOVE EMPTY DOCUMENTS =====
row_totals <- slam::row_sums(dtm)
non_empty_rows <- row_totals > 0

# Filter DTM and dataframe
dtm <- dtm[non_empty_rows, ]
df <- df[non_empty_rows, ]

# ===== FEATURE CONSTRUCTION =====
X_dtm <- as.data.frame(as.matrix(dtm))

# Add back engineered features (already numeric)
X_dtm$URL <- df$URL
X_dtm$PHONE <- df$PHONE
#X_dtm$EMAIL <- df$EMAIL

# Scale features
X_scaled <- as.data.frame(scale(X_dtm))

# Target vector
y_clean <- df$target_binary

# Bar plot for binary target
ggplot(df, aes(x = factor(target_binary))) +
  geom_bar(fill = c("lightblue", "salmon")) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  labs(title = "Distribution of Phishing vs. Non-Phishing Messages",
       x = "Label (0 = Ham, 1 = Phishing)",
       y = "Number of Messages") +
  theme_minimal()

# ===== TRAIN/TEST SPLIT =====
set.seed(123)

# Split the data into train and test sets (70% train, 30% test)
train_indices <- sample(seq_len(nrow(X_scaled)), size = floor(0.7 * nrow(X_scaled)))
X_train <- X_scaled[train_indices, ]
y_train <- y_clean[train_indices]
X_test <- X_scaled[-train_indices, ]
y_test <- y_clean[-train_indices]

# ===== TRAIN/VALID SPLIT =====
train_indices_final <- sample(seq_len(nrow(X_train)), size = floor(0.8 * nrow(X_train)))
X_train_final <- X_train[train_indices_final, ]
y_train_final <- y_train[train_indices_final]
X_valid <- X_train[-train_indices_final, ]
y_valid <- y_train[-train_indices_final]

# Ensure both training and validation sets have the same features
# This step ensures that no feature is missing from either set
X_valid <- X_valid[, colnames(X_train_final), drop = FALSE]


# ===== SVM Cross-Validation WITHOUT LDA =====
set.seed(123)
num_folds <- 10
folds <- createFolds(y_train_final, k = num_folds)

results_no_lda <- data.frame(Fold = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(), 
                             Accuracy = numeric(), AUC = numeric(), Precision = numeric(), 
                             Recall = numeric(), F1_Score = numeric(), Training_Time = numeric())

for (i in seq_along(folds)) {
  cat("Processing Fold", i, "\n")
  
  # Fold split
  test_idx <- folds[[i]]
  train_idx <- setdiff(seq_len(nrow(X_train_final)), test_idx)
  
  X_train_fold <- X_train_final[train_idx, ]
  y_train_fold <- y_train_final[train_idx]
  X_test_fold  <- X_train_final[test_idx, ]
  y_test_fold  <- y_train_final[test_idx]
  
  # Ensure valid rows
  valid_train <- complete.cases(X_train_fold) & apply(X_train_fold, 1, function(row) all(is.finite(row)))
  valid_test  <- complete.cases(X_test_fold)  & apply(X_test_fold, 1, function(row) all(is.finite(row)))
  
  X_train_fold <- X_train_fold[valid_train, , drop = FALSE]
  y_train_fold <- y_train_fold[valid_train]
  X_test_fold  <- X_test_fold[valid_test, , drop = FALSE]
  y_test_fold  <- y_test_fold[valid_test]
  
  if (nrow(X_train_fold) < 5 || length(unique(y_train_fold)) < 2) {
    cat("Skipping Fold", i, ": insufficient data or only one class.\n")
    next
  }
  
  # Train SVM
  start_time <- Sys.time()
  svm_model <- svm(X_train_fold, as.factor(y_train_fold), kernel = "radial", probability = TRUE)
  end_time <- Sys.time()
  training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  # Predict
  preds <- predict(svm_model, X_test_fold)
  pred_probs <- attr(predict(svm_model, X_test_fold, probability = TRUE), "probabilities")[, 2]
  
  # Metrics
  conf <- confusionMatrix(preds, as.factor(y_test_fold), positive = "1")
  auc_score <- auc(y_test_fold, pred_probs)
  
  results_no_lda <- rbind(results_no_lda, data.frame(
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

print(results_no_lda)
summary(results_no_lda)

# ===================================================
# ----- SVM WITH LDA -----
# ===================================================
set.seed(123)
folds <- createFolds(y_train_final, k = 10, returnTrain = FALSE)

# Initialize the result dataframe to store metrics
results_lda <- data.frame(Fold = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(),
                          Accuracy = numeric(), AUC = numeric(), Precision = numeric(),
                          Recall = numeric(), F1_Score = numeric(), Training_Time = numeric())

# Initialize empty vectors to store overall predictions, probabilities, and actuals
all_preds_lda <- c()
all_proba_lda <- c()
all_actual_lda <- c()  # Initialize an empty vector for actual labels

for (i in seq_along(folds)) {
  cat("Processing Fold", i, "\n")
  
  train_idx <- unlist(folds[-i])
  test_idx <- unlist(folds[i])
  
  # Train and test data for this fold
  X_train_fold <- X_train_final[train_idx, ]
  y_train_fold <- y_train_final[train_idx]
  X_test_fold <- X_train_final[test_idx, ]
  y_test_fold <- y_train_final[test_idx]
  
  # Remove constant features (no variance)
  non_constant_cols <- sapply(X_train_fold, function(col) var(col, na.rm = TRUE) != 0)
  X_train_fold <- X_train_fold[, non_constant_cols, drop = FALSE]
  X_test_fold <- X_test_fold[, non_constant_cols, drop = FALSE]
  
  # Remove highly correlated features (correlation > 0.99)
  cor_matrix <- cor(X_train_fold)
  high_corr <- findCorrelation(cor_matrix, cutoff = 0.99)
  if (length(high_corr) > 0) {
    X_train_fold <- X_train_fold[, -high_corr, drop = FALSE]
    X_test_fold <- X_test_fold[, -high_corr, drop = FALSE]
  }
  
  # ===== Apply SMOTE using smotefamily =====
  y_train_fold_num <- as.numeric(as.character(y_train_fold))  # Convert to numeric for SMOTE
  smote_out <- smotefamily::SMOTE(X_train_fold, y_train_fold_num, K = 5, dup_size = 2)
  
  X_train_fold <- smote_out$data[, -ncol(smote_out$data)]
  y_train_fold <- as.factor(smote_out$data$class)
  
  # ===== Fit LDA =====
  lda_model <- lda(y_train_fold ~ ., data = data.frame(y_train_fold, X_train_fold))
  X_train_lda <- as.matrix(X_train_fold) %*% lda_model$scaling
  X_test_lda <- as.matrix(X_test_fold) %*% lda_model$scaling
  
  # ===== Train SVM =====
  start_time <- Sys.time()
  svm_model_lda <- svm(X_train_lda, as.factor(y_train_fold), kernel = "radial", probability = TRUE)
  end_time <- Sys.time()
  training_time_lda <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  # Make predictions and get probabilities
  pred_lda <- predict(svm_model_lda, X_test_lda)
  pred_proba_lda <- attr(predict(svm_model_lda, X_test_lda, probability = TRUE), "probabilities")[, 2]
  
  # Confusion matrix for performance metrics
  conf_matrix_lda <- confusionMatrix(pred_lda, as.factor(y_test_fold))
  
  # Store results for the current fold
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
  
  # Store predictions, probabilities, and actual labels across all folds
  all_preds_lda <- c(all_preds_lda, as.character(pred_lda))
  all_proba_lda <- c(all_proba_lda, pred_proba_lda)
  all_actual_lda <- c(all_actual_lda, as.character(y_test_fold))
}

# Print results
print(results_lda)
summary(results_lda)

# Check structure and lengths of all predictions and actual labels
cat("Structure of all_preds_lda:\n")
str(all_preds_lda)

cat("Structure of all_actual_lda:\n")
str(all_actual_lda)

cat("Length of all_preds_lda:", length(all_preds_lda), "\n")
cat("Length of all_actual_lda:", length(all_actual_lda), "\n")
cat("Length of all_proba_lda:", length(all_proba_lda), "\n")
cat("Levels in actual:", levels(factor(all_actual_lda)), "\n")

# Check if predictions and actual labels are valid
cat("Unique values in predicted labels:", unique(all_preds_lda), "\n")
cat("Unique values in actual labels:", unique(all_actual_lda), "\n")

# Convert predictions and actual labels to factors with appropriate levels
# Use unique values from all_preds_lda and all_actual_lda to set levels, ensuring they overlap
all_preds_lda_factor <- factor(all_preds_lda, levels = c("0", "1"))
all_actual_lda_factor <- factor(all_actual_lda, levels = c("0", "1"))

# Check levels after conversion
cat("Levels in predicted labels after conversion:", levels(all_preds_lda_factor), "\n")
cat("Levels in actual labels after conversion:", levels(all_actual_lda_factor), "\n")

# Calculate confusion matrix again
cm_overall_lda <- caret::confusionMatrix(all_preds_lda_factor, all_actual_lda_factor)
cm_df_lda <- as.data.frame(cm_overall_lda$table)

# ----- Plot the Confusion Matrix -----
plt_cm_lda <- ggplot(data = cm_df_lda, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(label = Freq), vjust = 0.5) +
  labs(title = "Confusion Matrix (SVM with LDA - Overall)", fill = "Frequency") +
  theme_minimal()
print(plt_cm_lda)

# ----- ROC Curve (Overall) -----
if (length(unique(all_actual_lda_factor)) > 1) {
  # Create the ROC object
  roc_obj_lda <- roc(all_actual_lda_factor, as.numeric(all_proba_lda))
  
  # Plot ROC Curve
  plt_roc_lda <- ggplot(data.frame(FPR = 1 - roc_obj_lda$specificities, TPR = roc_obj_lda$sensitivities),
                        aes(x = FPR, y = TPR)) +
    geom_line(color = "blue") +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
    labs(title = "ROC Curve (SVM with LDA - Overall)",
         x = "False Positive Rate", y = "True Positive Rate",
         caption = paste("AUC =", round(auc(roc_obj_lda), 3))) +
    theme_minimal()
  print(plt_roc_lda)
} else {
  cat("Warning: Cannot plot ROC curve as all true labels are the same.\n")
}

# ----- Precision-Recall Curve (Overall) -----
if (length(unique(all_actual_lda_factor)) > 1) {
  # Create Precision-Recall curve
  pr_data_lda <- pr.curve(scores.class0 = all_proba_lda, weights.class0 = as.numeric(all_actual_lda_factor) - 1, curve = TRUE)
  
  # Plot Precision-Recall Curve
  plt_pr_lda <- ggplot(data.frame(Recall = pr_data_lda$curve[, 1], Precision = pr_data_lda$curve[, 2]),
                       aes(x = Recall, y = Precision)) +
    geom_line(color = "blue") +
    geom_hline(yintercept = mean(as.numeric(all_actual_lda_factor) == 1), linetype = "dashed", color = "red") + # Baseline
    labs(title = "Precision-Recall Curve (SVM with LDA - Overall)",
         x = "Recall", y = "Precision",
         caption = paste("AUC-PR =", round(pr_data_lda$auc.integral, 3))) +
    theme_minimal()
  print(plt_pr_lda)
} else {
  cat("Warning: Cannot plot Precision-Recall curve as all true labels are the same.\n")
}

# ===================================================
# ----- FEATURE IMPORTANCE FROM LDA -----
# ===================================================

importance <- lda_model$scaling
importance_df <- data.frame(Feature = rownames(importance),
                            Importance = abs(importance[, 1]))  # LD1 only (for binary)
top_n <- 20  # or any number you want
importance_df <- importance_df[order(-importance_df$Importance), ]
top_features <- head(importance_df, top_n)

ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 Features LD",
       x = "Feature", y = "Importance (|coefficient|)") +
  theme_minimal()


# Fit LDA on full training set to get feature importances
lda_model <- lda(y_train_final ~ ., data = data.frame(y_train_final, X_train_final))

# Get LDA coefficients (feature weights)
W <- lda_model$scaling
W_vec <- as.vector(W)

# Weight original features using LDA coefficients
X_train_weighted <- sweep(X_train_final, 2, W_vec, `*`)
X_valid_weighted <- sweep(X_valid, 2, W_vec, `*`)


# ncol(X_weighted_data)
# length(W_vec)
# 
# sum(is.na(X_weighted_data))

########### TUNE ##########################
tuned <- tune.svm(x = X_train_final, y = as.factor(y_train_final),
                  kernel = "radial",
                  gamma = 2^(-5:2), cost = 2^(-1:4))

best_model <- tuned$best.model

best_gamma <- best_model$gamma
best_cost <- best_model$cost

cat("Best Gamma:", best_gamma, "\n")
cat("Best Cost:", best_cost, "\n")

###### FIND THE BEST FEATURE TO SELECT ################
best_gamma = 0.03125 
best_cost = 4

# Setup parallel backend
cores <- parallel::detectCores() - 1
cl <- makeCluster(cores)
registerDoParallel(cl)

# Setup data and parameters
set.seed(123)
num_folds <- 10
folds <- createFolds(y_train_final, k = num_folds)
X_weighted_data <- as.matrix(X_train_weighted)
y_weighted_data <- y_train_final
feature_counts <- c(3, 4, 5, 8, 10, 14, 15, 18, 20, 21, 25, 30)
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
}

# Stop parallel backend
stopCluster(cl)

# ===================================================
# ----- SVM WITH LDA-WEIGHTED TOP 10 FEATURES -----
# ===================================================




# ===================================================
# ----- SVM WITH LDA-WEIGHTED TOP 30 FEATURES -----
# ===================================================

# Get top 10 features based on absolute LDA coefficients
top_features <- order(abs(W_vec), decreasing = TRUE)[1:30]

# Convert to matrix for CV loop
X_weighted_data <- as.matrix(X_train_weighted)
y_weighted_data <- y_train_final

set.seed(123)

num_folds <- 10
folds <- createFolds(y_weighted_data, k = num_folds, list = TRUE, returnTrain = FALSE)

# Initialize results container for Top 10 Features
results_top10 <- data.frame(Fold = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(),
                            Accuracy = numeric(), AUC = numeric(), Precision = numeric(),
                            Recall = numeric(), F1_Score = numeric(), Training_Time = numeric(),
                            User_CPU_Time = numeric(), System_CPU_Time = numeric())

# Accumulators for overall evaluation
all_preds_top10 <- c()
all_proba_top10 <- c()
all_actual_top10 <- c()

for (i in 1:num_folds) {
  
  cat("Processing Fold", i, "\n")
  
  valid_idx <- folds[[i]]
  
  X_train_fold <- X_weighted_data[-valid_idx, ]
  y_train_fold <- y_weighted_data[-valid_idx]
  
  X_valid_fold <- X_weighted_data[valid_idx, ]
  y_valid_fold <- y_weighted_data[valid_idx]
  
  # Select top 10 LDA-weighted features
  X_train_selected <- X_train_fold[, top_features]
  X_valid_selected <- X_valid_fold[, top_features]
  
  # Train SVM and measure CPU usage
  start_cpu <- proc.time()  # Capture CPU time before training
  start_time <- Sys.time()  # Capture wall-clock time
  svm_model_top10 <- svm(X_train_selected, as.factor(y_train_fold), kernel = "radial", probability = TRUE)
  end_time <- Sys.time()
  end_cpu <- proc.time()  # Capture CPU time after training
  
  # Calculate training time and CPU usage
  training_time_top10 <- as.numeric(difftime(end_time, start_time, units = "secs"))
  user_cpu_time <- (end_cpu - start_cpu)["user.self"]  # User CPU time
  system_cpu_time <- (end_cpu - start_cpu)["sys.self"]  # System CPU time
  
  pred_top10 <- predict(svm_model_top10, X_valid_selected)
  pred_proba_top10 <- attr(predict(svm_model_top10, X_valid_selected, probability = TRUE), "probabilities")[, 2]
  
  # Confusion matrix
  conf_matrix_top10 <- confusionMatrix(pred_top10, as.factor(y_valid_fold))
  
  # Collect metrics
  results_top10 <- rbind(results_top10, data.frame(
    Fold = i,
    TP = conf_matrix_top10$table["1", "1"],
    TN = conf_matrix_top10$table["0", "0"],
    FP = conf_matrix_top10$table["1", "0"],
    FN = conf_matrix_top10$table["0", "1"],
    Accuracy = conf_matrix_top10$overall["Accuracy"],
    AUC = auc(y_valid_fold, pred_proba_top10),
    Precision = conf_matrix_top10$byClass["Precision"],
    Recall = conf_matrix_top10$byClass["Recall"],
    F1_Score = conf_matrix_top10$byClass["F1"],
    Training_Time = training_time_top10,
    User_CPU_Time = user_cpu_time,
    System_CPU_Time = system_cpu_time
  ))
  
  # Accumulate all predictions and labels
  all_preds_top10 <- c(all_preds_top10, as.character(pred_top10))
  all_proba_top10 <- c(all_proba_top10, pred_proba_top10)
  all_actual_top10 <- c(all_actual_top10, as.character(y_valid_fold))
  
  # Output per-fold results
  cat(sprintf("Fold %d -> Accuracy: %.4f | AUC: %.4f | Precision: %.4f | Recall: %.4f | F1 Score: %.4f | User CPU: %.2f s | System CPU: %.2f s\n",
              i,
              conf_matrix_top10$overall["Accuracy"],
              auc(y_valid_fold, pred_proba_top10),
              conf_matrix_top10$byClass["Precision"],
              conf_matrix_top10$byClass["Recall"],
              conf_matrix_top10$byClass["F1"],
              user_cpu_time,
              system_cpu_time))
}

# Final factor conversion
all_preds_top10 <- factor(all_preds_top10, levels = c("0", "1"))
all_actual_top10 <- factor(all_actual_top10, levels = c("0", "1"))

# ===== Summary of Results =====
cat("\n===== Cross-Validation Summary for Top 10 Features =====\n")
cat("Mean Accuracy:", mean(results_top10$Accuracy, na.rm = TRUE), "±", sd(results_top10$Accuracy, na.rm = TRUE), "\n")
cat("Mean AUC:", mean(results_top10$AUC, na.rm = TRUE), "±", sd(results_top10$AUC, na.rm = TRUE), "\n")
cat("Mean Precision:", mean(results_top10$Precision, na.rm = TRUE), "±", sd(results_top10$Precision, na.rm = TRUE), "\n")
cat("Mean Recall:", mean(results_top10$Recall, na.rm = TRUE), "±", sd(results_top10$Recall, na.rm = TRUE), "\n")
cat("Mean F1 Score:", mean(results_top10$F1_Score, na.rm = TRUE), "±", sd(results_top10$F1_Score, na.rm = TRUE), "\n")
cat("Mean Training Time:", mean(results_top10$Training_Time, na.rm = TRUE), "±", sd(results_top10$Training_Time, na.rm = TRUE), "\n")
cat("Mean User CPU Time:", mean(results_top10$User_CPU_Time, na.rm = TRUE), "±", sd(results_top10$User_CPU_Time, na.rm = TRUE), "\n")
cat("Mean System CPU Time:", mean(results_top10$System_CPU_Time, na.rm = TRUE), "±", sd(results_top10$System_CPU_Time, na.rm = TRUE), "\n")

print(results_top10)

write.csv(results_top10, "results_top10_new.csv", row.names = FALSE)
# ===== PLOTS =====
# Confusion Matrix
cm_overall_top10 <- caret::confusionMatrix(all_preds_top10, all_actual_top10)
cm_df_top10 <- as.data.frame(cm_overall_top10$table)
plt_cm_top10 <- ggplot(data = cm_df_top10, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(label = Freq), vjust = 0.5) +
  labs(title = "Confusion Matrix (Top 10 Features)", fill = "Frequency") +
  theme_minimal()
print(plt_cm_top10)

# ROC Curve
roc_obj_top10 <- roc(as.numeric(as.character(all_actual_top10)), all_proba_top10)
plt_roc_top10 <- ggplot(data.frame(FPR = 1 - roc_obj_top10$specificities, TPR = roc_obj_top10$sensitivities),
                        aes(x = FPR, y = TPR)) +
  geom_line(color = "blue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(title = "ROC Curve (Top 10 Features)",
       x = "False Positive Rate", y = "True Positive Rate",
       caption = paste("AUC =", round(auc(roc_obj_top10), 3))) +
  theme_minimal()
print(plt_roc_top10)

# Precision-Recall Curve
all_actual_numeric <- as.numeric(as.character(all_actual_top10))

# Compute PR curve — class0 = positive (label 1), class1 = negative (label 0)
pr_data_top10 <- pr.curve(
  scores.class0 = all_proba_top10[all_actual_numeric == 1],
  scores.class1 = all_proba_top10[all_actual_numeric == 0],
  curve = TRUE
)

# Plot
plt_pr_top10 <- ggplot(
  data.frame(Recall = pr_data_top10$curve[, 1], Precision = pr_data_top10$curve[, 2]),
  aes(x = Recall, y = Precision)
) +
  geom_line(color = "blue") +
  geom_hline(yintercept = mean(all_actual_numeric == 1), linetype = "dashed", color = "red") +
  labs(
    title = "Precision-Recall Curve (Top 10 Features)",
    x = "Recall", y = "Precision",
    caption = paste("AUC-PR =", round(pr_data_top10$auc.integral, 3))
  ) +
  theme_minimal()

print(plt_pr_top10)

# Performance by Fold
results_long_top10 <- tidyr::gather(
  results_top10,
  key = "Metric",
  value = "Value",
  Accuracy, AUC, Precision, Recall, F1_Score
)

plt_metrics_top10 <- ggplot(results_long_top10, aes(x = Fold, y = Value, color = Metric)) +
  geom_line() +
  geom_point() +
  facet_wrap(~ Metric, scales = "free_y") +
  labs(
    title = "Performance Metrics vs. Fold (SVM with LDA-Weighted Top 10)",
    x = "Fold", y = "Value", color = "Metric"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(plt_metrics_top10)



# === Performance Metrics vs. Fold ===
results_long_top10 <- gather(results_top10, key = "Metric", value = "Value",
                             Accuracy, AUC, Precision, Recall, F1_Score)

plt_metrics_top10 <- ggplot(results_long_top10, aes(x = Fold, y = Value, color = Metric)) +
  geom_line() +
  geom_point() +
  facet_wrap(~ Metric, scales = "free_y") +
  labs(title = "Performance Metrics vs. Fold (SVM with LDA-Weighted Top 10)",
       x = "Fold", y = "Value", color = "Metric") +
  theme_minimal() +
  theme(legend.position = "bottom")
print(plt_metrics_top10)


# # ===== Final Model Training for Test Prediction =====
cat("\nTraining Final SVM Model on All Training Data with Top 10 LDA-Weighted Features\n")

X_train_final_selected <- X_weighted_data[, top_features]
y_train_final <- y_weighted_data

svm_model_top10_final <- svm(X_train_final_selected, as.factor(y_train_final), kernel = "radial",
                              gamma = best_gamma,
                              cost = best_cost, probability = TRUE)

###### BATCH #######

#Get top 10 features based on absolute LDA coefficients
top_features <- order(abs(W_vec), decreasing = TRUE)[1:30]

#Convert to matrix for batch CV loop
X_weighted_data <- as.matrix(X_train_weighted)
y_weighted_data <- y_train_final

set.seed(123)

num_batches <- 10
valid_proportion <- 0.2  # 20% of data for validation in each batch
n_samples <- nrow(X_weighted_data)

#Initialize results container for batch CV
results_batch <- data.frame(Batch = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(),
                            Accuracy = numeric(), AUC = numeric(), Precision = numeric(),
                            Recall = numeric(), F1_Score = numeric(), Training_Time = numeric(),
                            User_CPU_Time = numeric(), System_CPU_Time = numeric())

#Accumulators for overall evaluation
all_preds_batch <- c()
all_proba_batch <- c()
all_actual_batch <- c()

for (i in 1:num_batches) {
  
  cat("Processing Batch", i, "\n")
  
  #Randomly sample validation indices
  valid_idx <- sample(1:n_samples, size = round(valid_proportion * n_samples), replace = FALSE)
  
  X_train_batch <- X_weighted_data[-valid_idx, ]
  y_train_batch <- y_weighted_data[-valid_idx]
  
  X_valid_batch <- X_weighted_data[valid_idx, ]
  y_valid_batch <- y_weighted_data[valid_idx]
  
  #Select top 10 LDA-weighted features
  X_train_selected <- X_train_batch[, top_features]
  X_valid_selected <- X_valid_batch[, top_features]
  
  #Train SVM and measure CPU usage
  start_cpu <- proc.time()  # Capture CPU time before training
  start_time <- Sys.time()  # Capture wall-clock time
  svm_model_batch <- svm(X_train_selected, as.factor(y_train_batch), kernel = "radial", probability = TRUE)
  end_time <- Sys.time()
  end_cpu <- proc.time()  # Capture CPU time after training
  
  #Calculate training time and CPU usage
  training_time_batch <- as.numeric(difftime(end_time, start_time, units = "secs"))
  user_cpu_time <- (end_cpu - start_cpu)["user.self"]  # User CPU time
  system_cpu_time <- (end_cpu - start_cpu)["sys.self"]  # System CPU time
  
  pred_batch <- predict(svm_model_batch, X_valid_selected)
  pred_proba_batch <- attr(predict(svm_model_batch, X_valid_selected, probability = TRUE), "probabilities")[, 2]
  
  #Confusion matrix
  conf_matrix_batch <- confusionMatrix(pred_batch, as.factor(y_valid_batch))
  
  #Collect metrics
  results_batch <- rbind(results_batch, data.frame(
    Batch = i,
    TP = conf_matrix_batch$table["1", "1"],
    TN = conf_matrix_batch$table["0", "0"],
    FP = conf_matrix_batch$table["1", "0"],
    FN = conf_matrix_batch$table["0", "1"],
    Accuracy = conf_matrix_batch$overall["Accuracy"],
    AUC = auc(y_valid_batch, pred_proba_batch),
    Precision = conf_matrix_batch$byClass["Precision"],
    Recall = conf_matrix_batch$byClass["Recall"],
    F1_Score = conf_matrix_batch$byClass["F1"],
    Training_Time = training_time_batch,
    User_CPU_Time = user_cpu_time,
    System_CPU_Time = system_cpu_time
  ))
  
  #Accumulate all predictions and labels
  all_preds_batch <- c(all_preds_batch, as.character(pred_batch))
  all_proba_batch <- c(all_proba_batch, pred_proba_batch)
  all_actual_batch <- c(all_actual_batch, as.character(y_valid_batch))
  
  #Output per-batch results
  cat(sprintf("Batch %d -> Accuracy: %.4f | AUC: %.4f | Precision: %.4f | Recall: %.4f | F1 Score: %.4f | User CPU: %.2f s | System CPU: %.2f s\n",
              i,
              conf_matrix_batch$overall["Accuracy"],
              auc(y_valid_batch, pred_proba_batch),
              conf_matrix_batch$byClass["Precision"],
              conf_matrix_batch$byClass["Recall"],
              conf_matrix_batch$byClass["F1"],
              user_cpu_time,
              system_cpu_time))
}

#Final factor conversion
all_preds_batch <- factor(all_preds_batch, levels = c("0", "1"))
all_actual_batch <- factor(all_actual_batch, levels = c("0", "1"))

#===== Summary of Results =====
cat("\n===== Batch Cross-Validation Summary for Top 10 Features =====\n")
cat("Mean Accuracy:", mean(results_batch$Accuracy, na.rm = TRUE), "±", sd(results_batch$Accuracy, na.rm = TRUE), "\n")
cat("Mean AUC:", mean(results_batch$AUC, na.rm = TRUE), "±", sd(results_batch$AUC, na.rm = TRUE), "\n")
cat("Mean Precision:", mean(results_batch$Precision, na.rm = TRUE), "±", sd(results_batch$Precision, na.rm = TRUE), "\n")
cat("Mean Recall:", mean(results_batch$Recall, na.rm = TRUE), "±", sd(results_batch$Recall, na.rm = TRUE), "\n")
cat("Mean F1 Score:", mean(results_batch$F1_Score, na.rm = TRUE), "±", sd(results_batch$F1_Score, na.rm = TRUE), "\n")
cat("Mean Training Time:", mean(results_batch$Training_Time, na.rm = TRUE), "±", sd(results_batch$Training_Time, na.rm = TRUE), "\n")
cat("Mean User CPU Time:", mean(results_batch$User_CPU_Time, na.rm = TRUE), "±", sd(results_batch$User_CPU_Time, na.rm = TRUE), "\n")
cat("Mean System CPU Time:", mean(results_batch$System_CPU_Time, na.rm = TRUE), "±", sd(results_batch$System_CPU_Time, na.rm = TRUE), "\n")

print(results_batch)

write.csv(results_batch, "results_batch_new.csv", row.names = FALSE)

# Get the absolute values of LDA loadings
lda_weights <- abs(lda_model$scaling[, 1])

# Get indices of the top 10 features
top_10_indices <- order(lda_weights, decreasing = TRUE)[1:30]

# If you're using the real LDA-weighted data
X_train_weighted_top10 <- X_train_weighted[, top_10_indices]

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
    pROC::auc(pROC::roc(y_true, y_scores))
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

incremental_learning_pipeline_qr_eval_batch1_modified <- function(X_train_qr, y_train_qr, batch_size_qr, gamma_manual_qr, cost_manual_qr) {
  num_samples <- nrow(X_train_qr)
  num_batches <- ceiling(num_samples / batch_size_qr)
  
  stats_qr <- list(cov = NULL, mean = NULL, n = NULL)
  svm_model_qr <- NULL
  results_qr <- data.frame()
  
  all_predictions_qr <- list()
  all_true_labels_qr <- list()
  all_probabilities_qr <- list()
  
  # === Batch 1 (initialize stats only) ===
  X_batch1_qr <- X_train_qr[1:batch_size_qr, ]
  y_batch1_qr <- y_train_qr[1:batch_size_qr]
  
  stats_qr$n <- nrow(X_batch1_qr)
  stats_qr$mean <- colMeans(X_batch1_qr)
  stats_qr$cov <- cov(X_batch1_qr)
  
  qr_result_qr <- qr(stats_qr$cov)
  projection_qr <- qr.Q(qr_result_qr)[, 1:min(ncol(X_batch1_qr), stats_qr$n - 1), drop = FALSE]
  X_transformed_batch1_qr <- X_batch1_qr %*% projection_qr
  
  svm_model_qr <- e1071::svm(X_transformed_batch1_qr, as.factor(y_batch1_qr),
                             kernel = "radial", gamma = gamma_manual_qr,
                             cost = cost_manual_qr, probability = TRUE,
                             class.weights = c('0' = 1, '1' = 1))
  
  # === Loop through remaining batches ===
  for (batch_idx_qr in 2:num_batches) {
    start_idx <- (batch_idx_qr - 1) * batch_size_qr + 1
    end_idx <- min(batch_idx_qr * batch_size_qr, num_samples)
    X_batch_qr <- X_train_qr[start_idx:end_idx, ]
    y_batch_qr <- y_train_qr[start_idx:end_idx]
    
    # QR update timing
    start_qr_time <- Sys.time()
    update_result_qr <- incremental_qr_lda(stats_qr$cov, stats_qr$mean, stats_qr$n,
                                           X_batch_qr, n_components_qr = min(ncol(X_batch_qr), stats_qr$n - 1))
    projection_qr <- update_result_qr$projection_qr
    stats_qr$n <- update_result_qr$updated_n_qr
    stats_qr$mean <- update_result_qr$updated_mean_qr
    stats_qr$cov <- update_result_qr$updated_cov_qr
    end_qr_time <- Sys.time()
    
    # Transform and SVM training
    X_transformed_qr <- X_batch_qr %*% projection_qr
    X_accumulated_qr <- rbind(X_transformed_batch1_qr, X_transformed_qr)
    y_accumulated_qr <- c(y_batch1_qr, y_batch_qr)
    
    start_svm_time <- Sys.time()
    svm_model_qr <- e1071::svm(X_accumulated_qr, as.factor(y_accumulated_qr),
                               kernel = "radial", gamma = gamma_manual_qr,
                               cost = cost_manual_qr, probability = TRUE,
                               class.weights = c('0' = 1, '1' = 1))
    end_svm_time <- Sys.time()
    
    # Evaluation on current batch
    pred_qr <- predict(svm_model_qr, X_transformed_qr)
    prob_scores_qr <- attr(predict(svm_model_qr, X_transformed_qr, probability = TRUE), "probabilities")[, 2]
    
    metrics_qr <- compute_metrics_qr(y_batch_qr, pred_qr, prob_scores_qr)
    metrics_qr$QR_Update_Time <- as.numeric(difftime(end_qr_time, start_qr_time, units = "secs"))
    metrics_qr$SVM_Training_Time <- as.numeric(difftime(end_svm_time, start_svm_time, units = "secs"))
    metrics_qr$Total_Training_Time_qr <- metrics_qr$QR_Update_Time + metrics_qr$SVM_Training_Time
    metrics_qr$Batch <- batch_idx_qr
    
    results_qr <- rbind(results_qr, metrics_qr)
    
    all_predictions_qr[[batch_idx_qr - 1]] <- pred_qr
    all_true_labels_qr[[batch_idx_qr - 1]] <- y_batch_qr
    all_probabilities_qr[[batch_idx_qr - 1]] <- prob_scores_qr
  }
  
  return(list(
    results = results_qr,
    final_model = svm_model_qr,
    final_projection = projection_qr,
    final_stats = stats_qr,
    all_preds_qr = unlist(all_predictions_qr),
    all_actual_qr = unlist(all_true_labels_qr),
    all_proba_qr = unlist(all_probabilities_qr)
  ))
}

# ==== RUNNING MODIFIED QR INC ====

set.seed(42)
X_train_weighted_qr <- X_train_weighted_top10
y_train_final_qr <- y_train_final

batch_size_qr <- floor(0.1 * nrow(X_train_weighted_qr))

batch_results_qr_modified <- incremental_learning_pipeline_qr_eval_batch1_modified(
  as.matrix(X_train_weighted_qr),
  as.factor(y_train_final_qr),
  batch_size = batch_size_qr,
  gamma_manual = best_gamma,
  cost_manual = best_cost
)

print(batch_results_qr_modified$results)

# Save results to CSV
write.csv(batch_results_qr_modified$results, "results_incremental_qr_new.csv", row.names = FALSE)

all_preds_qr <- batch_results_qr_modified$all_preds_qr
all_actual_qr <- batch_results_qr_modified$all_actual_qr
all_proba_qr <- batch_results_qr_modified$all_proba_qr

# Make sure predictions and actuals are factors with same levels
all_preds_qr <- factor(all_preds_qr, levels = c("0", "1"))
all_actual_qr <- factor(all_actual_qr, levels = c("0", "1"))

# Confusion Matrix
cm_overall_qr <- confusionMatrix(all_preds_qr, all_actual_qr)
cm_df_qr <- as.data.frame(cm_overall_qr$table)

# Plot
plt_cm_qr <- ggplot(data = cm_df_qr, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(label = Freq), vjust = 0.5) +
  labs(title = "Confusion Matrix (QR Incremental)", fill = "Frequency") +
  theme_minimal()
print(plt_cm_qr)

# Compute ROC
roc_obj_qr <- roc(as.numeric(as.character(all_actual_qr)), all_proba_qr)

# Plot ROC
plt_roc_qr <- ggplot(
  data.frame(FPR = 1 - roc_obj_qr$specificities, TPR = roc_obj_qr$sensitivities),
  aes(x = FPR, y = TPR)
) +
  geom_line(color = "blue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(
    title = "ROC Curve (QR Incremental)",
    x = "False Positive Rate",
    y = "True Positive Rate",
    caption = paste("AUC =", round(auc(roc_obj_qr), 3))
  ) +
  theme_minimal()

print(plt_roc_qr)

# Convert actual labels to numeric
all_actual_qr_numeric <- as.numeric(as.character(all_actual_qr))

# Compute PR Curve — class0 = positive (label 1), class1 = negative (label 0)
pr_data_qr <- pr.curve(
  scores.class0 = all_proba_qr[all_actual_qr_numeric == 1],
  scores.class1 = all_proba_qr[all_actual_qr_numeric == 0],
  curve = TRUE
)

# Plot PR Curve
plt_pr_qr <- ggplot(
  data.frame(Recall = pr_data_qr$curve[, 1], Precision = pr_data_qr$curve[, 2]),
  aes(x = Recall, y = Precision)
) +
  geom_line(color = "blue") +
  geom_hline(yintercept = mean(all_actual_qr_numeric == 1), linetype = "dashed", color = "red") +
  labs(
    title = "Precision-Recall Curve (QR Incremental)",
    x = "Recall",
    y = "Precision",
    caption = paste("AUC-PR =", round(pr_data_qr$auc.integral, 3))
  ) +
  theme_minimal()

print(plt_pr_qr)

final_model_qr <- batch_results_qr_modified$final_model
final_projection_qr <- batch_results_qr_modified$final_projection

################### TEST DATA #################################
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

# ========== Main Function: Incremental QR on Test (Evaluating on New Batch) ==========
incremental_qr_on_test_eval_new_batch_v2 <- function(X_test_full, y_test_full,
                                                     initial_model, initial_proj, initial_stats,
                                                     gamma_manual, cost_manual,
                                                     verbose = TRUE) {
  if (!requireNamespace("ps", quietly = TRUE)) {
    warning("Package 'ps' not installed. CPU usage will not be measured.")
    cpu_monitoring <- FALSE
  } else {
    library(ps)
    cpu_monitoring <- TRUE
  }
  
  # Define batches
  total_samples <- nrow(X_test_full)
  batch_size <- floor(0.03 * total_samples)
  num_batches <- 10
  
  if (verbose) {
    cat("Total test samples:", total_samples, "\n")
    cat("Batch size (3%):", batch_size, "\n")
    cat("Number of batches:", num_batches, "\n")
  }
  
  results <- data.frame()
  svm_model <- initial_model
  projection_matrix <- initial_proj
  stats <- initial_stats
  
  # Storage for all predictions and actuals
  all_preds_test_qr <- factor(levels = levels(y_test_full))
  all_actual_test_qr <- numeric()
  all_proba_test_qr <- numeric()
  
  for (batch_idx in 1:num_batches) {
    if (verbose) cat("Batch", batch_idx, "...\n")
    
    # Determine batch indices
    start_idx <- (batch_idx - 1) * batch_size + 1
    end_idx <- min(batch_idx * batch_size, total_samples)
    if (start_idx > total_samples) break
    
    X_batch <- X_test_full[start_idx:end_idx, , drop = FALSE]
    y_batch <- y_test_full[start_idx:end_idx]
    
    # CPU usage before
    if (cpu_monitoring) {
      cpu_before <- ps_cpu_times()
    }
    
    # QR update
    start_qr <- proc.time()
    qr_update <- update_qr_stats(stats, X_batch, n_components_qr = ncol(X_batch))
    end_qr <- proc.time()
    
    projection_matrix <- qr_update$projection
    stats <- qr_update$stats
    
    qr_update_time <- (end_qr - start_qr)["elapsed"]
    
    # Prepare data for retraining
    if (batch_idx == 1) {
      X_train_transformed <- X_batch %*% projection_matrix
      y_train_new <- y_batch
    } else {
      X_seen <- X_test_full[1:end_idx, , drop = FALSE]
      y_seen <- y_test_full[1:end_idx]
      X_train_transformed <- X_seen %*% projection_matrix
      y_train_new <- y_seen
    }
    
    # SVM training
    start_train <- proc.time()
    svm_model <- train_svm_model(X_train_transformed, y_train_new, gamma_manual, cost_manual)
    end_train <- proc.time()
    
    svm_train_time <- (end_train - start_train)["elapsed"]
    total_batch_time <- qr_update_time + svm_train_time
    
    # CPU usage after
    if (cpu_monitoring) {
      cpu_after <- ps_cpu_times()
      cpu_usage <- cpu_after["user"] - cpu_before["user"]
    } else {
      cpu_usage <- NA
    }
    
    # Evaluate
    X_proj_batch <- X_batch %*% projection_matrix
    pred_probs <- predict(svm_model, X_proj_batch, probability = TRUE)
    preds <- predict(svm_model, X_proj_batch)
    prob_scores <- attr(pred_probs, "probabilities")[, 2]
    
    # Store cumulative predictions
    all_preds_test_qr <- factor(c(as.character(all_preds_test_qr), as.character(preds)), levels = levels(y_test_full))
    all_actual_test_qr <- c(all_actual_test_qr, as.numeric(as.character(y_batch)))
    all_proba_test_qr <- c(all_proba_test_qr, prob_scores)
    
    # Compute metrics
    metrics_row <- compute_metrics_qr(y_batch, preds, prob_scores)
    
    metrics_row$QR_Update_Time <- qr_update_time
    metrics_row$SVM_Train_Time <- svm_train_time
    metrics_row$Total_Batch_Time <- total_batch_time
    metrics_row$CPU_Usage_User_Seconds <- cpu_usage
    metrics_row$Batch <- batch_idx
    
    results <- rbind(results, metrics_row)
  }
  
  return(list(
    results = results,
    all_preds = all_preds_test_qr,
    all_actual = all_actual_test_qr,
    all_proba = all_proba_test_qr
  ))
}


# ----- Apply feature weighting -----
X_test_weighted_qr <- sweep(X_test, 2, W_vec, `*`)

# ----- Select top features -----
X_test_selected_qr <- X_test_weighted_qr[, top_features, drop = FALSE]

# ----- Convert to matrix -----
X_test_matrix <- as.matrix(X_test_selected_qr)

# ----- Run the updated incremental QR evaluation -----
incremental_results_test_qr_v2 <- incremental_qr_on_test_eval_new_batch_v2(
  X_test_matrix, as.factor(y_test),
  initial_model = final_model_qr,
  initial_proj = final_projection_qr,
  initial_stats = batch_results_qr_modified$final_stats,
  gamma_manual = best_gamma,
  cost_manual = best_cost,
  verbose = TRUE
)

print(incremental_results_test_qr_v2$results)

# Save results to CSV
write.csv(incremental_results_test_qr_v2$results, "results_incremental_qr_test.csv", row.names = FALSE)

cat("Length of all_preds:", length(incremental_results_test_qr_new_eval$all_preds), "\n")
cat("Length of all_actual:", length(incremental_results_test_qr_new_eval$all_actual), "\n")


# ----- Plotting -----
if (require(ggplot2) && require(caret) && require(pROC) && require(PRROC)) {
  # ----- Confusion Matrix Visualization (Final Batch) -----
  final_preds_eval_new <- incremental_results_test_qr_new_eval$all_preds
  final_actual_eval_new <- as.factor(incremental_results_test_qr_new_eval$all_actual)
  
  if (length(levels(final_preds_eval_new)) > 1 && length(levels(final_actual_eval_new)) > 1) {
    cm_final_eval_new <- caret::confusionMatrix(final_preds_eval_new, final_actual_eval_new)
    cm_df_eval_new <- as.data.frame(cm_final_eval_new$table)
    plt_cm_eval_new <- ggplot(data = cm_df_eval_new, aes(x = Prediction, y = Reference, fill = Freq)) +
      geom_tile() +
      scale_fill_gradient(low = "white", high = "steelblue") +
      geom_text(aes(label = Freq), vjust = 0.5) +
      labs(title = "Confusion Matrix on Test Data (Incremental QR)", fill = "Frequency") +
      theme_minimal()
    print(plt_cm_eval_new)
  } else {
    cat("Warning: Cannot plot confusion matrix due to insufficient unique levels in predictions or actuals.\n")
  }
  
  
  # ----- ROC Curve (Aggregated over Batches) -----
  all_probs_eval_new <- incremental_results_test_qr_new_eval$all_proba
  all_labels_eval_new <- incremental_results_test_qr_new_eval$all_actual
  
  if (length(unique(all_labels_eval_new)) > 1 && length(unique(all_probs_eval_new)) > 1) {
    roc_obj_eval_new <- roc(all_labels_eval_new, all_probs_eval_new)
    plt_roc_eval_new <- ggplot(data.frame(FPR = 1 - roc_obj_eval_new$specificities, TPR = roc_obj_eval_new$sensitivities),
                               aes(x = FPR, y = TPR)) +
      geom_line(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      labs(title = "ROC Curve on Test Data (Incremental QR)",
           x = "False Positive Rate", y = "True Positive Rate",
           caption = paste("AUC =", round(auc(roc_obj_eval_new), 3))) +
      theme_minimal()
    print(plt_roc_eval_new)
  } else {
    cat("Warning: Cannot plot ROC curve as all true labels or probabilities are the same.\n")
  }
  
  # ----- Precision-Recall Curve (Aggregated over Batches) -----
  if (require(PRROC) && length(unique(all_labels_eval_new)) > 1 && length(unique(all_probs_eval_new)) > 1) {
    pr_data_eval_new <- pr.curve(all_labels_eval_new, all_probs_eval_new, curve = TRUE)
    plt_pr_eval_new <- ggplot(data.frame(Recall = pr_data_eval_new$curve[, 1], Precision = pr_data_eval_new$curve[, 2]),
                              aes(x = Recall, y = Precision)) +
      geom_line(color = "blue") +
      geom_hline(yintercept = mean(all_labels_eval_new), linetype = "dashed", color = "red") + # Baseline
      labs(title = "Precision-Recall Curve on Test Data (Incremental QR)",
           x = "Recall", y = "Precision",
           caption = paste("AUC-PR =", round(pr_data_eval_new$auc.integral, 3))) +
      theme_minimal()
    print(plt_pr_eval_new)
  } else if (!require(PRROC)) {
    cat("Please install the 'PRROC' package to plot Precision-Recall curves.\n")
  } else {
    cat("Warning: Cannot plot Precision-Recall curve as all true labels or probabilities are the same.\n")
  }
  
  # ----- Performance Metrics vs. Batch -----
  results_long_eval_new <- gather(incremental_results_test_qr_new_eval$results, key = "Metric", value = "Value",
                                  Accuracy, AUC, Precision, Recall, F1_Score)
  
  plt_metrics_eval_new <- ggplot(results_long_eval_new, aes(x = Batch, y = Value, color = Metric)) +
    geom_line() +
    geom_point() +
    facet_wrap(~ Metric, scales = "free_y") +
    labs(title = "Performance Metrics vs. Batch (Incremental QR - Evaluating New Batch)",
         x = "Batch", y = "Value", color = "Metric") +
    theme_minimal() +
    theme(legend.position = "bottom")
  print(plt_metrics_eval_new)
  
} else {
  cat("Please install 'ggplot2', 'caret', 'pROC', and 'PRROC' packages to plot.\n")
}



# === Prepare Test Data with Top 10 Features ===
# === Apply LDA Weighting and Select Top 10 for Test Set ===
# === Prepare Test Data with Top 10 Features ===
# Assume X_test and W_vec are already defined

# Load required library
if (!requireNamespace("ps", quietly = TRUE)) {
  warning("Package 'ps' not installed. CPU usage will not be measured.")
  cpu_monitoring <- FALSE
} else {
  library(ps)
  cpu_monitoring <- TRUE
}

# Ensure test labels are factor
y_test <- as.factor(y_test)

# Apply feature weighting and selection
X_test_weighted <- sweep(X_test, 2, W_vec, `*`)
X_test_selected <- X_test_weighted[, top_10_indices, drop = FALSE]

# Split test set into 10 batches
batches <- split(1:nrow(X_test_selected), cut(1:nrow(X_test_selected), breaks = 10, labels = FALSE))

# Initialize result storage
test_results_top10 <- data.frame(
  Batch = integer(), TP = integer(), TN = integer(), FP = integer(), FN = integer(),
  Accuracy = numeric(), AUC = numeric(), Precision = numeric(),
  Recall = numeric(), F1_Score = numeric(), Inference_Time = numeric(),
  CPU_Usage_User_Seconds = numeric()
)

# Initialize accumulators
all_probabilities_test_top10 <- numeric()
all_true_labels_test_top10 <- numeric()
all_predictions_test_top10 <- factor(levels = levels(y_test))

# Use pretrained model: svm_model_top10_final
for (i in seq_along(batches)) {
  cat("Processing Test Batch", i, "\n")
  
  idx <- batches[[i]]
  X_batch <- X_test_selected[idx, ]
  y_batch <- y_test[idx]
  
  # Record CPU usage before inference
  if (cpu_monitoring) {
    cpu_before <- ps_cpu_times()
  }
  
  # Inference
  start_time <- Sys.time()
  preds <- predict(svm_model_top10_final, X_batch)
  preds_proba <- attr(predict(svm_model_top10_final, X_batch, probability = TRUE), "probabilities")[, 2]
  end_time <- Sys.time()
  
  inference_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  # Record CPU usage after inference
  if (cpu_monitoring) {
    cpu_after <- ps_cpu_times()
    cpu_usage <- cpu_after["user"] - cpu_before["user"]
  } else {
    cpu_usage <- NA
  }
  
  # Confusion Matrix
  cm <- table(Predicted = preds, Actual = y_batch)
  TP <- ifelse("1" %in% rownames(cm) & "1" %in% colnames(cm), cm["1", "1"], 0)
  TN <- ifelse("0" %in% rownames(cm) & "0" %in% colnames(cm), cm["0", "0"], 0)
  FP <- ifelse("1" %in% rownames(cm) & "0" %in% colnames(cm), cm["1", "0"], 0)
  FN <- ifelse("0" %in% rownames(cm) & "1" %in% colnames(cm), cm["0", "1"], 0)
  
  # Metrics
  accuracy <- mean(preds == y_batch)
  auc_val <- tryCatch({
    auc(as.numeric(as.character(y_batch)), preds_proba)
  }, error = function(e) {
    warning(paste("AUC calculation failed for batch", i, ":", e$message))
    NA
  })
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  recall <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  # Store batch result
  test_results_top10 <- rbind(test_results_top10, data.frame(
    Batch = i, TP = TP, TN = TN, FP = FP, FN = FN,
    Accuracy = accuracy, AUC = auc_val,
    Precision = precision, Recall = recall,
    F1_Score = f1, Inference_Time = inference_time,
    CPU_Usage_User_Seconds = cpu_usage
  ))
  
  # Collect predictions
  all_probabilities_test_top10 <- c(all_probabilities_test_top10, preds_proba)
  all_true_labels_test_top10 <- c(all_true_labels_test_top10, as.numeric(as.character(y_batch)))
  all_predictions_test_top10 <- factor(c(as.character(all_predictions_test_top10), as.character(preds)), levels = levels(y_test))
}

# Review results
print(test_results_top10)
summary(test_results_top10)

# Optional: Save to CSV
write.csv(test_results_top10, file = "results_test_top10_new.csv", row.names = FALSE)


# === Plotting ===
if (require(ggplot2) && require(caret) && require(pROC) && require(PRROC) && require(tidyr)) {
  # ----- Confusion Matrix Visualization (Overall) -----
  cat("Levels of all_predictions_test_top10:", levels(all_predictions_test_top10), "\n")
  cat("Levels of y_test:", levels(y_test), "\n")
  cm_overall_test_top10 <- caret::confusionMatrix(all_predictions_test_top10, y_test)
  cm_df_test_top10 <- as.data.frame(cm_overall_test_top10$table)
  plt_cm_test_top10 <- ggplot(data = cm_df_test_top10, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), vjust = 0.5) +
    labs(title = "Confusion Matrix on Test Data (Top 10 Features)", fill = "Frequency") +
    theme_minimal()
  print(plt_cm_test_top10)
  
  # ----- ROC Curve (Overall) -----
  if (length(unique(all_true_labels_test_top10)) > 1) {
    roc_obj_test_top10 <- roc(all_true_labels_test_top10, all_probabilities_test_top10)
    plt_roc_test_top10 <- ggplot(data.frame(FPR = 1 - roc_obj_test_top10$specificities, TPR = roc_obj_test_top10$sensitivities),
                                 aes(x = FPR, y = TPR)) +
      geom_line(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
      labs(title = "ROC Curve on Test Data (Top 10 Features)",
           x = "False Positive Rate", y = "True Positive Rate",
           caption = paste("AUC =", round(auc(roc_obj_test_top10), 3))) +
      theme_minimal()
    print(plt_roc_test_top10)
  } else {
    cat("Warning: Cannot plot ROC curve as all true labels are the same.\n")
  }
  
  # ----- Precision-Recall Curve (Overall) -----
  if (require(PRROC) && length(unique(all_true_labels_test_top10)) > 1) {
    pr_data_test_top10 <- pr.curve(all_true_labels_test_top10, all_probabilities_test_top10, curve = TRUE)
    plt_pr_test_top10 <- ggplot(data.frame(Recall = pr_data_test_top10$curve[, 1], Precision = pr_data_test_top10$curve[, 2]),
                                aes(x = Recall, y = Precision)) +
      geom_line(color = "blue") +
      geom_hline(yintercept = mean(all_true_labels_test_top10), linetype = "dashed", color = "red") + # Baseline
      labs(title = "Precision-Recall Curve on Test Data (Top 10 Features)",
           x = "Recall", y = "Precision",
           caption = paste("AUC-PR =", round(pr_data_test_top10$auc.integral, 3))) +
      theme_minimal()
    print(plt_pr_test_top10)
  } else if (!require(PRROC)) {
    cat("Please install the 'PRROC' package to plot Precision-Recall curves.\n")
  } else {
    cat("Warning: Cannot plot Precision-Recall curve as all true labels are the same.\n")
  }
  
  # ----- Performance Metrics vs. Batch -----
  results_long_test_top10 <- gather(test_results_top10, key = "Metric", value = "Value",
                                    Accuracy, AUC, Precision, Recall, F1_Score)
  
  plt_metrics_test_top10 <- ggplot(results_long_test_top10, aes(x = Batch, y = Value, color = Metric)) +
    geom_line() +
    geom_point() +
    facet_wrap(~ Metric, scales = "free_y") +
    labs(title = "Performance Metrics vs. Batch (Top 10 Features on Test Data)",
         x = "Batch", y = "Value", color = "Metric") +
    theme_minimal() +
    theme(legend.position = "bottom")
  print(plt_metrics_test_top10)
  
} else {
  cat("Please install 'ggplot2', 'caret', 'pROC', 'PRROC', and 'tidyr' packages to plot.\n")
}


# # ========== Helper: Simulated Incremental SVM Training ==========
# train_svm_with_buffer <- function(buffer_X, buffer_y, gamma, cost) {
#   # Combine all batches in buffer
#   X_combined <- do.call(rbind, buffer_X)
#   y_combined <- unlist(buffer_y)
#   
#   # Train SVM on the combined buffer
#   e1071::svm(
#     X_combined, as.factor(y_combined),
#     kernel = "radial",
#     gamma = gamma,
#     cost = cost,
#     probability = TRUE,
#     class.weights = c('0' = 1, '1' = 1)
#   )
# }
# 
# # ========== Main Function: Incremental QR with Buffer-based SVM ==========
# incremental_qr_with_buffer_svm <- function(X_test_full, y_test_full,
#                                            initial_proj, initial_stats,
#                                            gamma_manual, cost_manual,
#                                            buffer_size = 3,
#                                            verbose = TRUE) {
#   
#   total_samples <- nrow(X_test_full)
#   batch_size <- floor(0.03 * total_samples)
#   num_batches <- 10
#   
#   if (verbose) {
#     cat("Total test samples:", total_samples, "\n")
#     cat("Batch size (3%):", batch_size, "\n")
#     cat("Number of batches:", num_batches, "\n")
#   }
#   
#   results <- data.frame()
#   projection_matrix <- initial_proj
#   stats <- initial_stats
#   
#   # Batch buffer
#   buffer_X <- list()
#   buffer_y <- list()
#   
#   for (batch_idx in 1:num_batches) {
#     if (verbose) cat("Batch", batch_idx, "...\n")
#     
#     start_idx <- (batch_idx - 1) * batch_size + 1
#     end_idx <- min(batch_idx * batch_size, total_samples)
#     if (start_idx > total_samples) break
#     
#     X_batch <- X_test_full[start_idx:end_idx, , drop = FALSE]
#     y_batch <- y_test_full[start_idx:end_idx]
#     
#     start_time <- Sys.time()
#     
#     # Update QR stats
#     qr_update <- update_qr_stats(stats, X_batch, n_components_qr = ncol(X_batch))
#     projection_matrix <- qr_update$projection
#     stats <- qr_update$stats
#     
#     # Project current batch
#     X_batch_proj <- X_batch %*% projection_matrix
#     
#     # Update batch buffer
#     buffer_X <- c(buffer_X, list(X_batch_proj))
#     buffer_y <- c(buffer_y, list(y_batch))
#     
#     # Keep buffer size fixed
#     if (length(buffer_X) > buffer_size) {
#       buffer_X <- buffer_X[(length(buffer_X) - buffer_size + 1):length(buffer_X)]
#       buffer_y <- buffer_y[(length(buffer_y) - buffer_size + 1):length(buffer_y)]
#     }
#     
#     # Train on buffer
#     svm_model <- train_svm_with_buffer(buffer_X, buffer_y, gamma_manual, cost_manual)
#     
#     # Evaluate on the current batch
#     metrics_row <- evaluate_model_qr(svm_model, X_batch, y_batch, projection_matrix)
#     
#     end_time <- Sys.time()
#     metrics_row$Training_Time_qr <- as.numeric(difftime(end_time, start_time, units = "secs"))
#     metrics_row$Batch <- batch_idx
#     
#     results <- rbind(results, metrics_row)
#   }
#   
#   return(results)
# }
# 
# 
# incremental_results_buffered_qr <- incremental_qr_with_buffer_svm(
#   X_test_matrix, as.factor(y_test),
#   initial_proj = final_projection_qr,
#   initial_stats = batch_results_qr$final_stats,
#   gamma_manual = best_gamma_test,
#   cost_manual = best_cost_test,
#   buffer_size = 3,
#   verbose = TRUE
# )
# 
# print(incremental_results_buffered_qr)
