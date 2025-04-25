library(readr)
library(dplyr)
library(caret)
library(MASS)
library(ggplot2)
library(e1071)
library(ucimlrepo)
library(car)
library(glmnet)
library(randomForest)
library(Boruta)
library(randomForestExplainer)
library(rminer)
library(iml)


df <- fetch_ucirepo(id = 45)

X <- df$data$features
y <- df$data$targets
df <- cbind(X, y)

colnames(df)= c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","heatdiseaseorno")
summary(df)

df$heatdiseaseorno <- ifelse(df$heatdiseaseorno == 0, 0, 1)
summary(df)
colSums(is.na(df))

df <- na.omit(df)
head(df)

df$heatdiseaseorno <- as.factor(df$heatdiseaseorno)
df$cp <- as.factor(df$cp)
df$sex <- as.factor(df$sex)
df$fbs <- as.factor(df$fbs)
df$restecg <- as.factor(df$restecg)
df$exang <- as.factor(df$exang)
df$slope <- as.factor(df$slope)
df$thal <- as.factor(df$thal)

# Function to clean the data for EDA
cleaning_for_EDA <- function(df) {
  # Remove missing values
  df <- na.omit(df)
  # Remove duplicate rows
  df <- distinct(df)
  # Remove outliers using z-scores
  numeric_cols <- sapply(df, is.numeric)
  z_scores <- abs(scale(df[, numeric_cols]))
  df <- df[rowSums(z_scores < 3) == ncol(z_scores), ]
  
  return(df)
}

# Clean the data for EDA
df_cleaned <- cleaning_for_EDA(df)

# View the cleaned data
head(df_cleaned)

#########EXPLORATORY DATA ANALYSIS#####################

#2.1 Data Description
summary(df_cleaned)

#2.2a Data Visualization 1 Variable

#Fitur Numerik
numeric_cols <- names(df_cleaned)[sapply(df_cleaned, is.numeric)]

for (feature in numeric_cols) {
  p <- ggplot(df_cleaned, aes_string(x = feature)) +
    geom_histogram(fill = "skyblue", color = "darkblue", bins = 30) + # Use geom_histogram for histograms
    labs(
      title = paste('Distribution of', feature),
      x = feature,
      y = 'Frequency'
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  print(p)
}

#Fitur Kategori
non_numeric_columns <- names(df_cleaned)[!sapply(df_cleaned, is.numeric)]

for (feature in non_numeric_columns) {
  p <- ggplot(df_cleaned, aes_string(x = feature)) +
    geom_bar(fill = "skyblue", color = "darkblue") +
    labs(
      title = paste('Distribution of', feature),
      x = feature,
      y = 'Frequency'
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  print(p)
}


#2.2b Data Visualization 2 Variables (X and Y)

#Fitur Numerik
for (feature in numeric_cols) {
  if (feature %in% c("ca")) {
    p <-ggplot(df_cleaned, aes_string(x = feature, fill = "heatdiseaseorno")) +
      geom_bar(position = "dodge") +
      labs(title = paste('Distribution of', feature), x = feature, y = 'Frequency') +
      theme_minimal()
    
  } else {
    # Create a boxplot for other features
    p <- ggplot(df_cleaned, aes_string(x = "heatdiseaseorno", y = feature)) +
      geom_boxplot(fill = "skyblue", color = "darkblue", outlier.color = "red", outlier.shape = 16) +
      labs(
        title = paste('Boxplot of', feature, 'by Class'),
        x = "Class",
        y = feature
      ) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  }
  print(p)
}

#Fitur Kategori
non_numeric_columns = non_numeric_columns[1:7]
for (feature in non_numeric_columns) {
  p <-ggplot(df_cleaned, aes_string(x = feature, fill = "heatdiseaseorno")) +
    geom_bar(position = "dodge") +
    labs(title = paste('Distribution of', feature), x = feature, y = 'Frequency') +
    theme_minimal()
  print(p)
}



#2.3 Uji SIgnifikansi antara Fitur dengan Target Variabel

for (feature in numeric_cols) {
  standardized_feature <- scale(df_cleaned[[feature]])
  ks_result <- ks.test(standardized_feature, "pnorm", mean = 0, sd = 1)
  cat("Kolmogorov-Smirnov Test for", feature, ":\n")
  print(ks_result)
  cat("\n")
}
#Thalach and chol is Normal, whereas the other features does not follow normal distribution


#ANOVA
anova_result <- aov(thalach ~ heatdiseaseorno, data = df_cleaned)
cat("ANOVA Results:\n")
summary(anova_result)
anova_result <- aov(chol ~ heatdiseaseorno, data = df_cleaned)
cat("ANOVA Results:\n")
summary(anova_result)

# Kruskal-Wallis
for (feature in numeric_cols) {
  kruskal_result <- kruskal.test(df_cleaned[[feature]] ~ df_cleaned$heatdiseaseorno)
  cat("\nKruskal-Wallis Test Results for", feature, ":\n")
  print(kruskal_result)
}
#Chi-Square Test
for (feature in non_numeric_columns) {
  chi_result <- chisq.test(table(df_cleaned[[feature]], df_cleaned$heatdiseaseorno))
  cat("\nChi-Square Test Results for", feature, ":\n")
  print(chi_result)
}

#ANOVA: kadar kolesterol (chol) tidak mempengaruhi keberadaan seangan jantung
#Kruskal Wallis: tekanan darah kondisi istirahat (trestbps)) tidak mempengaruhi keberadaan seangan jantung
#Chi-Square: keberadaan gula darah tinggi tidak mempengaruhi keberadaan seangan jantung

df_cleaned <- df_cleaned[, !(colnames(df_cleaned) == "chol")]
df_cleaned <- df_cleaned[, !(colnames(df_cleaned) == "trestbps")]
df_cleaned <- df_cleaned[, !(colnames(df_cleaned) == "fbs")]


#2.4 Data Normalization
X = df_cleaned[, !(colnames(df_cleaned) == "heatdiseaseorno")]
normalize <- preProcess(X, method = c("center", "scale"))
X_normalized <- predict(normalize, X)
y <- df_cleaned[, (colnames(df_cleaned) == "heatdiseaseorno")]


############# MODELLING and EVALUATIONS ########################

#3.1 LASSO LOGISTIC REGRESSION
X_matrix <- model.matrix(~ . - 1, data = X_normalized)
y <- df_cleaned$heatdiseaseorno

set.seed(42)
folds <- createFolds(y, k = 5, list = TRUE, returnTrain = TRUE)

#TRAINING
train_accuracy_list <- c()
train_precision_list <- c()
train_recall_list <- c()
train_f1_list <- c()

#TESTING
accuracy_list <- c()
lambda_list <- c()
precision_list <- c()
recall_list <- c()
f1_list <- c()

coefficients_list <- list()

for (i in seq_along(folds)) {
  train_indices <- folds[[i]]
  test_indices <- setdiff(seq_len(nrow(X_matrix)), train_indices)
  
  X_train <- X_matrix[train_indices, ]
  y_train <- y[train_indices]
  X_test <- X_matrix[test_indices, ]
  y_test <- y[test_indices]
  
  cv.lasso <- cv.glmnet(X_train, y_train, alpha = 1, family = "binomial", nfolds = 5)
  lambda_list[i]<-cv.lasso$lambda.min
  model <- glmnet(X_train, y_train, alpha = 1, family = "binomial", lambda = cv.lasso$lambda.min)
  
  coefficients <- coef(model)
  coefficients_list[[i]] <- coefficients
  
  #DATA TRAINING
  y_pred_train <- predict(model, newx = X_train, type = "response")
  y_pred_train <- ifelse(y_pred_train > 0.5, 1, 0)
  y_train <- as.numeric(as.character(y_train))       
  y_pred_train <- as.numeric(as.character(y_pred_train[,1]))  
  
  train_true_positives <- sum(y_train == 1 & y_pred_train == 1)
  train_predicted_positives <- sum(y_pred_train == 1)
  train_actual_positives <- sum(y_train == 1)
  
  train_accuracy_list[i] <- mean(y_train == y_pred_train)
  train_precision_list[i] <- ifelse(train_predicted_positives == 0, NA, train_true_positives / train_predicted_positives)
  train_recall_list[i] <- ifelse(train_actual_positives == 0, NA, train_true_positives / train_actual_positives)
  train_f1_list[i] <- ifelse(is.na(train_precision_list[i]) | is.na(train_recall_list[i]) | 
                         (train_precision_list[i] + train_recall_list[i]) == 0, NA, 
                       2 * train_precision_list[i] * train_recall_list[i] / (train_precision_list[i] + train_recall_list[i]))
  
  
  #DATA TESTING
  y_pred_test <- predict(model, newx = X_test, type = "response")
  y_pred_test <- ifelse(y_pred_test > 0.5, 1, 0)
  y_test <- as.numeric(as.character(y_test))       
  y_pred_test <- as.numeric(as.character(y_pred_test[,1]))  
  
  true_positives <- sum(y_test == 1 & y_pred_test == 1)
  predicted_positives <- sum(y_pred_test == 1)
  actual_positives <- sum(y_test == 1)
  
  accuracy_list[i] <- mean(y_test == y_pred_test)
  precision_list[i] <- ifelse(predicted_positives == 0, NA, true_positives / predicted_positives)
  recall_list[i] <- ifelse(actual_positives == 0, NA, true_positives / actual_positives)
  f1_list[i] <- ifelse(is.na(precision_list[i]) | is.na(recall_list[i]) | 
                         (precision_list[i] + recall_list[i]) == 0, NA, 
                       2 * precision_list[i] * recall_list[i] / (precision_list[i] + recall_list[i]))
}

# Average metrics TRAINING
train_average_accuracy <- mean(train_accuracy_list)
train_average_precision <- mean(train_precision_list)
train_average_recall <- mean(train_recall_list)
train_average_f1 <- mean(train_f1_list)

# Print results TRAINING
cat("Train Average Accuracy:", train_average_accuracy, "\n")
cat("Train Average Precision:", train_average_precision, "\n")
cat("Train Average Recall:", train_average_recall, "\n")
cat("Train Average F1 Score:", train_average_f1, "\n")

# Average metrics TESTING
average_accuracy <- mean(accuracy_list)
average_precision <- mean(precision_list)
average_recall <- mean(recall_list)
average_f1 <- mean(f1_list)

# Print results

cat("Test Average Accuracy:", average_accuracy, "\n")
cat("Test Average Precision:", average_precision, "\n")
cat("Test Average Recall:", average_recall, "\n")
cat("Test Average F1 Score:", average_f1, "\n")


# Coefficients summary
all_coefficients <- do.call(cbind, coefficients_list)
coeff_names <- c("Intercept",colnames(X_matrix))
average_coefficients <- rowMeans(all_coefficients, na.rm = TRUE)
max_coefficients <- apply(all_coefficients, 1, max, na.rm = TRUE)
min_coefficients <- apply(all_coefficients, 1, min, na.rm = TRUE)

cat("Coefficient Summary:\n")
cat("Average Coefficients:\n")
print(data.frame(Coefficient = coeff_names, Average = average_coefficients))

cat("Max Coefficients:\n")
print(data.frame(Coefficient = coeff_names, Max = max_coefficients))
cat("\n")

cat("Min Coefficients:\n")
print(data.frame(Coefficient = coeff_names, Min = min_coefficients))
cat("\n")



#3.2 SVM
library(e1071)
# untuk menilai kernel terbaik, akan dicoba run random sample train dan test, 
# dan akan dilihat kernel mana yang accuracy dari 30 kali random sampling train test
# yang accuracynya tertinggi
train_accuracy=matrix(0,30,4)
test_accuracy=matrix(0,30,4)

set.seed(42)
for(i in 1:30){
  train_indices <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X_normalized[train_indices, ]
  X_test <- X_normalized[-train_indices, ]
  y_train <- y[train_indices]
  y_test <- y[-train_indices]
  train=cbind(X_train,y_train)
  test = cbind(X_test,y_test)
  
  svm_mod1=svm(y_train~., data=train, kernel='radial', cost=1, gamma=1)
  svm_mod2=svm(y_train~., data=train, kernel='linear', cost=1, gamma=1)
  svm_mod3=svm(y_train~., data=train, kernel='polynomial', cost=1, gamma=1)
  svm_mod4=svm(y_train~., data=train, kernel='sigmoid', cost=1, gamma=1)

  #TRAINING
  train_y_svm1=predict(svm_mod1, newdata=train)
  train_y_svm2=predict(svm_mod2, newdata=train)
  train_y_svm3=predict(svm_mod3, newdata=train)
  train_y_svm4=predict(svm_mod4, newdata=train)
  train_result1=table(actual=train$y_train, predicted=train_y_svm1)
  train_result2=table(actual=train$y_train, predicted=train_y_svm2)
  train_result3=table(actual=train$y_train, predicted=train_y_svm3)
  train_result4=table(actual=train$y_train, predicted=train_y_svm4)
  train_accuracy[i,1]=(train_result1[1,1]+train_result1[2,2])/length(train$y_train)
  train_accuracy[i,2]=(train_result2[1,1]+train_result2[2,2])/length(train$y_train)
  train_accuracy[i,3]=(train_result3[1,1]+train_result3[2,2])/length(train$y_train)
  train_accuracy[i,4]=(train_result4[1,1]+train_result4[2,2])/length(train$y_train)
  
  #TESTING
  test_y_svm1=predict(svm_mod1, newdata=test)
  test_y_svm2=predict(svm_mod2, newdata=test)
  test_y_svm3=predict(svm_mod3, newdata=test)
  test_y_svm4=predict(svm_mod4, newdata=test)
  test_result1=table(actual=test$y_test, predicted=test_y_svm1)
  test_result2=table(actual=test$y_test, predicted=test_y_svm2)
  test_result3=table(actual=test$y_test, predicted=test_y_svm3)
  test_result4=table(actual=test$y_test, predicted=test_y_svm4)
  test_accuracy[i,1]=(test_result1[1,1]+test_result1[2,2])/length(test$y_test)
  test_accuracy[i,2]=(test_result2[1,1]+test_result2[2,2])/length(test$y_test)
  test_accuracy[i,3]=(test_result3[1,1]+test_result3[2,2])/length(test$y_test)
  test_accuracy[i,4]=(test_result4[1,1]+test_result4[2,2])/length(test$y_test)
}

#TRAINING
mean(train_accuracy[,1]) #rata-rata accuracy kernel radial
mean(train_accuracy[,2]) #rata-rata accuracy kernel linear
mean(train_accuracy[,3]) #rata-rata accuracy kernel polynomial
mean(train_accuracy[,4]) #rata-rata accuracy kernel sigmoid
# kernel linear yang terbaik 

#TESTING
mean(test_accuracy[,1]) #rata-rata accuracy kernel radial
mean(test_accuracy[,2]) #rata-rata accuracy kernel linear
mean(test_accuracy[,3]) #rata-rata accuracy kernel polynomial
mean(test_accuracy[,4]) #rata-rata accuracy kernel sigmoid
# kernel linear yang terbaik (radial and polynomial overfitting, sigmoid underfitting)

#Mencari cost terbaik
set.seed(42)
train_control <- trainControl(method = "cv", number = 10)
best_costs=matrix(0,10,1)
for(i in 1:10){
  tuned_svm <- train(
    y_train ~ ., 
    data = train, 
    method = "svmLinear",
    trControl = train_control,
    tuneGrid = expand.grid(C = c(0.1, 1, 10, 100))
  )
  best_params <- tuned_svm$bestTune
  best_costs[i] <- best_params$C
}
best_costs=names(which.max(table(best_costs)))

# Evaluasi model terbaik pada data test
train_accuracy=matrix(0,30,1)
train_precision=matrix(0,30,1)
train_recall=matrix(0,30,1)
train_f1_score=matrix(0,30,1)

test_accuracy=matrix(0,30,1)
test_precision=matrix(0,30,1)
test_recall=matrix(0,30,1)
test_f1_score=matrix(0,30,1)

importance_matrix <- matrix(0, ncol = 16, nrow = 30)
importance_signs <- matrix(0, ncol = 16, nrow = 30)

set.seed(42)
for (i in 1:30) {
  train_indices <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X_normalized[train_indices, ]
  X_test <- X_normalized[-train_indices, ]
  y_train <- y[train_indices]
  y_test <- y[-train_indices]
  train <- data.frame(X_train, y_train)
  test <- data.frame(X_test, y_test)
  
  svm_best <- svm(y_train ~ ., data = train, kernel = "linear", cost = best_costs)
  
  importance  <- t(svm_best$SV)%*% (svm_best$coefs)
  colnames(importance_matrix) <- rownames(importance)
  importance_matrix[i,] <- importance
  importance_signs[i, ] <- ifelse(importance > 0, "Increase", "Decrease")
  colnames(importance_signs) <- rownames(importance)
  
  # TRAINING
  train_y_svm_best <- predict(svm_best, newdata = train)
  result <- table(actual = train$y_train, predicted = train_y_svm_best)
  
  train_accuracy[i] <- sum(diag(result)) / sum(result)
  train_precision[i] <- result[2, 2] / sum(result[, 2])
  train_recall[i] <- result[2, 2] / sum(result[2, ])
  train_f1_score[i] <- 2 * (train_precision[i] * train_recall[i]) / (train_precision[i] + train_recall[i])
  
  # TESTING
  test_y_svm_best <- predict(svm_best, newdata = test)
  result <- table(actual = test$y_test, predicted = test_y_svm_best)
  
  test_accuracy[i] <- sum(diag(result)) / sum(result)
  test_precision[i] <- result[2, 2] / sum(result[, 2])
  test_recall[i] <- result[2, 2] / sum(result[2, ])
  test_f1_score[i] <- 2 * (test_precision[i] * test_recall[i]) / (test_precision[i] + test_recall[i])
}


cat("Mean Train Accuracy: ",mean(train_accuracy), "\n")
cat("Mean Train Precision: ",mean(train_precision), "\n")
cat("Mean Train Recall: ",mean(train_recall), "\n")
cat("Mean Train f1 score: ",mean(train_f1_score), "\n")

cat("Mean Test Accuracy: ",mean(test_accuracy), "\n")
cat("Mean Test Precision: ",mean(test_precision), "\n")
cat("Mean Test Recall: ",mean(test_recall), "\n")
cat("Mean Test f1 score: ",mean(test_f1_score), "\n")


average_importance <- colMeans(importance_matrix)
print(sort(average_importance))

barplot(
  sort(abs(colMeans(importance_matrix))), 
  main = "Average Variable Importance",
  ylab = "Abs.Weights",
  las = 2,  # Rotate axis labels
  col = "skyblue"
)

barplot(
  sort(average_importance), 
  main = "Average Variable Importance",
  ylab = "Weights",
  las = 2,  # Rotate axis labels
  col = "skyblue"
)

importance_summary <- apply(importance_signs, 2, function(x) {
  table(x)
})
print(importance_summary)

#3.3 Random Forest
# Langkah 1: Tuning Hyperparameter
# Cross-validation control
set.seed(42)
train_control <- trainControl(method = "cv", number = 5, savePredictions = "final")

# Latih model menggunakan caret untuk tuning
tuned_rf <- train(
  y_train ~ ., 
  data = train, 
  method = "rf", 
  trControl = train_control, 
  importance = TRUE
)

# Model terbaik berdasarkan tuning
print(tuned_rf)
best_mtry <- tuned_rf$bestTune$mtry

# Langkah 2: Latih ulang model dengan parameter terbaik
train_accuracy=matrix(0,30,1)
train_precision=matrix(0,30,1)
train_recall=matrix(0,30,1)
train_f1_score=matrix(0,30,1)

test_accuracy=matrix(0,30,1)
test_precision=matrix(0,30,1)
test_recall=matrix(0,30,1)
test_f1_score=matrix(0,30,1)

importance_matrix <- matrix(0, ncol = ncol(X_normalized), nrow = 30)
colnames(importance_matrix) <- colnames(X_normalized)
shap_values_list <- list()

set.seed(42)
for(i in 1:30){
  train_indices <- createDataPartition(y, p = 0.7, list = FALSE)
  X_train <- X_normalized[train_indices, ]
  X_test <- X_normalized[-train_indices, ]
  y_train <- y[train_indices]
  y_test <- y[-train_indices]
  train=cbind(X_train,y_train)
  test = cbind(X_test,y_test)
  rf_mod <- randomForest(
    y_train ~ ., 
    data = train, 
    mtry = best_mtry, 
    importance = TRUE
  )
  train_y_rf <- predict(rf_mod, newdata = train)
  conf_matrix <- confusionMatrix(factor(train_y_rf), factor(train$y_train))
  train_accuracy[i]=(conf_matrix$table[1,1]+conf_matrix$table[2,2])/sum(conf_matrix$table)
  train_precision[i]=conf_matrix$table[2,2]/(conf_matrix$table[1,2]+conf_matrix$table[2,2])
  train_recall[i]=conf_matrix$table[2,2]/(conf_matrix$table[2,1]+conf_matrix$table[2,2])
  train_f1_score[i] <- 2 * (train_precision[i] * train_recall[i]) / (train_precision[i] + train_recall[i])

  test_y_rf <- predict(rf_mod, newdata = test)
  conf_matrix <- confusionMatrix(factor(test_y_rf), factor(test$y_test))
  test_accuracy[i]=(conf_matrix$table[1,1]+conf_matrix$table[2,2])/sum(conf_matrix$table)
  test_precision[i]=conf_matrix$table[2,2]/(conf_matrix$table[1,2]+conf_matrix$table[2,2])
  test_recall[i]=conf_matrix$table[2,2]/(conf_matrix$table[2,1]+conf_matrix$table[2,2])
  test_f1_score[i] <- 2 * (test_precision[i] * test_recall[i]) / (test_precision[i] + test_recall[i])
  
  importance_matrix[i, ] <- importance(rf_mod, type = 1)
  
  predictor <- Predictor$new(rf_mod, data = X_train, y = y_train)
  shap_values <- sapply(1:nrow(X_test), function(x) {
    shap_instance <- Shapley$new(predictor, x.interest = X_test[x, , drop = FALSE])
    as.numeric(shap_instance$results$phi)
  })
  shap_values_list[[i]] <- t(shap_values) 
}
             
cat("Mean Train Accuracy: ",mean(train_accuracy), "\n")
cat("Mean Train Precision: ",mean(train_precision), "\n")
cat("Mean Train Recall: ",mean(train_recall), "\n")
cat("Mean Train f1 score: ",mean(train_f1_score), "\n")

cat("Mean Test Accuracy: ",mean(test_accuracy), "\n")
cat("Mean Test Precision: ",mean(test_precision), "\n")
cat("Mean Test Recall: ",mean(test_recall), "\n")
cat("Mean Test f1 score: ",mean(test_f1_score), "\n")

average_importance <- colMeans(importance_matrix)

print(sort(average_importance))
barplot(
  sort(average_importance), 
  main = "Average Variable Importance",
  ylab = "Mean Decrease Accuracy",
  las = 2,  # Rotate axis labels
  col = "skyblue"
)

shap_values_matrix <- do.call(rbind, shap_values_list)
mean_shap_values <- colMeans(shap_values_matrix)
feature_summary <- data.frame(
  Feature = rep(colnames(X_normalized), each = nrow(shap_values_matrix)),
  SHAP_Value = as.vector(shap_values_matrix)
)

library(ggbeeswarm)
ggplot(feature_summary, aes(x = SHAP_Value, y = Feature)) +
  geom_beeswarm(aes(color = Feature), size = 2, alpha = 0.6) +
  labs(title = "Swarm Plot of SHAP Values",
       x = "SHAP Value",
       y = "Feature") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))

barplot(
  feature_summary$Mean_SHAP_Value,
  main = "Average SHAP Value",
  ylab = "Mean SHAP Value",
  col = "skyblue",
  names.arg = feature_summary$Feature,
  las = 2)


predictor <- Predictor$new(rf_mod, data = X_train)
shap_values <- sapply(1:nrow(X_test), function(x) {
  shap_instance <- Shapley$new(predictor, x.interest = X_test[x, , drop = FALSE])
  as.numeric(shap_instance$results$phi)
})


# Convert SHAP values to a data frame
shap_values_df <- as.data.frame(t(shap_values))
colnames(shap_values_df) <- colnames(X_test)  # Set column names based on features
library(reshape2)
library(ggbeeswarm)

X_test_df <- as.data.frame(X_test)
X_test_long <- melt(X_test_df, variable.name = "Feature", value.name = "Feature_Value")

# Prepare data for plotting
shap_long <- melt(shap_values_df, variable.name = "Feature", value.name = "SHAP_Value")
shap_long$Feature_Value <- X_test_long$Feature_Value
shap_long <- shap_long[1:860, ]
ggplot(shap_long, aes(x = Feature, y = SHAP_Value, color = Feature_Value)) +
  geom_beeswarm(size = 2, alpha = 0.6, dodge.width = 0.5) +
  scale_color_gradient(low = "blue", high = "red", name = "Feature Value") +
  labs(title = "Swarm Plot of SHAP Values with Feature Values",
       x = "Features",
       y = "SHAP Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))

X_test
