
library(class)

students <- read.csv("StudentsPerformance.csv")

students$average_score <- (
  students$math.score +
    students$reading.score +
    students$writing.score
) / 3

students$result <- factor(
  ifelse(students$average_score >= 60,
         "Pass", "Needs Improvement")
)

# Select input features
x <- model.matrix(
  ~ gender + race.ethnicity +
    parental.level.of.education + lunch +
    test.preparation.course,
  data = students
)[, -1, drop = FALSE]

y <- students$result

# Split data: 80% train, 20% test
set.seed(123)

index <- sample(seq_len(nrow(students)),
                floor(0.8 * nrow(students)))

x_train_all <- x[index, , drop = FALSE]
y_train_all <- y[index]

x_test <- x[-index, , drop = FALSE]
y_test <- y[-index]

# Split training data into training and validation
val_index <- sample(
  seq_len(nrow(x_train_all)),
  floor(0.25 * nrow(x_train_all))
)

x_val <- x_train_all[val_index, , drop = FALSE]
y_val <- y_train_all[val_index]

x_subtrain <- x_train_all[-val_index, , drop = FALSE]
y_subtrain <- y_train_all[-val_index]

# Remove constant features using training data only
feature_sd <- apply(x_subtrain, 2, sd)
keep <- !is.na(feature_sd) & feature_sd > 0

x_subtrain <- x_subtrain[, keep, drop = FALSE]
x_val <- x_val[, keep, drop = FALSE]
x_train_all <- x_train_all[, keep, drop = FALSE]
x_test <- x_test[, keep, drop = FALSE]

# Scale using subtraining data
feature_mean <- colMeans(x_subtrain)
feature_sd <- apply(x_subtrain, 2, sd)

x_subtrain <- scale(
  x_subtrain, center = feature_mean, scale = feature_sd
)

x_val <- scale(
  x_val, center = feature_mean, scale = feature_sd
)

# Compare K values
k_values <- c(1, 3, 5, 7)
accuracy <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  
  pred <- knn(
    train = x_subtrain,
    test = x_val,
    cl = y_subtrain,
    k = k_values[i]
  )
  
  accuracy[i] <- mean(pred == y_val) * 100
}

comparison <- data.frame(
  K = k_values,
  Validation_Accuracy = round(accuracy, 2)
)

print(comparison)

# Choose the best K using validation accuracy
best_k <- k_values[which.max(accuracy)]

print(paste("Best K:", best_k))

# Scale full training data using its own statistics
feature_mean <- colMeans(x_train_all)
feature_sd <- apply(x_train_all, 2, sd)

x_train_scaled <- scale(
  x_train_all, center = feature_mean, scale = feature_sd
)

x_test_scaled <- scale(
  x_test, center = feature_mean, scale = feature_sd
)

# Final prediction on unseen test data
final_pred <- knn(
  train = x_train_scaled,
  test = x_test_scaled,
  cl = y_train_all,
  k = best_k
)

print(table(Actual = y_test, Predicted = final_pred))

test_accuracy <- mean(final_pred == y_test) * 100

print(paste("Final Test Accuracy:",
            round(test_accuracy, 2), "%"))


# Save K value comparison
write.csv(
  comparison,
  "KNN_Model_Comparison.csv",
  row.names = FALSE
)

# Save confusion matrix


conf_matrix <- table(
  Actual = y_test,
  Predicted = final_pred
)

write.csv(
  as.data.frame.matrix(conf_matrix),
  "KNN_Confusion_Matrix.csv"
)

# Save final accuracy and best K


results_summary <- data.frame(
  Best_K = best_k,
  Test_Accuracy = round(test_accuracy, 2)
)

write.csv(
  results_summary,
  "KNN_Final_Results.csv",
  row.names = FALSE
)