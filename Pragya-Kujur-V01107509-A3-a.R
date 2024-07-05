# Part A - Conduct a logistic regression analysis

# 1. Load Libraries
library(tidyverse)
library(caret)
install.packages("pROC")
library(pROC)
library(e1071)
library(tm)
install.packages("rpart.plot")
library(rpart.plot)
library(rpart)


library(caret)
install.packages("caret")
library(caret)
install.packages("tm")
library(tm)
install.packages("SnowballC")
library(SnowballC)
library(rpart.plot)


# 2. Import Dataset
setwd('C://Users//Home//Downloads')
spam_df <- read.csv('spam.csv', stringsAsFactors = FALSE)

# 3. Data Cleaning and Exploration
# Rename columns to have uniform names
names(spam_df) <- c("Category", "Message")

# Check for class imbalance
table(spam_df$Category)

# 4. Encode Categorical Variables
spam_df$Category <- as.factor(spam_df$Category)

# 5. Train-Test Split
set.seed(42)
index <- createDataPartition(spam_df$Category, p = 0.8, list = FALSE)
train_data <- spam_df[index, ]
test_data <- spam_df[-index, ]

# 6. Feature Extraction
# Convert messages to a document-term matrix
train_corpus <- VCorpus(VectorSource(train_data$Message)) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stemDocument)

dtm_train <- DocumentTermMatrix(train_corpus)
dtm_train <- removeSparseTerms(dtm_train, 0.99)
train_matrix <- as.matrix(dtm_train)
train_df <- as.data.frame(train_matrix)
train_df$Category <- train_data$Category


# Same for test data
test_corpus <- VCorpus(VectorSource(test_data$Message)) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stemDocument)

dtm_test <- DocumentTermMatrix(test_corpus, control = list(dictionary = Terms(dtm_train)))
test_matrix <- as.matrix(dtm_test)
test_df <- as.data.frame(test_matrix)
test_df$Category <- test_data$Category

# 7. Logistic Regression Model
log_model <- glm(Category ~ ., data = train_df, family = binomial)
summary(log_model)

# Predict on test set
log_pred <- predict(log_model, newdata = test_df, type = "response")
log_pred_class <- ifelse(log_pred > 0.5, "spam", "ham")

# 8. Evaluate Model
# Confusion Matrix
conf_matrix <- confusionMatrix(as.factor(log_pred_class), test_df$Category)
print(conf_matrix)

# ROC Curve and AUC
roc_log <- roc(test_df$Category, log_pred, levels = rev(levels(test_df$Category)))
plot(roc_log, col = "darkorange")
auc(roc_log)

# Part B - Decision Tree Analysis

# 1. Train Decision Tree Model
dt_model <- rpart(Category ~ ., data = train_df, method = "class")
rpart.plot(dt_model)

# Predict on test set
dt_pred <- predict(dt_model, newdata = test_df, type = "class")

# 2. Evaluate Model
# Confusion Matrix
conf_matrix_dt <- confusionMatrix(dt_pred, test_df$Category)
print(conf_matrix_dt)

# ROC Curve and AUC
dt_pred_prob <- predict(dt_model, newdata = test_df, type = "prob")[,2]
roc_dt <- roc(test_df$Category, dt_pred_prob, levels = rev(levels(test_df$Category)))
plot(roc_dt, col = "blue")
auc(roc_dt)

# 3. Comparison of Logistic Regression and Decision Tree
comparison <- data.frame(
  Model = c("Logistic Regression", "Decision Tree"),
  Accuracy = c(conf_matrix$overall['Accuracy'], conf_matrix_dt$overall['Accuracy']),
  AUC = c(auc(roc_log), auc(roc_dt))
)

print(comparison)
