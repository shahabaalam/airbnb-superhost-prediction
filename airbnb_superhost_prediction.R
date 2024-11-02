# Load necessary packages
library(dplyr)
library(e1071)
library(glmnet)
library(broom)
library(tidyr)

# Read the dataset
data <- read.csv("airbnb_data.csv")

# Check the structure of the dataset
str(data)

# Get a summary of the dataset
summary(data)

# Get the number of rows and columns
dim(data)

# Check for any missing values
colSums(is.na(data))


######Handling Missing Values
#Imputation of Numeric Columns
# Replace missing numeric values with median
data$host_response_rate[is.na(data$host_response_rate)] <- median(data$host_response_rate, na.rm = TRUE)
data$host_acceptance_rate[is.na(data$host_acceptance_rate)] <- median(data$host_acceptance_rate, na.rm = TRUE)
data$price[is.na(data$price)] <- median(data$price, na.rm = TRUE)

# Similarly for review scores
data$review_scores_rating[is.na(data$review_scores_rating)] <- median(data$review_scores_rating, na.rm = TRUE)
# Repeat for other review scores: accuracy, cleanliness, checkin, etc.
# Check for any missing values
colSums(is.na(data))

#Imputation of Categorical Columns
# Replace missing categorical values (assuming 'not a superhost' for NA)
data$host_is_superhost[is.na(data$host_is_superhost)] <- 0

# Correct Data Types
# Convert date columns
data$host_since <- as.Date(data$host_since, format = "%Y-%m-%d")
data$first_review <- as.Date(data$first_review, format = "%Y-%m-%d")
data$last_review <- as.Date(data$last_review, format = "%Y-%m-%d")

# Convert categorical columns to logical
data$instant_bookable <- ifelse(data$instant_bookable == "t", TRUE, FALSE)
data$host_has_profile_pic <- ifelse(data$host_has_profile_pic == "t", TRUE, FALSE)
data$host_identity_verified <- ifelse(data$host_identity_verified == "t", TRUE, FALSE)

#Removing Irrelevant Columns
# Drop irrelevant columns
data <- data %>% select(-X)

#Check for Duplicates
# Remove duplicates
data <- data %>% distinct()

library(tidyr)
data <- data %>% drop_na(host_since, host_listings_count, host_total_listings_count)
data <- data %>%
  mutate(beds = replace_na(beds, median(beds, na.rm = TRUE)))

# Replace missing review scores and reviews_per_month with 0
data <- data %>%
  mutate(
    across(
      starts_with("review_scores"), 
      ~ replace_na(., 0)
    ),
    reviews_per_month = replace_na(reviews_per_month, 0)
  )

# Replace missing first_review and last_review with 0
data <- data %>%
  mutate(
    first_review = replace_na(first_review, as.Date(0)),  # Replace with the default date
    last_review = replace_na(last_review, as.Date(0)),    # Replace with the default date
    reviews_per_month = replace_na(reviews_per_month, 0)   # Replace with 0
  )



# Check the remaining missing values
colSums(is.na(data))

###


# 1. Preliminary Data Cleaning:

# (a) Recode 'host_identity_verified' with 't' as 1 and 'f' as 0
data <- data %>%
  mutate(host_identity_verified = ifelse(host_identity_verified == TRUE, 1, 0))


# (b) Remove observations with NA in key variables
data <- data %>%
  drop_na(review_scores_rating, review_scores_accuracy, review_scores_value)

str(data)
colSums(is.na(data))  # Check for any remaining NA values


# 2. Analysis:

# (a) Set seed and randomly allocate 10% observations to a test set
# Set the seed for reproducibility
set.seed(0)

# Create a random sample of indices for the test set (10% of the data)
test_indices <- sample(1:nrow(data), size = 0.1 * nrow(data))

# Create the test set
test_set <- data[test_indices, ]

# Create the training set (the remaining 90%)
train_set <- data[-test_indices, ]

# Check the sizes of the resulting datasets
cat("Training set size:", nrow(train_set), "\n")
cat("Test set size:", nrow(test_set), "\n")


# (b) Estimate a linear probability model with review_scores_rating as the only predictor
# Fit the linear probability model
lpm_model <- lm(host_is_superhost ~ review_scores_rating, data = train_set)

# Summarize the model results
summary(lpm_model)


# (c) Estimate a logit model
# Fit the logit model
logit_model <- glm(host_is_superhost ~ review_scores_rating, data = train_set, family = binomial)

# Summarize the model results
summary(logit_model)


# (d) Estimate a probit model
# Fit the probit model
probit_model <- glm(host_is_superhost ~ review_scores_rating, data = train_set, family = binomial(link = "probit"))

# Summarize the model results
summary(probit_model)


# (e) Show the coefficients from the linear, logit, and probit models
# Extract coefficients from each model
linear_coefficients <- summary(lpm_model)$coefficients[, "Estimate"]
logit_coefficients <- summary(logit_model)$coefficients[, "Estimate"]
probit_coefficients <- summary(probit_model)$coefficients[, "Estimate"]

# Combine coefficients into a data frame
coefficients_table <- data.frame(
  Model = c("Linear Probability Model", "Logit Model", "Probit Model"),
  Coefficient = c(linear_coefficients["review_scores_rating"],
                  logit_coefficients["review_scores_rating"],
                  probit_coefficients["review_scores_rating"])
)

# Print the coefficients table
print(coefficients_table)


#set seed, cost
install.packages("caret")
library(caret)

# (f) SVM with radial kernel, gamma = 0.1, and 10-fold CV for cost parameter selection
# Calculate sigma based on the given gamma value
gamma <- 0.1
sigma_value <- 1 / (2 * gamma)

# Convert the outcome variable to a factor for classification
train_set$host_is_superhost <- as.factor(train_set$host_is_superhost)

# Set seed for reproducibility
set.seed(0)

# Define the train control for 10-fold CV
train_control <- trainControl(method = "cv", number = 10)

# Define the cost grid: {10^-2, 10^-1, 10^0, 10^1, 10^2}, with a constant sigma
grid <- expand.grid(sigma = sigma_value, C = 10^(-2:2))

# Train the SVM with radial kernel, gamma = 0.1
svm_model <- train(host_is_superhost ~ review_scores_rating, 
                   data = train_set, 
                   method = "svmRadial",
                   trControl = train_control, 
                   tuneGrid = grid,
                   preProcess = c("center", "scale"),  # Scaling and centering the data
                   metric = "Accuracy", 
                   tuneLength = 5)

# Print the best model
print(svm_model)


# (g) SVM with gamma = 10, repeat cross-validation for cost parameter selection
# Setting gamma to 10
# Load caret library

# Set gamma and calculate corresponding sigma
gamma <- 10
sigma_value <- 1 / (2 * gamma)  # sigma = 0.05

# Set seed for reproducibility
set.seed(0)

# Train the SVM model with radial kernel using calculated sigma (equivalent to gamma = 10)
svm_model_gamma10 <- train(host_is_superhost ~ review_scores_rating, 
                           data = train_set, 
                           method = "svmRadial",
                           trControl = train_control, 
                           preProcess = c("center", "scale"),  # Scaling and centering the data
                           metric = "Accuracy", 
                           tuneGrid = expand.grid(sigma = sigma_value, C = 10^(-2:2)))

# Print the result of the SVM model
print(svm_model_gamma10)


# (h) L1 regularized logistic regression with interactions and squared terms
# Load necessary libraries
library(dplyr)  # For the pipe operator and mutate function
library(glmnet)  # For the LASSO model
# Calculate host_experience in years (assuming today's date is 2024-10-25)
train_set <- train_set %>%
  mutate(
    host_experience = as.numeric(difftime(Sys.Date(), host_since, units = "days")) / 365
  )

# Create new variables for squared terms and interactions
train_set <- train_set %>%
  mutate(
    review_scores_rating_sq = review_scores_rating^2,
    host_experience_sq = host_experience^2,
    review_scores_accuracy_sq = review_scores_accuracy^2,
    beds_sq = beds^2,
    review_scores_value_sq = review_scores_value^2,
    interaction_ratings_experience = review_scores_rating * host_experience,
    interaction_ratings_accuracy = review_scores_rating * review_scores_accuracy,
    interaction_ratings_beds = review_scores_rating * beds,
    interaction_ratings_value = review_scores_rating * review_scores_value,
    interaction_experience_accuracy = host_experience * review_scores_accuracy,
    interaction_experience_beds = host_experience * beds,
    interaction_experience_value = host_experience * review_scores_value,
    interaction_accuracy_beds = review_scores_accuracy * beds,
    interaction_accuracy_value = review_scores_accuracy * review_scores_value,
    interaction_beds_value = beds * review_scores_value
  )

# Prepare data for glmnet
x_train <- model.matrix(host_is_superhost ~ review_scores_rating + host_experience + 
                          review_scores_accuracy + beds + review_scores_value + 
                          review_scores_rating_sq + host_experience_sq + 
                          review_scores_accuracy_sq + beds_sq + 
                          review_scores_value_sq + 
                          interaction_ratings_experience + interaction_ratings_accuracy + 
                          interaction_ratings_beds + interaction_ratings_value + 
                          interaction_experience_accuracy + interaction_experience_beds + 
                          interaction_experience_value + interaction_accuracy_beds + 
                          interaction_accuracy_value + interaction_beds_value,
                        data = train_set)[, -1]  # Remove intercept

y_train <- train_set$host_is_superhost

# Fit the lasso model using cross-validation
set.seed(0)
lasso_cv_model <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", nfolds = 10)

# Check the optimal lambda value (no need to report)
optimal_lambda <- lasso_cv_model$lambda.min
cat("Optimal Lambda from CV:", optimal_lambda, "\n")


# (i) Calculate mean classification error for all models
# Load necessary library for creating a table
library(dplyr)

# 1. Predictions for each model
# Linear Probability Model
linear_pred <- predict(lpm_model, newdata = test_set)
linear_pred_class <- ifelse(linear_pred > 0.5, 1, 0)

# Logit Model
logit_pred <- predict(logit_model, newdata = test_set, type = "response")
logit_pred_class <- ifelse(logit_pred > 0.5, 1, 0)

# Probit Model
probit_pred <- predict(probit_model, newdata = test_set, type = "response")
probit_pred_class <- ifelse(probit_pred > 0.5, 1, 0)

# SVM Model with radial kernel
svm_pred <- predict(svm_model, newdata = test_set)

# SVM Model with gamma set to 10
svm_gamma10_pred <- predict(svm_model_gamma10, newdata = test_set)

# Ensure test_set has the same transformations as the train_set
# Convert host_since to numeric host_experience (years since the host started)
test_set <- test_set %>%
  mutate(
    host_experience = as.numeric(difftime(Sys.Date(), host_since, units = "days")) / 365,
    
    # Create squared terms and interaction terms
    review_scores_rating_sq = review_scores_rating^2,
    host_experience_sq = host_experience^2,
    review_scores_accuracy_sq = review_scores_accuracy^2,
    beds_sq = beds^2,
    review_scores_value_sq = review_scores_value^2,
    
    interaction_ratings_experience = review_scores_rating * host_experience,
    interaction_ratings_accuracy = review_scores_rating * review_scores_accuracy,
    interaction_ratings_beds = review_scores_rating * beds,
    interaction_ratings_value = review_scores_rating * review_scores_value,
    
    interaction_experience_accuracy = host_experience * review_scores_accuracy,
    interaction_experience_beds = host_experience * beds,
    interaction_experience_value = host_experience * review_scores_value,
    
    interaction_accuracy_beds = review_scores_accuracy * beds,
    interaction_accuracy_value = review_scores_accuracy * review_scores_value,
    interaction_beds_value = beds * review_scores_value
  )

# L1 Regularized Logistic Regression Prediction
lasso_pred <- predict(lasso_model, newx = as.matrix(test_set[, -which(names(test_set) == "host_is_superhost")]), type = "response")
lasso_pred_class <- ifelse(lasso_pred > 0.5, 1, 0)

# Continue with classification error calculations as before
classification_errors <- data.frame(
  Model = c("Linear Probability Model", 
            "Logit Model", 
            "Probit Model", 
            "SVM with Radial Kernel", 
            "SVM with Gamma = 10", 
            "L1 Regularized Logistic Regression"),
  
  Error = c(
    mean(linear_pred_class != test_set$host_is_superhost),
    mean(logit_pred_class != test_set$host_is_superhost),
    mean(probit_pred_class != test_set$host_is_superhost),
    mean(svm_pred != test_set$host_is_superhost),
    mean(svm_gamma10_pred != test_set$host_is_superhost),
    mean(lasso_pred_class != test_set$host_is_superhost)
  )
)

# Show the table of classification errors
print(classification_errors)





