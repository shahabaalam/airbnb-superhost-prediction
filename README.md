# Airbnb Superhost Prediction

## Overview
This project is part of a machine learning for economics problem set. The objective is to predict whether a host is a "superhost" on Airbnb based on various listing and host characteristics. The analysis utilizes several predictive models, including logistic regression, support vector machines (SVM), and regularized regression, to understand the factors influencing superhost status.

## Problem Description
The task involves building and comparing machine learning models to classify hosts as superhosts or not, based on cleaned data from Airbnb listings. Key steps include data preprocessing, model training, and parameter tuning. The analysis aims to identify the best predictive model for superhost status using metrics like classification accuracy and cross-validation error.

## Features

### Library Imports
Uses R packages such as:
- `dplyr` and `tidyr` for data manipulation and cleaning
- `e1071` for SVM modeling and parameter tuning
- `glmnet` for L1-regularized logistic regression
- `broom` for tidying model outputs

### Data Preparation
- **Loading and Initial Inspection**: Reads `airbnb_data.csv` and inspects the structure, missing values, and distribution of key variables.
- **Handling Missing Values**:
  - Replaces missing values in numeric fields with medians.
  - Assigns default values in categorical fields where applicable (e.g., treating missing `host_is_superhost` as non-superhost).
- **Data Type Corrections**: Converts date columns to `Date` types and recasts logical fields.
- **Removing Duplicates and Irrelevant Columns**: Streamlines the data by removing unnecessary columns and duplicate records.

### Predictive Modeling
- **Model Selection**: Trains several models to classify superhosts, including:
  - **Linear Probability Model**: Using `review_scores_rating` as the sole predictor.
  - **Logit and Probit Models**: Logistic and probit regression models for binary classification.
  - **Support Vector Machine (SVM)**: Using a radial kernel and tuning parameters via cross-validation.
  - **Regularized Logistic Regression**: An L1-regularized logistic regression model, incorporating squared and interaction terms for key predictors.
- **Parameter Tuning**: Cross-validation is used to select optimal parameters (e.g., cost for SVM, lambda for L1-regularization).

### Model Comparison
- **Performance Evaluation**: Calculates mean classification error for each model and compares them to identify the most accurate predictive model.
- **Interpretability**: The coefficients of linear, logit, and probit models are interpreted to understand the relationship between `review_scores_rating` and superhost status.

## Requirements
- **R** (version 4.0 or higher)
- **R packages**: `dplyr`, `e1071`, `glmnet`, `broom`, `tidyr`

Install the required packages using:
```r
install.packages(c("dplyr", "e1071", "glmnet", "broom", "tidyr"))
```
## How to Use
1. Place the `airbnb_data.csv` file in the project directory.
2. Run the script in R to clean and preprocess the data, train models, and evaluate their performance.
3. Use the output to analyze which characteristics best predict superhost status.

## Future Enhancements
- Extend feature engineering to explore additional interactions and non-linear terms.
- Automate the process for deploying the model for real-time predictions.

## License
This project is open-source and available for modification under the MIT License.
