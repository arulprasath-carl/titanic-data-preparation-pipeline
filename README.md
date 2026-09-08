# Titanic Data Preparation Pipeline

This project is a modular Python-based data preprocessing pipeline built using the Titanic dataset. The main goal of the project is to transform raw passenger data into a clean, structured, fully numerical, and machine-learning-ready format.

The pipeline focuses completely on data preparation and does not perform any model training or prediction. It is designed to demonstrate the important preprocessing steps that are usually required before applying machine learning algorithms.

The project begins by loading the Titanic dataset directly using Seaborn. It then validates whether all the required columns are present in the dataset and raises a custom error if any expected column is missing. Text and categorical values are cleaned and normalized to maintain consistency across the dataset.

Unnecessary, redundant, highly-missing, and target-leaking columns are removed to improve data quality and prevent data leakage. Missing numerical values are handled using median imputation, while missing categorical values are filled using the most frequent value or mode.

Categorical features are then converted into numerical form using suitable encoding techniques such as binary mapping, ordinal representation, and one-hot encoding. After preprocessing, the dataset is divided into training and testing sets using a stratified 80/20 split so that the survival class distribution remains approximately consistent in both datasets.

The final preprocessing step applies `StandardScaler` to standardize the input features. The scaler is fitted only on the training data and then used to transform both the training and testing sets. This ensures that information from the test set does not influence the training process and helps prevent data leakage.

The final output of this project consists of clean, encoded, scaled, and properly split feature and target datasets that are ready to be used by any downstream machine learning classification model.

## Key Features

- Loads the Titanic dataset using Seaborn
- Validates required dataset columns
- Uses custom exception handling for validation errors
- Cleans and normalizes text data
- Removes unnecessary and target-leaking columns
- Handles missing numerical values using median
- Handles missing categorical values using mode
- Encodes categorical variables into numerical format
- Performs stratified 80/20 train-test splitting
- Uses `random_state` for reproducible results
- Applies `StandardScaler` to the feature data
- Fits preprocessing statistics only on training data
- Prevents data leakage
- Produces an ML-ready dataset
- Uses a modular structure with separate preprocessing functions
- Includes exception handling and clear console output

## Project Workflow

Raw Titanic Dataset  
→ Data Validation  
→ Text Cleaning  
→ Remove Unnecessary Columns  
→ Handle Missing Values  
→ Encode Categorical Features  
→ Separate Features and Target  
→ Stratified Train-Test Split  
→ Feature Scaling  
→ Machine-Learning-Ready Dataset

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Scikit-learn

## Project Objective

The objective of this project is to demonstrate a complete and reusable data preprocessing workflow. It highlights how raw real-world data can be cleaned, transformed, validated, and prepared correctly before being passed into a machine learning model.

No prediction model is trained in this project. The focus is entirely on building a clean, reliable, modular, and leakage-free data preparation pipeline.
