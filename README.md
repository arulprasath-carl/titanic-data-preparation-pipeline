Titanic Data Preparation Pipeline

This project is a modular Python-based data preprocessing pipeline built using the Titanic dataset. The main goal of the project is to transform raw passenger data into a clean, structured, fully numerical, and machine-learning-ready format.

The pipeline focuses completely on data preparation and does not perform any model training or prediction. It is designed to demonstrate the important preprocessing steps that are usually required before applying machine learning algorithms.

The project begins by loading the Titanic dataset directly using Seaborn. It then validates whether all the required columns are present in the dataset and raises a custom error if any expected column is missing. Text and categorical values are cleaned and normalized to maintain consistency across the dataset.

Unnecessary, redundant, highly-missing, and target-leaking columns are removed to improve data quality and prevent data leakage. Missing numerical values are handled using median imputation, while missing categorical values are filled using the most frequent value or mode.

Categorical features are then converted into numerical form using suitable encoding techniques such as binary mapping, ordinal representation, and one-hot encoding. After preprocessing, the dataset is divided into training and testing sets using a stratified 80/20 split so that the survival class distribution remains approximately consistent in both datasets.

The final preprocessing step applies StandardScaler to standardize the input features. The scaler is fitted only on the training data and then used to transform both the training and testing sets. This ensures that information from the test set does not influence the training process and helps prevent data leakage.

The final output of this project consists of clean, encoded, scaled, and properly split feature and target datasets that are ready to be used by any downstream machine learning classification model.

Key Features
Loads the Titanic dataset using Seaborn
Validates required dataset columns
Uses custom exception handling for validation errors
Cleans and normalizes text data
Removes unnecessary and target-leaking columns
Handles missing numerical values using median
Handles missing categorical values using mode
Encodes categorical variables into numerical format
Performs stratified 80/20 train-test splitting
Uses random_state for reproducible results
Applies StandardScaler to the feature data
Fits preprocessing statistics only on training data
Prevents data leakage
Produces an ML-ready dataset
Uses a modular structure with separate preprocessing functions
Includes exception handling and clear console output
Project Workflow
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
File Structure
titanic-data-preparation-pipeline/
│
├── main.py                   # Entry point — imports and runs the full pipeline
├── load_data.py               # Loads the Titanic dataset from Seaborn
├── validate_data.py           # Validates required columns; custom DataValidationError
├── clean_text.py               # Cleans and normalizes text/categorical values
├── drop_useless.py             # Drops redundant, mostly-empty, or leaky columns
├── handle_missing.py           # Fills missing values (median / mode)
├── encode_categoricals.py      # Encodes categorical features into numeric form
├── split_data.py                # Performs stratified 80/20 train-test split
├── scale_features.py            # Scales features using StandardScaler (train-only fit)
└── README.md                    # Project documentation

Each file contains a single, focused function so the pipeline stays modular, readable, and easy to test or extend.

How to Run

1. Install the required libraries:

bash
pip install pandas numpy seaborn scikit-learn

2. Make sure all the project files are in the same folder:

main.py, load_data.py, validate_data.py, clean_text.py, drop_useless.py,
handle_missing.py, encode_categoricals.py, split_data.py, scale_features.py

3. Run the pipeline:

bash
python main.py
Sample Output
============================================================
TITANIC SURVIVOR DATA PREPARATION PIPELINE
============================================================

[1] Original dataset shape: (891, 15)
Validation passed: all required columns are present.
Cleaned text formatting in columns: ['sex', 'embarked', 'class', 'who', 'deck', 'embark_town', 'alive']
Dropped columns: ['deck', 'embark_town', 'class', 'who', 'adult_male', 'alive']
Missing values handled (numeric -> median, categorical -> mode).
Encoded categorical columns. Remaining nominal columns one-hot encoded: ['embarked']

[2] Final processed feature shape (including target): (891, 10)
Data split into train/test sets (test_size=0.2, stratified on 'survived').

[3] X_train shape: (712, 9)
    X_test shape:  (179, 9)
    y_train shape: (712,)
    y_test shape:  (179,)

[4] Target distribution (full dataset):
survived
0    0.616
1    0.384

Features scaled using StandardScaler (fit on training data only).

============================================================
PREPROCESSING COMPLETED SUCCESSFULLY
============================================================
Final scaled X_train shape: (712, 9)
Final scaled X_test shape:  (179, 9)
Technologies Used
Python
Pandas
NumPy
Seaborn
Scikit-learn
Project Objective

The objective of this project is to demonstrate a complete and reusable data preprocessing workflow. It highlights how raw real-world data can be cleaned, transformed, validated, and prepared correctly before being passed into a machine learning model.

No prediction model is trained in this project. The focus is entirely on building a clean, reliable, modular, and leakage-free data preparation pipeline.

Author

Arulprasath GitHub: https://github.com/arulprasath-carl

License

This project is licensed under the MIT License — free to use, modify, and share for educational or personal purposes.

