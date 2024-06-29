# Financial Distress ML Pipeline

## Overview
This project is a machine learning pipeline designed to analyze financial distress data. It includes scripts for ETL (Extract, Transform, Load), feature engineering, model training, and visualization.

## Project Structure
```
financial-distress-ml-pipeline/
├── codes/
│   ├── etl.py
│   ├── featureEngineering.py
│   ├── train.py
│   └── viz.py
├── data/
│   ├── credit-data.csv
│   ├── Data Dictionary.xls
│   └── data_types.json
├── processed_data/
│   ├── col_names.pickle
│   ├── X.npz
│   └── y.npz
├── requirements.txt
└── test_etl.py
    ├── test_featureEngineering.py
    └── test_viz.py
```

## Setup
1. **Install Dependencies**: Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run ETL Script**: This script processes the raw data and saves it to `processed_data/`.
   ```bash
   python codes/etl.py
   ```

3. **Run Feature Engineering Script**: This script transforms the processed data.
   ```bash
   python codes/featureEngineering.py
   ```

4. **Train Models**: This script trains various machine learning models and evaluates their performance.
   ```bash
   python codes/train.py
   ```

5. **Visualize Results**: This script generates visualizations of the data and model results.
   ```bash
   python codes/viz.py
   ```

## Scripts Documentation

### etl.py
- **Purpose**: Processes raw credit data.
- **Functions**:
  - `translate_data_type(data_type)`: Translates data types from the dictionary to pandas DataFrame types.
  - `save_data_types(data_dictonary, header_row)`: Saves translated data types to a JSON file.
  - `perform_etl()`: Orchestrates the ETL process.

### featureEngineering.py
- **Purpose**: Engineers features for machine learning models.
- **Functions**:
  - `drop_max_outliers(data, drop_vars)`: Drops rows with large outliers.
  - `convert_to_binary(data, to_bin_vars)`: Converts specified columns to binary variables.
  - `combine_binary_variables(data, combine_vars, ordered)`: Combines binary variables into a single categorical or ordinal variable.
  - `apply_one_hot_encoding(data, cat_vars)`: Applies one-hot encoding to categorical variables.
  - `split_data(data, test=True, split_params=SPLIT_PARAMS)`: Splits data into training and testing sets.
  - `impute_missing_values(X_train, X_test, data_types, ask=True)`: Imputes missing values in the training and testing sets.
  - `discretize_continuous_variables(X_train, X_test, cont_vars, right_inclusive)`: Discretizes continuous variables into bins.
  - `scale_features(X_train, X_test, ask=True, scale_test=False)`: Scales features using a specified scaler.
  - `save_processed_data(X_train, X_test, y_train, y_test, col_names)`: Saves processed data to NumPy arrays and a pickle file.
  - `process_data()`: Orchestrates the entire feature engineering process.

### train.py
- **Purpose**: Trains machine learning models and evaluates their performance.
- **Functions**:
  - `ask_user()`: Prompts the user for a classifier and metric index.
  - `load_preprocessed_features(dir_path=INPUT_DIR, test=True)`: Loads pre-processed feature matrices.
  - `build_default_benchmark(data, metric_index)`: Builds and evaluates the default decision tree benchmark model.
  - `predict_probabilities(clf, X_test)`: Predicts probabilities for the test set.
  - `perform_cross_validation(clf, skf, data, metric_index, threshold)`: Performs cross-validation and evaluates the model.
  - `find_best_threshold(model_index, metric_index, train_data, verbose=True, plot=True)`: Finds the best threshold for the model.
  - `tune_hyperparameters(model_index, metric_index, train_data, best_threshold, n_folds=10, verbose=True)`: Tunes hyperparameters using grid search and cross-validation.
  - `calculate_precision_at_k(y_true, y_scores, k)`: Calculates precision at a given threshold.
  - `evaluate_tuned_model(model_index, metric_index, best_threshold, best_grid, data, plot=True, verbose=True, write=False, output=None)`: Evaluates the tuned model on the test set.
  - `train_and_evaluate_model(model_index, metric_index, data, train_data, write=False, output=None)`: Trains and evaluates a specific model.

### viz.py
- **Purpose**: Generates visualizations of the data and model results.
- **Functions**:
  - `read_clean_data(data_types_json, data_file, dir_path=INPUT_DIR)`: Reads clean data from CSV files.
  - `bar_plot(ax, data, column, sub=True, labels=["", "", ""], x_tick=[None, None])`: Generates a bar plot for a given column.
  - `hist_plot(ax, ds, col, cut=False)`: Generates a histogram plot for a given column.
  - `hist_panel(data, panel_title="", cut=False)`: Generates a panel of histograms for multiple columns.
  - `corr_triangle(ax, data, sub=False, plot_title="")`: Generates a correlation triangle plot.
  - `box_plot(data)`: Generates a box plot for the data.
  - `plot_predicted_scores(cv_scores, title="")`: Plots predicted scores against actual values.
  - `plot_precision_recall(y_true, y_score, baseline, title="")`: Plots precision-recall curve.
  - `plot_auc_roc(clf, data, title="")`: Plots ROC curve and AUC score.
  - `plot_feature_importances(importances, col_names, n=5, title="")`: Plots feature importances.

## Testing
- **Test ETL**: Ensures the ETL process runs successfully.
  ```bash
  pytest test_etl.py
  ```

- **Test Feature Engineering**: Ensures feature engineering scripts run successfully.
  ```bash
  pytest test_featureEngineering.py
  ```

- **Test Visualization**: Ensures visualization scripts run successfully.
  ```bash
  pytest test_viz.py
  ```

## Contributing
Contributions are welcome! Please follow the guidelines in `CONTRIBUTING.md`.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
