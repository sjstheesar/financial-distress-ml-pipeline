import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from viz import read_data


pd.set_option('mode.chained_assignment', None)


INPUT_DIR = "../data/"
OUTPUT_DIR = "../processed_data/"

DATA_FILE = "credit-data.csv"
DATA_TYPES = "data_types.json"

MAX_OUTLIERS = {'MonthlyIncome': 1,
                'DebtRatio': 1}

TO_BINARIES = {'NumberOfTime30-59DaysPastDueNotWorse': 0,
               'NumberOfTime60-89DaysPastDueNotWorse': 0,
               'NumberOfTimes90DaysLate': 0}

TO_COMBINE = {'PastDue': ['NumberOfTimes90DaysLate',
                          'NumberOfTime60-89DaysPastDueNotWorse',
                          'NumberOfTime30-59DaysPastDueNotWorse']}
TO_ORDINAL = {'PastDue': True}

TO_DESCRETIZE = {'age': 5}
RIGHT_INCLUSIVE = {'age': True}

TO_ONE_HOT = ['zipcode']

TARGET = 'SeriousDlqin2yrs'
TO_DROP = ['PersonID', 'SeriousDlqin2yrs']

SPLIT_PARAMS = {'test_size': 0.25, 'random_state': 123}

SCALERS = [StandardScaler, MinMaxScaler]


#----------------------------------------------------------------------------#
def drop_max_outliers(data, drop_vars):
    """
    Drop rows with large outliers in specified columns.

    Parameters:
        data (DataFrame): The input DataFrame.
        drop_vars (dict): A dictionary mapping column names to the number of
                          outliers to drop.

    Returns:
        DataFrame: The modified DataFrame with outliers dropped.
    """
    for col_name, n in drop_vars.items():
        data.drop(data.nlargest(n, col_name, keep='all').index,
                  axis=0, inplace=True)

    return data


def convert_to_binary(data, to_bin_vars):
    """
    Convert specified columns to binary variables based on a cut point.

    Parameters:
        data (DataFrame): The input DataFrame.
        to_bin_vars (dict): A dictionary mapping column names to cut points.

    Returns:
        DataFrame: The modified DataFrame with binary variables added.
    """
    for var, cut_point in to_bin_vars.items():
        data[var] = np.where(data[var] > cut_point, 1, 0)

    return data


def combine_binary_variables(data, combine_vars, ordered):
    """
    Combine specified binary columns into a single ordinal or categorical variable.

    Parameters:
        data (DataFrame): The input DataFrame.
        combine_vars (dict): A dictionary mapping lists of column names to the
                             resulting variable name.
        ordered (dict): A dictionary mapping resulting variable names to whether they
                        should be ordered.

    Returns:
        DataFrame: The modified DataFrame with combined variables added.
    """
    for col_name, to_combine in combine_vars.items():
        dummies = data[to_combine]
        dummies['negative'] = np.where((dummies == 1).sum(axis=1) > 0, 0, 1)

        data[col_name] = pd.Categorical(dummies.idxmax(axis=1),
                                        categories=(to_combine + ['negative'])[::-1],
                                        ordered=ordered[col_name]).codes
        data.drop(to_combine, axis=1, inplace=True)

    return data


def apply_one_hot_encoding(data, cat_vars):
    """
    Apply one-hot encoding to specified categorical columns.

    Parameters:
        data (DataFrame): The input DataFrame.
        cat_vars (list): A list of column names to apply one-hot encoding.

    Returns:
        DataFrame: The modified DataFrame with one-hot encoded variables added.
    """
    for var in cat_vars:
        dummies = pd.get_dummies(data[var], prefix=var)
        data.drop(var, axis=1, inplace=True)
        data = pd.concat([data, dummies], axis=1)

    return data


def split_data(data, test=True, split_params=SPLIT_PARAMS):
    """
    Split the data into training and testing sets.

    Parameters:
        data (DataFrame): The input DataFrame.
        test (bool): Whether to perform a train-test split.
        split_params (dict): A dictionary of parameters for the train-test split.

    Returns:
        tuple: X_train, X_test, y_train, y_test, col_names
    """
    data.dropna(axis=0, subset=[TARGET], inplace=True)
    y = data[TARGET]
    data.drop(TO_DROP, axis=1, inplace=True)
    X = data

    if test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)
        return X_train, X_test, y_train, y_test, X.columns

    return X, None, y, None, X.columns


def impute_missing_values(X_train, X_test, data_types, ask=True):
    """
    Impute missing values in the training and testing sets.

    Parameters:
        X_train (DataFrame): The training features.
        X_test (DataFrame): The testing features.
        data_types (dict): A dictionary mapping column names to their data types.
        ask (bool): Whether to prompt the user for imputation method.

    Returns:
        tuple: X_train, X_test
    """
    if ask:
        imputer_index = int(input(("Up till now we support:\n"
                                   "\t1. Imputing with column mean\n"
                                   "\t2. Imputing with column median\n"
                                   "Please input the index (1 or 2) of your"
                                   " imputation method:\n")))
    else:
        imputer_index = 1
    data_types = {col: data_type for col, data_type in data_types.items()
                  if col in X_train.columns}

    if imputer_index == 1:
        X_train = X_train.fillna(X_train.mean()).astype(data_types)
        X_test = X_test.fillna(X_test.mean()).astype(data_types)
    else:
        X_train = X_train.fillna(X_test.median()).astype(data_types)
        X_test = X_test.fillna(X_test.median()).astype(data_types)
    print("Finished imputing missing values with feature {}.\n".\
          format(["means", "medians"][imputer_index - 1]))

    return X_train, X_test


def discretize_continuous_variables(X_train, X_test, cont_vars, right_inclusive):
    """
    Discretize continuous variables into a specified number of bins.

    Parameters:
        X_train (DataFrame): The training features.
        X_test (DataFrame): The testing features.
        cont_vars (dict): A dictionary mapping column names to the number of bins.
        right_inclusive (dict): A dictionary mapping column names to whether
                                 the bins should be right inclusive.

    Returns:
        tuple: X_train, X_test
    """
    for col_name, n in cont_vars.items():
        X_train[col_name] = pd.cut(X_train[col_name], n,
                                   right=right_inclusive[col_name]).cat.codes
        X_test[col_name] = pd.cut(X_test[col_name], n,
                                  right=right_inclusive[col_name]).cat.codes

    return X_train, X_test


def scale_features(X_train, X_test, ask=True, scale_test=False):
    """
    Scale the features using a specified scaler.

    Parameters:
        X_train (DataFrame): The training features.
        X_test (DataFrame): The testing features.
        ask (bool): Whether to prompt the user for the scaler method.
        scale_test (bool): Whether to scale the test set as well.

    Returns:
        tuple: X_train, X_test
    """
    if ask:
        scaler_index = int(input(("\nUp till now we support:\n"
                                  "\t1. StandardScaler\n"
                                  "\t2. MinMaxScaler\n"
                                  "Please input a scaler index (1 or 2):\n")))
    else:
        scaler_index = 1

    scaler = SCALERS[scaler_index - 1]()
    X_train = scaler.fit_transform(X_train.values.astype(float))
    if scale_test:
        X_test = scaler.transform(X_test.values.astype(float))
    print("Finished scaling the features.\n")

    return X_train, X_test


def save_processed_data(X_train, X_test, y_train, y_test, col_names):
    """
    Save the processed data to NumPy arrays and a pickle file.

    Parameters:
        X_train (DataFrame): The training features.
        X_test (DataFrame): The testing features.
        y_train (Series): The training target.
        y_test (Series): The testing target.
        col_names (list): A list of column names.

    Returns:
        None
    """
    if "processed_data" not in os.listdir("../"):
        os.mkdir("processed_data")

    np.savez(OUTPUT_DIR + 'X.npz', train=X_train, test=X_test)
    np.savez(OUTPUT_DIR + 'y.npz', train=y_train.values.astype(float),
             test=y_test.values.astype(float))
    with open(OUTPUT_DIR + 'col_names.pickle', 'wb') as handle:
        pickle.dump(col_names, handle)

    print(("Saved the resulting NumPy matrices to directory {}. Features are"
           " in 'X.npz' and target is in 'y.npz'. Column names are saved as"
           " 'col_names.pickle'.").format(OUTPUT_DIR))


def process_data():
    """
    Process the data by reading it, dropping outliers, converting variables,
    combining binaries, applying one-hot encoding, splitting into train-test sets,
    imputing missing values, discretizing continuous variables, scaling features,
    and saving the processed data.

    Returns:
        None
    """
    # Load data
    try:
        data, data_types = read_data(DATA_TYPES, DATA_FILE, dir_path=INPUT_DIR)
        print("Finished reading cleaned data.\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Drop rows with large outliers
    try:
        data = drop_max_outliers(data, MAX_OUTLIERS)
        print("Finished dropping extreme large values:")
        for col_name, n in MAX_OUTLIERS.items():
            print("\tDropped {} observations with extreme large values on '{}'.".\
                  format(n, col_name))
    except Exception as e:
        print(f"Error dropping outliers: {e}")
        return

    # Convert some variables to binary
    try:
        data = convert_to_binary(data, TO_BINARIES)
        print("\nFinished transforming the following variables: {}.\n".\
              format(list(TO_BINARIES.keys())))
    except Exception as e:
        print(f"Error converting to binary: {e}")
        return

    # Combine some binaries into categoricals or ordinals
    try:
        data = combine_binary_variables(data, TO_COMBINE, TO_ORDINAL)
        print("Finished combining binaries:")
        for col_name, to_combine in TO_COMBINE.items():
            print("\tCombined {} into a {} variable '{}'.".format(to_combine, \
                ["categorical", "ordinal"][int(TO_ORDINAL[col_name])], col_name))
    except Exception as e:
        print(f"Error combining binaries: {e}")
        return

    # Apply one-hot encoding on categoricals
    try:
        data = apply_one_hot_encoding(data, TO_ONE_HOT)
        print("\nFinished one-hot encoding the following categorical variables: {}\n".\
              format(TO_ONE_HOT))
    except Exception as e:
        print(f"Error applying one-hot encoding: {e}")
        return

    # Split the data into training and test sets
    try:
        X_train, X_test, y_train, y_test, col_names = split_data(data)
    except Exception as e:
        print(f"Error splitting data: {e}")
        return

    # Do imputation to fill in missing values
    try:
        X_train, X_test = impute_missing_values(X_train, X_test, data_types)
    except Exception as e:
        print(f"Error imputing missing values: {e}")
        return

    # Discretize some continuous features into ordinals
    try:
        X_train, X_test = discretize_continuous_variables(X_train, X_test, TO_DESCRETIZE, RIGHT_INCLUSIVE)
        print("Finished discretizing some continuous variables:")
        for col_name, n in TO_DESCRETIZE.items():
            print("\tDiscretized '{}' into {} bins.".format(col_name, n))
    except Exception as e:
        print(f"Error discretizing continuous variables: {e}")
        return

    # Scale training and test data
    try:
        X_train, X_test = scale_features(X_train, X_test, scale_test=True)
    except Exception as e:
        print(f"Error scaling features: {e}")
        return

    try:
        save_processed_data(X_train, X_test, y_train, y_test, col_names)
    except Exception as e:
        print(f"Error saving processed data: {e}")


#----------------------------------------------------------------------------#
if __name__ == "__main__":
    process_data()
