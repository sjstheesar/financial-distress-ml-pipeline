import os
import pytest
import featureEngineering

from test_etl import check_file
from viz import read_clean_data


INPUT_DIR = "./clean_data/"
TEST_DIR = "./processed_data/"
TEST_ACCESS = os.listdir(TEST_DIR)
TEST_DISCRETIZE = [('age', 5), ('NumberOfDependents', 3)]
TEST_ONE_HOT = [['zipcode'], ['age'], ['zipcode', 'age']]
TEST_TO_BINARY = [{'NumberOfTime30-59DaysPastDueNotWorse': 0,
                   'NumberOfTime60-89DaysPastDueNotWorse': 0,
                   'NumberOfTimes90DaysLate': 0},
                  {'age': 18},
                  {'NumberOfDependents': 0}]


#----------------------------------------------------------------------------#
@pytest.mark.parametrize("file_name", TEST_ACCESS)
def test_accessibility(file_name):
    """
    Test whether the output data files are accessible for further analysis.

    Parameters:
        file_name (str): Name of the file to be tested.

    Returns:
        None: Raises an AssertionError if any condition fails.
    """
    full_path = os.path.join(TEST_DIR, file_name)
    check_file(full_path)


@pytest.mark.parametrize("var,bins", TEST_DISCRETIZE)
def test_discretize(var, bins):
    """
    Test whether the function for discretizing a continuous variable into a
    specific number of bins works properly.

    Parameters:
        var (str): Name of the variable to be discretized.
        bins (int): Number of bins.

    Returns:
        None: Raises an AssertionError if any condition fails.
    """
    data = read_clean_data("data_types.json", "credit-clean.csv",
                           dir_path=INPUT_DIR)
    discretized = featureEngineering.discretize(data[var], bins)

    if not len(discretized.value_counts()) == bins:
        raise AssertionError(f"Discretization of {var} into {bins} bins failed.")


@pytest.mark.parametrize("cat_vars", TEST_ONE_HOT)
def test_one_hot(cat_vars):
    """
    Test whether the function to create binary/dummy variables from
    categorical variables works properly.

    Parameters:
        cat_vars (list): List of categorical variable names.

    Returns:
        None: Raises an AssertionError if any condition fails.
    """
    data = read_clean_data("data_types.json", "credit-clean.csv",
                           dir_path=INPUT_DIR)

    col_counts = data.shape[1]
    drop_counts = len(cat_vars)
    add_counts = 0
    for var in cat_vars:
        add_counts += len(data[var].value_counts())

    processed_data = featureEngineering.one_hot(data, cat_vars)
    if col_counts - drop_counts + add_counts != processed_data.shape[1]:
        raise AssertionError(f"One-hot encoding of {cat_vars} failed.")


@pytest.mark.parametrize("to_bin_vars", TEST_TO_BINARY)
def test_to_binary(to_bin_vars):
    """
    Test whether the function to transform variables to binaries
    works properly.

    Parameters:
        to_bin_vars (dict): Dictionary mapping variable names to cut points.

    Returns:
        None: Raises an AssertionError if any condition fails.
    """
    data = read_clean_data("data_types.json", "credit-clean.csv",
                           dir_path=INPUT_DIR)
    data = featureEngineering.to_binary(data, to_bin_vars)

    for var in to_bin_vars:
        if len(data[var].value_counts()) > 2:
            raise AssertionError(f"Binary transformation of {var} failed.")
