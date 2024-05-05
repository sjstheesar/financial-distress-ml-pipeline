import os
import subprocess
import pytest
import pandas as pd


subprocess.call('python etl.py', shell=True)

TEST_DIR = "./clean_data/"

TEST_ACCESS = os.listdir(TEST_DIR)
TEST_NO_MISSING = ["credit-clean.csv"]


#----------------------------------------------------------------------------#
def check_file(full_path):
    """
    Check whether a file contains at least some information and is readable.

    Parameters:
        full_path (str): Path to the file.

    Returns:
        None: Raises an AssertionError if any condition fails.
    """
    if not os.path.getsize(full_path) > 0:
        raise AssertionError(f"File {full_path} is empty.")
    if not os.path.isfile(full_path) and os.access(full_path, os.R_OK):
        raise AssertionError(f"File {full_path} does not exist or is not readable.")


#----------------------------------------------------------------------------#
@pytest.mark.parametrize("file_name", TEST_ACCESS)
def test_file_accessible(file_name):
    """
    Test whether the data file is accessible for further analysis.

    Parameters:
        file_name (str): Name of the file to be tested.

    Returns:
        None: Raises an AssertionError if any condition fails.
    """
    full_path = os.path.join(TEST_DIR, file_name)
    check_file(full_path)


@pytest.mark.parametrize("file_name", TEST_NO_MISSING)
def test_no_missing(file_name):
    """
    Test whether there are no missing values in the clean data.

    Parameters:
        file_name (str): Name of the file to be tested.

    Returns:
        None: Raises an AssertionError if any condition fails.
    """
    full_path = os.path.join(TEST_DIR, file_name)
    check_file(full_path)

    data = pd.read_csv(full_path)
    if not all(data.isnull().sum() == 0):
        raise AssertionError(f"File {full_path} contains missing values.")
