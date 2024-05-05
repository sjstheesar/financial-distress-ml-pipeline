import os
import json
import pandas as pd


INPUT_DIR = "../data/"
OUTPUT_DIR = "../data/"

DATA_DICTIONARY = "Data Dictionary.xls"
HEADER_ROW = 1


#----------------------------------------------------------------------------#
def translate_data_type(data_type):
    """
    Translate a data type from the data dictionary to a domain in
    pandas DataFrame.

    Parameters:
        data_type (str): Data type in the data dictionary.

    Returns:
        str: Data type for pandas DataFrame.
    """
    if data_type in ["integer", "Y/N"]:
        return "int"

    if data_type in ["percentage", "real"]:
        return "float"

    return "object"


def save_data_types(data_dictonary, header_row):
    """
    Load data into a pandas DataFrame, extract data types from the data
    dictionary, fill missing values and modify data types.

    Parameters:
        data_dictonary (str): Name of the data dictionary.
        header_row (int): Row of the dictionary headers.

    Returns:
        None
    """
    try:
        data_dict = pd.read_excel(INPUT_DIR + data_dictonary, header=header_row)
        types = data_dict.Type.apply(translate_data_type)
        data_types = dict(zip(data_dict['Variable Name'], types))

        with open(OUTPUT_DIR + "data_types.json", 'w') as file:
            json.dump(data_types, file)

        print(("ETL process finished. Data dictionary wrote to 'data_types.json'"
               " under the directory {}.".format(OUTPUT_DIR)))
    except Exception as e:
        print(f"An error occurred: {e}")


def execute_etl():
    """
    Execute the ETL process.

    Parameters:
        None

    Returns:
        None
    """
    save_data_types(DATA_DICTIONARY, HEADER_ROW)


#----------------------------------------------------------------------------#
if __name__ == "__main__":
    execute_etl()
