### find_data_folds.py
import os

from typing import List

def find_data_folds(directory: str, prefix: str) -> List[str]:
    """
    Searches a specified directory for CSV files that begin with a given prefix and returns a sorted list of these files.

    Args:
        directory (str): The path to the directory where the files are located.
        prefix (str): The prefix that the CSV files should start with to be included in the list.

    Returns:
        List[str]: A list of sorted filenames that match the given prefix and end with '.csv'.

    Example:
        >>> find_data_folds('/path/to/data', 'train_fold')
        ['train_fold1.csv', 'train_fold2.csv', 'train_fold3.csv']
    """
    files = os.listdir(directory)
    folds = [f for f in files if f.startswith(prefix) and f.endswith('.csv')]
    folds.sort()
    return folds