### util.py

import os
import pandas as pd
from typing import List
from argparse import Namespace
import re


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
    #find all files which contain the prefix, they dont have to start with it
    folds = [f for f in files if prefix in f and f.endswith('.csv')]
    folds.sort()
    return folds

def create_dfs(data_path: str, train_fold: str, val_fold: str, data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a list of tuples containing training and validation dataframes.

    Args:
        data_path (str): The path to the dataframes.
        train_fold (str): The name of the training fold.
        val_fold (str): The name of the validation fold.
        data_dir (str): The name of the data sub_directory.

    Returns:
        tuple: (train_df, val_df) DataFrames for the training and validation datasets.
    """

    if data_dir in {'breast', 'acl'}:
        train_df = pd.read_csv(os.path.join(data_path, train_fold))
        val_df = pd.read_csv(os.path.join(data_path, val_fold))

    elif data_dir == 'hemorrhage':
        train_df = pd.read_csv(os.path.join(data_path, train_fold), usecols=['dir', 'label'])[['dir', 'label']]
        val_df = pd.read_csv(os.path.join(data_path, val_fold), usecols=['dir', 'label'])[['dir', 'label']]
        print(train_df.head())
        
    return train_df, val_df

def validate_args(args: Namespace) -> None:
    """
    Validates the arguments provided to the script, ensuring that they refer to valid directories, databases,
    model structures, and model names.

    Args:
        args (Namespace): Command line arguments parsed by argparse.

    Raises:
        FileNotFoundError: If the data directory does not exist.
        ValueError: If the provided database, structure, or model name is invalid.
    """
    # Validate data directory
    full_path = os.path.join('data', args.data_dir)
    valid_dirs = ['acl', 'breast', 'hemorrhage']
    if not os.path.isdir(full_path):
        raise FileNotFoundError(f"The directory specified does not exist: {full_path}")
    if args.data_dir not in valid_dirs:
        raise ValueError(f"Data directory '{args.data_dir}' not yet supported. Please choose from {valid_dirs}.")

    # Validate database choice
    valid_databases = ['RadImageNet', 'ImageNet']
    if args.database not in valid_databases:
        raise ValueError(f"Pre-trained database '{args.database}' does not exist. Please choose from {valid_databases}.")

    # Validate structure option
    if args.structure not in ['unfreezeall', 'freezeall'] and not re.match(r'unfreezetop\d+', args.structure):
        raise Exception('Freeze any layers? Choose to unfreezeall/freezeall/unfreezetop{i} layers for the network.')

    # Validate model name
    valid_models = ['ResNet50', 'DenseNet121', 'InceptionV3']
    if args.model_name not in valid_models:
        raise ValueError(f"Pre-trained network '{args.model_name}' does not exist. Please choose from {valid_models}.")