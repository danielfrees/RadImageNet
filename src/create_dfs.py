### create_dfs.py

import pandas as pd
import os

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