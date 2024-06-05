#!/usr/bin/env python
# coding: utf-8

"""util.py

Handles train/test/validation splits. Handles argument parsing validation for main.

"""

import os
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split


def merge_folds(files, directory, prefix):
    """
    Merge all folds into a single dataframe, handling duplicates.

    Args:
        files (list): List of filenames to be merged.
        directory (str): Directory path.
        prefix (str): Prefix to filter and sort files by.

    Returns:
        pd.DataFrame: Combined dataframe.
    """
    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(directory, file))
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset="filename")

    return combined_df


def check_and_resplit_data(train_df, val_df, test_df, verbose, target_column):
    """
    Check for overlap and re-split the data if necessary, stratifying based on the target column.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        verbose (bool): Whether to print verbose output.
        target_column (str): The name of the target column to stratify on.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Re-split train, val, and test data.
    """
    if verbose:
        print("\nChecking for overlaps...")

    combined_df = pd.concat(
        [train_df, val_df, test_df], ignore_index=True
    ).drop_duplicates()

    if len(combined_df) < (len(train_df) + len(val_df) + len(test_df)):
        if verbose:
            print("Data leakage detected! Re-splitting the data...")

        train_df, temp_df = train_test_split(
            combined_df,
            test_size=0.25,
            random_state=42,
            stratify=combined_df[target_column],
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.4, random_state=42, stratify=temp_df[target_column]
        )  # Updated to have more val data

        total_size = len(combined_df)
        train_size = len(train_df)
        val_size = len(val_df)
        test_size = len(test_df)

        if verbose:
            print(
                (
                    f"\nRe-split sizes - Train: {train_size / total_size * 100:.2f}%, "
                    f"Val: {val_size / total_size * 100:.2f}%, "
                    f"Test: {test_size / total_size * 100:.2f}%\n"
                )
            )

    return train_df, val_df, test_df


def get_full_data(
    directory: str,
    force_reload_data: bool = False,
    verbose: bool = False,
    target_column: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get (and create, if needed) the full data for the specified directory.

    Args:
        directory (str): path to the desired data. e.g. 'data/acl/dataframe'
        force_reload_data (bool): whether to force reloading the data even if it exists.
        verbose (bool): whether to print verbose output.
        target_column (str): The name of the target column to stratify on.


    Returns:
        tuple[pd.DataFrame]: the training, validation, and test dataframes containing the
            full training, validation, and test data, respectively.

    Example:
        >>> train_df, val_df, test_df = get_full_data(data_path)
    """
    if verbose:
        print("============== Loading Combined (Full) Data ===================")
    files = os.listdir(directory)

    data_dict = {"train": None, "val": None, "test": None}

    for split in data_dict.keys():
        split_path = os.path.join(directory, f"{split}.csv")
        if not os.path.exists(split_path) or force_reload_data:
            folds = get_sorted_folds(files, split)
            if verbose:
                print(f"Processing {split} folds: {folds}")
            data_dict[split] = merge_folds(folds, directory, split)
            data_dict[split].to_csv(split_path, index=False)
        else:
            data_dict[split] = pd.read_csv(split_path)

    train_df, val_df, test_df = check_and_resplit_data(
        data_dict["train"], data_dict["val"], data_dict["test"], verbose, target_column
    )
    if verbose:
        print("Full train, validation, test CSVs created and DFs loaded.")
        print("===========================================================\n")

    # Save the resplit dataframes to their respective paths
    train_df.to_csv(os.path.join(directory, "train.csv"), index=False)
    val_df.to_csv(os.path.join(directory, "val.csv"), index=False)
    test_df.to_csv(os.path.join(directory, "test.csv"), index=False)

    return train_df, val_df, test_df


def get_sorted_folds(files, prefix):
    """
    Helper func. Given a list of files, and a prefix, find all data fold csvs.

    prefix should be one of 'train', 'val', 'test'

    Returns the sorted csvs.
    """
    folds = [
        f for f in files if prefix in f and "fold" in f and f.endswith(".csv")
    ]  # adjusted to make sure we don't accidentally use full data too
    folds.sort()
    return folds


def find_data_folds(directory: str, prefix: str) -> List[str]:
    """
    Searches a specified directory for CSV files that begin with a given prefix
    and returns a sorted list of these files.

    Args:
        directory (str): The path to the directory where the files are located.
        prefix (str): The prefix that the CSV files should start with to be included in the list.

    Returns:
        List[str]: A list of sorted filenames that match the given prefix and end with '.csv'.

    Example:
        >>> find_data_folds('/path/to/data', 'train')
        ['train_fold1.csv', 'train_fold2.csv', 'train_fold3.csv']
    """
    files = os.listdir(directory)
    # find all files which contain the prefix, they dont have to start with it
    folds = get_sorted_folds(files, prefix)
    return folds


def create_dfs(
    data_path: str, train_fold: str, val_fold: str, data_dir: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    if data_dir in {"breast", "acl"}:
        train_df = pd.read_csv(os.path.join(data_path, train_fold))
        val_df = pd.read_csv(os.path.join(data_path, val_fold))

    elif data_dir == "hemorrhage":
        train_df = pd.read_csv(
            os.path.join(data_path, train_fold), usecols=["dir", "label"]
        )[["dir", "label"]]
        val_df = pd.read_csv(
            os.path.join(data_path, val_fold), usecols=["dir", "label"]
        )[["dir", "label"]]
        print(train_df.head())

    return train_df, val_df


# ================== fun w hyperparams (logging and loading tools) ===============
def generate_model_param_str(
    data_dir,
    backbone_model,
    pretrain,
    clf,
    structure,
    lr,
    batch_size,
    dropout_prob,
    fc_hidden_size_ratio,
    num_filters,
    kernel_size,
    epoch,
    image_size,
    lr_decay_method,
    lr_decay_beta,
):
    """
    Generates a model parameter string for a unique set of hyperparameters.
    """
    model_param_str = (
        f"{data_dir}_backbone_{backbone_model}_pretrain_{pretrain}_clf_{clf}_fold_full_"
        f"structure_{structure}_lr_{lr}_batchsize_{batch_size}_"
        f"dropprob_{dropout_prob}_fcsizeratio_{fc_hidden_size_ratio}_"
        f"numfilters_{num_filters}_kernelsize_{kernel_size}_epochs_{epoch}_"
        f"imagesize_{image_size}_lrdecay_{lr_decay_method}_lrbeta_{lr_decay_beta}"
    )
    return model_param_str


def parse_hyperparams(hyperparams_str):
    """
    Parses the model hyperparameters from the string into a dict.

    Args:
        hyperparams_str (str): The model hyperparameters string.

    Returns:
        dict: A dictionary of hyperparameters.
    """
    params = hyperparams_str.split("_")
    hyperparams = {}
    hyperparams["task"] = params[0]
    i = 1
    while i < len(params):
        if params[i] == "backbone":
            hyperparams["backbone"] = params[i + 1]
            i += 2
        elif params[i] == "pretrain":
            hyperparams["pretrain"] = params[i + 1]
            i += 2
        elif params[i] == "clf":
            hyperparams["clf"] = params[i + 1]
            i += 2
        elif params[i] == "fold":
            hyperparams["fold"] = params[i + 1]
            i += 2
        elif params[i] == "structure":
            hyperparams["structure"] = params[i + 1]
            i += 2
        elif params[i] == "lr":
            hyperparams["lr"] = params[i + 1]
            i += 2
        elif params[i] == "batchsize":
            hyperparams["batchsize"] = params[i + 1]
            i += 2
        elif params[i] == "dropprob":
            hyperparams["dropprob"] = params[i + 1]
            i += 2
        elif params[i] == "fcsizeratio":
            hyperparams["fcsizeratio"] = params[i + 1]
            i += 2
        elif params[i] == "numfilters":
            hyperparams["numfilters"] = params[i + 1]
            i += 2
        elif params[i] == "kernelsize":
            hyperparams["kernelsize"] = params[i + 1]
            i += 2
        elif params[i] == "epochs":
            hyperparams["epochs"] = params[i + 1]
            i += 2
        elif params[i] == "imagesize":
            hyperparams["imagesize"] = params[i + 1]
            i += 2
        elif params[i] == "lrdecay":
            hyperparams["lrdecay"] = params[i + 1]
            i += 2
        elif params[i] == "lrbeta":
            hyperparams["lrbeta"] = params[i + 1]
            i += 2
        elif params[i] == "amp":
            hyperparams["amp"] = params[i + 1]
            i += 2
        elif params[i] == "logevery":
            hyperparams["logevery"] = params[i + 1]
            i += 2
        elif params[i] == "usefolds":
            hyperparams["usefolds"] = params[i + 1]
            i += 2
        elif params[i] == "verbose":
            hyperparams["verbose"] = params[i + 1]
            i += 2
        else:
            i += 1
    return hyperparams
