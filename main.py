#!/usr/bin/env python
# coding: utf-8

"""main.py

Wrapper for collecting args passed for experimental details, then iterating
through data (or train and val folds, if --use_folds passed), setting up dataloaders,
triggering training.

Also sets up the device automatically with pref cuda > mps > cpu.
"""

import argparse
import os
import torch
from src.util import get_full_data, find_data_folds, create_dfs
from src.data import create_dataloaders
from src.model import get_compiled_model, run_model
from src.argparser import create_parser, validate_args


def main() -> None:
    """
    Main function to parse command line arguments, set up data loaders,
    compile the model, and run the training process.
    """
    parser = create_parser()
    args = parser.parse_args()
    validate_args(args, verbose=True)

    # ====== Set Device, priority cuda > mps > cpu =======
    # discard parallelization for now
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = (
            True  # Enable cuDNN benchmark for optimal performance
        )
        torch.backends.cudnn.deterministic = (
            False  # Set to False to allow for the best performance
        )
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.verbose:
        print("\n=====================")
        print(f"Device Located: {device}")
        print(f"Loading data from directory: {args.data_dir}")
        print("=====================\n")

    # Set the path to your dataframes
    partial_path = os.path.join("data", args.data_dir)
    data_path = os.path.join(partial_path, "dataframe")

    # ============= Train Expts Using Folds like original RadImageNet work ==========
    if args.use_folds:
        # Determine available folds for training and validation
        train_folds = find_data_folds(data_path, "train")
        val_folds = find_data_folds(data_path, "val")

        if args.verbose:
            print(
                f"Found {len(train_folds)} training folds and {len(val_folds)} validation folds."
            )

        # Process each corresponding pair of train and validation folds
        fold = 0
        for train_file, val_file in zip(train_folds, val_folds):
            fold += 1
            if args.verbose:
                print(
                    f"Processing training fold: {train_file} and validation fold: {val_file}"
                )

            train_df, val_df = create_dfs(
                data_path, train_file, val_file, args.data_dir
            )

            train_loader, val_loader = create_dataloaders(
                train_df, val_df, args.batch_size, args.image_size, partial_path
            )

            if args.verbose:
                print("Data loaders created")

            model, optimizer, loss_fn = get_compiled_model(args, device)

            if args.verbose:
                print("Model compiled")

            run_model(
                model,
                optimizer,
                loss_fn,
                train_loader,
                val_loader,
                args,
                device,
                partial_path,
                args.database,
                fold,
            )

    # ============== Train Expts Using Complete Train, Validation Datasets (Recommended) ==========
    else:
        fold = "full"  # for logging filenames

        if args.data_dir == "acl":
            target_column = "acl_label"
        elif args.data_dir == "breast":
            target_column = "label"
        else:
            raise ValueError("Invalid Data Dir. Cannot determine target label.")

        train_df, val_df, test_df = get_full_data(
            data_path,
            force_reload_data=False,
            verbose=True,
            target_column=target_column,
        )

        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df, args.batch_size, args.image_size, partial_path
        )

        if args.verbose:
            print("Data loaders created")

        model, optimizer, loss_fn = get_compiled_model(args, device)

        if args.verbose:
            print("Model compiled")

        run_model(
            model,
            optimizer,
            loss_fn,
            train_loader,
            val_loader,
            test_loader,
            args,
            device,
            partial_path,
            args.database,
            fold,
        )


if __name__ == "__main__":
    main()
