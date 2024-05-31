#!/usr/bin/env python
# coding: utf-8

### main.py

import argparse
import os
import pandas as pd
import torch
from src.util import get_full_data, validate_args, find_data_folds, create_dfs
from src.data import create_dataloaders
from src.model import get_compiled_model, run_model



def main() -> None:
    """
    Main function to parse command line arguments, set up data loaders, 
    compile the model, and run the training process.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Name of the data directory, e.g., acl')
    parser.add_argument('--database', type=str, required=True, help='Choose RadImageNet or ImageNet')
    parser.add_argument('--backbone_model_name', type=str, required=True, help='Choose ResNet50, DenseNet121, or InceptionV3')
    parser.add_argument('--clf', type=str, required=True, help='Classifier type. Choose Linear, Nonlinear, or Conv.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--epoch', type=int, default=30, help='Number of epochs')
    parser.add_argument('--structure', type=str, default='unfreezeall', help='Structure: unfreezeall, freezeall, or unfreezetop10')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Prob. of dropping nodes in dropout layers')
    parser.add_argument('--fc_hidden_size_ratio', type=float, default=0.5, help='Ratio of hidden size to features for FC intermediate layers.')
    parser.add_argument('--num_filters', type = int, default = 4, help = "Number of Filters used in convolutional layers.")
    parser.add_argument('--kernel_size', type = int, default = 2, help = "Size of Kernel used in convolutional layers.")
    parser.add_argument('--log_every', type=int, default = 100)
    parser.add_argument('--use_folds', action='store_true', default = False, 
                        help = 'Run separate models for different train and validation folds. Useful for matching original RadImageNet baselines, but messy.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    validate_args(args)


    # ====== Set Device, priority cuda > mps > cpu =======
    # discard parallelization for now
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for optimal performance
        torch.backends.cudnn.deterministic = False  # Set to False to allow for the best performance
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    

    if args.verbose:
        print("\n=====================")
        print(f"Device Located: {device}")
        print(f"Loading data from directory: {args.data_dir}")
        print("=====================\n")

    # Set the path to your dataframes
    partial_path = os.path.join('data', args.data_dir)
    data_path = os.path.join(partial_path, 'dataframe')

    # ============= Train Expts Using Folds like original RadImageNet work ==========
    if args.use_folds:
        # Determine available folds for training and validation
        train_folds = find_data_folds(data_path, 'train')
        val_folds = find_data_folds(data_path, 'val')

        if args.verbose:
            print(f"Found {len(train_folds)} training folds and {len(val_folds)} validation folds.")

        # Process each corresponding pair of train and validation folds
        fold = 0
        for train_file, val_file in zip(train_folds, val_folds):
            fold += 1
            if args.verbose:
                print(f"Processing training fold: {train_file} and validation fold: {val_file}")

            train_df, val_df = create_dfs(data_path, train_file, val_file, args.data_dir)

            train_loader, val_loader = create_dataloaders(train_df, val_df, args.batch_size, args.image_size, partial_path)

            if args.verbose:
                print("Data loaders created")

            model, optimizer, loss_fn = get_compiled_model(args, device)

            if args.verbose:
                print("Model compiled")

            run_model(model, optimizer, loss_fn, train_loader, val_loader, args, device, partial_path, args.database, fold)

    # ============== Train Expts Using Complete Train, Validation Datasets (Recommended) ==========
    else:
        fold = 'full'  # for logging filenames

        train_df, val_df, _ = get_full_data(data_path, force_reload_data= False, verbose=True)

        train_loader, val_loader = create_dataloaders(train_df, val_df, args.batch_size, args.image_size, partial_path)

        if args.verbose:
            print("Data loaders created")

        model, optimizer, loss_fn = get_compiled_model(args, device)

        if args.verbose:
            print("Model compiled")

        run_model(model, optimizer, loss_fn, train_loader, val_loader, args, device, partial_path, args.database, fold)



if __name__ == "__main__":
    main()
