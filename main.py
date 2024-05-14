#!/usr/bin/env python
# coding: utf-8

### main.py

import argparse
from src.validate_args import validate_args
from src.RadDataSet import create_dataloaders
from src.run_model import run_model
from src.FineTuneModel import get_compiled_model
from src.find_data_folds import find_data_folds
import os
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Name of the data directory, e.g., acl')
    parser.add_argument('--database', type=str, help='choose RadImageNet or ImageNet')
    parser.add_argument('--model_name', type=str, help='choose ResNet50/DenseNet121/InceptionV3')
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--image_size', type=int, help='image size', default=256)
    parser.add_argument('--epoch', type=int, help='number of epochs', default=30)
    parser.add_argument('--structure', type=str, help='unfreezeall/freezeall/unfreezetop10', default='unfreezeall')
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    args = parser.parse_args()

    validate_args(args)

    # Set Cuda Device in Pytorch, discard parallelization for now
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the path to your dataframes
    partial_path = os.path.join('data', args.data_dir)
    data_path = os.path.join(partial_path, 'dataframe')

    # Determine available folds for training and validation
    train_folds = find_data_folds(data_path, 'train_fold')
    print(train_folds)
    val_folds = find_data_folds(data_path, 'val_fold')

    # Process each corresponding pair of train and validation folds
    for train_file, val_file in zip(train_folds, val_folds):
        train_df = pd.read_csv(os.path.join(data_path, train_file))
        print(train_df.head())
        val_df = pd.read_csv(os.path.join(data_path, val_file))
        
        train_loader, val_loader = create_dataloaders(train_df, val_df, args.batch_size, args.image_size, partial_path)
        
        model, optimizer, loss_fn = get_compiled_model(args, device)
        run_model(model, optimizer, loss_fn, train_loader, val_loader, args.epoch, device)

if __name__ == "__main__":
    main()