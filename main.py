#!/usr/bin/env python
# coding: utf-8

### main.py

import argparse
from validate_args import validate_args
from RadDataSet import create_dataloaders
from run_model import run_model
from FineTuneModel import get_compiled_model
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Name of the data directory, e.g., acl')
    parser.add_argument('--database', type=str, help='choose RadImageNet or ImageNet')
    parser.add_argument('--model_name', type=str, help='choose IRV2/ResNet50/DenseNet121/InceptionV3')
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--image_size', type=int, help='image size', default=256)
    parser.add_argument('--epoch', type=int, help='number of epochs', default=30)
    parser.add_argument('--structure', type=str, help='unfreezeall/freezeall/unfreezetop10', default=30)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    args = parser.parse_args()

    validate_args(args)

    # Set Cuda Device in Pytorch, discard parallelization for now
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = pd.read_csv(f'{args.data_dir}dataframe/train_fold1.csv')
    val_df = pd.read_csv(f'{args.data_dir}dataframe/val_fold1.csv')
    train_loader, val_loader = create_dataloaders(train_df, val_df, args.batch_size, args.image_size)

    model, optimizer, loss_fn = get_compiled_model()
    run_model(model, optimizer, loss_fn, train_loader, val_loader, args.epoch, device)

if __name__ == "__main__":
    main()