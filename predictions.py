#!/usr/bin/env python
# coding: utf-8

"""
predictions.py

Simple script to load a particular model, generate all of the test predictions,
and save to csv along with the true labels.
"""

import argparse
import os
import torch
import pandas as pd
from src.util import generate_model_param_str, get_full_data
from src.argparser import create_parser, validate_args
from src.data import create_dataloaders
from src.model import load_model


def main() -> None:
    """
    Load the best checkpointed model specified by the CLI args, eval on test data,
    and save all preds and labels to predictions/preds_{MODEL_PARAM_STR}.csv.
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

    # ============ load the data ===================
    partial_path = os.path.join("data", args.data_dir)
    data_path = os.path.join(partial_path, "dataframe")
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

    _, _, test_loader = create_dataloaders(
        train_df, val_df, test_df, args.batch_size, args.image_size, partial_path
    )

    if args.verbose:
        print("Data loaders created.\n")

    # =============== locate and load the model ====================
    MODEL_PARAM_STR = generate_model_param_str(
        data_dir=args.data_dir,
        backbone_model=args.backbone_model_name,
        pretrain=args.database,
        clf=args.clf,
        structure=args.structure,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout_prob=args.dropout_prob,
        fc_hidden_size_ratio=args.fc_hidden_size_ratio,
        num_filters=args.num_filters,
        kernel_size=args.kernel_size,
        epoch=args.epoch,
        image_size=args.image_size,
        lr_decay_method=args.lr_decay_method,
        lr_decay_beta=args.lr_decay_beta,
    )

    model_name = f"best_model_{MODEL_PARAM_STR}.pth"
    model_path = os.path.join(partial_path, "models", model_name)
    assert os.path.exists(
        model_path
    ), f"Model weights not found at {model_path}. Make sure to run this experiment before using predictions.py!"

    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]

    # ==== load in model  ======
    model = load_model(device, args)
    model.load_state_dict(model_state_dict)
    model.eval()  # set to eval mode, no training

    if args.verbose:
        print("======\nModel loaded!\n======\n")

    all_preds = []
    all_labels = []

    # ================= run model to generate all preds (test set) =================
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Save all model predictions and the true labels to CSV
    os.makedirs("predictions", exist_ok=True)
    output_file = os.path.join("predictions", f"preds_{MODEL_PARAM_STR}.csv")
    df = pd.DataFrame({"Prediction": all_preds, "Label": all_labels})
    df.to_csv(output_file, index=False)

    if args.verbose:
        print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
