#!/usr/bin/env python
# coding: utf-8

"""
argparser.py

Setup and validate CLI arguments for various scripts.
"""

import argparse
from argparse import Namespace
import os
import re


def create_parser():
    """
    Main function to manage creating the argparser for args shared across
    scripts. Mainly controls hyperparameters around model training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Name of the data directory, e.g., acl",
    )
    parser.add_argument(
        "--database", type=str, required=True, help="Choose RadImageNet or ImageNet"
    )
    parser.add_argument(
        "--backbone_model_name",
        type=str,
        required=True,
        help="Choose ResNet50, DenseNet121, or InceptionV3",
    )
    parser.add_argument(
        "--clf",
        type=str,
        required=True,
        help="Classifier type. Choose Linear, Nonlinear, Conv, or ConvSkip.",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--epoch", type=int, default=30, help="Number of epochs")
    parser.add_argument(
        "--structure",
        type=str,
        default="unfreezeall",
        help="Structure: unfreezeall, freezeall, or unfreezetop10",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--lr_decay_method",
        type=str,
        default=None,
        help=(
            "LR Decay Method. Choose from None (default), 'cosine' for Cosine "
            "Annealing, 'beta' for multiplying LR by lr_decay_beta each epoch"
        ),
    )
    parser.add_argument(
        "--lr_decay_beta",
        type=float,
        default=0.5,
        help="Beta for LR Decay. Multiply LR by lr_decay_beta each epoch if lr_decay_method = 'beta'",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.5,
        help="Prob. of dropping nodes in dropout layers",
    )
    parser.add_argument(
        "--fc_hidden_size_ratio",
        type=float,
        default=0.5,
        help="Ratio of hidden size to features for FC intermediate layers.",
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        default=4,
        help="Number of Filters used in convolutional layers.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=2,
        help="Size of Kernel used in convolutional layers.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Enable AMP for faster mixed-precision training. Need CUDA + "
            "recommend batch size of 256+ to use throughput gains if running AMP."
        ),
    )
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--use_folds",
        action="store_true",
        default=False,
        help=(
            "Run separate models for different train and validation folds. "
            "Useful for matching original RadImageNet baselines, but messy."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser


def validate_args(args: Namespace, verbose: bool = False) -> None:
    """
    Validates the arguments provided to the script, ensuring that they refer to
    valid directories, databases, model structures, and model names.

    Args:
        args (Namespace): Command line arguments parsed by argparse.

    Raises:
        FileNotFoundError: If the data directory does not exist.
        ValueError: If the provided database, structure, or model name is invalid.
    """
    # Validate data directory
    full_path = os.path.join("data", args.data_dir)
    valid_dirs = ["acl", "breast"]
    if not os.path.isdir(full_path):
        raise FileNotFoundError(f"The directory specified does not exist: {full_path}")
    if args.data_dir not in valid_dirs:
        raise ValueError(
            f"Data directory '{args.data_dir}' not yet supported. Please choose from {valid_dirs}."
        )

    # Validate database choice
    valid_databases = ["RadImageNet", "ImageNet"]
    if args.database not in valid_databases:
        raise ValueError(
            f"Pre-trained database '{args.database}' does not exist. Please choose from {valid_databases}."
        )

    # Validate structure option
    if args.structure not in ["unfreezeall", "freezeall"] and not re.match(
        r"unfreezetop\d+", args.structure
    ):
        raise ValueError(
            "Freeze any layers? Choose to unfreezeall/freezeall/unfreezetop{i} layers for the network."
        )

    # Validate model name
    valid_models = ["ResNet50", "DenseNet121", "InceptionV3"]
    if args.backbone_model_name not in valid_models:
        raise ValueError(
            (
                f"Pre-trained network '{args.backbone_model_name}' does not exist. "
                "Please choose from {valid_models}."
            )
        )

    # Validate clf name
    valid_clfs = ["Linear", "NonLinear", "Conv", "ConvSkip"]
    if args.clf not in valid_clfs:
        raise ValueError(
            f"Clf '{args.clf}' does not exist. Please choose from {valid_clfs}."
        )

    # Validate batch size
    if args.batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    # Validate image size
    if args.image_size <= 0:
        raise ValueError("Image size must be a positive integer.")

    # Validate number of epochs
    if args.epoch <= 0:
        raise ValueError("Number of epochs must be a positive integer.")

    # Validate learning rate
    if args.lr <= 0:
        raise ValueError("Learning rate must be a positive float.")

    # Validate learning rate decay method
    valid_lr_decay_methods = [None, "cosine", "beta"]
    if args.lr_decay_method not in valid_lr_decay_methods:
        raise ValueError(
            (
                f"LR Decay Method '{args.lr_decay_method}' does not exist. "
                f"Please choose from {valid_lr_decay_methods}."
            )
        )

    # Validate learning rate decay beta if method is 'beta'
    if args.lr_decay_method == "beta" and (
        args.lr_decay_beta <= 0 or args.lr_decay_beta >= 1
    ):
        raise ValueError(
            "LR Decay Beta must be a float between 0 and 1 when using 'beta' decay method."
        )

    # Validate dropout probability
    if not (0 <= args.dropout_prob <= 1):
        raise ValueError("Dropout probability must be between 0 and 1.")

    # Validate fully connected hidden size ratio
    if not (0 <= args.fc_hidden_size_ratio):
        raise ValueError("Fully connected hidden size ratio must be greater than 0.")

    # Validate number of filters
    if args.num_filters <= 0:
        raise ValueError("Number of filters must be a positive integer.")

    # Validate kernel size
    if args.kernel_size <= 0:
        raise ValueError("Kernel size must be a positive integer.")

    if verbose:
        print("All arguments are valid.")
