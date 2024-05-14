### validate_args.py

import os
from argparse import Namespace

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
    if not os.path.isdir(full_path):
        raise FileNotFoundError(f"The directory specified does not exist: {full_path}")

    # Validate database choice
    valid_databases = ['RadImageNet', 'ImageNet']
    if args.database not in valid_databases:
        raise ValueError(f"Pre-trained database '{args.database}' does not exist. Please choose from {valid_databases}.")

    # Validate structure option
    valid_structures = ['unfreezeall', 'freezeall', 'unfreezetop10']
    if args.structure not in valid_structures:
        raise ValueError(f"Invalid structure option '{args.structure}'. Choose to unfreezeall, freezeall, or unfreezetop10 layers.")

    # Validate model name
    valid_models = ['IRV2', 'ResNet50', 'DenseNet121', 'InceptionV3']
    if args.model_name not in valid_models:
        raise ValueError(f"Pre-trained network '{args.model_name}' does not exist. Please choose from {valid_models}.")
