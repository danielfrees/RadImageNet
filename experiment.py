"""
experiment.py

Overall wrapper for running ablation experiments for the medical classification
transfer learning.

Usage:
    $ python experiment.py --method runall         # run all experiments, save results and models
    $ python experiment.py --method summarize      # take all experiments and create overall results csv
    $ python experiment.py --method visualize      # run visualizations
"""
import argparse
import itertools
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from main import main as run_experiment


def summarize_results(results_dirs, verbose=False):
    """
    Summarizes the results from the experiments.

    Args:
        results_dirs (list): List of directories where the results are stored.
        verbose (bool): Whether to print verbose output.

    Returns:
        pd.DataFrame: DataFrame containing the summarized results.
    """
    if verbose:
        print("Summarizing results...")

    summary_rows = []

    for results_dir in results_dirs:
        if not os.path.exists(results_dir):
            if verbose:
                print(f"Directory {results_dir} does not exist.")
            continue

        for filename in os.listdir(results_dir):
            if filename.startswith("training_history_") and filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(results_dir, filename))
                # Extract hyperparameters from the filename
                hyperparams = filename[len("training_history_") : -len(".csv")]
                best_row = df.loc[df["val_auc"].idxmax()].copy()

                # Extract hyperparameters into individual columns
                hyperparam_dict = parse_hyperparams(hyperparams)
                best_row = pd.concat([best_row, pd.Series(hyperparam_dict)])

                summary_rows.append(best_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join("results", "results.csv"), index=False)
    if verbose:
        print("Summary saved to results/results.csv")
    return summary_df


def parse_hyperparams(hyperparams_str):
    """
    Parses the hyperparameters from the string.

    Args:
        hyperparams_str (str): The hyperparameters string.

    Returns:
        dict: A dictionary of hyperparameters.
    """
    params = hyperparams_str.split("_")
    hyperparams = {}
    i = 0
    while i < len(params):
        if params[i] == "backbone":
            hyperparams["backbone"] = params[i + 1]
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
        else:
            i += 1
    return hyperparams


def create_visualizations(summary_df, verbose=False):
    """
    Creates visualizations from the summarized results.

    Args:
        summary_df (pd.DataFrame): DataFrame containing the summarized results.
        verbose (bool): Whether to print verbose output.
    """
    if verbose:
        print("Creating visualizations...")

    # Create visualizations for different hyperparameters
    hyperparameters = [
        "backbone",
        "clf",
        "structure",
        "lr",
        "batchsize",
        "dropprob",
        "fcsizeratio",
        "numfilters",
        "kernelsize",
        "epochs",
    ]

    for hyperparam in hyperparameters:
        if hyperparam in summary_df.columns:
            plt.figure(figsize=(10, 6))
            summary_df.boxplot(column="val_auc", by=hyperparam)
            plt.title(f"Validation AUC by {hyperparam}")
            plt.suptitle("")
            plt.xlabel(hyperparam)
            plt.ylabel("Validation AUC")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join("results", f"validation_auc_by_{hyperparam}.png"))
            if verbose:
                print(f"Visualization for {hyperparam} saved to results/")


def main():
    """
    Main function to handle different methods: runall, summarize, visualize.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["runall", "summarize", "visualize"],
        help=(
            "Choose one of: runall, summarize, visualize. runall runs the grid "
            "search of experiments specified in main of experiment.py."
            "summarize combines all hyperparameter experiment results found "
            "and their loss / performance vals at point of best validation "
            "performance, and combines this into results/results.csv. "
            "visualize generates visualizations based on results.csv"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    DATA_DIRS = ["breast", "acl"]
    DATABASES = ["ImageNet", "RadImageNet"]
    BACKBONE_MODELS = ["ResNet50", "DenseNet121"]
    CLFS = ["Linear", "NonLinear", "Conv", "ConvSkip"]
    LEARNING_RATES = [1e-4]
    BATCH_SIZES = [128]
    IMAGE_SIZES = [256]
    EPOCHS = [30]
    STRUCTURES = ["freezeall"]
    LR_DECAY_METHODS = ["beta", "cosine"]
    LR_DECAY_BETAS = [0.8]
    DROPOUT_PROBS = [0.5]
    FC_HIDDEN_SIZE_RATIOS = [0.5]
    NUM_FILTERS = [4]
    KERNEL_SIZES = [2]
    AMPS = [True]
    USE_FOLDS = [False]
    LOG_EVERY = [100]

    if args.method == "runall":
        if args.verbose:
            print("Running all experiments...")
        for (
            data_dir,
            database,
            backbone_model,
            clf,
            lr,
            batch_size,
            image_size,
            epoch,
            structure,
            lr_decay_method,
            lr_decay_beta,
            dropout_prob,
            fc_hidden_size_ratio,
            num_filters,
            kernel_size,
            amp,
            use_folds,
            log_every,
        ) in itertools.product(
            DATA_DIRS,
            DATABASES,
            BACKBONE_MODELS,
            CLFS,
            LEARNING_RATES,
            BATCH_SIZES,
            IMAGE_SIZES,
            EPOCHS,
            STRUCTURES,
            LR_DECAY_METHODS,
            LR_DECAY_BETAS,
            DROPOUT_PROBS,
            FC_HIDDEN_SIZE_RATIOS,
            NUM_FILTERS,
            KERNEL_SIZES,
            AMPS,
            USE_FOLDS,
            LOG_EVERY,
        ):
            if args.verbose:
                print(
                    f"Running experiment with {data_dir}, {database}, {backbone_model}, {clf}, {lr}, "
                    f"{batch_size}, {image_size}, {epoch}, {structure}, {lr_decay_method}, {lr_decay_beta}, "
                    f"{dropout_prob}, {fc_hidden_size_ratio}, {numfilters}, {kernelsize}, {amp}, {use_folds}, "
                    f"{log_every}"
                )
            # Set sys.argv for the run_experiment call
            sys.argv = (
                [
                    "main.py",
                    "--data_dir",
                    data_dir,
                    "--database",
                    database,
                    "--backbone_model_name",
                    backbone_model,
                    "--clf",
                    clf,
                    "--batch_size",
                    str(batch_size),
                    "--image_size",
                    str(image_size),
                    "--epoch",
                    str(epoch),
                    "--structure",
                    structure,
                    "--lr",
                    str(lr),
                    "--lr_decay_method",
                    lr_decay_method if lr_decay_method is not None else "",
                    "--lr_decay_beta",
                    str(lr_decay_beta),
                    "--dropout_prob",
                    str(dropout_prob),
                    "--fc_hidden_size_ratio",
                    str(fc_hidden_size_ratio),
                    "--num_filters",
                    str(num_filters),
                    "--kernel_size",
                    str(kernel_size),
                    "--log_every",
                    str(log_every),
                ]
                + (["--amp"] if amp else [])
                + (["--use_folds"] if use_folds else [])
                + (["--verbose"] if args.verbose else [])
            )
            run_experiment()

    elif args.method == "summarize":
        results_dirs = [
            os.path.join("data", "acl", "models"),
            os.path.join("data", "breast", "models"),
        ]
        summary_df = summarize_results(results_dirs, args.verbose)

    elif args.method == "visualize":
        results_csv = os.path.join("results", "results.csv")
        if not os.path.exists(results_csv):
            raise FileNotFoundError(
                "The results summary CSV does not exist. Run `summarize` first."
            )
        summary_df = pd.read_csv(results_csv)
        create_visualizations(summary_df, args.verbose)


if __name__ == "__main__":
    main()
