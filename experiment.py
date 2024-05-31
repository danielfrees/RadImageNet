""" 
experiment.py

Overall wrapper for running ablation experiments for the medical classification 
transfer learning. 

Usage:
    $ python experiment.py --runall         # run all experiments, collect results
    $ python experiment.py --visualize      # run visualizations
"""

import os 
import argparse
import itertools


def main():
    """ 

    ===== If run with flag --runall: =====

    Runs all below specified experiments, stores resulting models and corresponding
    train and validation metrics into the results/models/ directory. 
     
    Updates the overall experimental results in results/results.csv with columns 
    for training and validation and test metrics, as well as hyperparameters.

    ===== If run with flag --visualize: =====

    Intakes the overall experimental results from results/results.csv and visualizes
    various useful breakdowns of the ablations for reporting purposes. 

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, 
                        help="Choose one of: runall, summarize, visualize. runall runs the grid search of experiments specified in main of experiment.py. summarize ")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()


    DATA_DIRS = ['breast', 'acl']
    LEARNING_RATES = []
    


    pass

    


if __name__ == "__main__":
    main()