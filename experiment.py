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
    pass

    


if __name__ == "__main__":
    main()