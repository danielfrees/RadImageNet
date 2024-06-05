# Towards Optimal Convolutional Transfer Learning Architectures for Downstream Medical Classification Tasks

## Main Scripts

`experiment.py` is the most comprehensive wrapper script for our analysis. With `--method runall`, this manages gridsearching for optimal architectures by passing cross-products of hyperparameters and architecture choices to main(). With `--method summarize --filter key value` the script produces a `results/results.csv` file containing all metrics for each hyperparameter combination where key = value (e.g. if we set --filter epochs 10, we will retrieve all results for experiments run with --epochs 10). Note that each row in the results.csv corresponds to the 'best'/checkpointed epoch from model training, as selected by highest validation AUC. With `--method visualize` the script produces a set of visualizations in `results/acl/`, `results/breast/`, and `results/overall/` which provide box-plot + experimental scatters comparing hyperparameter choices' affect on metrics.

*Example (also triggerable as a sequence with `./run_experiment.sh`):*

```bash
# Run all experiments defined in experiment.py loops
python experiment.py --method runall --verbose

# Summarize all epoch=10 experiments into a single CSV
python experiment.py --method summarize --verbose --filter epochs 10

# Create visualizations from the summarized results
python experiment.py --method visualize --verbose
```

`main.py` handles dataloader setup and device setup, and serves as a point of contact for users to trigger new experiments from the CLI (and for experiment.py to start grid search experiments).

*Examples (Best Breast Model and Best ACL Model):*

```bash
$ python main.py --data_dir breast --database ImageNet --backbone_model_name ResNet50 --clf ConvSkip --structure unfreezetop5 --verbose --dropout_prob 0.5 --fc_hidden_size_ratio 1.0 --num_filters 16 --kernel_size 2 --epoch 30 --batch_size 64 --lr_decay_method cosine --amp --lr 5e-4

$ python main.py --data_dir acl --database ImageNet --backbone_model_name ResNet50 --clf ConvSkip --structure unfreezetop5 --verbose --dropout_prob 0.5 --fc_hidden_size_ratio 0.5 --num_filters 16 --kernel_size 4 --epoch 30 --batch_size 64 --lr_decay_method cosine --amp --lr 1e-3
```

`interpret.py` handles all Grad-CAM logic for generating and visualizing Grad-CAM heatmaps to interpret model results.

*Example:*
```bash
$ python interpret.py --data_dir breast --database ImageNet --backbone_model_name ResNet50 --clf ConvSkip --structure unfreezetop5 --verbose --dropout_prob 0.5 --fc_hidden_size_ratio 1.0 --num_filters 16 --kernel_size 2 --epoch 30 --batch_size 64 --lr_decay_method cosine --amp --lr 5e-4 --image_index 0
```

`predictions.py` is a simple script for producing `predictions/preds_{MODEL_PARAM_STR}.csv` files with all the test predictions for a particular model.

*Example:*
```bash
$ python predictions.py --data_dir breast --database ImageNet  --backbone_model_name ResNet50 --clf ConvSkip --structure unfreezetop5 --verbose --dropout_prob 0.5 --fc_hidden_size_ratio 1.0 --num_filters 16 --kernel_size 2 --epoch 30 --batch_size 64 --lr_decay_method cosine --amp --lr 5e-4
```

## Source Code

`src/` contains the source code for argument parsing, dataloader setup, model architecture building in PyTorch, and other utils.

## Other Important Directories

`data/` contains the data for all of the downstream classification tasks. We focus primarily on `data/breast/` and `/data/acl/`. Each of this subdirectories contains folders `datafram/e`, `images/`, and `models/`. The dataframe folder contains the five-fold splits used by RadImageNet, as well as combined, re-split 75/15/10 train/val/test stratified (on target) splits that we generate and use. Each row contains a label and an image path, which points to an image in `images/`. `models/` contains training histories (performance metrics throughout training) as well as checkpointed models, though much of this is not uploaded to github due to filesize constraints.

`logs/` is used for TensorBoard logging, and should also be mostly empty on github.

`predictions/` contains predictions for our best breast and ACL models, as well as their less performant RadImageNet initialized counterparts.

`results/` contains gridsearch and unfreezing experiment results and visualizations.

`tflow_replicated_expts/` contains debugged code from the original RadImageNet repo, used to compare results for our Linear baselines models.





##  ====== Internal Usage for Authors =======
Updates History:

PyTorch v3 had fixes to Caffe preprocessing, train dataloader shuffling (especailly important for ACL), and a handful of other fixes.

PyTorch v4 architecture removes the softmax from the classifier appended to the backbone, relying instead on SoftmaxLoss so that we don't do a double softmax. This massively improves breast performance.

After Refactor May 29:

Example usage: ```python main.py --data_dir acl --database RadImageNet --backbone_model_name ResNet50 --clf NonLinear  --structure freezeall --verbose --dropout_prob 0.5 --fc_hidden_size_ratio 0.5 --num_filters 8 --kernel_size 2 --epoch 5 --batch_size 64```

See main.py for the full list of arguments. `model.py` handles training the models, as well as defines the `Backbone` and `Classifier` layers. `util.py` validates arguments and provides functions for loading data. `main.py` parses arguments, sets device, and iterates through training and validation folds.

====== 05/31: ======
Aditri added Convolutions with Skip Connections as an option. Daniel added data prep options to run against full train/val/test splits and re-split and aggregated the data to ensure no leakage.

Daniel added LR scheduling, more dynamic model checkpointing for all hyperparameters.

Daniel added linting.

TBD: Daniel adding SWA, experiment.py for running a vast grid of experiments + summarizing experiments into overall results/results.csv, visualizations for report.