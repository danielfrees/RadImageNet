# Results


## Scripts:

`vis-unfreezing.py` is a simple script which visualizes the trends of unfreezing across the ACL and Breast tasks.

## GridSearch Versioning Descriptions:

v0- slightly buggy gridsearch

v1- debugged the gridsearch (partial indicates the expt did not fully finish)

v2- improved the model slightly and added more interesting set of gridsearch hyperparams. Fixed expt tracking. Previously some results got overwritten.

v3- added stratified splitting for the data re-splitting to ensure good target balance across splits. prevents noisy checkpointing on val set and noisy results that have more to do with easy splits/ bad local minima (which causes shitty F1 scores). Improves F1 across the board. Debugged the experiment so results cache properly. PIL still randomly breaks when loading images occasionally, the only way to dewbug this would be a slightly dangerous try/ except around the dataloader.

v4- fixed a bug where pretraining weights were being overwritten. can now compare RadImageNet and ImageNet