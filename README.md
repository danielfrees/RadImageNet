# Towards Optimal Architectures for Transfer Learning Computer Vision Models to Perform Radiological Diagnostics

Updates History: 

PyTorch v3 had fixes to Caffe preprocessing, train dataloader shuffling (especailly important for ACL), and a handful of other fixes. 

PyTorch v4 architecture removes the softmax from the classifier appended to the backbone, relying instead on SoftmaxLoss so that we don't do a double softmax. This massively improves breast performance. 

After Refactor May 29:

Example usage: ```python main.py --data_dir acl --database RadImageNet --backbone_model_name ResNet50 --clf NonLinear  --structure freezeall --verbose --dropout_prob 0.5 --fc_hidden_size_ratio 0.5 --num_filters 8 --kernel_size 2 --epoch 5 --batch_size 64```

See main.py for the full list of arguments. `model.py` handles training the models, as well as defines the `Backbone` and `Classifier` layers. `util.py` validates arguments and provides functions for loading data. `main.py` parses arguments, sets device, and iterates through training and validation folds. 

05/31: Aditri added Convolutions with Skip Connections as an option. Daniel added data prep options to run against full train/val/test splits and re-split and aggregated the data to ensure no leakage. Daniel added LR scheduling, more dynamic model checkpointing for all hyperparameters. 

TBD: Daniel adding SWA, experiment.py for running a vast grid of experiments + summarizing experiments into overall results/results.csv, visualizations for report. 