#!/bin/bash

# Test the baseline models


python main.py --data_dir acl --database RadImageNet --model_name InceptionV3 --structure freezeall --verbose
python main.py --data_dir acl --database ImageNet --model_name InceptionV3 --structure freezeall --verbose
python main.py --data_dir breast --database RadImageNet --model_name InceptionV3 --structure freezeall --verbose
python main.py --data_dir breast --database ImageNet --model_name InceptionV3 --structure freezeall --verbose

python main.py --data_dir acl --database RadImageNet --model_name ResNet50 --structure freezeall --verbose
python main.py --data_dir acl --database ImageNet --model_name ResNet50 --structure freezeall --verbose
python main.py --data_dir breast --database RadImageNet --model_name ResNet50 --structure freezeall --verbose
python main.py --data_dir breast --database ImageNet --model_name ResNet50 --structure freezeall --verbose


# Test unfreezing
python main.py --data_dir acl --database RadImageNet --model_name ResNet50 --structure unfreezetop1 --verbose
python main.py --data_dir breast --database RadImageNet --model_name ResNet50 --structure unfreezetop1 --verbose
