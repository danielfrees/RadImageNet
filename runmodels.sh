#!/bin/bash

# Test the baselines models
# python main.py --data_dir acl --database RadImageNet --model_name InceptionV3 --structure freezeall --verbose
# python main.py --data_dir acl --database ImageNet --model_name InceptionV3 --structure freezeall --verbose
# python main.py --data_dir breast --database RadImageNet --model_name InceptionV3 --structure freezeall --verbose
# python main.py --data_dir breast --database ImageNet --model_name InceptionV3 --structure freezeall --verbose

python main.py --data_dir acl --database RadImageNet --model_name InceptionV3 --structure unfreezetop2 --verbose
python main.py --data_dir acl --database ImageNet --model_name InceptionV3 --structure unfreezetop2 --verbose
