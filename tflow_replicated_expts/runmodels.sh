#!/bin/bash

# Breast Model
# cd breast
# python breast_train.py --database ImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
# python breast_eval.py --database ImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
# python breast_train.py --database RadImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
# python breast_eval.py --database RadImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
# cd ..

# ACL Model
cd acl
python acl_train.py --database ImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
python acl_eval.py --database ImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
python acl_eval.py --database ImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset test
python acl_train.py --database RadImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
python acl_eval.py --database RadImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
python acl_eval.py --database RadImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset test
cd ..
#
# # Covid19 Model
# cd covid19
# python covid19_train.py --database ImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
# python covid19_eval.py --database ImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
# python covid19_train.py --database RadImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
# python covid19_eval.py --database RadImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
# cd ..
#
# Meniscus Model
cd meniscus
python meniscus_train.py --database ImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
python meniscus_eval.py --database ImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
python meniscus_eval.py --database ImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset test
python meniscus_train.py --database RadImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
python meniscus_eval.py --database RadImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
python meniscus_eval.py --database RadImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset test
cd ..
#
# # Pneumonia Model
# cd pneumonia
# python pneumonia_train.py --database ImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
# python pneumonia_eval.py --database ImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
# python pneumonia_train.py --database RadImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
# python pneumonia_eval.py --database RadImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
# cd ..
#
# # Sarscovid2 Model
# cd sarscovid2
# python sarscovid2_train.py --database ImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
# python sarscovid2_eval.py --database ImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
# python sarscovid2_train.py --database RadImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
# python sarscovid2_eval.py --database RadImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
# cd ..
#
# Thyroid Model
cd thyroid
python thyroid_train.py --database ImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
python thyroid_eval.py --database ImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
python thyroid_eval.py --database ImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset test
python thyroid_train.py --database RadImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
python thyroid_eval.py --database RadImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
python thyroid_eval.py --database RadImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset test
cd ..

# Hemorrhage model
# cd hemorrhage
# python hemorrhage_train.py --database ImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
# python hemorrhage_eval.py --database ImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
# python hemorrhage_train.py --database RadImageNet --structure freezeall --gpu_node mps --model_name InceptionV3
# python hemorrhage_eval.py --database RadImageNet --gpu_node mps --model_name InceptionV3 --structure freezeall --dataset val
# cd ..
