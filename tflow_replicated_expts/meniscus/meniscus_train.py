#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    InceptionResNetV2,
    ResNet50,
    InceptionV3,
    DenseNet121,
)
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import argparse
from tqdm import tqdm

print("GPU?")
print(tf.config.list_physical_devices("GPU"))
print()

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_node", type=str, help="specify gpu nodes")
parser.add_argument("--database", type=str, help="choose RadImageNet or ImageNet")
parser.add_argument(
    "--model_name", type=str, help="choose IRV2/ResNet50/DenseNet121/InceptionV3"
)
parser.add_argument("--batch_size", type=int, help="batch size", default=256)
parser.add_argument("--image_size", type=int, help="image size", default=256)
parser.add_argument("--epoch", type=int, help="number of epochs", default=30)
parser.add_argument(
    "--structure", type=str, help="unfreezeall/freezeall/unfreezetop10", default=30
)
parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
args = parser.parse_args()

# Limit to the first GPU for this model
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_node

# Import pre-trained weights from ImageNet or RadImageNet
database = args.database
if database not in ["RadImageNet", "ImageNet"]:
    raise Exception(
        "Pre-trained database not exists. Please choose ImageNet or RadImageNet"
    )

if args.structure not in ["unfreezeall", "freezeall", "unfreezetop10"]:
    raise Exception(
        "Freeze any layers? Choose to unfreezeall/freezeall/unfreezetop10 layers for the network."
    )

# Set up training image size, batch size and number of epochs
image_size = args.image_size
batch_size = args.batch_size
num_epoches = args.epoch

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))


# Create model
def get_compiled_model():
    if args.model_name not in ["IRV2", "ResNet50", "DenseNet121", "InceptionV3"]:
        raise Exception(
            "Pre-trained network not exists. Please choose IRV2/ResNet50/DenseNet121/InceptionV3 instead"
        )
    else:
        if args.model_name == "IRV2":
            if database == "RadImageNet":
                model_dir = "../RadImageNet_models/RadImageNet-IRV2-notop.h5"
                base_model = InceptionResNetV2(
                    weights=model_dir,
                    input_shape=(image_size, image_size, 3),
                    include_top=False,
                    pooling="avg",
                )
            else:
                base_model = InceptionResNetV2(
                    weights="imagenet",
                    input_shape=(image_size, image_size, 3),
                    include_top=False,
                    pooling="avg",
                )
        elif args.model_name == "ResNet50":
            if database == "RadImageNet":
                model_dir = "../RadImageNet_models/RadImageNet-ResNet50-notop.h5"
                base_model = ResNet50(
                    weights=model_dir,
                    input_shape=(image_size, image_size, 3),
                    include_top=False,
                    pooling="avg",
                )
            else:
                base_model = ResNet50(
                    weights="imagenet",
                    input_shape=(image_size, image_size, 3),
                    include_top=False,
                    pooling="avg",
                )
        elif args.model_name == "DenseNet121":
            if database == "RadImageNet":
                model_dir = "../RadImageNet_models/RadImageNet-DenseNet121-notop.h5"
                base_model = DenseNet121(
                    weights=model_dir,
                    input_shape=(image_size, image_size, 3),
                    include_top=False,
                    pooling="avg",
                )
            else:
                base_model = DenseNet121(
                    weights="imagenet",
                    input_shape=(image_size, image_size, 3),
                    include_top=False,
                    pooling="avg",
                )
        elif args.model_name == "InceptionV3":
            if database == "RadImageNet":
                model_dir = "../RadImageNet_models/RadImageNet-InceptionV3-notop.h5"
                base_model = InceptionV3(
                    weights=model_dir,
                    input_shape=(image_size, image_size, 3),
                    include_top=False,
                    pooling="avg",
                )
            else:
                base_model = InceptionV3(
                    weights="imagenet",
                    input_shape=(image_size, image_size, 3),
                    include_top=False,
                    pooling="avg",
                )

    if args.structure == "freezeall":
        for layer in base_model.layers:
            layer.trainable = False
    elif args.structure == "unfreezeall":
        pass
    elif args.structure == "unfreezetop10":
        for layer in base_model.layers[:-10]:
            layer.trainable = False

    y = base_model.output
    y = Dropout(0.5)(y)
    predictions = Dense(2, activation="softmax")(y)
    model = Model(inputs=base_model.input, outputs=predictions)
    adam = Adam(learning_rate=args.lr)
    model.compile(
        optimizer=adam,
        loss=BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def run_model():
    # Open a strategy scope.
    with strategy.scope():
        model = get_compiled_model()

    train_steps = len(train_generator.labels) / batch_size
    val_steps = len(validation_generator.labels) / batch_size

    save_model_dir = "./models/"
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    filepath = (
        "models/meniscus-"
        + args.structure
        + "-fold"
        + str(i + 1)
        + "-"
        + database
        + "-"
        + args.model_name
        + "-"
        + str(image_size)
        + "-"
        + str(batch_size)
        + "-"
        + str(args.lr)
        + ".h5.keras"
    )

    checkpoint = ModelCheckpoint(
        filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    history = model.fit(
        train_generator,
        epochs=int(num_epoches),
        steps_per_epoch=int(train_steps),
        validation_data=validation_generator,
        validation_steps=int(val_steps),
        callbacks=[checkpoint],
    )

    # Save training loss
    train_auc = history.history["auc"]
    val_auc = history.history["val_auc"]
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    d_loss = pd.DataFrame(
        {
            "train_auc": train_auc,
            "val_auc": val_auc,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
    )
    save_loss_dir = "./loss"
    if not os.path.exists(save_loss_dir):
        os.mkdir(save_loss_dir)
    d_loss.to_csv(
        "loss/meniscus-"
        + args.structure
        + "-fold"
        + str(i + 1)
        + "-"
        + database
        + "-"
        + args.model_name
        + "-"
        + str(image_size)
        + "-"
        + str(batch_size)
        + "-"
        + str(args.lr)
        + ".csv",
        index=False,
    )


for i in tqdm(range(5)):
    df_train = pd.read_csv("dataframe/meniscus_train_fold" + str(i + 1) + ".csv")
    df_val = pd.read_csv("dataframe/meniscus_val_fold" + str(i + 1) + ".csv")
    train_data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        preprocessing_function=preprocess_input,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    data_generator = ImageDataGenerator(
        rescale=1.0 / 255, preprocessing_function=preprocess_input
    )
    train_generator = train_data_generator.flow_from_dataframe(
        dataframe=df_train,
        x_col="img_dir",
        y_col="label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode="categorical",
    )
    validation_generator = data_generator.flow_from_dataframe(
        dataframe=df_val,
        x_col="img_dir",
        y_col="label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=726,
        class_mode="categorical",
    )
    num_classes = len(train_generator.class_indices)
    run_model()
