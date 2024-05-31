#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    InceptionResNetV2,
    ResNet50,
    InceptionV3,
    DenseNet121,
)
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, accuracy_score
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_node", type=str, help="specify gpu nodes")
parser.add_argument("--database", type=str, help="choose RadImageNet or ImageNet")
parser.add_argument(
    "--model_name", type=str, help="choose IRV2/ResNet50/DenseNet121/InceptionV3"
)
parser.add_argument("--batch_size", type=int, help="batch size", default=256)
parser.add_argument("--image_size", type=int, help="image size", default=256)
parser.add_argument(
    "--structure", type=str, help="unfreezeall/freezeall/unfreezetop10", default=30
)
parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--dataset", type=str, help="dataset to eval on", default="val")
args = parser.parse_args()

# Limit to the first GPU for this model
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_node

# Set up image size and batch size
image_size = args.image_size
batch_size = args.batch_size

# Create a MirroredStrategy
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Set up the data generators
data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    preprocessing_function=keras.applications.inception_resnet_v2.preprocess_input,
)

# Initialize metrics
auc_metric = AUC(name="auc")


# Function to load and evaluate model
def evaluate_model(fold):
    # Load validation data
    df = pd.read_csv(f"dataframe/meniscus_{args.dataset}_fold{fold}.csv")
    generator = data_generator.flow_from_dataframe(
        dataframe=df,
        x_col="img_dir",
        y_col="label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
        class_mode="categorical",
    )

    # Load model
    model_path = f"models/meniscus-{args.structure}-fold{fold}-{args.database}-{args.model_name}-{image_size}-{batch_size}-{args.lr}.h5.keras"
    model = keras.models.load_model(model_path, compile=False)

    # Compile the model with necessary metrics
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss=BinaryCrossentropy(),
        metrics=[auc_metric],
    )

    # Evaluate model
    results = model.evaluate(generator, verbose=1)
    y_true = generator.classes
    y_pred = np.argmax(model.predict(generator), axis=1)

    # Calculate F1 score and accuracy
    f1 = f1_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)

    # Create a dictionary of metrics
    metrics = {name: result for name, result in zip(["loss", "auc"], results)}
    metrics["f1"] = f1
    metrics["accuracy"] = accuracy

    return metrics


# Create directory for fold results
fold_results_dir = "fold_results"
os.makedirs(fold_results_dir, exist_ok=True)

# Evaluate each fold and print the results
all_metrics = []
for i in tqdm(range(1, 6)):
    metrics = evaluate_model(i)
    all_metrics.append(metrics)
    print(f"Fold {i} results: {metrics}")

    # Save the metrics to a CSV file in fold_results directory
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        f"{fold_results_dir}/evaluation_metrics_{args.dataset}_fold_{i}.csv",
        index=False,
    )

# Calculate average metrics
average_metrics = {
    key: np.mean([fold_metrics[key] for fold_metrics in all_metrics])
    for key in all_metrics[0]
}
print(f"Average metrics across all folds: {average_metrics}")

# Save the average metrics to a .txt file
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
with open(
    os.path.join(
        results_dir,
        f"average_metrics_{args.dataset}_meniscus_{args.structure}-{args.database}-{args.model_name}-{image_size}-{batch_size}-{args.lr}.txt",
    ),
    "w",
) as f:
    for key, value in average_metrics.items():
        f.write(f"{key}: {value}\n")
