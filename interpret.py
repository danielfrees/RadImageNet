"""
Qualitative model interpretability using Grad-CAM and friends.

See: https://github.com/jacobgil/pytorch-grad-cam
and: https://arxiv.org/abs/1610.02391

for the inspiration.

Example Usage:
>>> python interpret.py \
    --data_dir breast \
    --database ImageNet \
    --backbone_model_name ResNet50 \
    --clf ConvSkip \
    --fold full \
    --structure freezeall \
    --lr 0.0001 \
    --batch_size 64 \
    --dropout_prob 0.5 \
    --fc_hidden_size_ratio 1.0 \
    --num_filters 16 \
    --kernel_size 2 \
    --epoch 5 \
    --image_size 256 \
    --lr_decay_method cosine \
    --lr_decay_beta 0.5 \
    --image_index 0
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import os
from argparse import Namespace
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib import pyplot as plt
from src.model import load_model
from src.util import validate_args, generate_model_param_str
from src.data import CaffeTransform
import argparse
import torch.nn.functional as F


def load_prep_image(image_path: str, device, args):
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            CaffeTransform(),
            transforms.Normalize(mean=[0, 0, 0], std=[0.225, 0.224, 0.229]),
        ]
    )
    image = Image.open(image_path).convert("RGB")  # Convert to RGB
    image = transform(image).unsqueeze(0).to(device)
    return image


def visualize_image_with_gradcam(image_index: int, args, device):
    """
    Use Grad-CAM to visualize importance of input image regions for img {image_index}
    in the validation dataset for the specified task and using the model with the specified
    hyperparameters.

    Hyperparameters and task defined in args.
    """
    data_path = f"./data/{args.data_dir}"
    model_dir = f"./data/{args.data_dir}/models"
    MODEL_PARAM_STR = generate_model_param_str(
        data_dir=args.data_dir,
        backbone_model=args.backbone_model_name,
        pretrain=args.database,
        clf=args.clf,
        structure=args.structure,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout_prob=args.dropout_prob,
        fc_hidden_size_ratio=args.fc_hidden_size_ratio,
        num_filters=args.num_filters,
        kernel_size=args.kernel_size,
        epoch=args.epoch,
        image_size=args.image_size,
        lr_decay_method=args.lr_decay_method,
        lr_decay_beta=args.lr_decay_beta,
    )
    model_name = f"best_model_{MODEL_PARAM_STR}.pth"
    model_path = os.path.join(model_dir, model_name)
    assert os.path.exists(model_path), f"Model weights not found at {model_path}"

    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]

    # ==== load in model and img ======

    model = load_model(device, args)
    print("======= Model Loaded ======")
    # for name, layer in model.named_children():   # used to determine target layers
    #     print(f"{name}: {layer}")
    print("===========================")
    model.load_state_dict(model_state_dict)
    model.eval()  # Set model to evaluation mode

    # !!! enable gradients. Essential to re-enable gradient flow when
    # transfer learning on a frozen backbone
    for param in model.parameters():
        param.requires_grad = True

    df_path = f"./data/{args.data_dir}/dataframe/val.csv"
    df = pd.read_csv(df_path)
    image_file, label = df.iloc[image_index]

    image_path = os.path.join(data_path, image_file)

    input_tensor = load_prep_image(image_path, device, args)
    input_tensor.requires_grad = True

    # Set target layers for Grad-CAM
    target_layers = []

    if args.backbone_model_name.startswith("DenseNet"):
        # Select denseblock4 from the backbone
        densenet_layers = list(model.backbone.backbone[0].children())
        target_layers += [densenet_layers[-2]]  # Targeting the denseblock4 specifically
        print("Target Layers")
        print(target_layers)
    elif args.backbone_model_name.startswith("ResNet"):
        # Last few layers of ResNet
        resnet_layers = list(model.backbone.backbone.children())
        target_layers += [
            resnet_layers[-3],
            resnet_layers[-2],
        ]  # avg. a few of the late resnet !convolutional! layers (the last layer is useless avg pool)
    else:
        raise ValueError("Unsupported model backbone")

    # set target class
    positive_class = "malignant" if args.data_dir == "breast" else "yes"
    negative_class = "benign" if args.data_dir == "breast" else "no"
    label_idx = None
    if label == positive_class:
        label_idx = 1
    elif label == negative_class:
        label_idx = 0
    else:
        raise ValueError(f"Unrecognized label: {label}")

    # fwd pass and backward pass to get the outputs and calculate gradients for Grad-CAM
    # !!! Don't be dumb and not compute gradients like me
    output = model(input_tensor)
    loss = nn.CrossEntropyLoss()(output, torch.tensor([label_idx]).to(device))
    model.zero_grad()
    loss.backward()

    # use predicted label to set the target class! want to interpret affect on this prediciton
    pred_idx = torch.argmax(output, dim=1)[0]
    print(f"Prediction: {pred_idx}")
    pred_str = (
        positive_class if pred_idx == 1 else negative_class
    )  # turn predicted label into string

    # run the grad-CAM algorithm
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    if not np.any(grayscale_cam):
        print("Warning: Grayscale CAM is all zeros. Check the model and target layers.")
    """ debugging: visualize the grayscale cam

    print("Grayscale CAM:")
    plt.figure(figsize=(10, 10))
    plt.imshow(grayscale_cam, cmap='gray')
    plt.title(f"Grayscale CAM for {positive_class} on {args.backbone_model_name}")
    plt.axis('off')
    plt.show()
    """

    # visualize grad-CAM heatmap overlayed on the original image
    rgb_img = (
        np.array(
            Image.open(image_path)
            .convert("RGB")
            .resize((args.image_size, args.image_size))
        )
        / 255.0
    )
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Original Image\nTrue label: {label}")
    axes[0].axis("off")
    axes[1].imshow(visualization)
    axes[1].set_title(
        f"Grad-CAM for target class = {pred_str} on {args.backbone_model_name}\nPredicted: {pred_str}"
    )
    axes[1].axis("off")
    plt.show()


def main():
    """
    Run gradcam visualization for request task, image, and args.

    Example usage:
    python script.py --data_dir acl --database RadImageNet --backbone_model_name DenseNet121 --clf Conv --fold full --structure freezeall --lr 0.0001 --batch_size 128 --dropout_prob 0.5 --fc_hidden_size_ratio 0.5 --num_filters 16 --kernel_size 2 --epoch 5 --image_index 0 --image_size 256
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Name of the data directory, e.g., acl",
    )
    parser.add_argument(
        "--database", type=str, required=True, help="Choose RadImageNet or ImageNet"
    )
    parser.add_argument(
        "--backbone_model_name",
        type=str,
        required=True,
        help="Choose ResNet50, DenseNet121, or InceptionV3",
    )
    parser.add_argument(
        "--clf",
        type=str,
        required=True,
        help="Classifier type. Choose Linear, Nonlinear, Conv, or ConvSkip",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--epoch", type=int, default=30, help="Number of epochs")
    parser.add_argument(
        "--structure",
        type=str,
        default="unfreezeall",
        help="Structure: unfreezeall, freezeall, or unfreezetop10",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--lr_decay_method",
        type=str,
        default=None,
        help=(
            "LR Decay Method. Choose from None (default), 'cosine' for Cosine "
            "Annealing, 'beta' for multiplying LR by lr_decay_beta each epoch"
        ),
    )
    parser.add_argument(
        "--lr_decay_beta",
        type=float,
        default=0.5,
        help="Beta for LR Decay. Multiply LR by lr_decay_beta each epoch if lr_decay_method = 'beta'",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.5,
        help="Prob. of dropping nodes in dropout layers",
    )
    parser.add_argument(
        "--fc_hidden_size_ratio",
        type=float,
        default=0.5,
        help="Ratio of hidden size to features for FC intermediate layers.",
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        default=4,
        help="Number of Filters used in convolutional layers.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=2,
        help="Size of Kernel used in convolutional layers.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Enable AMP for faster mixed-precision training. Need CUDA + "
            "recommend batch size of 256+ to use throughput gains if running AMP."
        ),
    )
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--use_folds",
        action="store_true",
        default=False,
        help=(
            "Run separate models for different train and validation folds. "
            "Useful for matching original RadImageNet baselines, but messy."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--image_index", type=int, required=True, help="Index of the image to visualize"
    )
    parser.add_argument("--fold", type=str, required=True, help="Fold of the model")

    args = parser.parse_args()

    validate_args(args, verbose=True)

    # ====== Set Device, priority cuda > mps > cpu =======
    # discard parallelization for now
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = (
            True  # Enable cuDNN benchmark for optimal performance
        )
        torch.backends.cudnn.deterministic = (
            False  # Set to False to allow for the best performance
        )
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.verbose:
        print("\n=====================")
        print(f"Device Located: {device}")
        print(f"Loading data from directory: {args.data_dir}")
        print("=====================\n")

    visualize_image_with_gradcam(args.image_index, args, device)


if __name__ == "__main__":
    main()
