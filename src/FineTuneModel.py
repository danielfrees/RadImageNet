### FineTuneModel.py

from argparse import Namespace
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from typing import Tuple

class FineTuneModel(nn.Module):
    """
    A PyTorch model class that fine-tunes a pre-trained network by replacing its classifier
    with a new fully connected layer for binary or multi-class classification.

    Attributes:
        base_model (nn.Sequential): The feature extractor part of the model, excluding the original classifier.
        dropout (nn.Dropout): Dropout layer to reduce overfitting.
        fc (nn.Linear): New fully connected layer for classification.
    """

    def __init__(self, base_model: nn.Module, num_classes: int = 2):
        """
        Initializes the FineTuneModel class by setting up the modified base model and the new classifier.

        Args:
            base_model (nn.Module): The pre-trained base model from which the last layer will be removed.
            num_classes (int): The number of classes for the new classifier.
        """
        super(FineTuneModel, self).__init__()
        # Extract the base model without the last layer
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.dropout = nn.Dropout(0.5)
        # Replace the classifier layer of the base model, already adjusted for num_classes
        # Initialize the new fully connected layer
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)
        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights of the newly added fully connected layer using kaiming normal initialization
        for the weights and sets biases to zero if present.
        """
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor containing batch of images.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        x = self.base_model(x)  # Pass input through the base model
        x = torch.flatten(x, 1)  # Flatten the output for the dropout layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)  # Pass through the new classifier
        return x
    
def get_compiled_model(args: Namespace, device: torch.device) -> Tuple[nn.Module, optim.Optimizer, nn.CrossEntropyLoss]:
    """
    Prepares and compiles the model by loading a base model, modifying its layers, setting the device,
    and preparing the optimizer and loss function for training.

    Args:
        args (Namespace): Command line arguments or other configuration that includes model_name, database, structure, and lr.
        device (torch.device): The device (CPU or GPU) the model should be moved to for training.

    Returns:
        tuple:
            - Module: The compiled and ready-to-train model.
            - Optimizer: The optimizer configured for the model.
            - CrossEntropyLoss: The loss function to be used during training.
    """
    # Load the base model with modified classifier layer
    base_model = load_base_model(args.model_name, args.database, device)

    # Modify the last layer to fit the binary classification task
    manage_layer_freezing(base_model, args.structure)
    print(base_model)

    # Move the model to the specified device
    model = FineTuneModel(base_model).to(device)

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Define loss function
    loss = nn.CrossEntropyLoss()

    return model, optimizer, loss
    
def load_base_model(model_name: str, database: str, device: nn.torch_device) -> nn.Module:
    """
    Loads a pre-trained model based on the specified model name and database, 
    and transfers it to the given device. It supports loading custom weights 
    for models trained with the RadImageNet dataset.

    Args:
        model_name (str): Name of the model to load (e.g., 'IRV2', 'ResNet50', 'DenseNet121').
        database (str): Indicates the dataset used to pre-train the model ('ImageNet' or 'RadImageNet').
        device (torch.device): The device (e.g., CPU or GPU) to which the model should be transferred.

    Returns:
        Module: The loaded and device-set PyTorch model.

    Raises:
        Exception: If the weights for RadImageNet models do not exist at the specified path.
    """
    base_model = None
    model_dir = f"../RadImageNet_models/RadImageNet-{model_name}_notop.h5"
    
    # Load the appropriate pre-trained model based on model name and whether it was trained on ImageNet
    if model_name == 'IRV2' or model_name == 'InceptionV3':
        base_model = models.inception_v3(pretrained=(database == 'ImageNet'))
    elif model_name == 'ResNet50':
        base_model = models.resnet50(pretrained=(database == 'ImageNet'))
    elif model_name == 'DenseNet121':
        base_model = models.densenet121(pretrained=(database == 'ImageNet'))
    
    # Transfer the model to the specified device
    base_model = base_model.to(device)
    
    # Load custom RadImageNet weights if specified and check if the file exists
    if database == 'RadImageNet' and os.path.exists(model_dir):
        base_model.load_state_dict(torch.load(model_dir, map_location=device))
    elif database == 'RadImageNet':
        raise Exception(f'RadImageNet model weights for {model_name} do not exist at specified path {model_dir}. Please ensure the file exists.')
    
    return base_model

def manage_layer_freezing(model: nn.Module, structure: str) -> None:
    """
    Adjusts the trainable status of layers in a model based on a specified structure command. This function
    can freeze all layers, unfreeze all layers, or unfreeze only the top N layers of the model.

    Args:
        model (Module): The PyTorch model whose layer training settings are to be modified.
        structure (str): A command string that dictates how layers should be frozen or unfrozen.
                         It can be 'freezeall', 'unfreezeall', or 'unfreezetopN' where N is an integer
                         indicating the number of top layers to unfreeze.

    Raises:
        ValueError: If the structure parameter does not follow the expected format or specifies an invalid option.
    """
    children = list(model.children())
    total_layers = len(children)
    
    if structure == 'freezeall':
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

    elif structure == 'unfreezeall':
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True

    elif structure.startswith('unfreezetop'):
        # Attempt to extract the number of layers to unfreeze from the structure string
        try:
            n_layers = int(structure[len('unfreezetop'):])
        except ValueError:
            raise ValueError("Invalid layer specification. Ensure it follows 'unfreezetopN' format where N is a number.")
        
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the last n_layers
        for i in range(total_layers - n_layers, total_layers):
            for param in children[i].parameters():
                param.requires_grad = True

    else:
        raise ValueError("Invalid structure parameter. Use 'freezeall', 'unfreezeall', or 'unfreezetopN' where N is a number.")