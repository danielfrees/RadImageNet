### BackBone.py

from argparse import Namespace
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from typing import Tuple
    
class Backbone(nn.Module):
    """
    A PyTorch model class serves as the backbone of the pre-trained network by removing its classifier.

    Attributes:
        backbone (nn.Sequential): The feature extractor part of the model, excluding the original classifier.
        dropout (nn.Dropout): Dropout layer to reduce overfitting.
    """

    def __init__(self, base_model: nn.Module, model_name: str):
        """
        Initializes the BackboneModel class by setting up the modified base model and the new classifier.

        Args:
            base_model (nn.Module): The pre-trained base model from which the last layer will be removed.
        """
        super(Backbone, self).__init__()
        self.model_name = model_name
        # Extract the base model without the last layer
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
                            
    def forward(self, x):
        x = self.backbone(x)
        # DenseNet121 requires additional operations for dimensionality reduction
        if self.model_name == 'DenseNet121':
            x = nn.functional.relu(x, inplace=True)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    
class Classifier(nn.Module):
    def __init__(self, num_in_features, num_class):
        super(Classifier, self).__init__()
        self.drop_out = nn.Dropout()
        self.linear = nn.Linear(num_in_features, num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.drop_out(x)
        x = self.linear(x)
        x = self.softmax(x)
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
    model = load_base_model(args.model_name, args.database, device, args)

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Define loss function
    loss = nn.CrossEntropyLoss()

    return model, optimizer, loss
    
def load_base_model(model_name: str, database: str, device: torch.device, args: Namespace) -> nn.Module:
    """
    Loads a pre-trained model based on the specified model name and database, 
    and transfers it to the given device. It supports loading custom weights 
    for models trained with the RadImageNet dataset.

    Args:
        model_name (str): Name of the model to load (e.g., 'IRV2', 'ResNet50', 'DenseNet121').
        database (str): Indicates the dataset used to pre-train the model ('ImageNet' or 'RadImageNet').
        device (torch.device): The device (e.g., CPU or GPU) to which the model should be transferred.
        args (Namespace): Command line arguments or other configuration that includes model_name, database, structure, and lr.

    Returns:
        Module: The loaded and device-set PyTorch model.

    Raises:
        Exception: If the weights for RadImageNet models do not exist at the specified path.
    """
    base_model = None
    model_dir = f"./RadImageNet_pytorch/{model_name}.pt"
    
    if model_name == 'InceptionV3':
        weights = "IMAGENET1K_V1" if database == 'ImageNet' else None
        base_model = models.inception_v3(weights=weights)
        # Remove the auxiliary output layer to allow for smaller input sizes (75x75), otherwise it requires 299x299
        base_model.AuxLogits = None
    elif model_name == 'ResNet50':
        weights = "IMAGENET1K_V1" if database == 'ImageNet' else None
        base_model = models.resnet50(weights=weights)
    elif model_name == 'DenseNet121':
        weights = "IMAGENET1K_V1" if database == 'ImageNet' else None
        base_model = models.densenet121(weights=weights)
    # Determine the number of input features for the classifier
    num_in_features = list(base_model.children())[-1].in_features
    backbone = Backbone(base_model, model_name)

    # Load custom RadImageNet weights if specified and the file exists
    if database == 'RadImageNet' and os.path.exists(model_dir):
        backbone.load_state_dict(torch.load(model_dir, map_location=device))
    elif database == 'RadImageNet':
        raise Exception(f'RadImageNet model weights for {model_name} do not exist at specified path {model_dir}. Please ensure the file exists.')
    
    manage_layer_freezing(backbone, args.structure)
    classifier = Classifier(num_in_features, 2)

    model = nn.Sequential(backbone, classifier)
    model = model.to(device)

    return model

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