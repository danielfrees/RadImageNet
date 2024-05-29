### model.py

from argparse import Namespace
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from typing import Tuple
from argparse import Namespace
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


    
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
            nn.functional.relu(x, inplace=True)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    
class Classifier(nn.Module):
    def __init__(self, num_in_features, num_class):
        super(Classifier, self).__init__()
        # Intermediate layer size, can be adjusted
        intermediate_size = num_in_features // 2  
        self.relu = nn.LeakyReLU(negative_slope=0.01)

        # First fully connected layer
        self.fc1 = nn.Linear(num_in_features, intermediate_size)
        # Batch normalization for the first layer
        self.bn1 = nn.BatchNorm1d(intermediate_size)
        # Dropout layer
        self.dropout = nn.Dropout(0.5)  # Adjust dropout rate as needed

        # Second fully connected layer (output layer)
        self.fc2 = nn.Linear(intermediate_size, num_class)

        # Initialize weights using Kaiming He initialization, good practice for layers followed by ReLU
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        # Apply first fully connected layer with ReLU activation and batch normalization
        x = self.relu(self.bn1(self.fc1(x)))
        # Apply dropout
        x = self.dropout(x)
        # Apply second fully connected layer (output layer)
        x = self.fc2(x)
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 1e-5)

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
        base_model = models.inception_v3(weights=weights, 
                                         transform_input = False, 
                                         init_weights = False,     # using pretrained weights!!
                                         aux_logits = weights is not None)  # needs to be set true for imagenet for some reason
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


def run_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: Namespace,
    device: torch.device,
    partial_path: str,
    fold: int,
    database: str
) -> None:
    """
    Runs the training and validation process for a given model.

    Args:
        model (nn.Module): The neural network model to train.
        optimizer (optim.Optimizer): Optimizer for updating model weights.
        loss_fn (nn.Module): Loss function to measure the model's performance.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        num_epochs (int): Total number of epochs to train the model.
        device (torch.device): The device (CPU or GPU) to run the model on.
        verbose (bool): Enable verbose output.
        database (str): ImageNet or RadImageNet for the pretrained weights

    This function performs training and validation across the specified number of epochs,
    saving model checkpoints after each epoch and printing the loss values.
    """
    num_epochs = args.epoch
    verbose = args.verbose

    save_model_dir = os.path.join(partial_path, 'models')
    os.makedirs(save_model_dir, exist_ok=True)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    # log to tensorboard
    task = partial_path.split(os.sep)[1]
    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', f'{task}_{current_datetime}_{database}_fold_{fold}')
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.detach().cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        
        # Apply softmax to logits to get probabilities
        all_probs = torch.softmax(torch.tensor(all_preds), dim=1).numpy()
        
        train_auc = roc_auc_score(all_labels, all_probs[:, 1])
        history['train_auc'].append(train_auc)

        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training AUC: {train_auc:.4f}')
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('AUC/train', train_auc, epoch)

        # Perform validation
        model.eval()
        val_running_loss = 0.0
        val_labels = []
        val_preds = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                
                val_labels.append(labels.cpu().numpy())
                val_preds.append(outputs.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        val_labels = np.concatenate(val_labels)
        val_preds = np.concatenate(val_preds)
        
        # Apply softmax to logits to get probabilities
        val_probs = torch.softmax(torch.tensor(val_preds), dim=1).numpy()
        
        val_auc = roc_auc_score(val_labels, val_probs[:, 1])
        history['val_auc'].append(val_auc)

        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}')
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)

        # Save the model checkpoint if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_model_dir, f'best_{args.model_name}_{epoch+1}_fold_{fold}_structure_{args.structure}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
                print(f'Saved model with validation loss: {val_loss:.4f} at epoch {epoch+1}')

    # Save training and validation loss history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_model_dir, f'training_history_{args.model_name}_{database}_fold_{fold}_structure_{args.structure}.csv'), index=False)

    writer.close()