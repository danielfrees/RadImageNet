### run_model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# Function to run the model
def run_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device
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

    This function performs training and validation across the specified number of epochs,
    saving model checkpoints after each epoch and printing the loss values.
    """
    save_model_dir = './models/'
    os.makedirs(save_model_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        # Perform validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

        # Save the model checkpoint
        torch.save(model.state_dict(), os.path.join(save_model_dir, f'model_epoch_{epoch+1}.pth'))