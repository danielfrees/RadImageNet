### run_model.py

from argparse import Namespace
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Function to run the model
def run_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: Namespace,
    device: torch.device,
    partial_path: str,
    fold: int
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

    This function performs training and validation across the specified number of epochs,
    saving model checkpoints after each epoch and printing the loss values.
    """
    num_epochs = args.epoch
    verbose = args.verbose

    save_model_dir = os.path.join(partial_path, 'models')
    os.makedirs(save_model_dir, exist_ok=True)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    for epoch in range(num_epochs):
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
        
        train_auc = roc_auc_score(all_labels, all_preds[:, 1])
        history['train_auc'].append(train_auc)

        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training AUC: {train_auc:.4f}')

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
        
        val_auc = roc_auc_score(val_labels, val_preds[:, 1])
        history['val_auc'].append(val_auc)

        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}')

        # Save the model checkpoint if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_model_dir, f'best_{args.model_name}_{epoch+1}_fold_{fold}_structure_{args.structure}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
                print(f'Saved model with validation loss: {val_loss:.4f} at epoch {epoch+1}')

    # Save training and validation loss history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_model_dir, f'training_history_{args.model_name}_fold_{fold}_structure_{args.structure}.csv'), index=False)