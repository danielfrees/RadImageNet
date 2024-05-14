### RadDataSet.py

import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Custom Dataset class for handling image loading
class RadDataset(Dataset):
    def __init__(self, dataframe: torch.Tensor, partial_path: str, transform=None):
        """
        Initializes the dataset.

        Args:
            dataframe (torch.Tensor): A dataframe containing the paths to the images and their corresponding labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = dataframe
        self.transform = transform
        self.partial_path = partial_path

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int):
        """Fetches the image and label at the index `idx` and applies transformations if any.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: (image, label) where image is the transformed image, and label is the corresponding binary label.
        """
        image_path = os.path.join(self.partial_path, self.df.iloc[idx, 0])
        image = Image.open(image_path).convert('RGB')
        label = 1 if self.df.iloc[idx, 1] == 'yes' else 0

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataloaders(train_df, val_df, batch_size: int, image_size: int, partial_path: str) -> tuple[DataLoader, DataLoader]:
    """
    Creates training and validation data loaders.

    Args:
        train_df (DataFrame): Dataframe containing training data.
        val_df (DataFrame): Dataframe containing validation data.
        batch_size (int): The size of batches.
        image_size (int): The size to which each image is resized.

    Returns:
        tuple: (train_loader, val_loader) DataLoaders for the training and validation datasets.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RadDataset(train_df, transform=train_transform, partial_path=partial_path)
    val_dataset = RadDataset(val_df, transform=val_transform, partial_path=partial_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
