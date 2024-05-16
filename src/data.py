### data.py

import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

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
        label = 1 if self.df.iloc[idx, 1] in ['yes', 'malignant'] else 0

        if self.transform:
            image = self.transform(image)

        return image, label
    
class CaffeTransform:
    """ 
    Mimic the preprocessing of tf.keras.applications.imagenet_utils.preprocess_input 
    used by the original RadImageNet authors. 
    
    Caffe style image preprocessing. Built based on tf docs: 
    https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/imagenet_utils.py
    """
    def __init__(self, mean=[103.939, 116.779, 123.68]):   # set mean to imagenet caffe style means
        self.mean = torch.tensor(mean).view(3, 1, 1)   # convert to tensor
        self.mean = self.mean * 1./255   # rescale to apply to Tensor img in range [0,1]

    def __call__(self, img):
        # Convert from RGB to BGR
        img = img[[2, 1, 0], :, :]
        # Subtract mean based on ImageNet
        img = img - self.mean
        return img
    
class Standardize:
    """Standardize the image by scaling pixel values by 1/255."""
    def __call__(self, img):
        return img / 255.0
    

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
    transforms.RandomRotation(10, interpolation=InterpolationMode.NEAREST, fill = 128),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),  # matches
        scale=(0.9, 1.1),  # matches
        shear=0.1,   # matches https://github.com/keras-team/keras/blob/v3.1.0/keras/legacy/preprocessing/image.py
    ),
    transforms.ToTensor(),
    CaffeTransform(),   # mimic caffe style preprocess_image fnc from tensorflow
    #Standardize(),   # *1./255 (handled implicitly by ToTensor())
    # BGR
    # transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])

    # BGR with no mean (caffe already does the mean)
    transforms.Normalize(mean=[0, 0, 0], std=[0.225, 0.224, 0.229])

    # RGB
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        CaffeTransform(),   # mimic caffe style preprocess_image fnc from tensorflow
        #Standardize(),  # *1./255 (handled implicitly by ToTensor())
        #BGR
        #transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        # BGR with no mean (caffe already does the mean)
        transforms.Normalize(mean=[0, 0, 0], std=[0.225, 0.224, 0.229])
        #RGB
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = RadDataset(train_df, transform=train_transform, partial_path=partial_path)
    val_dataset = RadDataset(val_df, transform=val_transform, partial_path=partial_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 0, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers = 0, pin_memory = True)

    return train_loader, val_loader
