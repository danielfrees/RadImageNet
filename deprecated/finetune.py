### finetune.py

from argparse import Namespace
import torch
import torch.nn as nn

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
        self.activation = nn.LeakyReLU(0.1)
        self.bn = nn.BatchNorm1d(base_model.fc.in_features)  # Add BatchNorm1d before Dropout
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
        x = self.activation(x)
        x = self.fc(x)  # Pass through the new classifier
        return x