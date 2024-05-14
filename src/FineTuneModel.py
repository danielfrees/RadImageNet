import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Define the custom model class
class FineTuneModel(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super(FineTuneModel, self).__init__()
        self.base_model = base_model
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.dropout = nn.Dropout(0.5)
        # Replace the classifier layer of the base model, already adjusted for num_classes
        # Initialize the new fully connected layer
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)
        self._init_weights()

    def _init_weights(self):
        # Initialize weights to the fc layer
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
def get_compiled_model(args, device):
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
    
def load_base_model(model_name, database, device):
    # Check model existence
    if model_name not in ['IRV2', 'ResNet50', 'DenseNet121', 'InceptionV3']:
        raise Exception('Pre-trained network not exists. Please choose IRV2/ResNet50/DenseNet121/InceptionV3 instead')

    base_model = None
    model_dir = f"../RadImageNet_models/RadImageNet-{model_name}_notop.h5"
    
    # Load the appropriate pre-trained model
    if model_name == 'IRV2' or model_name == 'InceptionV3':
        base_model = models.inception_v3(pretrained=(database == 'ImageNet'))
    elif model_name == 'ResNet50':
        base_model = models.resnet50(pretrained=(database == 'ImageNet'))
    elif model_name == 'DenseNet121':
        base_model = models.densenet121(pretrained=(database == 'ImageNet'))
    
    # Move the model to the specified device immediately after creation and before loading custom weights
    base_model = base_model.to(device)
    
    # Load custom RadImageNet weights if specified and file exists
    if database == 'RadImageNet':
        if os.path.exists(model_dir):
            base_model.load_state_dict(torch.load(model_dir, map_location=device))  # Ensure the weights are loaded to the right device
        else:
            raise Exception(f'RadImageNet model weightsfor {model_name} does not exist at specified path {model_dir}. Please ensure the file exists.')
    
    return base_model

def manage_layer_freezing(model, structure):
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
        # Unfreeze the top N layers; expect structure like 'unfreezetop10'
        # Extract the number of layers to unfreeze from the structure string
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