import torch
import torch.nn as nn
import torchvision.models as models

class CNNModel(nn.Module):
    def __init__(self, pretrained=True):
        super(CNNModel, self).__init__()
        # Use a pre-trained ResNet-50 model for feature extraction
        self.resnet = models.resnet50(pretrained=pretrained)
        # Remove the fully connected layers at the end
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        # Freeze the pre-trained layers
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        # Extract features from the image using the ResNet-50 model
        features = self.resnet(images)
        # Resize the features to a fixed size
        features = features.view(features.size(0), -1)
        return features

def create_cnn_model(pretrained=True):
    return CNNModel(pretrained=pretrained)
