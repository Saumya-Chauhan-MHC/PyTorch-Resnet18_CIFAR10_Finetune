import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    # model = model.to('cuda')
    return model
