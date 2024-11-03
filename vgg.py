import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19, VGG16_Weights, VGG19_Weights
from collections import OrderedDict

__all__ = [
    'VGG',
    'vgg16',
    'vgg19',
    'VGG16',
    'VGG19',
]

layer_names = [
    ['conv1_1', 'conv1_2'], 'pool1', ['conv2_1', 'conv2_2'], 'pool2',
    ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4'], 'pool3', 
    ['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4'], 'pool4',
    ['conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'], 'pool5', 
    'flatten', 'fc1_relu', 'fc2_relu', 'outputs'
]

cfg = {
    'A': [[64], 'M', [128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'B': [[64, 64], 'M', [128, 128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'D': [[64, 64], 'M', [128, 128], 'M', [256, 256, 256], 'M', [512, 512, 512], 'M', [512, 512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'E': [[64, 64], 'M', [128, 128], 'M', [256, 256, 256, 256], 'M', [512, 512, 512, 512], 'M', [512, 512, 512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
}

mapped_cfg = {
    'vgg11': 'A',
    'vgg11_bn': 'A',
    'vgg13': 'B',
    'vgg13_bn': 'B',
    'vgg16': 'D',
    'vgg16_bn': 'D',
    'vgg19': 'E',
    'vgg19_bn': 'E'
}

class VGG(nn.Module):
    def __init__(self, layer_type, batch_norm=False, end_with='outputs', name=None):
        super(VGG, self).__init__()
        self.end_with = end_with
        config = cfg[mapped_cfg[layer_type]]
        self.features = self._make_layers(config, batch_norm, end_with)

    def _make_layers(self, config, batch_norm=False, end_with='outputs'):
        layers = []
        in_channels = 3
        is_end = False
        
        for layer_group_idx, layer_group in enumerate(config):
            if isinstance(layer_group, list):
                for idx, n_filter in enumerate(layer_group):
                    layer_name = layer_names[layer_group_idx][idx]
                    layers.append(('conv{}'.format(len(layers)), 
                                 nn.Conv2d(in_channels, n_filter, kernel_size=3, padding=1)))
                    if batch_norm:
                        layers.append(('bn{}'.format(len(layers)), 
                                     nn.BatchNorm2d(n_filter)))
                    layers.append(('relu{}'.format(len(layers)), nn.ReLU(inplace=True)))
                    in_channels = n_filter
                    if layer_name == end_with:
                        is_end = True
                        break
            else:
                layer_name = layer_names[layer_group_idx]
                if layer_group == 'M':
                    layers.append(('pool{}'.format(len(layers)), 
                                 nn.MaxPool2d(kernel_size=2, stride=2)))
                elif layer_group == 'O':
                    layers.append(('fc3', nn.Linear(4096, 1000)))
                elif layer_group == 'F':
                    layers.append(('flatten', nn.Flatten()))
                elif layer_group == 'fc1':
                    layers.append(('fc1', nn.Linear(512 * 7 * 7, 4096)))
                    layers.append(('relu_fc1', nn.ReLU(inplace=True)))
                elif layer_group == 'fc2':
                    layers.append(('fc2', nn.Linear(4096, 4096)))
                    layers.append(('relu_fc2', nn.ReLU(inplace=True)))
                if layer_name == end_with:
                    is_end = True
            if is_end:
                break
                
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        """
        Parameters:
            x : tensor
                Shape [batch_size, 3, height, width], values between [0, 1]
        """
        # Convert input range from [0,1] to VGG expected format
        x = x * 255.0
        mean = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1).to(x.device)
        x = x - mean
        
        x = self.features(x)
        if self.end_with == 'outputs':
            x = x.view(x.size(0), -1)
        return x

def load_pretrained_weights(model, model_path):
    """Load pretrained weights to model"""
    weights_dict = torch.load(model_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in weights_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def vgg16(pretrained=False, end_with='outputs', name=None):
    """VGG 16-layer model"""
    model = VGG(layer_type='vgg16', batch_norm=False, end_with=end_with)
    if pretrained:
        vgg16_pretrained = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        pretrained_dict = vgg16_pretrained.state_dict()
        model_dict = model.state_dict()
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # Update model with pretrained weights
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def vgg19(pretrained=False, end_with='outputs', name=None):
    """VGG 19-layer model"""
    model = VGG(layer_type='vgg19', batch_norm=False, end_with=end_with)
    if pretrained:
        vgg19_pretrained = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        pretrained_dict = vgg19_pretrained.state_dict()
        model_dict = model.state_dict()
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # Update model with pretrained weights
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

# Aliases for backward compatibility
VGG16 = vgg16
VGG19 = vgg19

class VGG19_simple_api(nn.Module):
    def __init__(self):
        super(VGG19_simple_api, self).__init__()
        # Conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        # Conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        # Conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        # Conv4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        # Conv5
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        # Dense layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True),
            nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        # Get intermediate conv4 features for perceptual loss
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        conv = self.conv4(x)
        x = self.conv5(conv)
        x = self.classifier(x)
        return x, conv