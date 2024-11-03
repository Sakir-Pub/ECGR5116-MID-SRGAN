import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return out

class SRGAN_g(nn.Module):
    def __init__(self):
        super(SRGAN_g, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock() for _ in range(16)])
        
        # Second convolution
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Upsampling blocks
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixelshuffle1 = nn.PixelShuffle(2)
        
        self.conv4 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixelshuffle2 = nn.PixelShuffle(2)
        
        # Final convolution
        self.conv5 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        identity = x
        
        x = self.residual_blocks(x)
        x = self.bn1(self.conv2(x))
        x = x + identity
        
        x = F.relu(self.pixelshuffle1(self.conv3(x)))
        x = F.relu(self.pixelshuffle2(self.conv4(x)))
        x = torch.tanh(self.conv5(x))
        
        return x

class SRGAN_d(nn.Module):
    def __init__(self, input_shape=(3, 128, 128)):
        super(SRGAN_d, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        
        # Calculate size after convolutions
        self._get_final_flattened_size(input_shape)
        
        self.dense1 = nn.Linear(self.flat_size, 1024)
        self.dense2 = nn.Linear(1024, 1)
        
    def _get_final_flattened_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.bn7(self.conv8(self.bn6(self.conv7(self.bn5(self.conv6(
            self.bn4(self.conv5(self.bn3(self.conv4(self.bn2(self.conv3(
            self.bn1(self.conv2(self.conv1(x)))))))))))))))
        self.flat_size = x.shape[1] * x.shape[2] * x.shape[3]
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn1(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.bn7(self.conv8(x)), 0.2)
        
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.dense1(x), 0.2)
        x = self.dense2(x)
        
        return torch.sigmoid(x), x

# VGG19 feature extraction model
class VGG19FeatureExtractor(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        self.vgg19 = nn.Sequential(*list(vgg19.features.children())[:35])  # up to conv4_4
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        return self.vgg19(x)