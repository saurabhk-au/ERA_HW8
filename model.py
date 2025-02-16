import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Output: [-1, 16, 32, 32]
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout2d(0.01)
        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)  # Output: [-1, 16, 32, 32]
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout2d(0.01)
        
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)  # Output: [-1, 16, 32, 32]
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout2d(0.01)
        
        # Second block with Depthwise Separable Convolution
        self.depthwise_conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)  # Depthwise
        self.pointwise_conv4 = nn.Conv2d(16, 32, kernel_size=1)  # Pointwise
        self.bn4 = nn.BatchNorm2d(32)
        self.dropout4 = nn.Dropout2d(0.01)
        
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # Output: [-1, 32, 32, 32]
        self.bn5 = nn.BatchNorm2d(32)
        self.dropout5 = nn.Dropout2d(0.01)
        
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)  # Output: [-1, 32, 16, 16]
        self.bn6 = nn.BatchNorm2d(32)
        self.dropout6 = nn.Dropout2d(0.01)
        
        # Third block
        self.depthwise_conv7 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)  # Depthwise
        self.pointwise_conv7 = nn.Conv2d(32, 64, kernel_size=1)  # Pointwise
        self.bn7 = nn.BatchNorm2d(64)
        self.dropout7 = nn.Dropout2d(0.01)
        
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # Output: [-1, 64, 16, 16]
        self.bn8 = nn.BatchNorm2d(64)
        self.dropout8 = nn.Dropout2d(0.01)
        
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # Output: [-1, 64, 8, 8]
        self.bn9 = nn.BatchNorm2d(64)
        self.dropout9 = nn.Dropout2d(0.01)
        
        # Fourth block
        self.conv10 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # Output: [-1, 32, 8, 8]
        self.bn10 = nn.BatchNorm2d(32)
        self.dropout10 = nn.Dropout2d(0.01)
        
        self.conv11 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # Output: [-1, 32, 8, 8]
        self.bn11 = nn.BatchNorm2d(32)
        self.dropout11 = nn.Dropout2d(0.01)
        
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3)  # Output: [-1, 32, 6, 6]
        
        self.gap = nn.AvgPool2d(6)  # Global Average Pooling
        self.linear = nn.Conv2d(32, 10, kernel_size=1)  # Output: [-1, 10, 1, 1]

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.pointwise_conv4(self.depthwise_conv4(x))))
        x = self.dropout4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout5(x)
        
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout6(x)
        
        x = F.relu(self.bn7(self.pointwise_conv7(self.depthwise_conv7(x))))
        x = self.dropout7(x)
        
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.dropout8(x)
        
        x = F.relu(self.bn9(self.conv9(x)))
        x = self.dropout9(x)
        
        x = F.relu(self.bn10(self.conv10(x)))
        x = self.dropout10(x)
        
        x = F.relu(self.bn11(self.conv11(x)))
        x = self.dropout11(x)
        
        x = self.conv12(x)
        x = self.gap(x)
        x = self.linear(x)
        x = x.view(x.size(0), 10)
        return F.log_softmax(x, dim=1)