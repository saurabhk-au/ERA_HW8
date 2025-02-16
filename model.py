import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv1x1_1 = nn.Conv2d(16, 16, kernel_size=1)
        self.bn1x1_1 = nn.BatchNorm2d(16)
        self.dropout1x1_1 = nn.Dropout(0.3)
        
        self.conv2 = DepthwiseSeparableConv(16, 32, kernel_size=3, padding=1, stride=1)
        
        self.conv1x1_2 = nn.Conv2d(32, 32, kernel_size=1)
        self.bn1x1_2 = nn.BatchNorm2d(32)
        self.dropout1x1_2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.3)
        
        self.conv1x1_3 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn1x1_3 = nn.BatchNorm2d(64)
        self.dropout1x1_3 = nn.Dropout(0.3)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(0.3)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn1x1_1(self.conv1x1_1(x)))
        x = self.dropout1x1_1(x)
        
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.bn1x1_2(self.conv1x1_2(x)))
        x = self.dropout1x1_2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn1x1_3(self.conv1x1_3(x)))
        x = self.dropout1x1_3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x