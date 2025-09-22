import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Simple 5-layer CNN (CIFAR-style)
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, k=64, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(k)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)

        self.conv2 = nn.Conv2d(k, 2*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2*k)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(2*k, 4*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4*k)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(4*k, 8*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(8*k)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=8, stride=8)

        self.fc = nn.Linear(8*k, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)



# -------------------------------
# Factory function
# -------------------------------
def simple_cnn(num_classes=10, k=64):
    """5-layer CNN with widths [k, 2k, 4k, 8k]"""
    return SimpleCNN(k=k, num_classes=num_classes)