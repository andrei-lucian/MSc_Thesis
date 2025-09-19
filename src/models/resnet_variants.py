import torch.nn as nn


# -------------------------------
# Pre-activation BasicBlock
# -------------------------------
class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # shortcut if shape mismatch
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))	 # <-- Hook sees preactivation here after bn1
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))	 # <-- Hook sees preactivation here after bn2
        out += shortcut
        return out


# -------------------------------
# Pre-activation ResNet
# -------------------------------
class PreActResNet(nn.Module):
    def __init__(self, block, layers, k=64, num_classes=100):
        super().__init__()
        self.in_planes = k

        # stem
        self.conv1 = nn.Conv2d(3, k, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(k)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # four stages with [k, 2k, 4k, 8k] channels
        self.layer1 = self._make_layer(block, k, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * k, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * k, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * k, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8 * k * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# -------------------------------
# Factory functions
# -------------------------------
def resnet18_preact(num_classes=100, k=64):
    """Pre-activation ResNet-18 with [k, 2k, 4k, 8k] channels"""
    return PreActResNet(PreActBasicBlock, [2, 2, 2, 2], k=k, num_classes=num_classes)


def resnet34_preact(num_classes=100, k=64):
    """Pre-activation ResNet-34 with [k, 2k, 4k, 8k] channels"""
    return PreActResNet(PreActBasicBlock, [3, 4, 6, 3], k=k, num_classes=num_classes)


def get_model(cfg, num_classes):
    arch = cfg.arch.lower()
    if arch == "resnet18":
        return resnet18_preact(num_classes=num_classes, k=int(64 * cfg.width_multiplier))
    elif arch == "resnet34":
        return resnet34_preact(num_classes=num_classes, k=int(64 * cfg.width_multiplier))
    else:
        raise ValueError(f"Unknown architecture: {cfg.arch}")