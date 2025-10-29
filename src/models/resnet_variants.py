import torch
import torch.nn as nn

# =========================================================
# === BaseNet18 (no residuals, for nonlinearity analysis) ==
# =========================================================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, nonlinear=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if nonlinear else nn.Identity()

    def forward(self, x):
        x = self.bn(self.conv(x))   # preactivation = output of BN
        x = self.relu(x)
        return x


class BaseNet18_CIFAR(nn.Module):
    """
    Simple CNN (BaseNet18) following Pinson et al. (2024) style.
    No residual connections; used for studying effective nonlinearity.
    """
    def __init__(self, first_n_linear=0, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Config: 18 conv layers total (same as ResNet18, but no skips)
        cfg = [(64, 1)] * 4 + [(128, 2)] + [(128, 1)] * 3 + \
              [(256, 2)] + [(256, 1)] * 3 + [(512, 2)] + [(512, 1)] * 3

        self.blocks = nn.ModuleList()
        in_channels = 64
        for i, (out_channels, stride) in enumerate(cfg):
            nonlinear = (i >= first_n_linear)
            self.blocks.append(BasicBlock(in_channels, out_channels, stride, nonlinear))
            in_channels = out_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =========================================================
# === Pre-activation ResNet (with residuals) ==============
# =========================================================
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

        # Shortcut if shape changes
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))   # <-- preactivation before first conv
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))  # <-- preactivation before second conv
        out += shortcut
        return out


class PreActResNet(nn.Module):
    """
    CIFAR-style Pre-activation ResNet (He et al. 2016).
    """
    def __init__(self, block, layers, k=64, num_classes=10):
        super().__init__()
        self.in_planes = k

        # CIFAR-style stem: 3×3 conv, stride=1, no maxpool
        self.conv1 = nn.Conv2d(3, k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(k)
        self.relu = nn.ReLU(inplace=True)

        # four stages with [k, 2k, 4k, 8k] channels
        self.layer1 = self._make_layer(block, k, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * k, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * k, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * k, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8 * k * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =========================================================
# === Factory functions ===================================
# =========================================================
def resnet18_preact(num_classes=10, k=64):
    """Pre-activation ResNet-18"""
    return PreActResNet(PreActBasicBlock, [2, 2, 2, 2], k=k, num_classes=num_classes)

def resnet34_preact(num_classes=10, k=64):
    """Pre-activation ResNet-34"""
    return PreActResNet(PreActBasicBlock, [3, 4, 6, 3], k=k, num_classes=num_classes)

def basenet18(num_classes=10, first_n_linear=0):
    """Non-residual CNN (BaseNet18)"""
    return BaseNet18_CIFAR(first_n_linear=first_n_linear, num_classes=num_classes)