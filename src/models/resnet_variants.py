import torch.nn as nn
import torchvision.models as models

def get_model(cfg, num_classes):
    if cfg.arch.lower() == "resnet18":
        model = models.resnet18(pretrained=cfg.pretrained)
    elif cfg.arch.lower() == "resnet34":
        model = models.resnet34(pretrained=cfg.pretrained)
    else:
        raise ValueError("Unknown architecture")
    
    # adjust width
    if cfg.width_multiplier != 1.0:
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                module.out_channels = int(module.out_channels * cfg.width_multiplier)
    
    # final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model