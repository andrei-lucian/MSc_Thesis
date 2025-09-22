from src.models.resnet_variants import resnet18_preact, resnet34_preact
from src.models.cnn import simple_cnn

def get_model(cfg, num_classes):
    arch = cfg.arch.lower()
    k = cfg.width
    if arch == "resnet18":
        return resnet18_preact(num_classes=num_classes, k=k)
    elif arch == "resnet34":
        return resnet34_preact(num_classes=num_classes, k=k)
    elif arch == "simplecnn":
        return simple_cnn(num_classes=num_classes, k=k)
    else:
        raise ValueError(f"Unknown architecture: {cfg.arch}")
