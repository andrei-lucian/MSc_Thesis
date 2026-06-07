# src/data/factory.py
import os
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.data.cifar import NoisyCIFAR, SubsetCIFAR, CIFAR100Coarse 
from src.data.iwslt import get_iwslt14 

def get_dataset(cfg, seed=0):
    name = cfg.name.lower()
    data_root = os.path.join(get_original_cwd(), "data")
    
    if name in ["cifar10", "cifar100"]:
        # ------------------------
        # CIFAR-10/100
        # ------------------------
        
        if name == "cifar10":
            num_classes = 10
        else:
            num_classes = 100

        # Augmentations
        if cfg.augment == "standard":
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:  # "none"
            transform = transforms.ToTensor()

        train_dataset = datasets.__dict__[cfg.name](
            root=data_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.__dict__[cfg.name](
            root=data_root, train=False, download=True, transform=transforms.ToTensor()
        )

        # ---------------------------------------------------------
        # 2. Coarse Labels Filter (CIFAR-100 Only)
        # ---------------------------------------------------------
        # We do this first so num_classes updates to 20 before noise/subsetting
        if name == "cifar100" and getattr(cfg, "coarse_labels", False):
            print("Converting CIFAR-100 fine labels (100) to coarse superclasses (20).")
            train_dataset = CIFAR100Coarse(train_dataset)
            test_dataset = CIFAR100Coarse(test_dataset)
            num_classes = 20  # Update class count for the model head and downstream wrappers

        subset_classes = getattr(cfg, "subset_classes", None)
        if subset_classes is not None and subset_classes < num_classes:
            print(f"Subsetting {name} to {subset_classes} classes.")
            train_dataset = SubsetCIFAR(train_dataset, num_classes=subset_classes)
            test_dataset = SubsetCIFAR(test_dataset, num_classes=subset_classes)
            
            # Update num_classes so the model head and noise wrapper scale correctly
            num_classes = subset_classes 

        # ---------------------------------------------------------
        # 4. Unstructured Complexity Filter (Label Noise)
        # ---------------------------------------------------------
        if getattr(cfg, "label_noise", 0.0) > 0:
            train_dataset = NoisyCIFAR(train_dataset, num_classes, noise_fraction=cfg.label_noise, seed=seed)

    # ---------------------------------------------------------
    # 2. SVHN (Street View House Numbers)
    # ---------------------------------------------------------
    elif name == "svhn":
        num_classes = 10
        # SVHN does not use CIFAR-style augmentations
        transform = transforms.ToTensor()
        
        # SVHN uses 'split' instead of 'train'
        train_dataset = datasets.SVHN(
            root=data_root, split='train', download=True, transform=transform
        )
        test_dataset = datasets.SVHN(
            root=data_root, split='test', download=True, transform=transform
        )

    # ---------------------------------------------------------
    # 3. IWSLT
    # ---------------------------------------------------------
    elif name == "iwslt14":
        # ------------------------
        # IWSLT’14 De–En translation
        # ------------------------
        train_loader, test_loader, (src_vocab_size, tgt_vocab_size) = get_iwslt14(cfg)
        return train_loader, test_loader, (src_vocab_size, tgt_vocab_size)

    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")

    # ---------------------------------------------------------
    # Final DataLoader Construction
    # ---------------------------------------------------------
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    
    return train_loader, test_loader, num_classes