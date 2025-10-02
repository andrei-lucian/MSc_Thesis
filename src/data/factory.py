# src/data/factory.py
import os
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.data.cifar import NoisyCIFAR
from src.data.iwslt import get_iwslt14 


def get_dataset(cfg, seed=0):
	name = cfg.name.lower()

	if name in ["cifar10", "cifar100"]:
		# ------------------------
		# CIFAR-10/100
		# ------------------------
		data_root = os.path.join(get_original_cwd(), "./data")
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

		# Inject label noise
		if getattr(cfg, "label_noise", 0.0) > 0:
			train_dataset = NoisyCIFAR(
				train_dataset, num_classes,
				noise_fraction=cfg.label_noise,
				seed=seed
			)

		train_loader = DataLoader(
			train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
		)
		test_loader = DataLoader(
			test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
		)
		return train_loader, test_loader, num_classes

	elif name == "iwslt14":
		# ------------------------
		# IWSLT’14 De–En translation
		# ------------------------
		train_loader, test_loader, (src_vocab_size, tgt_vocab_size) = get_iwslt14(cfg)
		return train_loader, test_loader, (src_vocab_size, tgt_vocab_size)

	else:
		raise ValueError(f"Unknown dataset: {cfg.name}")
