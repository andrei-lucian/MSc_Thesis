import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from hydra.utils import get_original_cwd

import random
from torch.utils.data import Dataset

class NoisyCIFAR(Dataset):
	def __init__(self, base_dataset, num_classes, noise_fraction=0.0, seed=0):
		self.base_dataset = base_dataset
		self.num_classes = num_classes
		self.noise_fraction = noise_fraction

		n_samples = len(base_dataset)
		n_noisy = int(noise_fraction * n_samples)

		rng = random.Random(seed)
		self.noisy_indices = rng.sample(range(n_samples), n_noisy)

		# Make a copy of labels
		self.targets = list(base_dataset.targets)

		for idx in self.noisy_indices:
			true_label = self.targets[idx]
			noisy_label = rng.randint(0, num_classes - 1)
			while noisy_label == true_label:
				noisy_label = rng.randint(0, num_classes - 1)
			self.targets[idx] = noisy_label

	def __len__(self):
		return len(self.base_dataset)

	def __getitem__(self, idx):
		img, _ = self.base_dataset[idx]
		label = self.targets[idx]
		return img, label

def get_dataset(cfg, seed):
	data_root = os.path.join(get_original_cwd(), "./data")  # ensure relative to project root
	if cfg.name.lower() == "cifar10":
		num_classes = 10
	elif cfg.name.lower() == "cifar100":
		num_classes = 100
	else:
		raise ValueError("Unknown dataset")

	transform_list = [transforms.ToTensor()]
	if cfg.augment == "standard":
		transform_list = [
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor()
		]
	if cfg.augment == "none":
		transform_list = [transforms.ToTensor()]	

	transform = transforms.Compose(transform_list)
	
	train_dataset = datasets.__dict__[cfg.name](
		root=data_root, train=True, download=True, transform=transform
	)
	test_dataset = datasets.__dict__[cfg.name](
		root=data_root, train=False, download=True, transform=transforms.ToTensor()
	)

	# Wrap train dataset with noise
	if getattr(cfg, "label_noise", 0.0) > 0:
		train_dataset = NoisyCIFAR(train_dataset, num_classes, noise_fraction=cfg.label_noise, seed=seed)

	train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

	return train_loader, test_loader, num_classes
