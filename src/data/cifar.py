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

class SubsetCIFAR(Dataset):
    """
    Filters a dataset to only include samples from the first 'num_classes'.
    Maintains the .targets attribute so it can be chained with NoisyCIFAR.
    """
    def __init__(self, base_dataset, num_classes):
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        
        # Filter indices and targets for the requested number of classes
        self.indices = []
        self.targets = []
        
        for i, target in enumerate(base_dataset.targets):
            if target < num_classes:
                self.indices.append(i)
                self.targets.append(target)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map the requested index to the original dataset's index
        original_idx = self.indices[idx]
        img, _ = self.base_dataset[original_idx]
        label = self.targets[idx]
        return img, label