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

class CIFAR100Coarse(Dataset):
    """
    A Dataset wrapper that converts CIFAR-100 fine labels (0-99) 
    into coarse superclass labels (0-19).
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
        # This is the official mapping from the CIFAR-100 fine classes to the 20 coarse classes.
        # Index = fine_label, Value = coarse_label
        self.fine_to_coarse_map = [
             4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  # 0-9
             3, 14,  9, 18,  7, 11,  3,  9,  7, 11,  # 10-19
             6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  # 20-29
             0, 11,  1, 10, 12, 14, 16,  9, 11,  5,  # 30-39
             5, 19,  8,  8, 15, 13, 14, 17, 18, 10,  # 40-49
            16,  4, 17,  4,  2,  0, 17,  4, 18, 17,  # 50-59
            10,  3,  2, 12, 12, 16, 12,  1,  9, 19,  # 60-69
             2, 10,  0,  1, 16, 12,  9, 13, 15, 13,  # 70-79
            16, 19,  2,  4,  6, 19,  5,  5,  8, 19,  # 80-89
            18,  1,  2, 15,  6,  0, 17,  8, 14, 13   # 90-99
        ]
        
        # We also overwrite the targets list so this behaves like a standard dataset
        # (Useful if you chain this with other wrappers later)
        if hasattr(base_dataset, 'targets'):
            self.targets = [self.fine_to_coarse_map[label] for label in base_dataset.targets]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, fine_label = self.base_dataset[idx]
        
        # Convert the fine label to the coarse label
        coarse_label = self.fine_to_coarse_map[fine_label]
        
        return img, coarse_label