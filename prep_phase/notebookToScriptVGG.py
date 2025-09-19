import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn.functional as Fchat
import random
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)  # disable inplace to retain pre-activation

        self.pre_act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        self.pre_act = x.clone().detach()  # capture before ReLU
        x = self.relu(x)
        self.pre_act += x.clone().detach()  # capture after ReLU
        self.pre_act /= 2
        return x
    
class VGGTracked(nn.Module):
    def __init__(self, cfg, num_classes=100, in_channels=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.feature_layers = self._make_layers(cfg, in_channels)
        
		# Infer the output shape to get the flattened feature size
        dummy_input = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            dummy_out = self.feature_layers(dummy_input)
        flattened_dim = dummy_out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                block = ConvBNReLU(in_channels, x)
                layers.append(block)
                self.blocks.append(block)
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_layers(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

    def get_activations(self):
        """Returns a list of (pre, post) activation tensors from each ConvBNReLU block."""
        return [(b.pre_act) for b in self.blocks]
	
class EarlyStopping:
	def __init__(self, patience=5, min_delta=0):
		"""
		Args:
			patience (int): How many epochs to wait before stopping if no improvement.
			min_delta (float): Minimum change to qualify as an improvement.
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.best_loss = float('inf')  # Track best loss
		self.counter = 0  # Count epochs without improvement

	def __call__(self, avg_loss):
		"""Returns True if training should stop."""
		if avg_loss < self.best_loss - self.min_delta:
			self.best_loss = avg_loss  # Update best loss
			self.counter = 0  # Reset counter
		else:
			self.counter += 1  # Increase counter if no improvement
		
		return self.counter >= self.patience  # Stop if patience is exceeded
    

def compute_accuracy(preds, labels):
	"""
	Computes the accuracy given predicted labels and true labels.
	
	Args:
		preds (torch.Tensor): Predicted labels (tensor of shape [batch_size])
		labels (torch.Tensor): True labels (tensor of shape [batch_size])
	
	Returns:
		float: Accuracy percentage
	"""
	correct = (preds == labels).sum().item()  # Count correct predictions
	total = labels.size(0)  # Total number of samples
	accuracy = correct / total * 100  # Compute percentage
	return accuracy


def get_cifar100_loaders(split_array = None, batch_size=500):
	transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
		transforms.ToTensor(),
		transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
	])
	
	trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
	testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
	
	if split_array is not None:
		total_size = len(trainset)
		split_sizes = [int(total_size * p) for p in split_array]

		# Ensure the sum does not exceed dataset size (due to rounding)
		split_sizes[-1] = total_size  # Ensure last split gets exactly the full dataset

		# Perform the splits
		sub_datasets = [random_split(trainset, [s, total_size - s])[0] for s in split_sizes]
		trainloader = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in sub_datasets]

	trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
	testloader = DataLoader(testset, batch_size=500, shuffle=True, num_workers=1) #testloader always has size 500
	
	return trainloader, testloader

# Function to update the running average
def update_running_avg(new_tensor, running_sum, count):
	count += 1
	running_sum += new_tensor
	running_avg = running_sum / count
	return running_avg, running_sum, count

def test(model, device, test_loader, criterion, data_file, epoch, run, name):
	"""Evaluates the model on the test dataset and computes loss & accuracy."""
	model.eval()  # Set model to evaluation mode
	total_loss = 0.0
	outputs = torch.tensor([]).to(device)
	mean_layer_act = torch.tensor([]).to(device)
	
	column_names = [f'Layer{i+1}' for i in range(len(model.blocks))] # Define column names for csv file
	column_names.extend(['acc', 'avg_loss', 'epoch', 'run', 'model_name']) # Add extra info columns

	# Initialize running sum and count
	running_sum = torch.zeros(len(model.blocks)).to(device)  # len(model.blocks) = Number of layers
	count = 0

	with torch.no_grad():  # Disable gradient calculation
		for inputs, labels in test_loader:
			inputs, labels = inputs.to(device), labels.to(device)
			for input, label in zip(inputs, labels):
				input, label = input.to(device), label.to(device)
				output = model(input.unsqueeze(0))
				outputs = torch.cat((outputs, output))
				pre_act = model.get_activations()
				for layer in pre_act:
					mean_layer_act = torch.cat((mean_layer_act,torch.mean(layer).unsqueeze(dim=0)))
				running_avg, running_sum, count = update_running_avg(mean_layer_act, running_sum, count)
				mean_layer_act = torch.tensor([]).to(device)

			loss = criterion(outputs, labels)  # Compute loss
			total_loss += loss.item()
			
			preds = torch.argmax(outputs, dim=1)  # Get predicted labels
			
			break # We only need one batch

	acc = compute_accuracy(preds, labels)
	avg_loss = total_loss / len(test_loader)  # Average loss

	numeric_data = torch.cat((running_avg, torch.tensor([acc, avg_loss, epoch, run]).to(device)))
	numeric_array = numeric_data.cpu().numpy()
	full_row = list(numeric_array) + [name]
	df = pd.DataFrame([full_row], columns=column_names)

	# Append to CSV, write header only if the file does not exist
	df.to_csv(data_file, mode='a', header=not os.path.exists(data_file), index=False)

	print(acc, avg_loss)
	return avg_loss, acc
	

def train(model, device, train_loader, test_loader, criterion, optimizer, run, name, epochs=5):
	model.train()
	# Define CSV file path
	data_file = 'activations_per_layer_vgg.csv'
	early_stopping = EarlyStopping(patience=10, min_delta=0.001)  # Adjust patience & delta

	dataset_size = len(train_loader.dataset)  # Get total dataset size
	progress_interval = int(dataset_size * 0.05)  # Compute 5% of dataset
	count = 0  # Initialize sample counter

	for epoch in range(epochs):

		for i, (inputs, labels) in enumerate(train_loader):
			count += len(inputs)  # Increment count by batch size
			inputs, labels = inputs.to(device), labels.to(device)
			
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
		if count % progress_interval < len(inputs):  # Print every 5% of dataset
			avg_loss, acc = test(model, device, test_loader, criterion, data_file, epoch, run, name)
		
		
		if early_stopping(avg_loss):
			print("Early stopping triggered. Stopping training.")
			break  # Stop training


if __name__ == '__main__':
	vgg_configs = {
		'vgg_tiny':      [32, 'M', 64, 'M'],                                    # Very small
		'vgg_small':     [64, 'M', 128, 'M'],                                   # Small
		'vgg_medium':    [64, 'M', 128, 'M', 256, 'M'],                         # Moderate depth
		'vgg_medium+':   [64, 64, 'M', 128, 'M', 256, 'M'],                     # Slightly deeper
		'vgg_large':     [64, 64, 'M', 128, 128, 'M', 256, 'M'],                # Deeper & more filters
		'vgg_large+':    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],           # More capacity
		'vgg_xlarge':    [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],      # VGG-19 like
		'vgg_xlarge+':   [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M'], # Very deep
		'vgg_huge':      [64, 64, 'M', 128, 128, 'M', 256, 256, 512, 'M'],      # Added 512 block
		'vgg_huge+':     [64, 64, 'M', 128, 128, 'M', 256, 256, 512, 512, 'M']  # Largest
	}


	'''Training loop for models of different sizes'''

	device = 'cuda'
	num_runs = 2

	train_loader, test_loader = get_cifar100_loaders(batch_size = 128)
	criterion = nn.CrossEntropyLoss()

	for name, config in vgg_configs.items():
		for run in range(num_runs):
			model = VGGTracked(config)
			model.to(device)
			optimizer = optim.Adam(model.parameters(), lr=0.001)

			train(model, device, test_loader, test_loader, criterion, optimizer, run, name, epochs=1) #Train for x epochs or until early stopping