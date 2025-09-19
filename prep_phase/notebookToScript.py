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

def get_cifar100_loaders(batch_size=500):
	transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),  # 🔥 Stronger augmentation
		transforms.ToTensor(),
		transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
	])
	trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
	testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
	

	# Trim dataset to make it divisible by 5
	# Define split sizes (cumulative percentages: 20%, 40%, 60%, 80%, 100%)
	total_size = len(trainset)
	split_sizes = [int(total_size * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]

	# Ensure the sum does not exceed dataset size (due to rounding)
	split_sizes[-1] = total_size  # Ensure last split gets exactly the full dataset

	# Perform the splits
	sub_datasets = [random_split(trainset, [s, total_size - s])[0] for s in split_sizes]
	sub_dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in sub_datasets]

	testloader = DataLoader(testset, batch_size=500, shuffle=True, num_workers=1) #testloader always has size 500
	
	return sub_dataloaders, testloader

class ResidualBlock(nn.Module):
	def __init__(self, inchannel, outchannel, stride=1):
		super(ResidualBlock, self).__init__()

		self.pre_activations1 = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(outchannel))
		
		self.pre_activations2 = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(outchannel)
		)

		self.shortcut = nn.Sequential()
		if stride != 1 or inchannel != outchannel:
			self.shortcut = nn.Sequential(
				nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(outchannel)
			)
			
	def forward(self, x):
		pre_activations = self.pre_activations1(x)
		pre_activations2 = self.pre_activations2(pre_activations)
		out = pre_activations2 + self.shortcut(x)
		out = F.relu(out)
		return out, pre_activations, pre_activations2

class ResNet(nn.Module):
	def __init__(self, ResidualBlock, num_classes=100):
		super(ResNet, self).__init__()
		self.inchannel = 64
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU()
		)
		self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
		self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
		self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
		self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
		self.fc = nn.Linear(512, num_classes)
		
	def make_layer(self, block, channels, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.inchannel, channels, stride))
			self.inchannel = channels
		return nn.ModuleList(layers)
	
	def forward(self, x):
		out = self.conv1(x)	
		pre_activations_list = []  # Store pre-activations
		
		pre_activations_list.append(out) # Pre-activation of the first layer

		# Iterate through each block to capture pre-activations
		for block in self.layer1:
			out, pre_act, pre_act2 = block(out)
			pre_activations_list.append(pre_act)
			pre_activations_list.append(pre_act2)	

		for block in self.layer2:
			out, pre_act, pre_act2 = block(out)
			pre_activations_list.append(pre_act)
			pre_activations_list.append(pre_act2)	
		
		for block in self.layer3:
			out, pre_act, pre_act2 = block(out)
			pre_activations_list.append(pre_act)
			pre_activations_list.append(pre_act2)	

		for block in self.layer4:
			out, pre_act, pre_act2 = block(out)
			pre_activations_list.append(pre_act)
			pre_activations_list.append(pre_act2)	
		

		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out, pre_activations_list
	
def ResNet18():
	return ResNet(ResidualBlock)

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

# Function to update the running average
def update_running_avg(new_tensor, running_sum, count):
	count += 1
	running_sum += new_tensor
	running_avg = running_sum / count
	return running_avg, running_sum, count

def test(model, device, test_loader, criterion, data_file, epoch, run, data_counter):
	"""Evaluates the model on the test dataset and computes loss & accuracy."""
	model.eval()  # Set model to evaluation mode
	total_loss = 0.0
	outputs = torch.tensor([]).to(device)
	mean_layer_act = torch.tensor([]).to(device)
	
	column_names = [f'Layer{i+1}' for i in range(17)] # Define column names for csv file
	column_names.extend(['acc', 'avg_loss', 'epoch', 'run', 'ammount_of_data']) # Add extra info columns

	# Initialize running sum and count
	running_sum = torch.zeros(17).to(device)  # 17 = Number of layers
	count = 0

	with torch.no_grad():  # Disable gradient calculation
		for inputs, labels in test_loader:
			inputs, labels = inputs.to(device), labels.to(device)
			for input, label in zip(inputs, labels):
				input, label = input.to(device), label.to(device)
				output, pre_act = model(input.unsqueeze(0))
				outputs = torch.cat((outputs, output))
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

	running_avg = torch.cat((running_avg, torch.tensor([acc, avg_loss, epoch, run, data_counter]).to(device))) # Add current epoch and run to the features
	
    # Convert to DataFrame (single row)
	df = pd.DataFrame([running_avg.cpu().numpy()], columns=column_names)

    # Append to CSV, write header only if the file does not exist
	df.to_csv(data_file, mode='a', header=not os.path.exists(data_file), index=False)

	print(acc, avg_loss)
	return avg_loss, acc

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
	
def train(model, device, train_loader, test_loader, criterion, optimizer, scheduler, run, data_counter, epochs=5):
	model.train()
	# Define CSV file path
	data_file = 'activations_per_layer.csv'
	early_stopping = EarlyStopping(patience=5, min_delta=0.001)  # Adjust patience & delta

	dataset_size = len(train_loader.dataset)  # Get total dataset size
	progress_interval = int(dataset_size * 0.05)  # Compute 5% of dataset
	count = 0  # Initialize sample counter

	for epoch in range(epochs):

		for i, (inputs, labels) in enumerate(train_loader):
			count += len(inputs)  # Increment count by batch size
			inputs, labels = inputs.to(device), labels.to(device)
			
			optimizer.zero_grad()
			outputs, _ = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			
		if count % progress_interval < len(inputs):  # Print every 5% of dataset
			avg_loss, acc = test(model, device, test_loader, criterion, data_file, epoch, run, data_counter)
		
		# scheduler.step()
		
		if early_stopping(avg_loss):
			print("Early stopping triggered. Stopping training.")
			break  # Stop training


device = 'cuda'
# model = ResNet18()
# model.to(device)
num_runs = 5

train_loaders, test_loader = get_cifar100_loaders(128)
criterion = nn.CrossEntropyLoss()

data_counter = 0
for train_loader in train_loaders:
	if data_counter >= 3:
		for run in range(num_runs):
			if run == 0 and data_counter == 3:
				run += 1
			model = ResNet18()
			model.to(device)
			optimizer = optim.Adam(model.parameters(), lr=0.001)
			# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  # For 200 epochs

			train(model, device, train_loader, test_loader, criterion, optimizer, None, run, data_counter, epochs=100) #Train for x epochs or until early stopping

	data_counter += 1