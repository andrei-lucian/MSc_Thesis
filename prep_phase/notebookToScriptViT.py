import torch
import timm
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import sys
from collections import defaultdict

# Training loop
def train_model(model, train_loader, val_loader, epochs=10):
	
	# Loss & optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

	block_acts = defaultdict(dict)
	avg_acts = []  # Final average activations (fc1 + fc2) per block

	def make_hook(block_idx, layer_name):
		def hook(module, inp, outp):
			if not module.training:
				act = outp.detach()[1:]  # Remove CLS token
				mean_val = act.mean().item()

				block_acts[block_idx][layer_name] = mean_val

				if 'fc1' in block_acts[block_idx] and 'fc2' in block_acts[block_idx]:
					avg_val = (block_acts[block_idx]['fc1'] + block_acts[block_idx]['fc2']) / 2
					avg_acts.append(avg_val)

		return hook

	for idx, blk in enumerate(model.blocks):
		blk.mlp.fc1.register_forward_hook(make_hook(idx, 'fc1'))
		blk.mlp.fc2.register_forward_hook(make_hook(idx, 'fc2'))
		
	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		correct = 0
		total = 0

		loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

		for images, labels in loop:
			images, labels = images.to(device), labels.to(device)

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * images.size(0)
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()

			loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

		train_loss = running_loss / len(train_loader.dataset)
		train_acc = 100. * correct / total

		# Validation
		model.eval()
		val_loss = 0.0
		val_correct = 0
		val_total = 0
		avg_acts = []

		with torch.no_grad():
			for images, labels in val_loader:
				images, labels = images.to(device), labels.to(device)
				outputs = model(images)
				loss = criterion(outputs, labels)
				val_loss += loss.item() * images.size(0)
				_, predicted = outputs.max(1)
				val_total += labels.size(0)
				val_correct += predicted.eq(labels).sum().item()
				break # Only do one batch for validation to grab the preactivations

		val_loss /= len(val_loader.dataset)
		val_acc = 100. * val_correct / val_total

		print(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")

		# Build DataFrame
		rows = []
		for block_idx, avg_activation in enumerate(avg_acts):
			row = {
				"epoch": epoch + 1,
				"block_idx": block_idx,
				"val_loss": val_loss,
				"val_acc": val_acc,
				"activation": avg_activation
			}
			rows.append(row)

		df = pd.DataFrame(rows)

		# Save as CSV
		os.makedirs("activations_csv", exist_ok=True)
		csv_path = "activations_csv/all_epochs.csv"
		if not os.path.exists(csv_path):
			df.to_csv(csv_path, index=False)
		else:
			df.to_csv(csv_path, mode='a', header=False, index=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) load a pretrained ViT
model = timm.create_model('vit_small_patch16_224', pretrained=True)
model.to(device)

transform = transforms.Compose([
	transforms.Resize((224, 224)), 
	transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
	transforms.ToTensor(),
	transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=1)
test_loader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=1) 

train_model(model, train_loader, test_loader, epochs=20)