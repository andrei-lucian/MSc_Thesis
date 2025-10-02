import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import csv
from torch.optim.lr_scheduler import LambdaLR
from src.metrics.preactivation import PreactivationLogger
import os


class Trainer:
	def __init__(self, model, train_loader: DataLoader, test_loader: DataLoader, cfg):
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.cfg = cfg

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)

		if cfg.training.optimizer == "sgd":
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr=cfg.training.lr,
				momentum=cfg.training.momentum,
				weight_decay=cfg.training.weight_decay,
			)
		elif cfg.training.optimizer == "adam":
			self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.training.lr)
		else:
			raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")
		
		# Learning rate schedule
		if cfg.training.lr_schedule == "inverse_sqrt":
			def lr_lambda(step):
				return 1.0 / ( (1 + step // 512) ** 0.5 )
			self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
		elif cfg.training.lr_schedule == "cosine":
			self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
				self.optimizer, T_max=cfg.training.epochs
			)
		else:
			self.scheduler = None

		# Loss
		self.criterion = nn.CrossEntropyLoss()

		# Preactivation logger
		self.preact_logger = PreactivationLogger(self.model)

		# Hydra automatically sets cwd to the run directory
		self.out_dir = Path(os.getcwd())
		self.out_dir.mkdir(parents=True, exist_ok=True)

		# Count parameters
		self.num_params = sum(p.numel() for p in self.model.parameters())

		# CSV log file
		self.log_file = self.out_dir / "metrics.csv"
		with open(self.log_file, "w", newline="") as f:
			writer = csv.writer(f)

			# Metadata row
			writer.writerow(["arch", "width", "seed", "noise", "params"])
			writer.writerow([
				cfg.model.arch,
				getattr(cfg.model, "width", None),
				cfg.experiment.seed,
				getattr(cfg.dataset, "label_noise", 0.0),
				self.num_params,
			])

			# Metrics header
			header = [
				"epoch", "train_loss", "train_acc",
				"test_loss", "test_acc"
			]
			header += [f"p_layer{i}" for i in range(len(self.preact_logger.layers))]
			writer.writerow(header)

	def train_one_epoch(self):
		self.model.train()
		total_loss, correct, total = 0, 0, 0

		for x, y in self.train_loader:
			x, y = x.to(self.device), y.to(self.device)
			
			self.optimizer.zero_grad()
			out = self.model(x)
			loss = self.criterion(out, y)
			loss.backward()
			self.optimizer.step()

			if self.scheduler is not None:
				self.scheduler.step()

			total_loss += loss.item() * x.size(0)
			_, preds = out.max(1)
			correct += preds.eq(y).sum().item()
			total += y.size(0)

		return total_loss / total, correct / total

	def evaluate(self):
		self.model.eval()
		total_loss, correct, total = 0, 0, 0

		with torch.no_grad():
			for x, y in self.test_loader:
				x, y = x.to(self.device), y.to(self.device)
				out = self.model(x)
				loss = self.criterion(out, y)

				total_loss += loss.item() * x.size(0)
				_, preds = out.max(1)
				correct += preds.eq(y).sum().item()
				total += y.size(0)

		return total_loss / total, correct / total

	def log_metrics(self, epoch, train_loss, train_acc, test_loss, test_acc):
		p_l_values = self.preact_logger.compute_metric(self.test_loader)

		with open(self.log_file, "a", newline="") as f:
			writer = csv.writer(f)
			row = [epoch, train_loss, train_acc, test_loss, test_acc]
			row += p_l_values
			writer.writerow(row)

	def run(self):
		for epoch in range(1, self.cfg.training.epochs + 1):
			train_loss, train_acc = self.train_one_epoch()
			test_loss, test_acc = self.evaluate()
			self.log_metrics(epoch, train_loss, train_acc, test_loss, test_acc)

			if epoch % 10 == 0:
				print(
					f"[{epoch}/{self.cfg.training.epochs}] "
					f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} "
					f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}"
				)
