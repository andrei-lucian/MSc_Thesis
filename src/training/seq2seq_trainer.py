import torch
import torch.nn as nn
import csv
from src.training.trainer import Trainer
from src.metrics.preactivation import PreactivationLogger
from pathlib import Path
import os

class Seq2SeqTrainer(Trainer):
	def __init__(self, model, train_loader, valid_loader, cfg):
		super().__init__(model, train_loader, valid_loader, cfg)

		# Replace classification loss with label-smoothed CE
		self.criterion = nn.CrossEntropyLoss(
			ignore_index=cfg.dataset.pad_idx,
			label_smoothing=getattr(cfg.training, "label_smoothing", 0.1),
		)

		# Own preactivation logger (use seq2seq-aware compute_metric)
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
			header += [f"p_{name}" for name in self.preact_logger.layers]
			writer.writerow(header)

	def train_one_epoch(self):
		self.model.train()
		total_loss, total_tokens, correct = 0, 0, 0

		for src, tgt, src_mask, tgt_mask in self.train_loader:
			src, tgt = src.to(self.device), tgt.to(self.device)
			src_mask, tgt_mask = src_mask.to(self.device), tgt_mask.to(self.device)

			# Teacher forcing: feed tgt[:-1] as input, predict tgt[1:]
			inp = tgt[:, :-1]
			target = tgt[:, 1:]

			self.optimizer.zero_grad()
			logits = self.model(
				src,
				inp,
				src_key_padding_mask=src_mask,
				tgt_key_padding_mask=tgt_mask[:, :-1],
			)
			loss = self.criterion(
				logits.reshape(-1, logits.size(-1)), target.reshape(-1)
			)
			loss.backward()
			self.optimizer.step()
			if self.scheduler:
				self.scheduler.step()

			valid_tokens = (target != self.cfg.dataset.pad_idx).sum().item()
			total_loss += loss.item() * valid_tokens

			preds = logits.argmax(-1)
			correct += (
				(preds == target)
				.masked_fill(target == self.cfg.dataset.pad_idx, 0)
				.sum()
				.item()
			)
			total_tokens += valid_tokens

		return total_loss / total_tokens, correct / total_tokens


	def evaluate(self):
		self.model.eval()
		total_loss, total_tokens, correct = 0, 0, 0

		with torch.no_grad():
			for src, tgt, src_mask, tgt_mask in self.test_loader:
				src, tgt = src.to(self.device), tgt.to(self.device)
				src_mask, tgt_mask = src_mask.to(self.device), tgt_mask.to(self.device)

				inp = tgt[:, :-1]
				target = tgt[:, 1:]

				logits = self.model(
					src,
					inp,
					src_key_padding_mask=src_mask,
					tgt_key_padding_mask=tgt_mask[:, :-1],
				)
				loss = self.criterion(
					logits.reshape(-1, logits.size(-1)), target.reshape(-1)
				)

				valid_tokens = (target != self.cfg.dataset.pad_idx).sum().item()
				total_loss += loss.item() * valid_tokens

				preds = logits.argmax(-1)
				correct += (
					(preds == target)
					.masked_fill(target == self.cfg.dataset.pad_idx, 0)
					.sum()
					.item()
				)
				total_tokens += valid_tokens

		return total_loss / total_tokens, correct / total_tokens


	def log_metrics(self, epoch, train_loss, train_acc, valid_loss, valid_acc):
		# Collect preactivation stats (seq2seq aware)
		p_l_values = self.preact_logger.compute_metric(
			self.test_loader, is_seq2seq=True, pad_idx=self.cfg.dataset.pad_idx
		)

		with open(self.log_file, "a", newline="") as f:
			writer = csv.writer(f)
			row = [epoch, train_loss, train_acc, valid_loss, valid_acc]
			row += p_l_values
			writer.writerow(row)

	def run(self):
		for epoch in range(1, self.cfg.training.epochs + 1):
			train_loss, train_acc = self.train_one_epoch()
			valid_loss, valid_acc = self.evaluate()
			self.log_metrics(epoch, train_loss, train_acc, valid_loss, valid_acc)

			if epoch % 10 == 0:
				print(
					f"[{epoch}/{self.cfg.training.epochs}] "
					f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} "
					f"Valid: loss={valid_loss:.4f}, acc={valid_acc:.4f}"
				)
