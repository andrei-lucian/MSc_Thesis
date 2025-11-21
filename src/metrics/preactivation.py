import torch
import torch.nn as nn
from typing import List, Tuple

class PreactivationLogger:
	def __init__(self, model, sample_size=500, device=None, order: str = "interleave"):
		"""
		Logs mean preactivation values AND fraction-active values.
		compute_metric()   -> mean preactivations (original API)
		compute_fraction_active() -> fraction-active per layer (new API)
		"""
		self.model = model
		self.sample_size = sample_size
		self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.order = order

		self._hooks: List[torch.utils.hooks.RemovableHandle] = []
		self._buffers = {}              # original: mean preactivation values
		self._buffers_active = {}       # fraction active values
		self.layers: List[str] = []     
		self._alias_to_raw = {}         

		collected: List[Tuple[int, int, str, nn.Module, str]] = []
		is_seq2seq = (model.__class__.__name__ == "Seq2SeqTransformer")

		# ---------------------------------------------------------
		# Collect nonlinearities
		# ---------------------------------------------------------
		for raw_name, module in model.named_modules():
			if not isinstance(module, (nn.ReLU, nn.GELU)):
				continue

			if is_seq2seq:
				if raw_name.startswith("encoder"):
					parts = raw_name.split(".")
					idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
					alias = f"enc_layer{idx}"
					group = 0
				elif raw_name.startswith("decoder"):
					parts = raw_name.split(".")
					idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
					alias = f"dec_layer{idx}"
					group = 1
				else:
					alias = raw_name
					group, idx = 2, 10**9      
			else:
				alias = raw_name
				group, idx = 0, len(collected)

			collected.append((group, idx, alias, module, raw_name))

		# ---------------------------------------------------------
		# Sort in stable deterministic order
		# ---------------------------------------------------------
		if self.order == "interleave":
			collected.sort(key=lambda t: (t[1], t[0], t[2]))
		else:
			collected.sort(key=lambda t: (t[0], t[1], t[2]))

		# ---------------------------------------------------------
		# Register hooks
		# ---------------------------------------------------------
		for group, idx, alias, module, raw_name in collected:
			h = module.register_forward_pre_hook(self._make_hook(alias))
			self._hooks.append(h)
			self.layers.append(alias)
			self._alias_to_raw[alias] = raw_name

	# -------------------------------------------------------------
	# Hook that logs (1) mean preactivation, (2) fraction active
	# -------------------------------------------------------------
	def _make_hook(self, alias: str):
		def hook(module, inputs):
			x = inputs[0].detach()
			dims = tuple(range(1, x.dim()))

			# mean preactivation
			x_mean = x.mean(dim=dims)
			self._buffers.setdefault(alias, []).append(x_mean.cpu())

			# fraction active
			active = (x > 0).float().mean(dim=dims)
			self._buffers_active.setdefault(alias, []).append(active.cpu())

		return hook

	# -------------------------------------------------------------
	# ORIGINAL PUBLIC INTERFACE (unchanged)
	# Returns ONLY mean preactivations
	# -------------------------------------------------------------
	def compute_metric(self, dataloader=None, is_seq2seq=False, pad_idx=None):
		if dataloader is None:
			raise ValueError("Need a dataloader to compute metrics.")

		self.model.eval()
		self._buffers.clear()
		self._buffers_active.clear()

		collected = 0

		# ---------------------------------------------------------
		# Run forward passes
		# ---------------------------------------------------------
		for batch in dataloader:
			if is_seq2seq:
				src, tgt, src_mask, tgt_mask = batch
				src = src.to(self.device)
				tgt = tgt.to(self.device)
				src_mask = src_mask.to(self.device)
				tgt_mask = tgt_mask.to(self.device)
				inp = tgt[:, :-1]
				with torch.no_grad():
					self.model(
						src, inp,
						src_key_padding_mask=src_mask,
						tgt_key_padding_mask=tgt_mask[:, :-1],
					)
				collected += src.size(0)
			else:
				x, _ = batch
				x = x.to(self.device)
				with torch.no_grad():
					self.model(x)
				collected += x.size(0)

			if collected >= self.sample_size:
				break

		# ---------------------------------------------------------
		# Aggregate mean preactivations
		# ---------------------------------------------------------
		out = []
		for alias in self.layers:
			if alias in self._buffers and len(self._buffers[alias]) > 0:
				vals = torch.cat(self._buffers[alias], dim=0)
				out.append(vals.mean().item())
			else:
				out.append(float("nan"))

		return out

	# -------------------------------------------------------------
	# NEW METHOD
	# Returns fraction-active values in same order & shape
	# -------------------------------------------------------------
	def compute_fraction_active(self):
		out = []
		for alias in self.layers:
			if alias in self._buffers_active and len(self._buffers_active[alias]) > 0:
				vals = torch.cat(self._buffers_active[alias], dim=0)
				out.append(vals.mean().item())
			else:
				out.append(float("nan"))
		return out
	
	def compute_both(self, dataloader=None, is_seq2seq=False, pad_idx=None):
		"""
		Runs the forward passes ONCE and returns:
		(mean_preactivations, fraction_active)
		"""
		means = self.compute_metric(dataloader, is_seq2seq, pad_idx)
		fracs = self.compute_fraction_active()
		return means, fracs

	# -------------------------------------------------------------
	# Cleanup
	# -------------------------------------------------------------
	def close(self):
		for h in self._hooks:
			h.remove()
