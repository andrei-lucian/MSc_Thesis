import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy
import hydra
from omegaconf import DictConfig
from src.data.factory import get_dataset
from src.models.factory import get_model
from src.metrics.preactivation import PreactivationLogger


def build_model_from_cfg_and_extra(cfg, extra):
    """
    Mirrors your main(): if extra is (src_vocab, tgt_vocab) -> seq2seq,
    else it's CIFAR with num_classes.
    """
    if isinstance(extra, tuple):          # IWSLT
        src_vocab_size, tgt_vocab_size = extra
        model = get_model(cfg.model, src_vocab_size, tgt_vocab_size)
    else:                                 # CIFAR
        num_classes = extra
        model = get_model(cfg.model, num_classes)
    return model


# ------------------------------
# Evaluation
# ------------------------------
def evaluate(device, model, test_loader, criterion=nn.CrossEntropyLoss()):
	model.eval()
	total_loss, correct, total = 0, 0, 0

	with torch.no_grad():
		for x, y in test_loader:
			x, y = x.to(device), y.to(device)
			out = model(x)
			loss = criterion(out, y)

			total_loss += loss.item() * x.size(0)
			_, preds = out.max(1)
			correct += preds.eq(y).sum().item()
			total += y.size(0)

	return total_loss / total, correct / total


def linearize_and_evaluate(cfg, checkpoint_dir, thresholds=[0.0, 0.1, 0.2]):
	"""
	For each checkpoint, compute mean preactivations and test performance
	for multiple linearization thresholds.
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	_, test_loader, extra = get_dataset(cfg.dataset, seed=0)
	ckpts = sorted(Path(checkpoint_dir).glob("*.pt"))
	results = []
	print(f"[Debug] Looking for checkpoints in: {checkpoint_dir}")
	print(list(Path(checkpoint_dir).glob("*")))

	# Path for the cumulative results file
	ckpt_root = Path(checkpoint_dir)
	results_csv = ckpt_root.parent / "linearization_results.csv"
	header_written = results_csv.exists()

	for ckpt_path in ckpts:

		print(f"\n[Analysis] Loading {ckpt_path.name}")
		try:
			step_or_epoch = int(ckpt_path.stem.split("_")[1])
		except Exception:
			step_or_epoch = -1  # fallback if not numeric
		
		# --- 1. Load model ---
		model = build_model_from_cfg_and_extra(cfg, extra)
		model.load_state_dict(torch.load(ckpt_path, map_location=device))
		model.to(device)
		model.eval()

		# --- 2. Compute mean preactivations ---
		preact_logger = PreactivationLogger(model)
		p_means = preact_logger.compute_metric(test_loader)  # returns list of floats
		

		# --- 3. Sweep over thresholds (sorted ascending) ---
		for tau in sorted(thresholds):
			linearized_layers = [i for i, m in enumerate(p_means) if m > tau]

			# Skip this and all remaining thresholds if no layers qualify
			if len(linearized_layers) == 0:
				print(f"[Skip τ={tau:.2f}] No layers above threshold, moving to next checkpoint.")
				break

			# Build linearized model only if needed
			lin_model = deepcopy(model).to(device).eval()

			# Replace ReLUs with Identity for selected layers
			relu_counter = 0
			for name, module in lin_model.named_modules():
				if isinstance(module, nn.ReLU):
					if relu_counter in linearized_layers:
						parent_name = ".".join(name.split(".")[:-1])
						attr_name = name.split(".")[-1]
						parent = lin_model.get_submodule(parent_name) if parent_name else lin_model
						setattr(parent, attr_name, nn.Identity())
					relu_counter += 1

			# --- 4. Evaluate ---
			test_loss, test_acc = evaluate(device, lin_model, test_loader)

			# --- 5. Append result to CSV ---
			df = pd.DataFrame([{
				"index": step_or_epoch,
				"checkpoint": ckpt_path.name,
				"threshold": float(tau),
				"num_linear_layers": len(linearized_layers),
				"test_loss": float(test_loss),
				"test_acc": float(test_acc),
			}])
			df.to_csv(results_csv, mode="a", header=not header_written, index=False)
			header_written = True

			print(f"[Saved] step={step_or_epoch}, τ={tau:.2f}, "
				f"linear_layers={len(linearized_layers)}, acc={test_acc:.4f}")

			
	print("\n[Analysis complete] Results saved to linearization_results.csv")
	return df


# ------------------------------
# Run using python -m src.metrics.analyze_linearization
# ------------------------------

@hydra.main(config_path='../../configs', config_name="analyze", version_base=None)
def main(cfg: DictConfig):
	# Run linearization sweep
	linearize_and_evaluate(cfg, cfg.experiment.checkpoint_dir, cfg.experiment.thresholds)

if __name__ == "__main__":
	main()