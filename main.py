import hydra
from omegaconf import DictConfig, OmegaConf
import os

# Import modules
from src.training.seq2seq_trainer import Seq2SeqTrainer
from src.utils.seed import set_seed
from src.data.factory import get_dataset
from src.models.factory import get_model
from src.training.trainer import Trainer
# from src.utils.ood_evaluator import OODEvaluator
import torch
from src.metrics.preactivation import PreactivationLogger

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
	# Print the config
	OmegaConf.set_struct(cfg, False) # Used to get the run directory when logging
	
	# Set seed for reproducibility
	set_seed(cfg.experiment.seed)

	# Build dataset
	train_loader, test_loader, extra = get_dataset(cfg.dataset, cfg.experiment.seed)
	
	# Build model
	if isinstance(extra, tuple):  # IWSLT
		src_vocab_size, tgt_vocab_size = extra
		model = get_model(cfg.model, src_vocab_size, tgt_vocab_size)
	else:  # CIFAR
		num_classes = extra
		model = get_model(cfg.model, num_classes)

	# 3. Check for OOD Mode
	if cfg.experiment.get("mode") == "ood_eval":
		print("--- Entering OOD Evaluation Mode ---")
		
		# Load trained weights (ensure path is in your config)
		if hasattr(cfg.experiment, "checkpoint_path"):
			model.load_state_dict(torch.load(cfg.experiment.checkpoint_path))
			print(f"Loaded weights from {cfg.experiment.checkpoint_path}")
		else:
			raise ValueError("OOD mode requires 'experiment.checkpoint_path' in config.")

		# Build OOD Dataset (e.g., SVHN or Noise)
		# We modify a copy of the dataset config to fetch the OOD source
		ood_dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
		ood_dataset_cfg['name'] = cfg.experiment.get("ood_dataset_name", "SVHN")
		
		# Re-using factory to get OOD loader
		_, ood_loader, _ = get_dataset(DictConfig(ood_dataset_cfg), cfg.experiment.seed)

		logger = PreactivationLogger(model, sample_size=500, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

		# # 2. Initialize Evaluator with the logger
		# evaluator = OODEvaluator(model, test_loader, ood_loader, cfg, logger)
		# evaluator.run()
		logger.close()
		
	else:
		# Standard Training Mode
		num_params = sum(p.numel() for p in model.parameters())
		print(f"Total parameters: {num_params:,}")
		
		if cfg.model.trainer == "seq2seq":
			trainer = Seq2SeqTrainer(model, train_loader, test_loader, cfg)
		else:
			trainer = Trainer(model, train_loader, test_loader, cfg)

		trainer.run()
	
	print(f"Experiment finished. Outputs saved to {os.getcwd()}")

if __name__ == "__main__":
	main()