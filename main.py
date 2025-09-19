import hydra
from omegaconf import DictConfig, OmegaConf
import os

# Import modules
from src.utils.seed import set_seed
from src.data.cifar import get_dataset
from src.models.resnet_variants import get_model
from src.training.trainer import Trainer

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
	# Print the config
	OmegaConf.set_struct(cfg, False) # Used to get the run directory when logging
	
	# Set seed for reproducibility
	set_seed(cfg.experiment.seed)

	# Build dataset
	train_loader, test_loader, num_classes = get_dataset(cfg.dataset, cfg.experiment.seed)
	
	# Build model
	model = get_model(cfg.model, num_classes)

	num_params = sum(p.numel() for p in model.parameters())
	print(f"Total parameters: {num_params:,}")
	
	# Build trainer and run
	trainer = Trainer(model, train_loader, test_loader, cfg)
	trainer.run()
	
	print(f"Experiment finished. Outputs saved to {os.getcwd()}")

if __name__ == "__main__":
	main()