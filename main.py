import hydra
from omegaconf import DictConfig, OmegaConf

# Import modules
from src.utils.seed import set_seed
from src.data.cifar import get_dataset
from src.models.resnet_variants import get_model
# from src.training.trainer import Trainer

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
	# Print the config
	# print(OmegaConf.to_yaml(cfg))
	
	# Set seed for reproducibility
	set_seed(cfg.experiment.seed)
	
	# Build dataset
	train_loader, test_loader, num_classes = get_dataset(cfg.dataset)
	
	print(train_loader)
	
	# Build model
	model = get_model(cfg.model, num_classes)

	print(model)
	
	# # Build trainer and run
	# trainer = Trainer(model, train_loader, test_loader, cfg)
	# trainer.run()
	
	print(f"Experiment finished. Outputs saved to {cfg.experiment.output_dir}")

if __name__ == "__main__":
	main()
