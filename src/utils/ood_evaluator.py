import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class OODEvaluator:
    def __init__(self, model, id_loader, ood_loader, cfg, logger):
        self.model = model
        self.id_loader = id_loader
        self.ood_loader = ood_loader
        self.cfg = cfg
        self.logger = logger 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Determine how many images to evaluate. 
        self.eval_sample_size = 100000 

    @torch.no_grad()
    def get_layer_distributions(self, loader, name="ID"):
        print(f"\n[OOD Eval] Processing {name} (Extracting Distributions)...")
        self.model.eval()
        
        # 1. Clear the logger's internal buffers manually for BOTH metrics
        self.logger._buffers_active.clear()
        self.logger._buffers_mean.clear()
        
        collected = 0
        
        # 2. Run the forward passes to fill the logger's buffers
        for batch in tqdm(loader):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
                
            x = x.to(self.device)
            _ = self.model(x) # Triggers the hooks
            
            collected += x.size(0)
            if collected >= self.eval_sample_size:
                break
                
        # 3. Extract the raw lists of tensors for BOTH metrics
        active_distributions = {}
        mean_distributions = {}
        
        for idx, alias in enumerate(self.logger.layers):
            # Extract Fraction Active
            if alias in self.logger._buffers_active and len(self.logger._buffers_active[alias]) > 0:
                vals_active = torch.cat(self.logger._buffers_active[alias], dim=0)
                active_distributions[idx] = vals_active.cpu().numpy().tolist()
            else:
                active_distributions[idx] = []
                
            # Extract Mean Pre-activations
            if alias in self.logger._buffers_mean and len(self.logger._buffers_mean[alias]) > 0:
                vals_mean = torch.cat(self.logger._buffers_mean[alias], dim=0)
                mean_distributions[idx] = vals_mean.cpu().numpy().tolist()
            else:
                mean_distributions[idx] = []
                
        return active_distributions, mean_distributions

    def run(self):
        # 1. Get raw distributions for both ID and OOD
        id_active, id_mean = self.get_layer_distributions(self.id_loader, name="CIFAR-10 (ID)")
        ood_active, ood_mean = self.get_layer_distributions(self.ood_loader, name="SVHN (OOD)")
        
        # 2. Plot Fraction Active
        self.plot_distributions(
            id_dict=id_active, 
            ood_dict=ood_active, 
            title_metric="Fraction Active Neurons", 
            y_label="Fraction Active (Linearity)", 
            filename="ood_violin_fraction_active.png",
            baseline=0.5
        )
        
        # 3. Plot Mean of Pre-activations
        self.plot_distributions(
            id_dict=id_mean, 
            ood_dict=ood_mean, 
            title_metric="Mean of Pre-activations", 
            y_label="Activation Magnitude", 
            filename="ood_violin_mean_preact.png",
            baseline=0.0 # Standard mean for BatchNorm is 0
        )

        # 4. Plot the Sample-wise Heatmap Barcode (using Mean Pre-activations)
        self.plot_activation_heatmap(
            id_dict=id_mean, 
            ood_dict=ood_mean, 
            num_samples=100
        )

    def plot_distributions(self, id_dict, ood_dict, title_metric, y_label, filename, baseline=None):
        print(f"\nFormatting data for {title_metric}...")
        data = []
        
        # Flatten ID dictionary
        for layer, vals in id_dict.items():
            for v in vals:
                data.append({"Layer Index": layer, title_metric: v, "Dataset": "CIFAR-10 (ID)"})
                
        # Flatten OOD dictionary
        for layer, vals in ood_dict.items():
            for v in vals:
                data.append({"Layer Index": layer, title_metric: v, "Dataset": "SVHN (OOD)"})
                
        df = pd.DataFrame(data)

        print(f"Rendering Violin Plot for {title_metric}...")
        plt.figure(figsize=(14, 7))
        
        # Create the split violin plot
        sns.violinplot(
            data=df, 
            x="Layer Index", 
            y=title_metric, 
            hue="Dataset", 
            split=True, 
            inner="quartile", 
            palette={"CIFAR-10 (ID)": "#1f77b4", "SVHN (OOD)": "#d62728"},
            linewidth=1.2,
            cut=0 
        )

        # Draw vertical lines at spatial bottlenecks (Layers 5, 9, 13 for BaseNet18)
        for b in [5, 9, 13]:
            plt.axvline(x=b, color='black', alpha=0.6, linestyle=':')
            plt.text(b, plt.ylim()[1]*0.95, 'Stride-2\nBottleneck', color='black', fontsize=9, ha='center')

        # Add optional baseline (e.g., 0.5 for active, 0.0 for mean)
        if baseline is not None:
            plt.axhline(y=baseline, color='gray', alpha=0.3, linestyle='-', label=f'Theoretical Baseline ({baseline})')

        plt.title(f"Distribution of {title_metric}: ID vs OOD (BaseNet18 Width={getattr(self.model, 'k', 'NA')})", fontsize=14)
        plt.xlabel("Layer Index (Depth)", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        
        plt.legend(loc='lower left')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        save_path = filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {os.path.join(os.getcwd(), save_path)}")

    def plot_activation_heatmap(self, id_dict, ood_dict, num_samples=100):
        print("\nGenerating Activation Heatmaps...")
        
        # Helper to extract a clean NumPy matrix from the layer dictionary
        def extract_matrix(data_dict):
            layers = sorted(data_dict.keys())
            actual_samples = min(num_samples, len(data_dict[layers[0]]))
            matrix = [data_dict[l][:actual_samples] for l in layers]
            return np.array(matrix).T # Transpose to shape (samples, layers)

        id_matrix = extract_matrix(id_dict)
        ood_matrix = extract_matrix(ood_dict)

        # Force symmetric vmin/vmax so absolute 0.0 is exactly the center of the colormap
        raw_min = min(id_matrix.min(), ood_matrix.min())
        raw_max = max(id_matrix.max(), ood_matrix.max())
        abs_max = max(abs(raw_min), abs(raw_max))
        vmin, vmax = -abs_max, abs_max

        fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
        
        # 'vlag' is an excellent diverging colormap: blue (negative) -> white (zero) -> red (positive)
        cmap = "vlag" 

        sns.heatmap(id_matrix, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax, 
                    cbar_kws={'label': 'Mean Pre-activation Magnitude'})
        axes[0].set_title("CIFAR-10 (ID)", fontsize=14)
        axes[0].set_xlabel("Layer Index (Depth)", fontsize=12)
        axes[0].set_ylabel(f"Individual Image Samples (n={id_matrix.shape[0]})", fontsize=12)
        
        # Add visual markers for the spatial bottlenecks
        for b in [5, 9, 13]:
            axes[0].axvline(x=b, color='black', linestyle=':', alpha=0.5)

        sns.heatmap(ood_matrix, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax, 
                    cbar_kws={'label': 'Mean Pre-activation Magnitude'})
        axes[1].set_title("SVHN (OOD)", fontsize=14)
        axes[1].set_xlabel("Layer Index (Depth)", fontsize=12)
        
        for b in [5, 9, 13]:
            axes[1].axvline(x=b, color='black', linestyle=':', alpha=0.5)

        plt.tight_layout()
        save_path = "ood_heatmap_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {os.path.join(os.getcwd(), save_path)}")