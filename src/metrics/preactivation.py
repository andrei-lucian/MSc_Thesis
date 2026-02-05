import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class PreactivationLogger:
    def __init__(self, model, sample_size=500, device=None, order: str = "interleave"):
        """
        Logs:
        1. Mean: Standard center measure
        2. Median: Robust center measure
        3. Cosine Similarity: Linearity measure (1.0 = Linear, ~0.707 = ReLU)
        4. Fraction Active: Raw sparsity measure (>0).
        """
        self.model = model
        self.sample_size = sample_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.order = order

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Buffers for all 4 metrics
        self._buffers_mean = {}
        self._buffers_median = {}
        self._buffers_cossim = {}
        self._buffers_active = {}  # <--- RE-ADDED
        
        self.layers: List[str] = []     
        self._alias_to_raw = {}         

        collected: List[Tuple[int, int, str, nn.Module, str]] = []
        
        # ---------------------------------------------------------
        # Collect nonlinearities AND Identity (linear) layers
        # ---------------------------------------------------------
        print(f"[PreactivationLogger] Scanning model for layers...")
        for raw_name, module in model.named_modules():
            # Check for ReLU, GELU, or Identity
            if not isinstance(module, (nn.ReLU, nn.GELU, nn.Identity)):
                continue

            alias = raw_name
            group, idx = 0, len(collected) 
            collected.append((group, idx, alias, module, raw_name))

        # ---------------------------------------------------------
        # Sort
        # ---------------------------------------------------------
        if self.order == "interleave":
            collected.sort(key=lambda t: (t[1], t[0], t[2]))
        else:
            collected.sort(key=lambda t: (t[0], t[1], t[2]))

        # ---------------------------------------------------------
        # Register hooks
        # ---------------------------------------------------------
        for group, idx, alias, module, raw_name in collected:
            h = module.register_forward_pre_hook(self._make_hook(alias, module))
            self._hooks.append(h)
            self.layers.append(alias)
            self._alias_to_raw[alias] = raw_name

    def _make_hook(self, alias: str, module: nn.Module):
        # Robust check for Identity (Linear) layers
        is_fixed_linear = isinstance(module, nn.Identity) or "Identity" in str(module.__class__)

        def hook(module, inputs):
            x = inputs[0].detach()
            # Flatten: [Batch, Channel, H, W] -> [Batch, Features]
            x_flat = x.view(x.size(0), -1)

            # 1. Mean
            self._buffers_mean.setdefault(alias, []).append(x_flat.mean(dim=1).cpu())

            # 2. Median
            median_val = x_flat.median(dim=1).values
            self._buffers_median.setdefault(alias, []).append(median_val.cpu())

            # 3. Linearity Score (Cosine Similarity)
            if is_fixed_linear:
                # Linear Layer: Input == Output, so similarity is exactly 1.0
                cossim = torch.ones(x.size(0), device=x.device)
            else:
                # ReLU Layer: Measure how much ReLU changes the vector angle
                x_relu = F.relu(x_flat)
                # eps=1e-8 prevents division by zero if vector is dead
                cossim = F.cosine_similarity(x_flat, x_relu, dim=1, eps=1e-8)
            self._buffers_cossim.setdefault(alias, []).append(cossim.cpu())

            # 4. Fraction Active
            # We calculate this raw (x > 0) to demonstrate the BN effect (pinned at 0.5)
            active = (x_flat > 0).float().mean(dim=1)
            self._buffers_active.setdefault(alias, []).append(active.cpu())

        return hook

    def _compute_buffer(self, buffer_dict):
        out = []
        for alias in self.layers:
            if alias in buffer_dict and len(buffer_dict[alias]) > 0:
                vals = torch.cat(buffer_dict[alias], dim=0)
                out.append(vals.mean().item())
            else:
                out.append(float("nan"))
        return out

    def compute_all(self, dataloader=None):
        if dataloader is None:
            raise ValueError("Need a dataloader to compute metrics.")

        self.model.eval()
        self._buffers_mean.clear()
        self._buffers_median.clear()
        self._buffers_cossim.clear()
        self._buffers_active.clear() # <--- Clear buffer

        collected = 0

        # Handle various dataloader formats
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                 x = batch[0]
            else:
                 x = batch
            
            x = x.to(self.device)
            with torch.no_grad():
                self.model(x)
            
            collected += x.size(0)
            if collected >= self.sample_size:
                break

        means = self._compute_buffer(self._buffers_mean)
        medians = self._compute_buffer(self._buffers_median)
        cossims = self._compute_buffer(self._buffers_cossim)
        fracs_active = self._compute_buffer(self._buffers_active) # <--- Compute

        # Returns 4 lists now
        return means, medians, cossims, fracs_active

    def close(self):
        for h in self._hooks:
            h.remove()