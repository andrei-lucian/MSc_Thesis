import torch
import torch.nn as nn

class PreactivationLogger:
    def __init__(self, model, sample_size=500, device=None):
        """
        Logs mean preactivation values p_l across layers.
        Args:
            model: torch.nn.Module (with ReLU/GELU activations)
            sample_size: number of samples to use per evaluation
            device: torch.device (defaults to GPU if available)
        """
        self.model = model
        self.sample_size = sample_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Will store per-layer preactivation tensors temporarily
        self._hooks = []
        self._buffers = {}

        # Register hooks
        self.layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU)):
                # The input to the activation is the preactivation
                hook = module.register_forward_pre_hook(self._make_hook(name))
                self._hooks.append(hook)
                self.layers.append(name)

    def _make_hook(self, name):
        def hook(module, inputs):
            # inputs is a tuple; take first element
            x = inputs[0].detach()
            # Average across channels & spatial dims → (batch,)
            dims = tuple(range(1, x.dim()))  # all except batch
            x_mean = x.mean(dim=dims)
            if name not in self._buffers:
                self._buffers[name] = []
            self._buffers[name].append(x_mean.cpu())
        return hook

    def compute_metric(self, dataloader=None):
        """
        Run a batch through the model and compute p_l for each hooked layer.
        Args:
            dataloader: DataLoader (validation/test). If None, returns last stored values.
        Returns:
            list of floats: mean preactivation per layer
        """
        if dataloader is None:
            raise ValueError("Need a dataloader to compute metrics.")

		# Make sure the model is in eval before logging pre-activation
        self.model.eval()	
        self._buffers.clear()

        # Grab one batch of data (or sample_size from multiple batches)
        collected = 0
        for x, _ in dataloader:
            x = x.to(self.device)
            with torch.no_grad():
                self.model(x)
            collected += x.size(0)
            if collected >= self.sample_size:
                break

        # Aggregate across samples
        p_values = []
        for name in self.layers:
            if name in self._buffers:
                vals = torch.cat(self._buffers[name], dim=0)  # (N,)
                p_values.append(vals.mean().item())
            else:
                p_values.append(float("nan"))
        return p_values

    def close(self):
        for h in self._hooks:
            h.remove()
