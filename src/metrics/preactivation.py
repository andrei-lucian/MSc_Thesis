import torch
import torch.nn as nn
from typing import List, Tuple

class PreactivationLogger:
    def __init__(self, model, sample_size=500, device=None, order: str = "interleave"):
        """
        Logs mean preactivation values p_l across layers.
        Works for both classifiers (x, y) and seq2seq (src, tgt, masks).

        order: "enc_first" (enc0..encN, dec0..decN) or "interleave" (enc0,dec0,enc1,dec1,...)
        """
        self.model = model
        self.sample_size = sample_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.order = order

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._buffers = {}
        self.layers: List[str] = []           # ordered aliases used for CSV header
        self._alias_to_raw = {}               # optional: alias -> raw module name

        # 1) Collect all (group, idx, alias, module) first
        collected: List[Tuple[int, int, str, nn.Module, str]] = []
        is_seq2seq = (model.__class__.__name__ == "Seq2SeqTransformer")

        for raw_name, module in model.named_modules():
            if not isinstance(module, (nn.ReLU, nn.GELU)):
                continue

            if is_seq2seq:
                if raw_name.startswith("encoder"):
                    # e.g. "encoder.layers.3.relu" -> idx = 3
                    parts = raw_name.split(".")
                    # guard for unexpected names
                    idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
                    alias = f"enc_layer{idx}"
                    group = 0
                elif raw_name.startswith("decoder"):
                    parts = raw_name.split(".")
                    idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
                    alias = f"dec_layer{idx}"
                    group = 1
                else:
                    # fallback for anything else in seq2seq model
                    alias = raw_name
                    group, idx = 2, 10**9  # push to the end
            else:
                alias = raw_name
                group, idx = 0, len(collected)  # preserve discovery order

            collected.append((group, idx, alias, module, raw_name))

        # 2) Sort deterministically
        if self.order == "interleave":
            # interleave by idx: (enc0, dec0, enc1, dec1, ...)
            # we’ll sort by idx, then by group so enc(0) comes before dec(0)
            collected.sort(key=lambda t: (t[1], t[0], t[2]))
        else:
            # enc_first (default): enc0..encN, then dec0..decN, then others
            collected.sort(key=lambda t: (t[0], t[1], t[2]))

        # 3) Register hooks using the alias as the key
        for group, idx, alias, module, raw_name in collected:
            h = module.register_forward_pre_hook(self._make_hook(alias))
            self._hooks.append(h)
            self.layers.append(alias)
            self._alias_to_raw[alias] = raw_name  # optional: handy for debugging

    def _make_hook(self, alias: str):
        def hook(module, inputs):
            x = inputs[0].detach()
            # average over all dims except batch: for seq2seq x is [B, L, d_ff]
            dims = tuple(range(1, x.dim()))
            x_mean = x.mean(dim=dims)
            self._buffers.setdefault(alias, []).append(x_mean.cpu())
        return hook

    def compute_metric(self, dataloader=None, is_seq2seq=False, pad_idx=None):
        """
        Returns a list of mean preactivations per alias in self.layers (ordered).
        """
        if dataloader is None:
            raise ValueError("Need a dataloader to compute metrics.")

        self.model.eval()
        self._buffers.clear()

        collected = 0
        for batch in dataloader:
            if is_seq2seq:
                src, tgt, src_mask, tgt_mask = batch
                src, tgt = src.to(self.device), tgt.to(self.device)
                src_mask, tgt_mask = src_mask.to(self.device), tgt_mask.to(self.device)

                inp = tgt[:, :-1]
                with torch.no_grad():
                    self.model(
                        src, inp,
                        src_key_padding_mask=src_mask,
                        tgt_key_padding_mask=tgt_mask[:, :-1]
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

        # Aggregate in the deterministic alias order
        out = []
        for alias in self.layers:
            if alias in self._buffers and len(self._buffers[alias]) > 0:
                vals = torch.cat(self._buffers[alias], dim=0)  # (N,)
                out.append(vals.mean().item())
            else:
                out.append(float("nan"))
        return out

    def close(self):
        for h in self._hooks:
            h.remove()
