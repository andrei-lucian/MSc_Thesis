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


# =========================================================
# === Model builder
# =========================================================
def build_model_from_cfg_and_extra(cfg, extra):

    if isinstance(extra, tuple):
        src_vocab_size, tgt_vocab_size = extra
        model = get_model(cfg.model, src_vocab_size, tgt_vocab_size)
    else:
        num_classes = extra
        model = get_model(cfg.model, num_classes)
    return model


# =========================================================
# === Evaluation helper
# =========================================================
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


# =========================================================
# === New: automatic τ computation per model
# =========================================================
def compute_global_tau_from_metrics(
    metrics_csv: Path,
    mode: str = "quantile",  # "quantile" or "robust_z"
    q: float = 0.8,
    z: float = 1.0,
    burn_in: int = 0,
    positive_only: bool = True,
):
    """Compute one adaptive τ value for a model's entire training run."""
    if not metrics_csv.exists():
        print(f"o metrics.csv found at {metrics_csv}, skipping τ computation.")
        return []

    df = pd.read_csv(metrics_csv, skiprows=2)
    df.columns = [c.strip().lower() for c in df.columns]

    # detect step/epoch column
    if "step" in df.columns:
        step_col = "step"
    elif "epoch" in df.columns:
        step_col = "epoch"
    else:
        step_col = next((c for c in df.columns if np.issubdtype(df[c].dtype, np.number)), None)

    if step_col and burn_in > 0 and step_col in df.columns:
        df = df[df[step_col] >= burn_in].copy()

    # extract preactivation columns
    layer_cols = [c for c in df.columns if c.startswith("p_layer")]
    if not layer_cols:
        print(f"No p_layer* columns found in {metrics_csv.name}, skipping τ computation.")
        return []

    mus = df[layer_cols].to_numpy().ravel()
    mus = mus[np.isfinite(mus)]
    if positive_only:
        mus = mus[mus > 0]
    if mus.size == 0:
        print(f"No valid activations found in {metrics_csv.name}.")
        return []

    if mode == "quantile":
        taus = [float(np.quantile(mus, q)) for q in [0.65, 0.75, 0.8, 0.9, 0.95]]
    elif mode == "robust_z":
        med = float(np.median(mus))
        mad = float(np.median(np.abs(mus - med))) or 1e-8
        taus = [med + z_thr * mad for z_thr in [0.5, 1.0, 1.5]]
    else:
        raise ValueError("Mode must be 'quantile' or 'robust_z'.")

    print(f"[τ Computation] {metrics_csv.name}: computed τs = {taus}")
    return taus


# =========================================================
# === Linearization + Evaluation
# =========================================================
def linearize_and_evaluate(cfg, checkpoint_dir, thresholds):
    """For each checkpoint, compute mean preactivations and test performance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, extra = get_dataset(cfg.dataset, seed=1)
    ckpts = sorted(Path(checkpoint_dir).glob("*.pt"))
    results = []
    print(f"[Debug] Looking for checkpoints in: {checkpoint_dir}")
    print(list(Path(checkpoint_dir).glob("*")))

    # cumulative results file
    ckpt_root = Path(checkpoint_dir)
    results_csv = ckpt_root.parent / "linearization_results.csv"
    header_written = results_csv.exists()

    for ckpt_path in ckpts:
        print(f"\n[Analysis] Loading {ckpt_path.name}")
        try:
            step_or_epoch = int(ckpt_path.stem.split("_")[1])
        except Exception:
            step_or_epoch = -1

        # 1. Load model
        model = build_model_from_cfg_and_extra(cfg, extra)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()

        # 2. Compute mean preactivations
        preact_logger = PreactivationLogger(model)
        p_means = preact_logger.compute_metric(test_loader)

        # 3. Sweep over thresholds
        for tau in sorted(thresholds):
            linearized_layers = [i for i, m in enumerate(p_means) if m > tau]
            if len(linearized_layers) == 0:
                print(f"[Skip τ={tau:.3f}] No layers above threshold → next checkpoint.")
                break

            lin_model = deepcopy(model).to(device).eval()

            # Replace ReLUs with Identity
            relu_counter = 0
            for name, module in lin_model.named_modules():
                if isinstance(module, nn.ReLU):
                    if relu_counter in linearized_layers:
                        parent_name = ".".join(name.split(".")[:-1])
                        attr_name = name.split(".")[-1]
                        parent = lin_model.get_submodule(parent_name) if parent_name else lin_model
                        setattr(parent, attr_name, nn.Identity())
                    relu_counter += 1

            # 4. Evaluate
            test_loss, test_acc = evaluate(device, lin_model, test_loader)

            # 5. Save to CSV
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

            print(f"[Saved] step={step_or_epoch}, τ={tau:.3f}, "
                  f"linear_layers={len(linearized_layers)}, acc={test_acc:.4f}")

    print(f"\n[Analysis complete] Results saved to {results_csv}")
    return results_csv



# =========================================================
# === Main entrypoint
# Run using python -m src.metrics.analyze_linearization
# =========================================================
@hydra.main(config_path='../../configs', config_name="analyze", version_base=None)
def main(cfg: DictConfig):
    """
    Finds all experiment subfolders under cfg.experiment.checkpoint_dir,
    parses width/seed/noise from folder names,
    and runs linearization for each one (skips if results already exist).
    """
    base_dir = Path(cfg.experiment.checkpoint_dir)
    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]

    print(f"[Analysis] Searching for experiments in {base_dir}")
    print(f"Found {len(subdirs)} experiment folders")

    for sub in subdirs:
        ckpt_dir = sub / "checkpoints"
        results_csv = sub / "linearization_results.csv"
        metrics_csv = sub / "metrics.csv"

        if results_csv.exists():
            print(f"[Skip] {sub.name} — results already exist.")
            continue
        if not ckpt_dir.exists():
            print(f"[Skip] {sub.name} — no checkpoints found.")
            continue

        # Parse experiment metadata from folder name
        name = sub.name
        try:
            width = int(name.split("width")[1].split("_")[0])
        except Exception:
            width = getattr(cfg.model, "width", 64)
        try:
            seed = int(name.split("seed")[1].split("_")[0])
        except Exception:
            seed = getattr(cfg.experiment, "seed", 0)
        try:
            noise = float(name.split("noise")[1])
        except Exception:
            noise = getattr(cfg.dataset, "label_noise", 0.0)

        cfg.model.width = width
        cfg.dataset.label_noise = noise

        print(f"\n[=== Running linearization for {sub.name} ===]")
        print(f"→ width={width}, seed={seed}, noise={noise}")

        # --- NEW: compute adaptive τs for this model ---
        adaptive_taus = compute_global_tau_from_metrics(metrics_csv)
        if not adaptive_taus:  # fallback if file missing or invalid
            adaptive_taus = cfg.experiment.thresholds

        # --- Run analysis with computed τs ---
        linearize_and_evaluate(cfg, ckpt_dir, adaptive_taus)


if __name__ == "__main__":
    main()
