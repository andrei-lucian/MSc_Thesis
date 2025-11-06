import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import csv
from torch.optim.lr_scheduler import LambdaLR
from src.metrics.preactivation import PreactivationLogger
import os


class Trainer:
    def __init__(self, model, train_loader: DataLoader, test_loader: DataLoader, cfg):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # ------------------------------
        # Optimizer setup
        # ------------------------------
        if cfg.training.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=cfg.training.lr,
                momentum=cfg.training.momentum,
                weight_decay=cfg.training.weight_decay,
            )
            self.scheduler = None

        elif cfg.training.optimizer == "adam":
            # Use Transformer paper settings if inverse_sqrt schedule
            if cfg.training.lr_schedule == "inverse_sqrt":
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=1.0,  # Base LR = 1, schedule handles scaling
                    betas=(0.9, 0.98),
                    eps=1e-9
                )

                def lr_lambda(step):
                    step = max(1, step)
                    d_model = getattr(cfg.model, "width", 512)
                    warmup = getattr(cfg.training, "warmup_steps", 4000)
                    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))

                self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

            elif cfg.training.lr_schedule == "cosine":
                self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.training.lr)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=cfg.training.epochs
                )
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.training.lr)
                self.scheduler = None

        else:
            raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")

        # ------------------------------
        # Loss function
        # ------------------------------
        self.criterion = nn.CrossEntropyLoss()

        # ------------------------------
        # Preactivation logger
        # ------------------------------
        self.preact_logger = PreactivationLogger(self.model)

        # ------------------------------
        # Output + logging setup
        # ------------------------------
        self.out_dir = Path(os.getcwd())
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.num_params = sum(p.numel() for p in self.model.parameters())

        self.log_file = self.out_dir / "metrics.csv"
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            # Metadata
            writer.writerow(["arch", "width", "seed", "noise", "params"])
            writer.writerow([
                cfg.model.arch,
                getattr(cfg.model, "width", None),
                cfg.experiment.seed,
                getattr(cfg.dataset, "label_noise", 0.0),
                self.num_params,
            ])

            # Dynamic header
            index_name = "step" if getattr(cfg.training, "use_steps", False) else "epoch"
            header = [index_name, "train_loss", "train_acc", "test_loss", "test_acc"]
            header += [f"p_layer{i}" for i in range(len(self.preact_logger.layers))]
            writer.writerow(header)

    # ------------------------------
    # Training for one epoch
    # ------------------------------
    def train_one_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for step, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item() * x.size(0)
            _, preds = out.max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)

        return total_loss / total, correct / total

    # ------------------------------
    # Evaluation
    # ------------------------------
    def evaluate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)

                total_loss += loss.item() * x.size(0)
                _, preds = out.max(1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)

        return total_loss / total, correct / total

    # ------------------------------
    # Metric logging
    # ------------------------------
    def log_metrics(self, index, train_loss, train_acc, test_loss, test_acc, index_name="epoch"):
        """
        Logs performance and preactivation metrics.
        index_name: "epoch" or "step"
        """
        p_l_values = self.preact_logger.compute_metric(self.test_loader)
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            row = [index, train_loss, train_acc, test_loss, test_acc]
            row += p_l_values
            writer.writerow(row)
            
    # ------------------------------
    # Checkpoint saving
    # ------------------------------
    def save_checkpoint(self, step_or_epoch):
        """
        Save model, optimizer, and scheduler state for later analysis or resume.
        """
        ckpt_dir = self.out_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        # choose file name based on whether training is step-based or epoch-based
        mode = "step" if getattr(self.cfg.training, "use_steps", False) else "epoch"
        ckpt_path = ckpt_dir / f"{mode}_{step_or_epoch:06d}.pt"

        torch.save({
            "index": step_or_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "cfg": self.cfg,
        }, ckpt_path)

        print(f"[Checkpoint] Saved model at {ckpt_path}")

    # ------------------------------
    # Training loop (supports epochs or steps)
    # ------------------------------
    def run(self):
        """
        Runs training. Supports both epoch-based and step-based control.

        cfg.training.use_steps: bool
        cfg.training.max_steps: int (if use_steps=True)
        cfg.training.log_interval: int (step logging frequency)
        """
        use_steps = getattr(self.cfg.training, "use_steps", False)
        max_steps = getattr(self.cfg.training, "max_steps", None)
        log_interval = getattr(self.cfg.training, "log_interval", 1000)
        save_interval = getattr(self.cfg.training, "save_interval", 1000)
        save_checkpoints = getattr(self.cfg.training, "save_checkpoints", True)
        ckpt_dir = self.out_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        if not use_steps:
            # ========================================================
            # === Standard EPOCH-BASED training (default)
            # ========================================================
            for epoch in range(1, self.cfg.training.epochs + 1):
                train_loss, train_acc = self.train_one_epoch()
                test_loss, test_acc = self.evaluate()
                self.log_metrics(epoch, train_loss, train_acc, test_loss, test_acc, "epoch")

                # --- Save checkpoint every N epochs ---
                if save_checkpoints and (step % save_interval == 0 or step == 1):
                    ckpt_path = ckpt_dir / f"epoch_{epoch:06d}.pt"
                    torch.save(self.model.state_dict(), ckpt_path)
                    print(f"[Checkpoint] Saved model → {ckpt_path}")

                if epoch % 10 == 0 or epoch == 1:
                    print(
                        f"[Epoch {epoch}/{self.cfg.training.epochs}] "
                        f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
                        f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}"
                    )

        else:
            # ========================================================
            # === STEP-BASED training (Transformer-style)
            # ========================================================
            step = 0
            epoch = 0
            print(f"[Trainer] Running step-based training for {max_steps} steps")

            while step < max_steps:
                epoch += 1
                self.model.train()

                for x, y in self.train_loader:
                    if step >= max_steps:
                        break

                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    out = self.model(x)
                    loss = self.criterion(out, y)
                    loss.backward()
                    self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    step += 1

                    # Periodic logging and evaluation
                    if step % log_interval == 0 or step == 1:
                        train_loss = loss.item()
                        train_acc = (out.argmax(1) == y).float().mean().item()
                        test_loss, test_acc = self.evaluate()
                        self.log_metrics(step, train_loss, train_acc, test_loss, test_acc, "step")

                        print(
                            f"[Step {step}/{max_steps}] "
                            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
                            f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}"
                        )

                    # --- Save checkpoint every N steps ---
                    if save_checkpoints and (step % save_interval == 0 or step == 1):
                        ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                        torch.save(self.model.state_dict(), ckpt_path)
                        print(f"[Checkpoint] Saved model → {ckpt_path}")

