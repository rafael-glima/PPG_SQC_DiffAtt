"""
trainer.py
----------
Training loop for PPG quality assessment models.

Features
~~~~~~~~
- Works with PPGWaveformNet, PPGFeatureNet, or PPGEnsembleNet
- Multi-task loss (score MSE + verdict cross-entropy)
- AdamW with cosine annealing LR schedule
- Gradient clipping
- Early stopping on validation loss
- Per-epoch metrics: MAE, accuracy, F1 per class
- Checkpoint saving (best model by val loss)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import PPGDataset, PPGAugment
from .models import (
    ModelOutput,
    PPGEnsembleNet,
    PPGFeatureNet,
    PPGQualityLoss,
    PPGWaveformNet,
)

AnyModel = Union[PPGWaveformNet, PPGFeatureNet, PPGEnsembleNet]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Hyper-parameters and training settings."""

    # Optimisation
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 60
    batch_size: int = 64
    grad_clip: float = 1.0

    # Loss weights
    score_weight: float = 1.0
    verdict_weight: float = 1.0
    label_smoothing: float = 0.05

    # Schedule
    lr_min: float = 1e-6
    warmup_epochs: int = 3

    # Regularisation
    use_augment: bool = True
    augment_prob: float = 0.8
    use_class_weights: bool = True

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Misc
    num_workers: int = 0
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    loss: float = 0.0
    score_loss: float = 0.0
    verdict_loss: float = 0.0
    score_mae: float = 0.0
    verdict_acc: float = 0.0
    f1_accept: float = 0.0
    f1_caution: float = 0.0
    f1_reject: float = 0.0

    @property
    def macro_f1(self) -> float:
        return (self.f1_accept + self.f1_caution + self.f1_reject) / 3.0

    def __str__(self) -> str:
        return (
            f"loss={self.loss:.4f} "
            f"score_mae={self.score_mae:.4f} "
            f"verdict_acc={self.verdict_acc:.3f} "
            f"macro_f1={self.macro_f1:.3f}"
        )


def _compute_metrics(
    all_scores: list[float],
    all_pred_scores: list[float],
    all_verdicts: list[int],
    all_pred_verdicts: list[int],
) -> dict[str, float]:
    """Compute MAE, accuracy, and per-class F1."""
    y_score = np.array(all_scores)
    y_pred_score = np.array(all_pred_scores)
    y_ver = np.array(all_verdicts)
    y_pred_ver = np.array(all_pred_verdicts)

    mae = float(np.abs(y_score - y_pred_score).mean())
    acc = float((y_ver == y_pred_ver).mean())

    f1s = []
    for cls in range(3):
        tp = ((y_pred_ver == cls) & (y_ver == cls)).sum()
        fp = ((y_pred_ver == cls) & (y_ver != cls)).sum()
        fn = ((y_pred_ver != cls) & (y_ver == cls)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(float(f1))

    return {"mae": mae, "acc": acc, "f1": f1s}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PPGTrainer:
    """
    Trains and evaluates PPG quality models.

    Parameters
    ----------
    model : AnyModel
        One of PPGWaveformNet, PPGFeatureNet, PPGEnsembleNet.
    config : TrainerConfig
    dataset : PPGDataset
        Full dataset; will be split internally.
    val_dataset : PPGDataset, optional
        Explicit validation set (overrides internal split).

    Examples
    --------
    >>> model = PPGFeatureNet()
    >>> trainer = PPGTrainer(model, TrainerConfig(), dataset)
    >>> history = trainer.fit()
    """

    def __init__(
        self,
        model: AnyModel,
        config: TrainerConfig,
        dataset: PPGDataset,
        val_dataset: Optional[PPGDataset] = None,
    ) -> None:
        self.config = config
        self.device = config.resolve_device()
        self.model = model.to(self.device)

        # Datasets
        if val_dataset is not None:
            self.train_ds, self.val_ds = dataset, val_dataset
        else:
            self.train_ds, self.val_ds = dataset.split(val_fraction=0.15)

        self.is_ensemble = isinstance(model, PPGEnsembleNet)

        # Augmentation
        aug = PPGAugment(p=config.augment_prob) if config.use_augment else None
        if aug is not None:
            self.train_ds.transform = aug

        # Sampler
        if config.use_class_weights:
            weights = self.train_ds.class_weights()
            sample_weights = weights[self.train_ds.verdicts]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            self.train_loader = DataLoader(
                self.train_ds, batch_size=config.batch_size,
                sampler=sampler, num_workers=config.num_workers,
            )
        else:
            self.train_loader = DataLoader(
                self.train_ds, batch_size=config.batch_size,
                shuffle=True, num_workers=config.num_workers,
            )

        self.val_loader = DataLoader(
            self.val_ds, batch_size=config.batch_size * 2,
            shuffle=False, num_workers=config.num_workers,
        )

        # Loss
        self.criterion = PPGQualityLoss(
            score_weight=config.score_weight,
            verdict_weight=config.verdict_weight,
            label_smoothing=config.label_smoothing,
        )

        # Optimiser
        self.optimiser = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # LR schedule: linear warmup then cosine decay
        total_steps = config.max_epochs * len(self.train_loader)
        warmup_steps = config.warmup_epochs * len(self.train_loader)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / (warmup_steps + 1)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return config.lr_min / config.lr + (1.0 - config.lr_min / config.lr) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda)

        # State
        self.history: list[dict] = []
        self._best_val_loss = math.inf
        self._patience_counter = 0

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> list[dict]:
        """
        Run training loop.

        Returns
        -------
        history : list of dicts with train/val metrics per epoch
        """
        cfg = self.config
        print(f"Training on {self.device} | "
              f"train={len(self.train_ds)} val={len(self.val_ds)} | "
              f"model params={sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        for epoch in range(1, cfg.max_epochs + 1):
            t0 = time.time()
            train_metrics = self._run_epoch(epoch, training=True)
            val_metrics = self._run_epoch(epoch, training=False)
            elapsed = time.time() - t0

            row = {
                "epoch": epoch,
                "train": vars(train_metrics),
                "val": vars(val_metrics),
                "lr": self.optimiser.param_groups[0]["lr"],
            }
            self.history.append(row)

            print(
                f"Epoch {epoch:3d}/{cfg.max_epochs} "
                f"[{elapsed:.1f}s] "
                f"train: {train_metrics} | "
                f"val: {val_metrics} | "
                f"lr={row['lr']:.2e}"
            )

            # Checkpoint
            improved = val_metrics.loss < self._best_val_loss - cfg.min_delta
            if improved:
                self._best_val_loss = val_metrics.loss
                self._patience_counter = 0
                self._save_checkpoint("best_model.pt")
            else:
                self._patience_counter += 1
                if self._patience_counter >= cfg.patience:
                    print(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                    break

        print(f"Training complete. Best val loss: {self._best_val_loss:.4f}")
        return self.history

    def evaluate(self, dataset: Optional[PPGDataset] = None) -> EpochMetrics:
        """Evaluate on val set (or provided dataset)."""
        if dataset is not None:
            loader = DataLoader(dataset, batch_size=self.config.batch_size * 2, shuffle=False)
        else:
            loader = self.val_loader
        return self._run_epoch_with_loader(loader, training=False)

    def load_best(self) -> None:
        """Restore best checkpoint weights."""
        path = Path(self.config.checkpoint_dir) / "best_model.pt"
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        print(f"Loaded best model from {path}")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int, training: bool) -> EpochMetrics:
        loader = self.train_loader if training else self.val_loader
        return self._run_epoch_with_loader(loader, training)

    def _run_epoch_with_loader(self, loader: DataLoader, training: bool) -> EpochMetrics:
        self.model.train(training)
        cfg = self.config

        total_loss = score_loss_sum = verdict_loss_sum = 0.0
        all_scores, all_pred_scores, all_verdicts, all_pred_verdicts = [], [], [], []
        n_batches = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                waveform = batch["waveform"].to(self.device)
                features = batch["features"].to(self.device)
                score_gt = batch["score"].to(self.device)
                verdict_gt = batch["verdict"].to(self.device)

                if self.is_ensemble:
                    output: ModelOutput = self.model(waveform, features)
                elif isinstance(self.model, PPGWaveformNet):
                    output = self.model(waveform)
                else:
                    output = self.model(features)

                losses = self.criterion(output, score_gt, verdict_gt)

                if training:
                    self.optimiser.zero_grad()
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    self.optimiser.step()
                    self.scheduler.step()

                total_loss += losses["total"].item()
                score_loss_sum += losses["score_loss"].item()
                verdict_loss_sum += losses["verdict_loss"].item()

                pred_verdicts = output.verdict_logits.argmax(dim=-1)
                all_scores.extend(score_gt.cpu().tolist())
                all_pred_scores.extend(output.quality_score.detach().cpu().tolist())
                all_verdicts.extend(verdict_gt.cpu().tolist())
                all_pred_verdicts.extend(pred_verdicts.cpu().tolist())
                n_batches += 1

        n = max(1, n_batches)
        m = _compute_metrics(all_scores, all_pred_scores, all_verdicts, all_pred_verdicts)
        return EpochMetrics(
            loss=total_loss / n,
            score_loss=score_loss_sum / n,
            verdict_loss=verdict_loss_sum / n,
            score_mae=m["mae"],
            verdict_acc=m["acc"],
            f1_accept=m["f1"][0],
            f1_caution=m["f1"][1],
            f1_reject=m["f1"][2],
        )

    def _save_checkpoint(self, filename: str) -> None:
        path = Path(self.config.checkpoint_dir) / filename
        torch.save(self.model.state_dict(), path)
