"""
dalia_trainer.py
----------------
Training loop for PPG quality classification on windowed PPG-DaLiA data.

Task
~~~~
  Binary classification per 3-second window:
    label = 0  →  clean signal
    label = 1  →  artefact

  Loss   : BCE with logits  (pos_weight handles class imbalance)
  Metrics: accuracy, F1, AUC-ROC, quality-score MAE
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dalia_dataset import DaLiADataset


@dataclass
class DaLiATrainerConfig:
    lr:             float = 3e-4
    weight_decay:   float = 1e-4
    max_epochs:     int   = 60
    batch_size:     int   = 64
    grad_clip:      float = 1.0
    lr_min:         float = 1e-6
    warmup_epochs:  int   = 3

    pos_weight:     float = 2.0   # upweight artifact class in BCE
    patience:       int   = 10
    min_delta:      float = 1e-4

    num_workers:    int   = 0
    checkpoint_dir: str   = "checkpoints/dalia"
    device:         str   = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():   return torch.device("cuda")
            if torch.backends.mps.is_available(): return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


# ---------------------------------------------------------------------------
# Binary classifier CNN
# ---------------------------------------------------------------------------

class PPGBinaryNet(nn.Module):
    """
    Lightweight 1-D CNN binary classifier.

    Input : (B, 1, win_size)
    Output: (B,)  raw logits  →  sigmoid for probability of artefact
    """

    def __init__(
        self,
        in_channels:  int   = 1,
        base_channels: int  = 32,
        dropout:      float = 0.2,
    ) -> None:
        super().__init__()
        C = base_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, C,    15, padding=7,  bias=False),
            nn.BatchNorm1d(C),   nn.GELU(),

            nn.Conv1d(C,    C*2,  7,  stride=2, padding=3,  bias=False),
            nn.BatchNorm1d(C*2), nn.GELU(),

            nn.Conv1d(C*2,  C*4,  5,  stride=2, padding=2,  bias=False),
            nn.BatchNorm1d(C*4), nn.GELU(),

            nn.Conv1d(C*4,  C*4,  5,  stride=2, padding=2,  bias=False),
            nn.BatchNorm1d(C*4), nn.GELU(),

            nn.Conv1d(C*4,  C*4,  3,  stride=2, padding=1,  bias=False),
            nn.BatchNorm1d(C*4), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(C*4 * 2, 128),   # mean+std pooling
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, features=None) -> torch.Tensor:
        h = self.encoder(x)
        pooled = torch.cat([h.mean(-1), h.std(-1)], dim=-1)
        return self.head(pooled).squeeze(-1)   # (B,)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _binary_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    probs = 1 / (1 + np.exp(-logits))          # sigmoid
    preds = (probs > 0.5).astype(int)
    lab   = labels.astype(int)

    acc = float((preds == lab).mean())
    tp  = int(((preds == 1) & (lab == 1)).sum())
    fp  = int(((preds == 1) & (lab == 0)).sum())
    fn  = int(((preds == 0) & (lab == 1)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    # AUC-ROC via trapezoidal rule
    try:
        order  = np.argsort(-probs)
        tpr, fpr = [0.], [0.]
        pos, neg = lab.sum(), (1 - lab).sum()
        tp_, fp_ = 0, 0
        for i in order:
            if lab[i]: tp_ += 1
            else:      fp_ += 1
            tpr.append(tp_ / max(pos, 1))
            fpr.append(fp_ / max(neg, 1))
        auc = float(np.trapz(tpr, fpr))
        auc = abs(auc)
    except Exception:
        auc = 0.5

    return {"acc": acc, "f1": float(f1), "auc": auc,
            "precision": float(prec), "recall": float(rec)}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DaLiATrainer:
    """
    Trains a binary PPG quality classifier on windowed DaLiA data.

    Parameters
    ----------
    model    : PPGBinaryNet  (or any module returning (B,) logits from (B,1,L))
    config   : DaLiATrainerConfig
    train_ds : DaLiADataset
    val_ds   : DaLiADataset, optional  (if None, 15% of train_ds is used)

    Examples
    --------
    >>> model   = PPGBinaryNet()
    >>> trainer = DaLiATrainer(model, DaLiATrainerConfig(), train_ds, val_ds)
    >>> history = trainer.fit()
    """

    def __init__(
        self,
        model:    PPGBinaryNet,
        config:   DaLiATrainerConfig,
        train_ds: DaLiADataset,
        val_ds:   Optional[DaLiADataset] = None,
    ) -> None:
        self.cfg    = config
        self.device = config.resolve_device()
        self.model  = model.to(self.device)

        if val_ds is None:
            train_ds, val_ds = train_ds.split(val_fraction=0.15)
        self.train_ds, self.val_ds = train_ds, val_ds

        # Weighted sampler to balance clean / artifact windows
        sample_w = torch.where(
            train_ds.labels_t == 1,
            torch.tensor(config.pos_weight),
            torch.tensor(1.0),
        )
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

        self.train_loader = DataLoader(
            train_ds, batch_size=config.batch_size,
            sampler=sampler, num_workers=config.num_workers,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=config.batch_size * 2,
            shuffle=False, num_workers=config.num_workers,
        )

        pw = torch.tensor([config.pos_weight], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.optimiser = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        total_steps  = config.max_epochs * len(self.train_loader)
        warmup_steps = config.warmup_epochs * len(self.train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / (warmup_steps + 1)
            p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * p))
            return config.lr_min / config.lr + (1 - config.lr_min / config.lr) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda)

        self.history: list[dict] = []
        self._best_val_loss = math.inf
        self._patience_ctr  = 0

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> list[dict]:
        cfg = self.cfg
        print(
            f"DaLiA training on {self.device} | "
            f"train={len(self.train_ds)}  val={len(self.val_ds)} | "
            f"params={self.model.n_parameters:,}"
        )
        for epoch in range(1, cfg.max_epochs + 1):
            t0 = time.time()
            tr = self._run(training=True)
            va = self._run(training=False)
            elapsed = time.time() - t0

            row = {"epoch": epoch, "train": tr, "val": va,
                   "lr": self.optimiser.param_groups[0]["lr"]}
            self.history.append(row)

            print(
                f"Epoch {epoch:3d}/{cfg.max_epochs} [{elapsed:.1f}s]  "
                f"train loss={tr['loss']:.4f} acc={tr['acc']:.3f} f1={tr['f1']:.3f}  |  "
                f"val   loss={va['loss']:.4f} acc={va['acc']:.3f} f1={va['f1']:.3f} "
                f"auc={va['auc']:.3f}  lr={row['lr']:.2e}"
            )

            improved = va["loss"] < self._best_val_loss - cfg.min_delta
            if improved:
                self._best_val_loss = va["loss"]
                self._patience_ctr  = 0
                torch.save(self.model.state_dict(),
                           Path(cfg.checkpoint_dir) / "best_model.pt")
            else:
                self._patience_ctr += 1
                if self._patience_ctr >= cfg.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Training complete. Best val loss: {self._best_val_loss:.4f}")
        return self.history

    def load_best(self) -> None:
        path = Path(self.cfg.checkpoint_dir) / "best_model.pt"
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded best model from {path}")

    def evaluate(self, dataset: Optional[DaLiADataset] = None) -> dict:
        loader = (DataLoader(dataset, batch_size=self.cfg.batch_size * 2, shuffle=False)
                  if dataset is not None else self.val_loader)
        return self._run_loader(loader, training=False)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run(self, training: bool) -> dict:
        return self._run_loader(
            self.train_loader if training else self.val_loader, training
        )

    def _run_loader(self, loader: DataLoader, training: bool) -> dict:
        self.model.train(training)
        total_loss = 0.0
        all_logits, all_labels = [], []
        n = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                wav   = batch["waveform"].to(self.device)
                label = batch["label"].to(self.device)

                feat   = batch["features"].to(self.device)
                logits = self.model(wav, feat)
                loss   = self.criterion(logits, label)

                if training:
                    self.optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimiser.step()
                    self.scheduler.step()

                total_loss += loss.item()
                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(label.cpu().numpy())
                n += 1

        logits_all = np.concatenate(all_logits)
        labels_all = np.concatenate(all_labels)
        metrics = _binary_metrics(logits_all, labels_all)
        metrics["loss"] = total_loss / max(1, n)
        return metrics
