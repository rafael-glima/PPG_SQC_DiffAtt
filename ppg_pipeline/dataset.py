"""
dataset.py
----------
PyTorch Dataset and synthetic data generation for PPG quality assessment.

Ground-truth quality score
~~~~~~~~~~~~~~~~~~~~~~~~~~
For synthetic data the true quality score is computed analytically from
the known generation parameters — no manual annotation needed.

  score = Σ wᵢ · sᵢ,   sᵢ ∈ [0, 1] component scores

  Component scores
  ----------------
  s_noise    = max(0, 1 - noise_std / 0.6)
  s_motion   = max(0, 1 - motion_scale / 1.2)
  s_wander   = max(0, 1 - wander_scale / 0.8)
  s_dropout  = max(0, 1 - dropout_prob / 0.08)
  s_perf     = perfusion_index / 20

  Weights: noise 0.25, motion 0.20, wander 0.15, dropout 0.20, perf 0.20

Verdict mapping
~~~~~~~~~~~~~~~
  score >= 0.75 → ACCEPT  (0)
  score >= 0.50 → CAUTION (1)
  score <  0.50 → REJECT  (2)
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .signal_generator import PPGSignalGenerator, SignalParams
from .preprocessor import PPGPreprocessor
from .feature_extractor import PPGFeatureExtractor, FeatureVector


# ---------------------------------------------------------------------------
# Quality label computation
# ---------------------------------------------------------------------------

VERDICT_ACCEPT = 0
VERDICT_CAUTION = 1
VERDICT_REJECT = 2

_SCORE_WEIGHTS = {
    "noise":   0.25,
    "motion":  0.20,
    "wander":  0.15,
    "dropout": 0.20,
    "perf":    0.20,
}

def compute_gt_quality(params: SignalParams) -> tuple[float, int]:
    """
    Compute ground-truth quality score and verdict from known signal parameters.

    Returns
    -------
    score : float in [0, 1]
    verdict : int  (ACCEPT=0, CAUTION=1, REJECT=2)
    """
    s_noise  = max(0.0, 1.0 - params.noise_std / 0.6)
    s_motion = max(0.0, 1.0 - params.motion_scale / 1.2)
    s_wander = max(0.0, 1.0 - params.wander_scale / 0.8)
    s_drop   = max(0.0, 1.0 - params.dropout_prob / 0.08)
    s_perf   = params.perfusion_index / 20.0

    score = (
        _SCORE_WEIGHTS["noise"]   * s_noise
        + _SCORE_WEIGHTS["motion"]  * s_motion
        + _SCORE_WEIGHTS["wander"]  * s_wander
        + _SCORE_WEIGHTS["dropout"] * s_drop
        + _SCORE_WEIGHTS["perf"]    * s_perf
    )
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.75:
        verdict = VERDICT_ACCEPT
    elif score >= 0.50:
        verdict = VERDICT_CAUTION
    else:
        verdict = VERDICT_REJECT

    return score, verdict


# ---------------------------------------------------------------------------
# Synthetic dataset builder
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_samples: int = 2000,
    fs: float = 100.0,
    duration: float = 5.0,
    balanced: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> "PPGDataset":
    """
    Generate a synthetic labelled PPG dataset.

    Parameters
    ----------
    n_samples : int
        Total number of segments.
    fs : float
        Sampling frequency (Hz).
    balanced : bool
        If True, enforce equal class counts across ACCEPT/CAUTION/REJECT.
    seed : int
        Master RNG seed.
    verbose : bool
        Print progress.

    Returns
    -------
    PPGDataset
    """
    rng = np.random.default_rng(seed)
    pre = PPGPreprocessor(fs=fs)
    feat = PPGFeatureExtractor(fs=fs)

    waveforms, feature_vecs, scores, verdicts = [], [], [], []
    n_per_class = n_samples // 3 if balanced else None
    class_counts = [0, 0, 0]

    attempts = 0
    max_attempts = n_samples * 10

    while len(waveforms) < n_samples and attempts < max_attempts:
        attempts += 1
        params = SignalParams.random(rng)
        params.fs = fs
        params.duration = duration
        params.seed = int(rng.integers(0, 2**31))

        score, verdict = compute_gt_quality(params)

        # Balance classes
        if balanced and n_per_class is not None:
            if class_counts[verdict] >= n_per_class:
                continue

        gen = PPGSignalGenerator(params)
        try:
            raw, clean, t = gen.generate()
            filtered, peaks = pre.process(raw)
            fv = feat.extract(filtered, peaks)
        except Exception:
            continue

        waveforms.append(filtered)
        feature_vecs.append(fv.to_array())
        scores.append(score)
        verdicts.append(verdict)
        class_counts[verdict] += 1

        if verbose and len(waveforms) % 500 == 0:
            print(f"  Generated {len(waveforms)}/{n_samples} segments "
                  f"[A:{class_counts[0]} C:{class_counts[1]} R:{class_counts[2]}]")

    if verbose:
        print(f"Dataset ready: {len(waveforms)} segments | "
              f"ACCEPT={class_counts[0]} CAUTION={class_counts[1]} REJECT={class_counts[2]}")

    return PPGDataset(
        waveforms=np.stack(waveforms),
        feature_vecs=np.stack(feature_vecs),
        scores=np.array(scores, dtype=np.float32),
        verdicts=np.array(verdicts, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class PPGDataset(Dataset):
    """
    PyTorch Dataset wrapping PPG segments.

    Items
    -----
    waveform  : torch.Tensor, shape (1, L)   – preprocessed signal
    features  : torch.Tensor, shape (24,)    – extracted feature vector
    score     : torch.Tensor, scalar float   – quality score [0, 1]
    verdict   : torch.Tensor, scalar int     – 0=ACCEPT, 1=CAUTION, 2=REJECT

    Parameters
    ----------
    waveforms : np.ndarray, shape (N, L)
    feature_vecs : np.ndarray, shape (N, 24)
    scores : np.ndarray, shape (N,)
    verdicts : np.ndarray, shape (N,)
    transform : optional callable applied to waveform tensor
    """

    VERDICT_NAMES = ["ACCEPT", "CAUTION", "REJECT"]

    def __init__(
        self,
        waveforms: np.ndarray,
        feature_vecs: np.ndarray,
        scores: np.ndarray,
        verdicts: np.ndarray,
        transform: Optional[Callable] = None,
    ) -> None:
        assert len(waveforms) == len(feature_vecs) == len(scores) == len(verdicts)
        self.waveforms = torch.from_numpy(waveforms).unsqueeze(1)   # (N, 1, L)
        self.feature_vecs = torch.from_numpy(feature_vecs.astype(np.float32))
        self.scores = torch.from_numpy(scores.astype(np.float32))
        self.verdicts = torch.from_numpy(verdicts.astype(np.int64))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        waveform = self.waveforms[idx]
        if self.transform is not None:
            waveform = self.transform(waveform)
        return {
            "waveform": waveform,
            "features": self.feature_vecs[idx],
            "score": self.scores[idx],
            "verdict": self.verdicts[idx],
        }

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for imbalanced training."""
        counts = torch.bincount(self.verdicts, minlength=3).float()
        weights = counts.sum() / (3.0 * counts.clamp(min=1))
        return weights

    def split(self, val_fraction: float = 0.15, seed: int = 0) -> tuple["PPGDataset", "PPGDataset"]:
        """Random train/val split."""
        rng = np.random.default_rng(seed)
        n = len(self)
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_fraction))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        def subset(indices):
            return PPGDataset(
                waveforms=self.waveforms[indices, 0].numpy(),
                feature_vecs=self.feature_vecs[indices].numpy(),
                scores=self.scores[indices].numpy(),
                verdicts=self.verdicts[indices].numpy(),
                transform=self.transform,
            )

        return subset(train_idx), subset(val_idx)


# ---------------------------------------------------------------------------
# Data augmentation transforms
# ---------------------------------------------------------------------------

class PPGAugment:
    """
    Waveform augmentation for training.

    Applies randomly with given probability:
    - Amplitude scaling          (random ×[0.7, 1.3])
    - Additive Gaussian noise    (σ ~ Uniform[0, 0.05])
    - Time shift                 (random roll ±5%)
    - Flip sign                  (with prob 0.2)
    """

    def __init__(self, p: float = 0.8) -> None:
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x : shape (1, L)"""
        if torch.rand(1).item() > self.p:
            return x
        x = x.clone()
        # Amplitude scaling
        x = x * torch.empty(1).uniform_(0.7, 1.3)
        # Additive noise
        sigma = torch.empty(1).uniform_(0.0, 0.05).item()
        x = x + torch.randn_like(x) * sigma
        # Time shift (circular roll)
        shift = int(torch.randint(-25, 26, (1,)).item())
        x = torch.roll(x, shift, dims=-1)
        # Random sign flip
        if torch.rand(1).item() < 0.2:
            x = -x
        return x
