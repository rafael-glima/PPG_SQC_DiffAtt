"""
signal_generator.py
-------------------
Synthetic PPG waveform generation with physiologically realistic morphology
and controllable artifact injection (noise, motion, baseline wander, dropouts).

PPG morphology model
~~~~~~~~~~~~~~~~~~~~
A real PPG pulse consists of:
  - Systolic peak     : rapid pressure rise after ventricular ejection
  - Dicrotic notch    : brief pressure drop as aortic valve closes
  - Diastolic peak    : reflected pressure wave from the periphery

We approximate this with a sum of three Gaussians per cardiac cycle.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignalParams:
    """Full parameter set for one synthetic PPG segment."""

    # Physiological
    heart_rate: float = 72.0        # beats per minute  [45, 120]
    perfusion_index: float = 12.0   # AC/DC ratio * 100 [1, 20]

    # Artifacts
    noise_std: float = 0.15         # Gaussian noise σ  [0, 1]
    motion_scale: float = 0.10      # Motion artifact amplitude [0, 1.2]
    motion_start: float = 1.5       # Motion start time (seconds)
    motion_duration: float = 0.8    # Motion duration (seconds)
    wander_scale: float = 0.08      # Baseline wander amplitude [0, 0.8]
    wander_freq: float = 0.08       # Baseline wander frequency (Hz)
    dropout_prob: float = 0.02      # Per-sample dropout probability [0, 0.1]

    # Acquisition
    fs: float = 100.0               # Sampling frequency (Hz)
    duration: float = 5.0           # Segment duration (seconds)
    seed: Optional[int] = None      # RNG seed for reproducibility

    @property
    def n_samples(self) -> int:
        return int(self.fs * self.duration)

    @property
    def freq(self) -> float:
        """Fundamental cardiac frequency (Hz)."""
        return self.heart_rate / 60.0

    @classmethod
    def clean(cls, heart_rate: float = 72.0, **kwargs) -> "SignalParams":
        """Factory: clean signal with no artifacts."""
        return cls(
            heart_rate=heart_rate,
            noise_std=0.02,
            motion_scale=0.0,
            wander_scale=0.0,
            dropout_prob=0.0,
            **kwargs,
        )

    @classmethod
    def random(cls, rng: Optional[np.random.Generator] = None) -> "SignalParams":
        """Factory: fully randomised parameters for dataset generation."""
        if rng is None:
            rng = np.random.default_rng()
        return cls(
            heart_rate=float(rng.uniform(45, 120)),
            perfusion_index=float(rng.uniform(1, 20)),
            noise_std=float(rng.uniform(0.0, 0.6)),
            motion_scale=float(rng.uniform(0.0, 1.2)),
            motion_start=float(rng.uniform(0.3, 3.5)),
            motion_duration=float(rng.uniform(0.3, 1.5)),
            wander_scale=float(rng.uniform(0.0, 0.8)),
            wander_freq=float(rng.uniform(0.05, 0.2)),
            dropout_prob=float(rng.uniform(0.0, 0.08)),
            seed=int(rng.integers(0, 2**31)),
        )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class PPGSignalGenerator:
    """
    Generates synthetic PPG waveforms.

    Parameters
    ----------
    params : SignalParams
        Full parameter set.

    Examples
    --------
    >>> params = SignalParams(heart_rate=75, noise_std=0.1)
    >>> gen = PPGSignalGenerator(params)
    >>> raw, clean, t = gen.generate()
    >>> raw.shape
    (500,)
    """

    # Gaussian component parameters (mu_fraction, sigma, weight) per heartbeat cycle
    # mu_fraction = position within [0, 1] normalised cycle
    _PULSE_COMPONENTS = [
        (0.20, 0.045, 1.00),   # systolic peak
        (0.45, 0.030, 0.35),   # dicrotic notch shoulder
        (0.62, 0.065, 0.18),   # diastolic peak
    ]

    def __init__(self, params: SignalParams) -> None:
        self.params = params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a PPG segment.

        Returns
        -------
        raw : np.ndarray, shape (N,)
            Noisy PPG signal as would be acquired from a sensor.
        clean : np.ndarray, shape (N,)
            Ideal clean PPG without motion, wander, or dropouts (light noise).
        t : np.ndarray, shape (N,)
            Time axis in seconds.
        """
        p = self.params
        rng = np.random.default_rng(p.seed)

        t = np.linspace(0.0, p.duration, p.n_samples, endpoint=False)
        amplitude = p.perfusion_index / 20.0 * 1.5 + 0.3

        clean_base = self._cardiac_signal(t, amplitude)
        clean = clean_base + rng.normal(0, p.noise_std * 0.15, p.n_samples)

        wander = self._baseline_wander(t, rng)
        noise = rng.normal(0, p.noise_std, p.n_samples)
        motion = self._motion_artifact(t, rng)
        dropouts = self._dropouts(p.n_samples, clean_base, rng)

        raw = clean_base + wander + noise + motion + dropouts
        return raw.astype(np.float32), clean.astype(np.float32), t.astype(np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cardiac_signal(self, t: np.ndarray, amplitude: float) -> np.ndarray:
        """Sum-of-Gaussians PPG morphology model."""
        p = self.params
        phase = (t * p.freq) % 1.0          # fractional position within cycle [0, 1)
        signal = np.zeros_like(t)
        for mu, sigma, weight in self._PULSE_COMPONENTS:
            delta = phase - mu
            # Wrap delta so it is always in [-0.5, 0.5] (circular distance)
            delta = delta - np.round(delta)
            signal += weight * np.exp(-0.5 * (delta / sigma) ** 2)
        return signal * amplitude

    def _baseline_wander(self, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Low-frequency respiratory baseline drift."""
        p = self.params
        phase2 = rng.uniform(0, 2 * math.pi)
        return (
            p.wander_scale * np.sin(2 * math.pi * p.wander_freq * t)
            + p.wander_scale * 0.4 * np.sin(2 * math.pi * (p.wander_freq * 1.7) * t + phase2)
        )

    def _motion_artifact(self, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Bandlimited motion artifact burst."""
        p = self.params
        if p.motion_scale == 0.0:
            return np.zeros_like(t)
        artifact = np.zeros_like(t)
        start, end = p.motion_start, p.motion_start + p.motion_duration
        mask = (t >= start) & (t <= end)
        if mask.any():
            t_local = t[mask] - start
            env = np.sin(math.pi * t_local / p.motion_duration)  # Hann envelope
            freq_m = rng.uniform(1.5, 5.0)
            phase_m = rng.uniform(0, 2 * math.pi)
            burst = p.motion_scale * env * np.sin(2 * math.pi * freq_m * t_local + phase_m)
            burst *= rng.uniform(0.6, 1.0, burst.shape)          # amplitude modulation
            artifact[mask] = burst
        return artifact

    def _dropouts(self, n: int, base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Signal dropouts: subtract current value to flatten to zero."""
        p = self.params
        if p.dropout_prob == 0.0:
            return np.zeros(n)
        mask = rng.random(n) < p.dropout_prob
        # Convolve mask with short rect window to create contiguous dropout blocks
        block = np.ones(8)
        mask_f = np.convolve(mask.astype(float), block, mode="same") > 0
        return np.where(mask_f, -base, 0.0)
