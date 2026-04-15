"""
feature_extractor.py
--------------------
Extracts a fixed-length feature vector from a preprocessed PPG segment.

Feature domains
~~~~~~~~~~~~~~~
1. Time-domain amplitude  (4 features)  – AC, DC, AC/DC, amplitude range
2. Statistical shape       (4 features)  – skewness, kurtosis, entropy, zero-crossing rate
3. HRV / rhythm            (5 features)  – mean RR, SDNN, RMSSD, pNN50, heart rate
4. Morphological           (4 features)  – rise time, pulse width, augmentation index proxy, dicrotic ratio
5. Spectral                (4 features)  – LF power, HF power, LF/HF, spectral entropy
6. Quality proxies         (3 features)  – peak regularity, completeness, SNR proxy

Total: 24 normalised features → ML input vector
"""

from __future__ import annotations

import math

# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both
import numpy as _np_compat
_trapz = getattr(_np_compat, "trapezoid", None) or getattr(_np_compat, "trapz")
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """Named feature vector with metadata."""

    # Time-domain amplitude
    dc_mean: float = 0.0
    ac_amplitude: float = 0.0
    ac_dc_ratio: float = 0.0
    amplitude_range: float = 0.0

    # Statistical shape
    skewness: float = 0.0
    excess_kurtosis: float = 0.0
    sample_entropy: float = 0.0
    zero_crossing_rate: float = 0.0

    # HRV / rhythm
    mean_rr_ms: float = 0.0
    sdnn_ms: float = 0.0
    rmssd_ms: float = 0.0
    pnn50: float = 0.0
    heart_rate_est: float = 0.0

    # Morphological
    mean_rise_time: float = 0.0
    mean_pulse_width: float = 0.0
    augmentation_index: float = 0.0
    dicrotic_ratio: float = 0.0

    # Spectral
    lf_power: float = 0.0
    hf_power: float = 0.0
    lf_hf_ratio: float = 0.0
    spectral_entropy: float = 0.0

    # Quality proxies
    peak_regularity: float = 0.0
    completeness: float = 0.0
    snr_proxy: float = 0.0

    NAMES: list[str] = field(default_factory=lambda: [
        "dc_mean", "ac_amplitude", "ac_dc_ratio", "amplitude_range",
        "skewness", "excess_kurtosis", "sample_entropy", "zero_crossing_rate",
        "mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50", "heart_rate_est",
        "mean_rise_time", "mean_pulse_width", "augmentation_index", "dicrotic_ratio",
        "lf_power", "hf_power", "lf_hf_ratio", "spectral_entropy",
        "peak_regularity", "completeness", "snr_proxy",
    ])

    def to_array(self) -> np.ndarray:
        """Return feature values as float32 numpy array (length 24)."""
        return np.array(
            [getattr(self, name) for name in self.NAMES if name != "NAMES"],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.NAMES if name != "NAMES"}

    @property
    def n_features(self) -> int:
        return len(self.NAMES) - 1  # exclude NAMES field itself


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class PPGFeatureExtractor:
    """
    Compute a 24-dimensional feature vector from a preprocessed PPG segment.

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz).
    expected_hr_range : tuple[float, float]
        Expected heart rate range (bpm) for completeness estimation.

    Examples
    --------
    >>> extractor = PPGFeatureExtractor(fs=100.0)
    >>> features = extractor.extract(filtered_signal, peaks)
    >>> vec = features.to_array()   # shape (24,)
    """

    # Spectral band boundaries (Hz)
    LF_BAND = (0.04, 0.15)
    HF_BAND = (0.15, 0.40)

    def __init__(
        self,
        fs: float = 100.0,
        expected_hr_range: tuple[float, float] = (40.0, 200.0),
    ) -> None:
        self.fs = fs
        self.expected_hr_range = expected_hr_range

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, signal: np.ndarray, peaks: np.ndarray) -> FeatureVector:
        """
        Extract all features.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Preprocessed (filtered + normalised) PPG signal.
        peaks : np.ndarray
            Sample indices of detected systolic peaks.

        Returns
        -------
        FeatureVector
        """
        fv = FeatureVector()
        rr = self._rr_intervals(peaks)

        # Domain computations
        self._time_domain(signal, fv)
        self._statistical_shape(signal, fv)
        self._hrv_rhythm(rr, peaks, signal, fv)
        self._morphological(signal, peaks, fv)
        self._spectral(signal, fv)
        self._quality_proxies(signal, peaks, rr, fv)

        return fv

    # ------------------------------------------------------------------
    # Domain methods
    # ------------------------------------------------------------------

    def _time_domain(self, x: np.ndarray, fv: FeatureVector) -> None:
        dc = float(np.mean(x))
        ac = float((np.max(x) - np.min(x)) / 2.0)
        fv.dc_mean = dc
        fv.ac_amplitude = ac
        fv.ac_dc_ratio = ac / (abs(dc) + 1e-8)
        fv.amplitude_range = float(np.max(x) - np.min(x))

    def _statistical_shape(self, x: np.ndarray, fv: FeatureVector) -> None:
        fv.skewness = float(skew(x))
        fv.excess_kurtosis = float(kurtosis(x))
        fv.sample_entropy = self._approx_entropy(x, m=2, r=0.2)
        diff_signs = np.diff(np.sign(x))
        fv.zero_crossing_rate = float(np.sum(diff_signs != 0)) / len(x)

    def _hrv_rhythm(
        self,
        rr: np.ndarray,
        peaks: np.ndarray,
        x: np.ndarray,
        fv: FeatureVector,
    ) -> None:
        if len(rr) < 2:
            fv.mean_rr_ms = 0.0
            fv.sdnn_ms = 0.0
            fv.rmssd_ms = 0.0
            fv.pnn50 = 0.0
            fv.heart_rate_est = 0.0
            return

        rr_ms = rr * 1000.0 / self.fs
        fv.mean_rr_ms = float(np.mean(rr_ms))
        fv.sdnn_ms = float(np.std(rr_ms, ddof=1))
        successive_diff = np.diff(rr_ms)
        fv.rmssd_ms = float(np.sqrt(np.mean(successive_diff ** 2)))
        fv.pnn50 = float(np.mean(np.abs(successive_diff) > 50.0))
        fv.heart_rate_est = 60_000.0 / (fv.mean_rr_ms + 1e-8)

    def _morphological(self, x: np.ndarray, peaks: np.ndarray, fv: FeatureVector) -> None:
        if len(peaks) < 2:
            return

        rise_times, pulse_widths, aug_indices, dic_ratios = [], [], [], []

        for i, pk in enumerate(peaks[:-1]):
            # Onset: minimum before peak in this beat window
            window_start = peaks[i - 1] if i > 0 else max(0, pk - int(0.5 * self.fs))
            segment = x[window_start:pk]
            if len(segment) == 0:
                continue
            onset = window_start + int(np.argmin(segment))

            # Rise time (onset → systolic peak)
            rt = (pk - onset) / self.fs
            rise_times.append(rt)

            # Pulse width: 50% amplitude threshold crossing
            amp_50 = x[onset] + 0.5 * (x[pk] - x[onset])
            next_pk = peaks[i + 1]
            descend = x[pk:next_pk]
            cross = np.where(descend < amp_50)[0]
            pw = (cross[0] / self.fs) if len(cross) else (next_pk - pk) / self.fs
            pulse_widths.append(pw)

            # Augmentation index proxy (secondary peak height / primary peak)
            seg_post = x[pk:next_pk]
            if len(seg_post) > 4:
                local_peaks, _ = sp_signal.find_peaks(seg_post, prominence=0.02)
                if len(local_peaks):
                    dic_ratios.append(float(seg_post[local_peaks[0]]) / (x[pk] + 1e-8))
                aug_indices.append(float(np.mean(seg_post)) / (x[pk] + 1e-8))

        if rise_times:
            fv.mean_rise_time = float(np.mean(rise_times))
        if pulse_widths:
            fv.mean_pulse_width = float(np.mean(pulse_widths))
        if aug_indices:
            fv.augmentation_index = float(np.mean(aug_indices))
        if dic_ratios:
            fv.dicrotic_ratio = float(np.mean(dic_ratios))

    def _spectral(self, x: np.ndarray, fv: FeatureVector) -> None:
        freqs, psd = sp_signal.welch(x, fs=self.fs, nperseg=min(256, len(x)))

        def band_power(lo, hi):
            mask = (freqs >= lo) & (freqs <= hi)
            return float(_trapz(psd[mask], freqs[mask])) if mask.any() else 0.0

        fv.lf_power = band_power(*self.LF_BAND)
        fv.hf_power = band_power(*self.HF_BAND)
        fv.lf_hf_ratio = fv.lf_power / (fv.hf_power + 1e-12)

        # Normalised spectral entropy
        psd_norm = psd / (psd.sum() + 1e-12)
        fv.spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / math.log2(len(psd_norm)))

    def _quality_proxies(
        self,
        x: np.ndarray,
        peaks: np.ndarray,
        rr: np.ndarray,
        fv: FeatureVector,
    ) -> None:
        # Peak regularity: 1 - CV of RR intervals (coefficient of variation)
        if len(rr) > 1:
            cv = np.std(rr, ddof=1) / (np.mean(rr) + 1e-8)
            fv.peak_regularity = float(max(0.0, 1.0 - cv))
        else:
            fv.peak_regularity = 0.0

        # Completeness: detected peaks / expected peaks
        duration_s = len(x) / self.fs
        if fv.heart_rate_est > 0:
            expected = (fv.heart_rate_est / 60.0) * duration_s
        else:
            expected = duration_s  # fallback
        fv.completeness = float(min(1.0, len(peaks) / (expected + 1e-8)))

        # SNR proxy: ratio of signal power in cardiac band vs total power
        freqs, psd = sp_signal.welch(x, fs=self.fs, nperseg=min(256, len(x)))
        total = float(_trapz(psd, freqs)) + 1e-12
        hr_hz = fv.heart_rate_est / 60.0
        cardiac_mask = (freqs >= max(0.5, hr_hz * 0.8)) & (freqs <= min(8.0, hr_hz * 3.5))
        cardiac = float(_trapz(psd[cardiac_mask], freqs[cardiac_mask])) if cardiac_mask.any() else 0.0
        fv.snr_proxy = cardiac / total

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _rr_intervals(self, peaks: np.ndarray) -> np.ndarray:
        """Inter-peak intervals in samples."""
        if len(peaks) < 2:
            return np.array([])
        return np.diff(peaks).astype(float)

    @staticmethod
    def _approx_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Approximate entropy (ApEn) — fast approximate implementation.
        r is scaled by the signal's standard deviation.
        """
        n = len(x)
        if n < 10:
            return 0.0
        r_abs = r * (x.std() + 1e-8)
        # Subsample for speed when signal is long
        if n > 300:
            step = n // 300
            x = x[::step]
            n = len(x)

        def phi(m_):
            templates = np.array([x[i: i + m_] for i in range(n - m_)])
            count = np.sum(
                np.max(np.abs(templates[:, None] - templates[None, :]), axis=-1) <= r_abs,
                axis=1,
            )
            return np.mean(np.log(count / (n - m_ + 1e-8) + 1e-8))

        try:
            return float(phi(m) - phi(m + 1))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Convenience: normalise a feature array to [-1, 1] using per-feature stats
# ---------------------------------------------------------------------------

class FeatureNormaliser:
    """
    Running stats normaliser: fit on training data, transform at inference.

    Uses robust scaling: (x - median) / IQR clipped to [-3, 3].
    """

    def __init__(self) -> None:
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureNormaliser":
        """X : shape (N, 24)."""
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        self.median_ = np.median(X, axis=0)
        self.iqr_ = q75 - q25
        self.iqr_[self.iqr_ < 1e-8] = 1.0     # prevent div-by-zero
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() before transform()"
        return np.clip((X - self.median_) / self.iqr_, -3.0, 3.0).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        np.savez(path, median=self.median_, iqr=self.iqr_)

    @classmethod
    def load(cls, path: str) -> "FeatureNormaliser":
        data = np.load(path)
        obj = cls()
        obj.median_ = data["median"]
        obj.iqr_ = data["iqr"]
        obj._fitted = True
        return obj
