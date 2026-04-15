"""
preprocessor.py
---------------
Signal conditioning stage of the PPG quality pipeline.

Steps
~~~~~
1. Detrend         – remove polynomial baseline drift
2. Bandpass filter – 0.5–8 Hz 4th-order Butterworth (retains cardiac + harmonic content)
3. Normalise       – zero-mean, unit-variance per segment
4. Peak detection  – adaptive threshold local maxima with refractory period
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


class PPGPreprocessor:
    """
    Preprocess a raw PPG segment.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    lowcut : float
        Lower bandpass cutoff (Hz). Default 0.5 Hz.
    highcut : float
        Upper bandpass cutoff (Hz). Default 8.0 Hz.
    filter_order : int
        Butterworth filter order. Default 4.
    detrend_degree : int
        Polynomial degree for baseline removal. Default 2.

    Examples
    --------
    >>> pre = PPGPreprocessor(fs=100.0)
    >>> filtered, peaks = pre.process(raw_signal)
    """

    def __init__(
        self,
        fs: float = 100.0,
        lowcut: float = 0.5,
        highcut: float = 8.0,
        filter_order: int = 4,
        detrend_degree: int = 2,
    ) -> None:
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.detrend_degree = detrend_degree

        nyq = 0.5 * fs
        self._sos = sp_signal.butter(
            filter_order,
            [lowcut / nyq, highcut / nyq],
            btype="band",
            output="sos",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline.

        Parameters
        ----------
        raw : np.ndarray, shape (N,)
            Raw PPG samples.

        Returns
        -------
        filtered : np.ndarray, shape (N,)
            Detrended, bandpass-filtered, normalised signal.
        peaks : np.ndarray
            Sample indices of detected systolic peaks.
        """
        detrended = self._detrend(raw)
        filtered = self._bandpass(detrended)
        normalised = self._normalise(filtered)
        peaks = self._detect_peaks(normalised)
        return normalised.astype(np.float32), peaks

    def detrend_only(self, raw: np.ndarray) -> np.ndarray:
        """Return detrended signal without bandpass or normalisation."""
        return self._detrend(raw).astype(np.float32)

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _detrend(self, x: np.ndarray) -> np.ndarray:
        """Polynomial detrend: fit and subtract low-degree polynomial."""
        n = len(x)
        t = np.linspace(0.0, 1.0, n)
        coeffs = np.polyfit(t, x, self.detrend_degree)
        baseline = np.polyval(coeffs, t)
        return x - baseline

    def _bandpass(self, x: np.ndarray) -> np.ndarray:
        """Zero-phase SOS Butterworth bandpass."""
        return sp_signal.sosfiltfilt(self._sos, x)

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        """Zero-mean, unit-variance normalisation."""
        std = x.std()
        if std < 1e-8:
            return x - x.mean()
        return (x - x.mean()) / std

    def _detect_peaks(self, x: np.ndarray, min_hr: float = 40.0, max_hr: float = 200.0) -> np.ndarray:
        """
        Adaptive threshold peak detection with refractory period.

        Uses scipy.signal.find_peaks with:
        - minimum inter-peak distance derived from max HR
        - minimum prominence = 0.3 × dynamic range
        """
        min_dist = int(self.fs * 60.0 / max_hr)
        prominence = max(0.3 * (x.max() - x.min()), 0.1)
        peaks, _ = sp_signal.find_peaks(
            x,
            distance=min_dist,
            prominence=prominence,
        )
        # Filter by maximum HR
        if len(peaks) > 1:
            rr_samples = np.diff(peaks)
            max_dist = int(self.fs * 60.0 / min_hr)
            keep = np.concatenate([[True], rr_samples <= max_dist])
            peaks = peaks[keep]
        return peaks
