"""
pipeline.py
-----------
End-to-end PPG signal quality assessment pipeline for inference.

Combines:
  1. Preprocessing
  2. Feature extraction
  3. Model inference (waveform CNN, feature MLP, or ensemble)
  4. Structured quality report with verdict, scores, and feature breakdown

Usage
~~~~~
>>> pipeline = PPGQualityPipeline.from_feature_net(model, normaliser)
>>> report = pipeline.assess(raw_ppg_signal)
>>> print(report)
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from .feature_extractor import FeatureNormaliser, FeatureVector, PPGFeatureExtractor
from .models import ModelOutput, PPGEnsembleNet, PPGFeatureNet, PPGWaveformNet
from .preprocessor import PPGPreprocessor

AnyModel = Union[PPGWaveformNet, PPGFeatureNet, PPGEnsembleNet]


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------

class Verdict(enum.Enum):
    ACCEPT = 0
    CAUTION = 1
    REJECT = 2

    @property
    def label(self) -> str:
        return self.name

    @property
    def description(self) -> str:
        return {
            Verdict.ACCEPT:  "Excellent quality — signal accepted for ML inference.",
            Verdict.CAUTION: "Moderate quality — use with confidence downweighting.",
            Verdict.REJECT:  "Poor quality — segment rejected, re-acquisition recommended.",
        }[self]


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Structured output from the pipeline."""

    # Core assessment
    quality_score: float           # Model-predicted score [0, 1]
    verdict: Verdict               # ACCEPT / CAUTION / REJECT
    verdict_probabilities: dict[str, float]  # softmax over 3 classes

    # Signal statistics
    n_samples: int
    n_peaks: int
    estimated_hr: float            # bpm
    completeness: float            # [0, 1]

    # Full feature vector
    features: dict[str, float]

    # Optional rule-based sub-scores (computed from features, model-agnostic)
    rule_scores: dict[str, float]

    # Metadata
    model_type: str
    fs: float

    # ------------------------------------------------------------------

    def __str__(self) -> str:
        bar_len = 30
        filled = int(self.quality_score * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        lines = [
            "=" * 60,
            f"  PPG Quality Report — {self.verdict.label}",
            "=" * 60,
            f"  Quality score  : {self.quality_score:.3f}  [{bar}]",
            f"  Verdict        : {self.verdict.label} — {self.verdict.description}",
            f"  Heart rate est : {self.estimated_hr:.1f} bpm",
            f"  Peaks detected : {self.n_peaks}",
            f"  Completeness   : {self.completeness:.2%}",
            "",
            "  Verdict probabilities:",
            *[f"    {k:8s}: {v:.3f}" for k, v in self.verdict_probabilities.items()],
            "",
            "  Rule-based sub-scores (0–1):",
            *[f"    {k:20s}: {v:.3f}" for k, v in self.rule_scores.items()],
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "quality_score": round(self.quality_score, 6),
            "verdict": self.verdict.label,
            "verdict_probabilities": {k: round(v, 6) for k, v in self.verdict_probabilities.items()},
            "n_samples": self.n_samples,
            "n_peaks": self.n_peaks,
            "estimated_hr": round(self.estimated_hr, 2),
            "completeness": round(self.completeness, 4),
            "features": {k: round(v, 6) for k, v in self.features.items()},
            "rule_scores": {k: round(v, 4) for k, v in self.rule_scores.items()},
            "model_type": self.model_type,
            "fs": self.fs,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_ml_input(self) -> np.ndarray:
        """Return raw feature vector (24,) for feeding into downstream models."""
        return np.array(list(self.features.values()), dtype=np.float32)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PPGQualityPipeline:
    """
    End-to-end PPG quality assessment pipeline.

    Parameters
    ----------
    model : AnyModel
        Trained PyTorch model.
    preprocessor : PPGPreprocessor
    feature_extractor : PPGFeatureExtractor
    normaliser : FeatureNormaliser, optional
        If provided, normalises features before passing to model.
    device : str or torch.device
        Inference device.

    Examples
    --------
    >>> pipeline = PPGQualityPipeline(model, pre, feat, normaliser)
    >>> report = pipeline.assess(raw_signal)
    >>> print(report)
    >>> vec = report.to_ml_input()   # normalised feature array for downstream
    """

    def __init__(
        self,
        model: AnyModel,
        preprocessor: PPGPreprocessor,
        feature_extractor: PPGFeatureExtractor,
        normaliser: Optional[FeatureNormaliser] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.normaliser = normaliser
        self.device = torch.device(device)
        self._is_ensemble = isinstance(model, PPGEnsembleNet)
        self._is_waveform = isinstance(model, PPGWaveformNet)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_feature_net(
        cls,
        model: PPGFeatureNet,
        normaliser: Optional[FeatureNormaliser] = None,
        fs: float = 100.0,
        device: str = "cpu",
    ) -> "PPGQualityPipeline":
        return cls(
            model=model,
            preprocessor=PPGPreprocessor(fs=fs),
            feature_extractor=PPGFeatureExtractor(fs=fs),
            normaliser=normaliser,
            device=device,
        )

    @classmethod
    def from_waveform_net(
        cls,
        model: PPGWaveformNet,
        fs: float = 100.0,
        device: str = "cpu",
    ) -> "PPGQualityPipeline":
        return cls(
            model=model,
            preprocessor=PPGPreprocessor(fs=fs),
            feature_extractor=PPGFeatureExtractor(fs=fs),
            normaliser=None,
            device=device,
        )

    @classmethod
    def from_ensemble(
        cls,
        model: PPGEnsembleNet,
        normaliser: Optional[FeatureNormaliser] = None,
        fs: float = 100.0,
        device: str = "cpu",
    ) -> "PPGQualityPipeline":
        return cls(
            model=model,
            preprocessor=PPGPreprocessor(fs=fs),
            feature_extractor=PPGFeatureExtractor(fs=fs),
            normaliser=normaliser,
            device=device,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def assess(self, raw_signal: np.ndarray) -> QualityReport:
        """
        Assess the quality of a raw PPG segment.

        Parameters
        ----------
        raw_signal : np.ndarray, shape (N,)
            Raw PPG samples at the configured sampling frequency.

        Returns
        -------
        QualityReport
        """
        # 1. Preprocess
        filtered, peaks = self.preprocessor.process(raw_signal)

        # 2. Extract features
        fv: FeatureVector = self.feature_extractor.extract(filtered, peaks)
        feat_array = fv.to_array()                      # (24,)

        # 3. Normalise features (if normaliser available)
        feat_norm = (
            self.normaliser.transform(feat_array[None])[0]
            if self.normaliser else feat_array
        )

        # 4. Build tensors
        with torch.no_grad():
            wave_t = torch.from_numpy(filtered[None, None]).to(self.device)    # (1, 1, L)
            feat_t = torch.from_numpy(feat_norm[None]).to(self.device)         # (1, 24)

            if self._is_ensemble:
                output: ModelOutput = self.model(wave_t, feat_t)
            elif self._is_waveform:
                output = self.model(wave_t)
            else:
                output = self.model(feat_t)

        quality_score = float(output.quality_score[0].cpu())
        probs = torch.softmax(output.verdict_logits[0], dim=0).cpu().tolist()
        verdict_idx = int(output.verdict_logits[0].argmax())
        verdict = Verdict(verdict_idx)

        # 5. Rule-based sub-scores
        rule_scores = self._rule_scores(fv)

        return QualityReport(
            quality_score=quality_score,
            verdict=verdict,
            verdict_probabilities={
                "ACCEPT": probs[0],
                "CAUTION": probs[1],
                "REJECT": probs[2],
            },
            n_samples=len(raw_signal),
            n_peaks=len(peaks),
            estimated_hr=fv.heart_rate_est,
            completeness=fv.completeness,
            features=fv.to_dict(),
            rule_scores=rule_scores,
            model_type=type(self.model).__name__,
            fs=self.preprocessor.fs,
        )

    def assess_batch(self, segments: list[np.ndarray]) -> list[QualityReport]:
        """Assess a list of PPG segments. Returns one report per segment."""
        return [self.assess(seg) for seg in segments]

    # ------------------------------------------------------------------
    # Rule-based sub-scores (interpretable, no model needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_scores(fv: FeatureVector) -> dict[str, float]:
        """Derive interpretable sub-scores from extracted features."""

        def clamp(v: float) -> float:
            return max(0.0, min(1.0, v))

        snr_score = clamp(fv.snr_proxy)
        peak_score = clamp(fv.peak_regularity)
        complete_score = clamp(fv.completeness)

        # Perfusion: AC/DC ratio [0, 0.5] → [0, 1]
        perf_score = clamp(fv.ac_dc_ratio / 0.5)

        # HR plausibility: penalise if outside [40, 180] bpm
        hr = fv.heart_rate_est
        hr_score = 1.0 if 40 <= hr <= 180 else clamp(1.0 - abs(hr - 110) / 110)

        # HRV regularity: low SDNN/mean_RR = regular
        if fv.mean_rr_ms > 0:
            cv = fv.sdnn_ms / (fv.mean_rr_ms + 1e-8)
            hrv_score = clamp(1.0 - cv * 3.0)
        else:
            hrv_score = 0.0

        # Spectral concentration: more power in cardiac band = better
        spectral_score = clamp(fv.snr_proxy * 2.0)

        return {
            "snr":              round(snr_score, 4),
            "peak_regularity":  round(peak_score, 4),
            "completeness":     round(complete_score, 4),
            "perfusion_index":  round(perf_score, 4),
            "hr_plausibility":  round(hr_score, 4),
            "hrv_regularity":   round(hrv_score, 4),
            "spectral_focus":   round(spectral_score, 4),
        }
