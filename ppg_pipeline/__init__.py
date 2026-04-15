"""
PPG Signal Quality Assessment Pipeline
=======================================
A modular pipeline for photoplethysmography (PPG) signal quality assessment
using PyTorch-based ML models.

Modules
-------
signal_generator  : Synthetic PPG generation with controllable artifacts
preprocessor      : Bandpass filtering, detrending, peak detection
feature_extractor : Time-domain, HRV, morphological, and spectral features
models            : CNN waveform encoder, MLP feature net, ensemble
dataset           : PyTorch Dataset for supervised training
trainer           : Training loop with metrics and early stopping
pipeline          : End-to-end inference pipeline

Lazy imports
------------
Heavy dependencies (torch, scipy) are only pulled in when the relevant
sub-module is first accessed, so ``import ppg_pipeline`` never fails due
to a missing optional dependency.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    # signal_generator
    "PPGSignalGenerator", "SignalParams",
    # preprocessor
    "PPGPreprocessor",
    # feature_extractor
    "PPGFeatureExtractor", "FeatureVector", "FeatureNormaliser",
    # models  (torch required)
    "PPGWaveformNet", "PPGFeatureNet", "PPGEnsembleNet",
    # dataset  (torch required)
    "PPGDataset", "PPGAugment", "generate_synthetic_dataset",
    # trainer  (torch required)
    "PPGTrainer", "TrainerConfig",
    # pipeline  (torch required)
    "PPGQualityPipeline", "QualityReport", "Verdict",
]

# ---------------------------------------------------------------------------
# Eagerly import pure-Python / numpy-only modules (always safe)
# ---------------------------------------------------------------------------
from .signal_generator import PPGSignalGenerator, SignalParams  # noqa: E402

# ---------------------------------------------------------------------------
# Lazily import everything that depends on scipy or torch
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    """PEP 562 module-level __getattr__ for lazy imports."""

    if name == "PPGPreprocessor":
        from .preprocessor import PPGPreprocessor
        return PPGPreprocessor

    if name in ("PPGFeatureExtractor", "FeatureVector", "FeatureNormaliser"):
        from .feature_extractor import PPGFeatureExtractor, FeatureVector, FeatureNormaliser
        return {"PPGFeatureExtractor": PPGFeatureExtractor,
                "FeatureVector": FeatureVector,
                "FeatureNormaliser": FeatureNormaliser}[name]

    if name in ("PPGWaveformNet", "PPGFeatureNet", "PPGEnsembleNet"):
        from .models import PPGWaveformNet, PPGFeatureNet, PPGEnsembleNet
        return {"PPGWaveformNet": PPGWaveformNet,
                "PPGFeatureNet": PPGFeatureNet,
                "PPGEnsembleNet": PPGEnsembleNet}[name]

    if name in ("PPGDataset", "PPGAugment", "generate_synthetic_dataset"):
        from .dataset import PPGDataset, PPGAugment, generate_synthetic_dataset
        return {"PPGDataset": PPGDataset,
                "PPGAugment": PPGAugment,
                "generate_synthetic_dataset": generate_synthetic_dataset}[name]

    if name in ("PPGTrainer", "TrainerConfig"):
        from .trainer import PPGTrainer, TrainerConfig
        return {"PPGTrainer": PPGTrainer, "TrainerConfig": TrainerConfig}[name]

    if name in ("PPGQualityPipeline", "QualityReport", "Verdict"):
        from .pipeline import PPGQualityPipeline, QualityReport, Verdict
        return {"PPGQualityPipeline": PPGQualityPipeline,
                "QualityReport": QualityReport,
                "Verdict": Verdict}[name]

    raise AttributeError(f"module 'ppg_pipeline' has no attribute {name!r}")
