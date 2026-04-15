# PPG Signal Quality Assessment Pipeline

A modular, production-ready pipeline for photoplethysmography (PPG) signal
quality assessment using PyTorch. Supports end-to-end waveform CNNs,
feature-based MLPs, and late-fusion ensembles.

---

## Installation

```bash
pip install torch scipy numpy matplotlib
pip install -e .
```

---

## Architecture

```
Raw PPG signal
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  PPGPreprocessor                                     │
│  ① Polynomial detrend  ② Butterworth bandpass       │
│  ③ Zero-mean normalise ④ Adaptive peak detection    │
└────────────────┬────────────────────────────────────┘
                 │ filtered signal + peak indices
     ┌───────────┴────────────┐
     │                        │
     ▼                        ▼
┌──────────────┐     ┌──────────────────────────────┐
│ PPGWaveform  │     │ PPGFeatureExtractor           │
│ Net (CNN)    │     │  • Time-domain amplitude (4)  │
│              │     │  • Statistical shape     (4)  │
│  1D ResNet   │     │  • HRV / rhythm          (5)  │
│  + SE blocks │     │  • Morphological         (4)  │
│  + stats pool│     │  • Spectral              (4)  │
└──────┬───────┘     │  • Quality proxies       (3)  │
       │             └──────────────┬───────────────┘
       │                           │ 24-dim vector
       │                           ▼
       │                  ┌─────────────────┐
       │                  │ PPGFeatureNet   │
       │                  │ MLP (128→64→QH) │
       │                  └──────┬──────────┘
       │                         │
       └──────────┬──────────────┘
                  │ late fusion
                  ▼
         ┌────────────────┐
         │ PPGEnsembleNet │
         │  fusion MLP    │
         └────────┬───────┘
                  │
                  ▼
         ┌──────────────────────────────┐
         │  QualityHead (shared)        │
         │  ├─ quality_score  [0, 1]    │
         │  └─ verdict_logits (3-class) │
         └──────────────────────────────┘
                  │
                  ▼
         ┌────────────────────┐
         │  QualityReport     │
         │  • Verdict (enum)  │
         │  • Sub-scores      │
         │  • Feature dict    │
         │  • ML input vector │
         └────────────────────┘
```

### Verdict thresholds

| Score   | Verdict  | Meaning                                      |
|---------|----------|----------------------------------------------|
| ≥ 0.75  | ACCEPT   | Excellent — use for inference                |
| ≥ 0.50  | CAUTION  | Moderate — downweight confidence             |
| < 0.50  | REJECT   | Poor — discard or re-acquire                 |

---

## Quick start

```python
from ppg_pipeline import (
    PPGSignalGenerator, SignalParams,
    PPGFeatureNet, PPGQualityPipeline,
    generate_synthetic_dataset, PPGTrainer, TrainerConfig,
    FeatureNormaliser,
)

# 1. Generate training data
dataset = generate_synthetic_dataset(n_samples=2000, fs=100.0, balanced=True)

# 2. Fit feature normaliser
import numpy as np
X = dataset.feature_vecs.numpy()
normaliser = FeatureNormaliser().fit(X)
dataset.feature_vecs = __import__("torch").from_numpy(normaliser.transform(X))

# 3. Train a feature MLP
model = PPGFeatureNet(n_features=24, hidden_dims=[128, 128, 64])
config = TrainerConfig(max_epochs=40, lr=3e-4, checkpoint_dir="checkpoints")
trainer = PPGTrainer(model, config, dataset)
trainer.fit()
trainer.load_best()

# 4. Build inference pipeline
pipeline = PPGQualityPipeline.from_feature_net(model, normaliser, fs=100.0)

# 5. Assess a signal
params = SignalParams(heart_rate=72, noise_std=0.15)
raw, _, _ = PPGSignalGenerator(params).generate()
report = pipeline.assess(raw)
print(report)
# → quality_score=0.87  verdict=ACCEPT
```

---

## Run the full demo

```bash
python demo.py           # full training + visualisation
python demo.py --quick   # fast (400 samples, 20 epochs)
python demo.py --no-plot # headless / CI mode
```

---

## Module reference

| Module               | Contents                                              |
|----------------------|-------------------------------------------------------|
| `signal_generator`   | `SignalParams`, `PPGSignalGenerator`                  |
| `preprocessor`       | `PPGPreprocessor`                                     |
| `feature_extractor`  | `PPGFeatureExtractor`, `FeatureVector`, `FeatureNormaliser` |
| `models`             | `PPGWaveformNet`, `PPGFeatureNet`, `PPGEnsembleNet`, `PPGQualityLoss` |
| `dataset`            | `PPGDataset`, `generate_synthetic_dataset`, `PPGAugment` |
| `trainer`            | `PPGTrainer`, `TrainerConfig`, `EpochMetrics`         |
| `pipeline`           | `PPGQualityPipeline`, `QualityReport`, `Verdict`      |

---

## Features extracted

| Group                | Features (24 total)                               |
|----------------------|---------------------------------------------------|
| Time-domain amp      | DC mean, AC amplitude, AC/DC ratio, amplitude range |
| Statistical shape    | Skewness, kurtosis, approx. entropy, ZCR          |
| HRV / rhythm         | Mean RR, SDNN, RMSSD, pNN50, estimated HR         |
| Morphological        | Rise time, pulse width, augmentation index, dicrotic ratio |
| Spectral             | LF power, HF power, LF/HF ratio, spectral entropy |
| Quality proxies      | Peak regularity, completeness, SNR proxy          |
