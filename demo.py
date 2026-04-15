#!/usr/bin/env python3
"""
demo.py
-------
End-to-end demonstration of the PPG signal quality assessment pipeline.

Sections
~~~~~~~~
1. Signal generation   – create synthetic PPG signals with varying quality
2. Preprocessing       – filter and detect peaks
3. Feature extraction  – compute 24-dimensional feature vector
4. Model training      – train PPGFeatureNet on synthetic data
5. Ensemble training   – train PPGEnsembleNet (CNN + MLP)
6. Inference           – run full pipeline on new signals
7. Visualisation       – plot signals, features, and quality scores

Run
~~~
    python demo.py [--quick] [--no-plot]

    --quick   : use small dataset (200 samples) for fast testing
    --no-plot : skip matplotlib visualisations
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="PPG Quality Pipeline Demo")
parser.add_argument("--quick",   action="store_true", help="Small dataset for fast testing")
parser.add_argument("--no-plot", action="store_true", help="Disable visualisations")
args = parser.parse_args()

QUICK = args.quick
PLOT  = not args.no_plot
N_TRAIN = 400 if QUICK else 2000

print("=" * 60)
print("  PPG Signal Quality Assessment Pipeline — Demo")
print("=" * 60)

# ---------------------------------------------------------------------------
# Imports (post-argument parsing so --help works without torch)
# ---------------------------------------------------------------------------

import torch
from ppg_pipeline import (
    FeatureNormaliser,
    PPGAugment,
    PPGDataset,
    PPGEnsembleNet,
    PPGFeatureExtractor,
    PPGFeatureNet,
    PPGQualityPipeline,
    PPGSignalGenerator,
    PPGTrainer,
    PPGWaveformNet,
    SignalParams,
    TrainerConfig,
    Verdict,
    generate_synthetic_dataset,
)
from ppg_pipeline.preprocessor import PPGPreprocessor

FS = 100.0
DURATION = 5.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")


# ===========================================================================
# SECTION 1 — Signal generation
# ===========================================================================
print("─" * 60)
print("§1 Signal Generation")
print("─" * 60)

scenarios = {
    "Clean  (HR=72)": SignalParams.clean(heart_rate=72),
    "Noisy  (σ=0.4)": SignalParams(heart_rate=72, noise_std=0.4),
    "Motion artifact": SignalParams(heart_rate=72, noise_std=0.1, motion_scale=0.9),
    "Low perfusion":   SignalParams(heart_rate=90, perfusion_index=2, noise_std=0.15),
    "Baseline wander": SignalParams(heart_rate=65, wander_scale=0.7, noise_std=0.1),
    "Poor (all)":      SignalParams(heart_rate=72, noise_std=0.5, motion_scale=1.0,
                                    wander_scale=0.6, dropout_prob=0.06),
}

generated_signals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
for name, params in scenarios.items():
    gen = PPGSignalGenerator(params)
    raw, clean, t = gen.generate()
    generated_signals[name] = (raw, clean, t)
    print(f"  {name:20s}: {len(raw)} samples  range=[{raw.min():.2f}, {raw.max():.2f}]")

print()


# ===========================================================================
# SECTION 2 — Preprocessing & Peak Detection
# ===========================================================================
print("─" * 60)
print("§2 Preprocessing")
print("─" * 60)

pre = PPGPreprocessor(fs=FS)
preprocessed: dict[str, tuple[np.ndarray, np.ndarray]] = {}

for name, (raw, _, t) in generated_signals.items():
    filtered, peaks = pre.process(raw)
    preprocessed[name] = (filtered, peaks)
    print(f"  {name:20s}: {len(peaks):2d} peaks detected  "
          f"signal std={filtered.std():.3f}")
print()


# ===========================================================================
# SECTION 3 — Feature Extraction
# ===========================================================================
print("─" * 60)
print("§3 Feature Extraction")
print("─" * 60)

feat_ex = PPGFeatureExtractor(fs=FS)
feature_vectors: dict[str, np.ndarray] = {}

for name, (filtered, peaks) in preprocessed.items():
    fv = feat_ex.extract(filtered, peaks)
    vec = fv.to_array()
    feature_vectors[name] = vec
    print(f"  {name:20s}: HR={fv.heart_rate_est:5.1f}bpm  "
          f"SDNN={fv.sdnn_ms:5.1f}ms  "
          f"SNR={fv.snr_proxy:.3f}  "
          f"peaks={fv.peak_regularity:.3f}")

print(f"\n  Feature vector shape: {vec.shape}")
print()


# ===========================================================================
# SECTION 4 — Dataset Generation & Model Training (PPGFeatureNet)
# ===========================================================================
print("─" * 60)
print("§4 Training PPGFeatureNet (MLP on features)")
print("─" * 60)

print(f"Generating {N_TRAIN} synthetic training segments …")
full_dataset = generate_synthetic_dataset(
    n_samples=N_TRAIN, fs=FS, duration=DURATION, balanced=True, seed=42, verbose=True
)

# Fit feature normaliser on training split
train_ds, val_ds = full_dataset.split(val_fraction=0.15, seed=0)
X_train = train_ds.feature_vecs.numpy()
normaliser = FeatureNormaliser().fit(X_train)
X_train_norm = normaliser.transform(X_train)
X_val_norm = normaliser.transform(val_ds.feature_vecs.numpy())

# Overwrite feature tensors with normalised values
train_ds.feature_vecs = torch.from_numpy(X_train_norm)
val_ds.feature_vecs   = torch.from_numpy(X_val_norm)

# Build and train feature net
feature_net = PPGFeatureNet(n_features=24, hidden_dims=[128, 128, 64], dropout=0.2)
print(f"\nPPGFeatureNet parameters: {feature_net.n_parameters:,}")

cfg_feat = TrainerConfig(
    max_epochs=20 if QUICK else 40,
    batch_size=64,
    lr=3e-4,
    patience=8,
    checkpoint_dir="checkpoints/feature_net",
    device=DEVICE,
)

trainer_feat = PPGTrainer(feature_net, cfg_feat, train_ds, val_dataset=val_ds)
history_feat = trainer_feat.fit()
trainer_feat.load_best()
print()


# ===========================================================================
# SECTION 5 — Ensemble Training (PPGWaveformNet + PPGFeatureNet)
# ===========================================================================
print("─" * 60)
print("§5 Training PPGEnsembleNet (CNN waveform + MLP features)")
print("─" * 60)

# Train waveform CNN
waveform_net = PPGWaveformNet(in_channels=1, base_channels=16 if QUICK else 32)
print(f"PPGWaveformNet parameters: {waveform_net.n_parameters:,}")

cfg_wave = TrainerConfig(
    max_epochs=15 if QUICK else 30,
    batch_size=64,
    lr=3e-4,
    patience=6,
    use_augment=True,
    checkpoint_dir="checkpoints/waveform_net",
    device=DEVICE,
)

# Regenerate datasets with normalised features
full_ds2 = generate_synthetic_dataset(
    n_samples=N_TRAIN, fs=FS, duration=DURATION, balanced=True, seed=99, verbose=False
)
tr2, val2 = full_ds2.split(val_fraction=0.15, seed=1)
tr2.feature_vecs   = torch.from_numpy(normaliser.transform(tr2.feature_vecs.numpy()))
val2.feature_vecs  = torch.from_numpy(normaliser.transform(val2.feature_vecs.numpy()))
tr2.transform = PPGAugment(p=0.8)

trainer_wave = PPGTrainer(waveform_net, cfg_wave, tr2, val_dataset=val2)
history_wave = trainer_wave.fit()
trainer_wave.load_best()

# Build ensemble
ensemble_net = PPGEnsembleNet(waveform_net, feature_net, freeze_backbones=True)
print(f"\nPPGEnsembleNet trainable parameters: {ensemble_net.n_parameters:,}")

cfg_ens = TrainerConfig(
    max_epochs=10 if QUICK else 20,
    batch_size=64,
    lr=1e-3,
    patience=5,
    checkpoint_dir="checkpoints/ensemble",
    device=DEVICE,
)

trainer_ens = PPGTrainer(ensemble_net, cfg_ens, tr2, val_dataset=val2)
history_ens = trainer_ens.fit()
trainer_ens.load_best()
print()


# ===========================================================================
# SECTION 6 — Full Pipeline Inference
# ===========================================================================
print("─" * 60)
print("§6 Inference — PPGQualityPipeline")
print("─" * 60)

pipeline_feat = PPGQualityPipeline.from_feature_net(feature_net, normaliser, fs=FS, device=DEVICE)
pipeline_ens  = PPGQualityPipeline.from_ensemble(ensemble_net, normaliser, fs=FS, device=DEVICE)

for name, (raw, _, _) in generated_signals.items():
    report_f = pipeline_feat.assess(raw)
    report_e = pipeline_ens.assess(raw)
    print(f"\n{'─'*40}")
    print(f"  Scenario: {name}")
    print(f"  FeatureNet  → score={report_f.quality_score:.3f}  verdict={report_f.verdict.label}")
    print(f"  EnsembleNet → score={report_e.quality_score:.3f}  verdict={report_e.verdict.label}")

# Detailed report for clean signal
print("\n\nDetailed report (clean signal):")
clean_raw = generated_signals["Clean  (HR=72)"][0]
report = pipeline_ens.assess(clean_raw)
print(report)
print("\nJSON export snippet:")
d = report.to_dict()
print(f'  "quality_score" : {d["quality_score"]}')
print(f'  "verdict"       : {d["verdict"]}')
print(f'  "estimated_hr"  : {d["estimated_hr"]}')
print(f'  "n_peaks"       : {d["n_peaks"]}')


# ===========================================================================
# SECTION 7 — Visualisation (optional)
# ===========================================================================
if PLOT:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(16, 14))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)
        fig.suptitle("PPG Signal Quality Assessment Pipeline", fontsize=14, fontweight="bold")

        colours = ["#1D9E75", "#378ADD", "#E24B4A", "#EF9F27", "#A32D2D", "#7F77DD"]
        scenario_names = list(generated_signals.keys())

        # Row 0-1: raw signals + preprocessed
        for i, (name, col) in enumerate(zip(scenario_names, colours)):
            ax = fig.add_subplot(gs[0, i % 3]) if i < 3 else fig.add_subplot(gs[1, i % 3])
            raw, _, t = generated_signals[name]
            filtered, peaks = preprocessed[name]
            ax.plot(t, raw, color=col, lw=0.8, alpha=0.5, label="raw")
            ax.plot(t, filtered * (raw.max() - raw.min()) / 2 + raw.mean(),
                    color=col, lw=1.2, label="filtered")
            ax.scatter(t[peaks], raw[peaks], c="red", s=15, zorder=5)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6)

        # Row 2: feature heatmap
        ax_heat = fig.add_subplot(gs[2, :])
        feat_matrix = np.stack([feature_vectors[n] for n in scenario_names])
        feat_matrix_norm = (feat_matrix - feat_matrix.mean(0)) / (feat_matrix.std(0) + 1e-8)
        fv_names = list(feat_ex.extract(preprocessed["Clean  (HR=72)"][0],
                                        preprocessed["Clean  (HR=72)"][1]).to_dict().keys())
        im = ax_heat.imshow(feat_matrix_norm, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
        ax_heat.set_yticks(range(len(scenario_names)))
        ax_heat.set_yticklabels([n[:15] for n in scenario_names], fontsize=7)
        ax_heat.set_xticks(range(len(fv_names)))
        ax_heat.set_xticklabels(fv_names, rotation=45, ha="right", fontsize=6)
        ax_heat.set_title("Normalised Feature Heatmap (green=high, red=low)", fontsize=9)
        plt.colorbar(im, ax=ax_heat, fraction=0.015)

        # Row 3: training curves and quality scores bar
        ax_loss = fig.add_subplot(gs[3, 0])
        val_losses_f = [ep["val"]["loss"] for ep in history_feat]
        train_losses_f = [ep["train"]["loss"] for ep in history_feat]
        ax_loss.plot(train_losses_f, label="train (feat)", color="#378ADD", lw=1.2)
        ax_loss.plot(val_losses_f,   label="val (feat)",   color="#378ADD", lw=1.2, ls="--")
        ax_loss.set_title("Training loss — FeatureNet", fontsize=9)
        ax_loss.set_xlabel("Epoch", fontsize=8); ax_loss.set_ylabel("Loss", fontsize=8)
        ax_loss.legend(fontsize=7); ax_loss.tick_params(labelsize=7)

        ax_acc = fig.add_subplot(gs[3, 1])
        val_acc_f = [ep["val"]["verdict_acc"] for ep in history_feat]
        ax_acc.plot(val_acc_f, color="#1D9E75", lw=1.5, label="val accuracy")
        ax_acc.axhline(0.75, color="gray", ls=":", lw=0.8)
        ax_acc.set_title("Validation Verdict Accuracy", fontsize=9)
        ax_acc.set_xlabel("Epoch", fontsize=8); ax_acc.set_ylim(0, 1)
        ax_acc.legend(fontsize=7); ax_acc.tick_params(labelsize=7)

        ax_bar = fig.add_subplot(gs[3, 2])
        scores_feat = [pipeline_feat.assess(generated_signals[n][0]).quality_score for n in scenario_names]
        scores_ens  = [pipeline_ens.assess(generated_signals[n][0]).quality_score  for n in scenario_names]
        x = np.arange(len(scenario_names))
        ax_bar.barh(x - 0.2, scores_feat, 0.35, label="FeatureNet",  color="#378ADD", alpha=0.85)
        ax_bar.barh(x + 0.2, scores_ens,  0.35, label="EnsembleNet", color="#1D9E75", alpha=0.85)
        ax_bar.axvline(0.75, color="#1D9E75", lw=1, ls="--", alpha=0.6, label="ACCEPT")
        ax_bar.axvline(0.50, color="#EF9F27", lw=1, ls="--", alpha=0.6, label="CAUTION")
        ax_bar.set_yticks(x)
        ax_bar.set_yticklabels([n[:12] for n in scenario_names], fontsize=7)
        ax_bar.set_xlabel("Quality score", fontsize=8)
        ax_bar.set_title("Model Quality Scores", fontsize=9)
        ax_bar.legend(fontsize=6); ax_bar.tick_params(labelsize=7)

        plt.savefig("ppg_pipeline_demo.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to ppg_pipeline_demo.png")
        plt.show()

    except ImportError:
        print("matplotlib not installed — skipping plots")

print("\nDemo complete.")
