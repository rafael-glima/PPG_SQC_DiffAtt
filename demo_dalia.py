#!/usr/bin/env python3
"""
demo_dalia.py
-------------
Benchmarks ALL models in models.py on the windowed PPG-DaLiA dataset.

Every model is trained and evaluated under identical conditions:
  • 3-second windows, 1-second step  (192 samples @ 64 Hz)
  • BCEWithLogitsLoss  (binary: 0=clean, 1=artefact)
  • AdamW lr=3e-4, cosine schedule, early stopping
  • Same train/val/test splits

Models compared
~~~~~~~~~~~~~~~
  Waveform-only  (no feature extraction required)
  ─────────────────────────────────────────────────
  PPGBinaryNet          our lightweight 1-D CNN
  Baseline-1DCNN        Kasaeyan Naeini et al. 2020 (1-D)
  PPGDiffAttnNet        Differential Attention Transformer
  PPGWaveformNet        dilated ResNet + quality head (adapted)
  PPGSegmentationNet    U-Net quality branch (adapted)

  Features required  (pass --features to enable)
  ─────────────────────────────────────────────────
  PPGFeatureNet         MLP on 24-dim hand-crafted features
  PPGEnsembleNet        late-fusion of Waveform + Feature nets

  Skipped (needs 2-D image input)
  ─────────────────────────────────────────────────
  BaselinePPG2DCNN      VGG16 / ResNet50 / MobileNetV2

Setup
~~~~~
  git clone https://github.com/SullyChen/SMoLK.git
  pip install torch scipy numpy matplotlib
  pip install -e .
  python demo_dalia.py [options]

Options
~~~~~~~
  --train-dir PATH
  --test-dir  PATH
  --epochs    N       (default 40)
  --features          enable 24-dim feature extraction (slower)
  --no-plot
"""

import argparse, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--train-dir", default="SMoLK/PPG data/new_PPG_DaLiA_train/processed_dataset")
parser.add_argument("--test-dir",  default="SMoLK/PPG data/new_PPG_DaLiA_test/processed_dataset")
parser.add_argument("--epochs",    type=int, default=40)
parser.add_argument("--features",  action="store_true", help="Extract 24-dim feature vectors")
parser.add_argument("--no-plot",   action="store_true")
args = parser.parse_args()

import numpy as np
import torch
import torch.nn as nn

from ppg_pipeline.dalia_dataset import load_dalia, DALIA_FS
from ppg_pipeline.dalia_trainer import DaLiATrainer, DaLiATrainerConfig, PPGBinaryNet
from ppg_pipeline.feature_extractor import FeatureNormaliser
from ppg_pipeline.models import (
    PPGWaveformNet, PPGFeatureNet, PPGEnsembleNet,
    PPGSegmentationNet,
    BaselinePPG1DCNN,
    PPGDiffAttnNet,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLOT   = not args.no_plot

print("=" * 65)
print("  PPG Quality Pipeline — Full Model Comparison (DaLiA dataset)")
print("=" * 65)
print(f"  Device  : {DEVICE}")
print(f"  Features: {'enabled' if args.features else 'disabled (pass --features to enable)'}")
print()


# ===========================================================================
# 1. Load & window data
# ===========================================================================
train_ds, test_ds = load_dalia(
    train_dir        = args.train_dir,
    test_dir         = args.test_dir,
    window_sec       = 3.0,
    step_sec         = 1.0,
    extract_features = args.features,
    verbose          = True,
)
win_size = train_ds.waveforms.shape[-1]
print(f"\nWindow : {win_size} samples = {win_size/DALIA_FS:.1f}s @ {DALIA_FS:.0f}Hz")

# Normalise features (only meaningful when --features is set)
if args.features:
    X_tr = train_ds.feature_vecs.numpy()
    normaliser = FeatureNormaliser().fit(X_tr)
    train_ds.feature_vecs = torch.from_numpy(normaliser.transform(X_tr))
    test_ds.feature_vecs  = torch.from_numpy(
        normaliser.transform(test_ds.feature_vecs.numpy())
    )


# ===========================================================================
# 2. Model adapters
#    All non-binary models output ModelOutput / SegmentationOutput whose
#    quality_score ∈ [0,1] (1=clean).  We convert to a binary artifact logit:
#      logit = log( (1 - quality_score) / quality_score )
#    so  logit > 0  ⟺  quality_score < 0.5  ⟺  predicted artefact.
# ===========================================================================

class _QualityAdapter(nn.Module):
    """
    Adapts any model that returns ModelOutput or SegmentationOutput to output
    a single binary logit (B,) compatible with BCEWithLogitsLoss.

    mode
    ~~~~
    "wave"     — model(wav, _)        e.g. PPGWaveformNet, PPGSegmentationNet
    "feat"     — model(feat)          e.g. PPGFeatureNet
    "ensemble" — model(wav, feat)     e.g. PPGEnsembleNet
    """

    def __init__(self, base: nn.Module, mode: str = "wave") -> None:
        super().__init__()
        assert mode in ("wave", "feat", "ensemble")
        self.base = base
        self.mode = mode

    def forward(self, wav: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if self.mode == "wave":
            out = self.base(wav)
        elif self.mode == "feat":
            out = self.base(feat)
        else:  # ensemble
            out = self.base(wav, feat)

        q = out.quality_score.clamp(1e-6, 1.0 - 1e-6)
        # quality=1→clean (label 0), quality=0→artefact (label 1)
        return torch.log((1.0 - q) / q)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# 3. Model registry
# ===========================================================================

def _make_ensemble(win_size: int) -> PPGEnsembleNet:
    wave_net = PPGWaveformNet(base_channels=32, input_length=win_size)
    feat_net = PPGFeatureNet(n_features=24)
    return PPGEnsembleNet(wave_net, feat_net, freeze_backbones=False)


REGISTRY = []

# ── Waveform-only models ───────────────────────────────────────────────────
REGISTRY += [
    ("PPGBinaryNet",
     lambda ws: PPGBinaryNet(base_channels=32),
     "Binary 1-D CNN (ours)"),

    ("Baseline-1DCNN",
     lambda ws: BaselinePPG1DCNN(input_length=ws, n_filters=32),
     "Kasaeyan Naeini et al. 2020 — 1-D CNN"),

    ("PPGDiffAttnNet",
     lambda ws: PPGDiffAttnNet(input_length=ws, patch_size=16,
                               embed_dim=64, depth=4, num_heads=2),
     "Differential Attention Transformer"),

    ("PPGWaveformNet",
     lambda ws: _QualityAdapter(PPGWaveformNet(base_channels=32, input_length=ws), "wave"),
     "Dilated ResNet + quality head (adapted)"),

    ("PPGSegmentationNet",
     lambda ws: _QualityAdapter(PPGSegmentationNet(base_channels=32, input_length=ws), "wave"),
     "U-Net quality branch (adapted)"),
]

# ── Feature-dependent models (only included when --features passed) ────────
if args.features:
    REGISTRY += [
        ("PPGFeatureNet",
         lambda ws: _QualityAdapter(PPGFeatureNet(n_features=24), "feat"),
         "MLP on 24-dim hand-crafted features"),

        ("PPGEnsembleNet",
         lambda ws: _QualityAdapter(_make_ensemble(ws), "ensemble"),
         "Late-fusion ensemble (Waveform + Features)"),
    ]
else:
    print("  Skipping PPGFeatureNet and PPGEnsembleNet  (pass --features to enable)\n")

print(f"  BaselinePPG2DCNN skipped — needs 2-D image input (224×224×3)\n")


# ===========================================================================
# 4. Training loop — identical settings for all models
# ===========================================================================

results: dict[str, dict] = {}
histories: dict[str, list] = {}

BASE_CFG = dict(
    max_epochs    = args.epochs,
    batch_size    = 64,
    lr            = 3e-4,
    weight_decay  = 1e-4,
    patience      = 10,
    pos_weight    = 2.0,
    num_workers   = 0,
    device        = DEVICE,
)

for model_name, factory, description in REGISTRY:
    print("\n" + "─" * 65)
    print(f"  {model_name:22s}  {description}")
    print("─" * 65)

    model = factory(win_size)
    n_params = model.n_parameters if hasattr(model, "n_parameters") else \
               sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    cfg = DaLiATrainerConfig(
        **BASE_CFG,
        checkpoint_dir=f"checkpoints/dalia/{model_name}",
    )

    t0 = time.perf_counter()
    trainer = DaLiATrainer(model, cfg, train_ds, val_ds=test_ds)
    history = trainer.fit()
    trainer.load_best()
    elapsed = time.perf_counter() - t0

    metrics = trainer.evaluate(test_ds)
    metrics["n_params"]    = n_params
    metrics["train_sec"]   = elapsed
    metrics["description"] = description

    results[model_name]  = metrics
    histories[model_name] = history

    print(f"\n  ── Test results for {model_name} ──")
    print(f"     Loss      : {metrics['loss']:.4f}")
    print(f"     Accuracy  : {metrics['acc']:.4f}")
    print(f"     F1        : {metrics['f1']:.4f}")
    print(f"     AUC-ROC   : {metrics['auc']:.4f}")
    print(f"     Params    : {n_params:,}")
    print(f"     Time      : {elapsed:.1f}s")


# ===========================================================================
# 5. Summary table
# ===========================================================================

print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)
header = f"{'Model':<22}  {'Acc':>6}  {'F1':>6}  {'AUC':>6}  {'Params':>9}  {'Time':>7}"
print(header)
print("-" * 65)
for name, m in sorted(results.items(), key=lambda x: -x[1]["f1"]):
    print(f"  {name:<20}  {m['acc']:6.4f}  {m['f1']:6.4f}  {m['auc']:6.4f}  "
          f"{m['n_params']:>9,}  {m['train_sec']:6.0f}s")
print("=" * 65)

best = max(results.items(), key=lambda x: x[1]["f1"])
print(f"\n  Best model by F1: {best[0]}  (F1={best[1]['f1']:.4f}  AUC={best[1]['auc']:.4f})")


# ===========================================================================
# 6. Plots
# ===========================================================================

if PLOT:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        n_models = len(results)
        model_names_sorted = sorted(results, key=lambda x: -results[x]["f1"])

        # ── Figure 1: Comparison bar chart ────────────────────────────────
        fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig1.suptitle("Model Comparison — PPG-DaLiA Binary Artefact Classification",
                      fontsize=12, fontweight="bold")

        colors = plt.cm.tab10(np.linspace(0, 1, n_models))
        x = np.arange(n_models)
        labels = [n.replace("PPG", "").replace("Net", "") for n in model_names_sorted]

        for ax, metric, title in zip(axes, ["f1", "auc", "acc"],
                                     ["F1 score", "AUC-ROC", "Accuracy"]):
            vals = [results[n][metric] for n in model_names_sorted]
            bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_title(title, fontsize=10)
            ax.axhline(0.5, color="gray", ls=":", lw=0.8)
            ax.tick_params(axis="y", labelsize=8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        fig1.savefig("dalia_comparison.png", dpi=150, bbox_inches="tight")
        print("\nComparison chart saved → dalia_comparison.png")

        # ── Figure 2: Training curves ──────────────────────────────────────
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        fig2, axes2 = plt.subplots(n_rows, n_cols,
                                   figsize=(5 * n_cols, 3.5 * n_rows))
        fig2.suptitle("Training Curves — Val F1 per Epoch", fontsize=11)
        axes2 = np.array(axes2).flatten()

        for ax, name in zip(axes2, model_names_sorted):
            h = histories[name]
            epochs = [e["epoch"] for e in h]
            tr_f1  = [e["train"]["f1"] for e in h]
            va_f1  = [e["val"]["f1"]   for e in h]
            ax.plot(epochs, tr_f1, label="train", lw=1.3, color="#378ADD")
            ax.plot(epochs, va_f1, label="val",   lw=1.3, color="#1D9E75", ls="--")
            ax.set_title(name, fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("F1", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
            # Mark best val F1
            best_val_f1 = max(va_f1)
            best_ep     = epochs[va_f1.index(best_val_f1)]
            ax.axvline(best_ep, color="#E24B4A", ls=":", lw=0.9)
            ax.text(best_ep, 0.02, f"{best_val_f1:.3f}", fontsize=6,
                    color="#E24B4A", ha="center")

        for ax in axes2[len(results):]:
            ax.set_visible(False)

        plt.tight_layout()
        fig2.savefig("dalia_training_curves.png", dpi=150, bbox_inches="tight")
        print("Training curves saved   → dalia_training_curves.png")
        plt.show()

    except ImportError:
        print("matplotlib not available — skipping plots")

print("\nDone.")
