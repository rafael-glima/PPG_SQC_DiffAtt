"""
models.py
---------
PyTorch models for PPG signal quality assessment.

Three architectures
~~~~~~~~~~~~~~~~~~~
PPGWaveformNet  – 1-D CNN operating directly on the raw/preprocessed waveform.
                  Suited for end-to-end learning without hand-crafted features.

PPGFeatureNet   – MLP operating on the 24-dimensional extracted feature vector.
                  Interpretable, fast, works with small training sets.

PPGEnsembleNet  – Late-fusion ensemble: runs both branches and combines with a
                  learned gating network. Best accuracy in practice.

Both PPGWaveformNet and PPGFeatureNet share a common output head
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  quality_score  : float in [0, 1]  (sigmoid)
  verdict_logits : shape (3,)       → ACCEPT / CAUTION / REJECT
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class ModelOutput(NamedTuple):
    quality_score: torch.Tensor   # shape (B,)   float [0, 1]
    verdict_logits: torch.Tensor  # shape (B, 3) raw logits


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """1-D residual block with optional downsampling."""

    def __init__(self, channels: int, kernel: int = 7, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.GELU(),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, L)
        w = x.mean(dim=-1)          # global average pool → (B, C)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class QualityHead(nn.Module):
    """Shared output head: quality score regression + verdict classification."""

    def __init__(self, in_features: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )
        self.verdict_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.shared(x)
        score = self.score_head(h).squeeze(-1)      # (B,)
        logits = self.verdict_head(h)               # (B, 3)
        return ModelOutput(quality_score=score, verdict_logits=logits)


# ---------------------------------------------------------------------------
# Model 1: PPGWaveformNet (1-D CNN)
# ---------------------------------------------------------------------------

class PPGWaveformNet(nn.Module):
    """
    End-to-end 1-D CNN for PPG quality assessment.

    Architecture
    ~~~~~~~~~~~~
    Stem conv → 4 residual stages with increasing dilation → SE attention →
    global pooling → QualityHead

    Parameters
    ----------
    in_channels : int
        Number of input channels. 1 for single-lead PPG; 2 if raw+filtered.
    base_channels : int
        Feature channels in first stage (doubles at each stage up to 4×).
    input_length : int
        Expected waveform length (used for architecture checks only).
    dropout : float
        Dropout probability in residual blocks and head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        input_length: int = 500,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        C = base_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, C, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(C),
            nn.GELU(),
        )

        # Dilated residual stages
        self.stage1 = nn.Sequential(
            ResBlock1D(C, kernel=7, dilation=1, dropout=dropout),
            ResBlock1D(C, kernel=7, dilation=2, dropout=dropout),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(C, C * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C * 2),
            nn.GELU(),
        )

        self.stage2 = nn.Sequential(
            ResBlock1D(C * 2, kernel=7, dilation=1, dropout=dropout),
            ResBlock1D(C * 2, kernel=7, dilation=4, dropout=dropout),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(C * 2, C * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C * 4),
            nn.GELU(),
        )

        self.stage3 = nn.Sequential(
            ResBlock1D(C * 4, kernel=5, dilation=1, dropout=dropout),
            ResBlock1D(C * 4, kernel=5, dilation=8, dropout=dropout),
        )
        self.down3 = nn.Sequential(
            nn.Conv1d(C * 4, C * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C * 4),
            nn.GELU(),
        )

        self.stage4 = ResBlock1D(C * 4, kernel=3, dilation=1, dropout=dropout)
        self.se = SEBlock(C * 4)

        # Global statistics pooling (mean + std)
        self.pool_dim = C * 4 * 2

        self.head = QualityHead(self.pool_dim, hidden=128, dropout=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, C, L)
            Waveform tensor. C=1 for single PPG channel.
        """
        h = self.stem(x)
        h = self.stage1(h);  h = self.down1(h)
        h = self.stage2(h);  h = self.down2(h)
        h = self.stage3(h);  h = self.down3(h)
        h = self.stage4(h)
        h = self.se(h)

        # Statistics pooling over time dimension
        mean = h.mean(dim=-1)
        std = h.std(dim=-1)
        pooled = torch.cat([mean, std], dim=-1)

        return self.head(pooled)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Model 2: PPGFeatureNet (MLP on extracted features)
# ---------------------------------------------------------------------------

class PPGFeatureNet(nn.Module):
    """
    Feed-forward MLP operating on the 24-dimensional extracted feature vector.

    Architecture
    ~~~~~~~~~~~~
    Input → BN → FC(128) → GELU → Dropout → FC(64) → GELU → QualityHead

    Parameters
    ----------
    n_features : int
        Input feature dimensionality (default 24).
    hidden_dims : list[int]
        Hidden layer sizes.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_features: int = 24,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        self.input_bn = nn.BatchNorm1d(n_features)

        layers: list[nn.Module] = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = QualityHead(in_dim, hidden=64, dropout=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, n_features)
        """
        x = self.input_bn(x)
        h = self.backbone(x)
        return self.head(h)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Model 3: PPGEnsembleNet (late fusion)
# ---------------------------------------------------------------------------

class PPGEnsembleNet(nn.Module):
    """
    Late-fusion ensemble combining CNN waveform features and MLP feature-vector outputs.

    Both sub-models are run in parallel; their embeddings are concatenated and
    passed through a learned gating MLP that produces the final prediction.

    Parameters
    ----------
    waveform_net : PPGWaveformNet
    feature_net  : PPGFeatureNet
    freeze_backbones : bool
        If True, freeze backbone weights and only train the fusion layer.
        Useful for fine-tuning on small datasets.
    """

    def __init__(
        self,
        waveform_net: PPGWaveformNet,
        feature_net: PPGFeatureNet,
        freeze_backbones: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.waveform_net = waveform_net
        self.feature_net = feature_net

        if freeze_backbones:
            for p in self.waveform_net.parameters():
                p.requires_grad = False
            for p in self.feature_net.parameters():
                p.requires_grad = False

        # Replace heads with embedding extractors
        wave_emb_dim = waveform_net.pool_dim
        feat_emb_dim = feature_net.backbone[-4].out_features  # last Linear before head

        self.fusion = nn.Sequential(
            nn.Linear(wave_emb_dim + feat_emb_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = QualityHead(64, hidden=64, dropout=dropout)

    def _waveform_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled embedding from waveform CNN (before head)."""
        net = self.waveform_net
        h = net.stem(x)
        h = net.stage1(h);  h = net.down1(h)
        h = net.stage2(h);  h = net.down2(h)
        h = net.stage3(h);  h = net.down3(h)
        h = net.stage4(h)
        h = net.se(h)
        mean = h.mean(dim=-1)
        std = h.std(dim=-1)
        return torch.cat([mean, std], dim=-1)

    def _feature_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding from feature MLP (before head)."""
        net = self.feature_net
        h = net.input_bn(x)
        h = net.backbone(h)
        return h

    def forward(
        self,
        waveform: torch.Tensor,
        features: torch.Tensor,
    ) -> ModelOutput:
        """
        Parameters
        ----------
        waveform : torch.Tensor, shape (B, C, L)
        features : torch.Tensor, shape (B, n_features)
        """
        w_emb = self._waveform_embed(waveform)
        f_emb = self._feature_embed(features)
        fused = torch.cat([w_emb, f_emb], dim=-1)
        h = self.fusion(fused)
        return self.head(h)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class PPGQualityLoss(nn.Module):
    """
    Multi-task loss for joint quality score regression and verdict classification.

    L = α · MSE(score, score_gt)  +  β · CE(logits, verdict_gt)

    Parameters
    ----------
    score_weight : float
        Weight for the regression loss (α). Default 1.0.
    verdict_weight : float
        Weight for the classification loss (β). Default 1.0.
    label_smoothing : float
        Label smoothing for cross-entropy. Default 0.05.
    """

    def __init__(
        self,
        score_weight: float = 1.0,
        verdict_weight: float = 1.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.score_weight = score_weight
        self.verdict_weight = verdict_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mse = nn.MSELoss()

    def forward(
        self,
        output: ModelOutput,
        score_gt: torch.Tensor,
        verdict_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys: total, score_loss, verdict_loss
        """
        score_loss = self.mse(output.quality_score, score_gt.float())
        verdict_loss = self.ce(output.verdict_logits, verdict_gt.long())
        total = self.score_weight * score_loss + self.verdict_weight * verdict_loss
        return {
            "total": total,
            "score_loss": score_loss,
            "verdict_loss": verdict_loss,
        }


# ---------------------------------------------------------------------------
# Segmentation output container
# ---------------------------------------------------------------------------

class SegmentationOutput(NamedTuple):
    quality_score:   torch.Tensor   # (B,)     segment-level quality [0,1]
    verdict_logits:  torch.Tensor   # (B, 3)   ACCEPT/CAUTION/REJECT
    artifact_logits: torch.Tensor   # (B, L)   per-sample artifact probability logits


# ---------------------------------------------------------------------------
# PPGSegmentationNet: CNN with dual head (quality + per-sample segmentation)
# ---------------------------------------------------------------------------

class PPGSegmentationNet(nn.Module):
    """
    End-to-end 1-D CNN for joint quality assessment and artifact segmentation.

    Architecture
    ~~~~~~~~~~~~
    Shared dilated ResNet encoder → two decoders:
      (a) QualityHead   – segment-level score + verdict (same as PPGWaveformNet)
      (b) SegHead       – per-sample sigmoid logits for artifact mask prediction

    The segmentation head uses transposed convolutions to upsample back to the
    original signal length after the three stride-2 downsampling stages.

    Parameters
    ----------
    in_channels : int
    base_channels : int
    input_length : int
        Must be divisible by 8 (three stride-2 layers). Default 256 for DaLiA.
    dropout : float
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        input_length: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        C = base_channels
        self.input_length = input_length

        # ---- Shared encoder (identical to PPGWaveformNet) ----
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, C, 15, padding=7, bias=False),
            nn.BatchNorm1d(C), nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            ResBlock1D(C, 7, dilation=1, dropout=dropout),
            ResBlock1D(C, 7, dilation=2, dropout=dropout),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(C, C*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C*2), nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            ResBlock1D(C*2, 7, dilation=1, dropout=dropout),
            ResBlock1D(C*2, 7, dilation=4, dropout=dropout),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(C*2, C*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C*4), nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            ResBlock1D(C*4, 5, dilation=1, dropout=dropout),
            ResBlock1D(C*4, 5, dilation=8, dropout=dropout),
        )
        self.down3 = nn.Sequential(
            nn.Conv1d(C*4, C*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C*4), nn.GELU(),
        )
        self.stage4 = ResBlock1D(C*4, 3, dilation=1, dropout=dropout)
        self.se     = SEBlock(C*4)

        pool_dim = C*4*2   # mean + std

        # ---- Quality head ----
        self.quality_head = QualityHead(pool_dim, hidden=128, dropout=dropout)

        # ---- Segmentation decoder: upsample x8 back to input_length ----
        self.seg_up1 = nn.Sequential(
            nn.ConvTranspose1d(C*4, C*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C*2), nn.GELU(),
        )
        self.seg_up2 = nn.Sequential(
            nn.ConvTranspose1d(C*2, C, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C), nn.GELU(),
        )
        self.seg_up3 = nn.Sequential(
            nn.ConvTranspose1d(C, C, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(C), nn.GELU(),
        )
        self.seg_out = nn.Conv1d(C, 1, kernel_size=1)   # (B, 1, L) → artifact logits

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> SegmentationOutput:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, L)

        Returns
        -------
        SegmentationOutput
          .quality_score   (B,)
          .verdict_logits  (B, 3)
          .artifact_logits (B, L)   — pass through sigmoid for probabilities
        """
        h = self.stem(x)
        h = self.stage1(h)
        e1 = h                      # skip connection
        h = self.down1(h)

        h = self.stage2(h)
        e2 = h
        h = self.down2(h)

        h = self.stage3(h)
        h = self.down3(h)
        h = self.stage4(h)
        h = self.se(h)

        # Quality branch: global statistics pooling
        pooled = torch.cat([h.mean(-1), h.std(-1)], dim=-1)
        quality_out = self.quality_head(pooled)

        # Segmentation decoder with skip connections
        d = self.seg_up1(h)
        # Crop/pad to match skip size if needed
        d = d[..., :e2.shape[-1]] if d.shape[-1] > e2.shape[-1] else d
        d = d + e2[..., :d.shape[-1]]

        d = self.seg_up2(d)
        d = d[..., :e1.shape[-1]] if d.shape[-1] > e1.shape[-1] else d
        d = d + e1[..., :d.shape[-1]]

        d = self.seg_up3(d)
        # Final crop/pad to match input length exactly
        L = x.shape[-1]
        if d.shape[-1] > L:
            d = d[..., :L]
        elif d.shape[-1] < L:
            d = F.pad(d, (0, L - d.shape[-1]))

        artifact_logits = self.seg_out(d).squeeze(1)   # (B, L)

        return SegmentationOutput(
            quality_score=quality_out.quality_score,
            verdict_logits=quality_out.verdict_logits,
            artifact_logits=artifact_logits,
        )

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Joint loss for segmentation + quality assessment
# ---------------------------------------------------------------------------

class PPGSegmentationLoss(nn.Module):
    """
    Multi-task loss for joint artifact segmentation and quality assessment.

    L = α · BCE(artifact_logits, mask)
      + β · MSE(quality_score, quality_gt)
      + γ · CE(verdict_logits, verdict_gt)

    Parameters
    ----------
    seg_weight     : float   weight for per-sample artifact BCE  (α)
    score_weight   : float   weight for quality score MSE         (β)
    verdict_weight : float   weight for verdict CE                (γ)
    pos_weight     : float   upweight artifact class in BCE (handles imbalance)
    """

    def __init__(
        self,
        seg_weight:     float = 1.0,
        score_weight:   float = 0.5,
        verdict_weight: float = 0.5,
        pos_weight:     float = 2.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.seg_weight     = seg_weight
        self.score_weight   = score_weight
        self.verdict_weight = verdict_weight
        self.register_buffer("pw", torch.tensor([pos_weight]))
        self.ce  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mse = nn.MSELoss()

    def forward(
        self,
        output: SegmentationOutput,
        mask_gt:    torch.Tensor,   # (B, L) binary float
        score_gt:   torch.Tensor,   # (B,)
        verdict_gt: torch.Tensor,   # (B,) long
    ) -> dict[str, torch.Tensor]:
        bce = F.binary_cross_entropy_with_logits(
            output.artifact_logits, mask_gt.float(),
            pos_weight=self.pw.to(output.artifact_logits.device),
        )
        score_loss   = self.mse(output.quality_score, score_gt.float())
        verdict_loss = self.ce(output.verdict_logits, verdict_gt.long())
        total = (
            self.seg_weight     * bce
            + self.score_weight   * score_loss
            + self.verdict_weight * verdict_loss
        )
        return {
            "total":        total,
            "seg_loss":     bce,
            "score_loss":   score_loss,
            "verdict_loss": verdict_loss,
        }

# ===========================================================================
# Baseline models — PyTorch ports of Kasaeyan Naeini et al. (2020)
#
# Original: classification.py  (Keras/TensorFlow)
# Source  : create_1Dcnn() and create_2Dcnn() functions
#
# Architecture notes
# ~~~~~~~~~~~~~~~~~~
# The original 1-D CNN was designed for 5-minute PPG signals at 20 Hz
# (input length = 60 s × 20 Hz × 5 min = 1 200 samples).  The class here
# accepts any input_length so it can be used directly on the 3-second
# DaLiA windows (192 samples @ 64 Hz) without modification.
#
# The original 2-D CNN used pretrained ImageNet backbones (VGG16, ResNet50,
# MobileNetV2) with frozen weights and a small classification head.  The
# PyTorch port uses torchvision equivalents and the same freeze strategy.
#
# Output convention (shared with the rest of this module)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# All baseline models output a single raw logit per sample (B,).
# Pass through torch.sigmoid() for probability, or use BCEWithLogitsLoss
# directly during training — identical to PPGBinaryNet.
# ===========================================================================


# ---------------------------------------------------------------------------
# Baseline 1-D CNN  (port of create_1Dcnn)
# ---------------------------------------------------------------------------

class BaselinePPG1DCNN(nn.Module):
    """
    PyTorch port of the 1-D CNN baseline from Kasaeyan Naeini et al. (2020).

    Original Keras architecture
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Conv1D(32, 3, relu) → BN → Conv1D(64, 3, relu) → BN →
    MaxPool1D(2) → Flatten → Dense(348, relu) → BN → Dense(2, sigmoid)

    Ported changes
    ~~~~~~~~~~~~~~
    - Two-class sigmoid head collapsed to a single logit output (B,) so the
      model integrates with BCEWithLogitsLoss used throughout this package.
    - Input length is parameterised (default 1200 for the original 5-min @
      20 Hz setting; use 192 for 3-second DaLiA windows @ 64 Hz).
    - He-normal initialisation preserved (kaiming_normal in PyTorch).
    - Adam lr=1e-5 was in the original; use the trainer's configured lr.

    Parameters
    ----------
    input_length : int
        Number of time-steps in each window.  Original paper: 1200 (5 min @
        20 Hz).  DaLiA 3-second windows: 192 (3 s @ 64 Hz).
    n_filters : int
        Number of filters in the first conv layer (doubles in second layer).
        Original: 32.
    """

    def __init__(self, input_length: int = 1200, n_filters: int = 32) -> None:
        super().__init__()
        self.input_length = input_length

        self.conv_block = nn.Sequential(
            # Block 1 — Conv1D(n_filters, 3, relu) + BN
            nn.Conv1d(1, n_filters, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters),

            # Block 2 — Conv1D(n_filters*2, 3, relu) + BN
            nn.Conv1d(n_filters, n_filters * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters * 2),

            # MaxPool1D(2)
            nn.MaxPool1d(kernel_size=2),
        )

        # Compute flattened size after conv + pool
        flat_size = (input_length // 2) * (n_filters * 2)

        self.classifier = nn.Sequential(
            # Dense(348, relu) + BN
            nn.Linear(flat_size, 348),
            nn.ReLU(),
            nn.BatchNorm1d(348),

            # Dense(1) — single logit (replaces Dense(2, sigmoid))
            nn.Linear(348, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # He-normal (matches kernel_initializer='he_normal' in Keras)
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, features=None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, L)   — single-channel PPG window

        Returns
        -------
        logit : (B,)   — raw logit; sigmoid → P(artefact)
        """
        h = self.conv_block(x)
        h = h.flatten(start_dim=1)
        return self.classifier(h).squeeze(-1)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Baseline 2-D CNN  (port of create_2Dcnn)
# ---------------------------------------------------------------------------

class BaselinePPG2DCNN(nn.Module):
    """
    PyTorch port of the 2-D CNN baseline from Kasaeyan Naeini et al. (2020).

    Original Keras architecture
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Frozen ImageNet backbone (VGG16 / ResNet50 / MobileNetV2) →
    Flatten → Dense(128, relu) → BN → Dense(64, relu) → Dense(2, sigmoid)

    Backbone options (torchvision equivalents)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    "VGG16"       → torchvision.models.vgg16
    "ResNet50"    → torchvision.models.resnet50
    "MobileNetV2" → torchvision.models.mobilenet_v2

    Ported changes
    ~~~~~~~~~~~~~~
    - Single logit output (B,) replacing Dense(2, sigmoid).
    - Backbone weights frozen by default (freeze_backbone=True), matching
      ``layer.trainable = False`` in the original.
    - Input expected as (B, 3, 224, 224) RGB images (same as original).

    Parameters
    ----------
    backbone_name : str
        One of "VGG16", "ResNet50", "MobileNetV2".
    freeze_backbone : bool
        If True, backbone weights are frozen (only head is trained).
        Set to False to fine-tune the full network.
    pretrained : bool
        Load ImageNet weights (default True).  Set False for unit tests.
    """

    _BACKBONE_NAMES = {"VGG16", "ResNet50", "MobileNetV2"}

    def __init__(
        self,
        backbone_name:   str  = "MobileNetV2",
        freeze_backbone: bool = True,
        pretrained:      bool = True,
    ) -> None:
        super().__init__()
        if backbone_name not in self._BACKBONE_NAMES:
            raise ValueError(
                f"backbone_name must be one of {self._BACKBONE_NAMES}, "
                f"got '{backbone_name}'."
            )
        self.backbone_name = backbone_name

        try:
            import torchvision.models as tv_models
        except ImportError as e:
            raise ImportError(
                "torchvision is required for BaselinePPG2DCNN.\n"
                "Install it with:  pip install torchvision"
            ) from e

        weights = "IMAGENET1K_V1" if pretrained else None

        # ── Load backbone ────────────────────────────────────────────────
        if backbone_name == "VGG16":
            backbone = tv_models.vgg16(weights=weights)
            # Remove VGG's own classifier; keep only features (conv layers)
            self.backbone = backbone.features
            # VGG16 features output: (B, 512, 7, 7) for 224×224 input
            backbone_out_features = 512 * 7 * 7

        elif backbone_name == "ResNet50":
            backbone = tv_models.resnet50(weights=weights)
            # Strip the final FC layer; keep everything up to avgpool
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            # ResNet50 avgpool output: (B, 2048, 1, 1) → flatten → 2048
            backbone_out_features = 2048

        else:  # MobileNetV2
            backbone = tv_models.mobilenet_v2(weights=weights)
            self.backbone = backbone.features
            # MobileNetV2 features output: (B, 1280, 7, 7) for 224×224 input
            # followed by adaptive avg pool → (B, 1280, 1, 1)
            self.backbone = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            backbone_out_features = 1280

        # ── Freeze backbone weights ──────────────────────────────────────
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Classification head ──────────────────────────────────────────
        # Matches: Flatten → Dense(128,relu) → BN → Dense(64,relu) → Dense(1)
        self.head = nn.Sequential(
            nn.Linear(backbone_out_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),   # single logit (replaces Dense(2, sigmoid))
        )

        self._init_head()

    def _init_head(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # glorot_uniform in Keras
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 224, 224)   — RGB image tensor, pixel values in [0, 1]

        Returns
        -------
        logit : (B,)   — raw logit; sigmoid → P(artefact)
        """
        features = self.backbone(x)
        flat = features.flatten(start_dim=1)
        return self.head(flat).squeeze(-1)

    def unfreeze_backbone(self) -> None:
        """Enable gradient flow through backbone for full fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def n_parameters_total(self) -> int:
        return sum(p.numel() for p in self.parameters())

# ===========================================================================
# PPGDiffAttnNet — Differential Attention Transformer for PPG classification
#
# Based on: "Differential Transformer" (Microsoft Research, 2024)
#           arXiv:2410.05258
# Original: multihead_diffattn.py  (MultiheadDiffAttn class)
#
# External dependencies removed / replaced
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# flash_attn        → standard torch.matmul attention (imported but unused
#                     in the original forward pass)
# apex.FusedRMSNorm → pure-PyTorch RMSNorm (below)
# kernel.rotary     → self-contained RoPE implementation (below)
#
# Architecture overview
# ~~~~~~~~~~~~~~~~~~~~~
# 1. Patch embedding  : split PPG window into fixed patches → Linear projection
# 2. CLS token        : prepended learnable classification token
# 3. N × DiffAttn blocks: LayerNorm → MultiheadDiffAttn → residual
#                          LayerNorm → FFN (SwiGLU) → residual
# 4. Head             : CLS token → Linear → single logit (B,)
#
# Output convention (shared with all models in this module)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Raw logit (B,).  Use torch.sigmoid() for probability or BCEWithLogitsLoss
# during training — identical to PPGBinaryNet and the two baseline models.
# ===========================================================================


# ---------------------------------------------------------------------------
# Building block 1 — RMSNorm (replaces apex.FusedRMSNorm)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation.

    Equivalent to apex.normalization.FusedRMSNorm with elementwise_affine=True.
    Identical interface to nn.LayerNorm but without mean-centering.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))   # elementwise_affine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.scale


# ---------------------------------------------------------------------------
# Building block 2 — Rotary Position Embedding (replaces kernel.rotary)
# ---------------------------------------------------------------------------

def _precompute_rope(head_dim: int, max_len: int, base: float = 10_000.0,
                     device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute cosine / sine tables for Rotary Position Embeddings (RoPE).

    Returns
    -------
    cos, sin : each shape (max_len, head_dim)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos   = torch.arange(max_len, device=device).float()
    freqs = torch.outer(pos, theta)              # (max_len, half)
    emb   = torch.cat([freqs, freqs], dim=-1)    # (max_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_emb(
    x:   torch.Tensor,        # (B, seq_len, n_heads, head_dim)
    cos: torch.Tensor,        # (seq_len, head_dim)
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to query or key tensor.

    Equivalent to apply_rotary_emb(..., interleaved=True) from the original
    kernel.rotary, but implemented entirely in standard PyTorch.
    """
    cos = cos[:x.shape[1]].unsqueeze(0).unsqueeze(2)   # (1, T, 1, D)
    sin = sin[:x.shape[1]].unsqueeze(0).unsqueeze(2)
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Building block 3 — lambda initialiser (unchanged from original)
# ---------------------------------------------------------------------------

def _lambda_init(depth: int) -> float:
    """Per-layer lambda initialisation from the paper: 0.8 − 0.6·exp(−0.3·depth)."""
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


# ---------------------------------------------------------------------------
# Building block 4 — MultiheadDiffAttn (ported from multihead_diffattn.py)
# ---------------------------------------------------------------------------

class _MultiheadDiffAttn(nn.Module):
    """
    Differential Multi-head Attention.

    Ported from MultiheadDiffAttn in multihead_diffattn.py with three changes:
      1. RMSNorm is the self-contained _RMSNorm defined above.
      2. Rotary embeddings use _apply_rotary_emb instead of kernel.rotary.
      3. flash_attn_func removed; standard scaled-dot-product attention used.

    Parameters
    ----------
    embed_dim : int
        Total model dimension.
    depth : int
        Zero-based layer index — controls lambda initialisation.
    num_heads : int
        Number of *differential* attention heads.  Set to half the number of
        heads in the equivalent standard Transformer (see paper §3).
    num_kv_heads : int | None
        For Grouped Query Attention.  None → standard MHA (num_kv_heads = num_heads).
    """

    def __init__(
        self,
        embed_dim:    int,
        depth:        int,
        num_heads:    int,
        num_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim    = embed_dim
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep        = self.num_heads // self.num_kv_heads
        self.head_dim     = embed_dim // num_heads // 2   # half-dim per sub-head
        self.scaling      = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(embed_dim, embed_dim,                       bias=False)
        self.k_proj   = nn.Linear(embed_dim, embed_dim // self.n_rep,         bias=False)
        self.v_proj   = nn.Linear(embed_dim, embed_dim // self.n_rep,         bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim,                       bias=False)

        # Learnable differential scalar (λ₁ − λ₂ + λ_init)
        self.lambda_init = _lambda_init(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))

        # Sub-layer norm applied to each head's output (original: RMSNorm(2*head_dim))
        self.subln = _RMSNorm(2 * self.head_dim)

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Expand key/value heads to match query heads (GQA → MHA)."""
        B, n_kv, T, D = x.shape
        if n_rep == 1:
            return x
        return (x.unsqueeze(2)
                 .expand(B, n_kv, n_rep, T, D)
                 .reshape(B, n_kv * n_rep, T, D))

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        x:         torch.Tensor,                  # (B, T, embed_dim)
        cos:       torch.Tensor,                  # (T, head_dim) RoPE tables
        sin:       torch.Tensor,
        attn_mask: torch.Tensor | None = None,    # (T, T) causal mask or None
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x)   # (B, T, embed_dim)
        k = self.k_proj(x)   # (B, T, embed_dim // n_rep)
        v = self.v_proj(x)   # (B, T, embed_dim // n_rep)

        # Reshape into heads — each head has TWO sub-heads of size head_dim
        q = q.view(B, T, 2 * self.num_heads,    self.head_dim)
        k = k.view(B, T, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(B, T,     self.num_kv_heads,  2 * self.head_dim)

        # Apply Rotary Position Embeddings
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # (B, n_heads*2, T, head_dim)
        q = q.transpose(1, 2)
        k = self._repeat_kv(k.transpose(1, 2), self.n_rep)
        v = self._repeat_kv(v.transpose(1, 2), self.n_rep)

        # Scaled dot-product attention
        q = q * self.scaling
        attn = torch.matmul(q, k.transpose(-1, -2))   # (B, 2*H, T, T)

        if attn_mask is None:
            # Causal mask (lower-triangular)
            attn_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=attn.dtype),
                diagonal=1,
            )
        attn = torch.nan_to_num(attn) + attn_mask
        attn = F.softmax(attn.float(), dim=-1).type_as(q)

        # Differential attention: λ = exp(q1·k1) − exp(q2·k2) + λ_init
        lam1 = torch.exp(
            (self.lambda_q1.float() * self.lambda_k1.float()).sum()
        ).type_as(q)
        lam2 = torch.exp(
            (self.lambda_q2.float() * self.lambda_k2.float()).sum()
        ).type_as(q)
        lam = lam1 - lam2 + self.lambda_init

        # Split the 2H heads into H pairs and subtract
        attn = attn.view(B, self.num_heads, 2, T, T)
        attn = attn[:, :, 0] - lam * attn[:, :, 1]   # (B, H, T, T)

        out = torch.matmul(attn, v)                   # (B, H, T, 2*head_dim)
        out = self.subln(out)
        out = out * (1 - self.lambda_init)
        out = out.transpose(1, 2).reshape(B, T, self.embed_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Building block 5 — SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class _SwiGLUFFN(nn.Module):
    """
    Feed-forward block with SwiGLU activation.

    SwiGLU(x) = SiLU(xW₁) ⊙ (xW₂),  projected back with W₃.
    Uses 4/3 × embed_dim hidden size (same param count as standard 4× FFN).
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(embed_dim * 4 / 3 * 2)   # SwiGLU hidden (two parallel gates)
        self.w1  = nn.Linear(embed_dim, hidden // 2, bias=False)
        self.w2  = nn.Linear(embed_dim, hidden // 2, bias=False)
        self.w3  = nn.Linear(hidden // 2, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ---------------------------------------------------------------------------
# Building block 6 — DiffAttn Transformer block
# ---------------------------------------------------------------------------

class _DiffAttnBlock(nn.Module):
    """One transformer block: pre-norm DiffAttn + pre-norm SwiGLU FFN."""

    def __init__(
        self,
        embed_dim:    int,
        depth:        int,
        num_heads:    int,
        num_kv_heads: int | None = None,
        ffn_dropout:  float      = 0.0,
        attn_dropout: float      = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(embed_dim)
        self.attn  = _MultiheadDiffAttn(embed_dim, depth, num_heads, num_kv_heads)
        self.norm2 = _RMSNorm(embed_dim)
        self.ffn   = _SwiGLUFFN(embed_dim, ffn_dropout)

    def forward(
        self,
        x:         torch.Tensor,
        cos:       torch.Tensor,
        sin:       torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# PPGDiffAttnNet — full model
# ---------------------------------------------------------------------------

class PPGDiffAttnNet(nn.Module):
    """
    Differential Attention Transformer for PPG quality classification.

    Pipeline
    ~~~~~~~~
    (B, 1, L)   PPG window
        │
        ▼  Patch embedding (1-D Conv, non-overlapping patches of size `patch_size`)
    (B, n_patches, embed_dim)
        │
        ▼  Prepend CLS token
    (B, n_patches+1, embed_dim)
        │
        ▼  N × _DiffAttnBlock  (RoPE applied inside each block)
        │
        ▼  RMSNorm → CLS token → Linear → scalar logit
    (B,)

    Parameters
    ----------
    input_length : int
        Number of samples per PPG window.  E.g. 192 for 3-second @ 64 Hz.
    patch_size : int
        Samples per patch.  Must divide input_length evenly.  Default 16.
    embed_dim : int
        Transformer model dimension.  Must be divisible by (num_heads * 2).
    depth : int
        Number of transformer blocks.
    num_heads : int
        Differential attention heads per block.  Equivalent to *half* the
        number of standard MHA heads (see paper §3).  Must divide embed_dim//2.
    num_kv_heads : int | None
        For Grouped Query Attention.  None → MHA.
    dropout : float
        Dropout applied in FFN and on patch embeddings.

    Output
    ------
    Raw logit (B,).  Use torch.sigmoid() or BCEWithLogitsLoss.

    Examples
    --------
    >>> model = PPGDiffAttnNet(input_length=192, patch_size=16, embed_dim=64,
    ...                        depth=4, num_heads=2)
    >>> x = torch.randn(8, 1, 192)
    >>> logits = model(x)   # shape (8,)
    """

    def __init__(
        self,
        input_length: int   = 192,
        patch_size:   int   = 16,
        embed_dim:    int   = 64,
        depth:        int   = 4,
        num_heads:    int   = 2,
        num_kv_heads: int | None = None,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        assert input_length % patch_size == 0, (
            f"input_length ({input_length}) must be divisible by patch_size ({patch_size})"
        )
        assert embed_dim % (num_heads * 2) == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads*2 ({num_heads*2})"
        )

        self.input_length = input_length
        self.patch_size   = patch_size
        self.embed_dim    = embed_dim
        self.n_patches    = input_length // patch_size
        seq_len           = self.n_patches + 1   # +1 for CLS token

        # Patch embedding via strided 1-D convolution
        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.LayerNorm(embed_dim),   # applied after transpose in forward
        )
        self.embed_drop = nn.Dropout(dropout)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            _DiffAttnBlock(
                embed_dim    = embed_dim,
                depth        = i,
                num_heads    = num_heads,
                num_kv_heads = num_kv_heads,
                ffn_dropout  = dropout,
            )
            for i in range(depth)
        ])

        # Pre-compute RoPE tables for the maximum sequence length
        head_dim = embed_dim // num_heads // 2
        cos, sin = _precompute_rope(head_dim, max_len=seq_len)
        self.register_buffer("rope_cos", cos)   # (seq_len, head_dim)
        self.register_buffer("rope_sin", sin)

        self.norm = _RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1, bias=True)

        self._init_weights()

    # ── initialisation ───────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, features=None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, L)   single-channel PPG window

        Returns
        -------
        logit : (B,)
        """
        B = x.shape[0]

        # Patch embedding: Conv1d → (B, embed_dim, n_patches) → transpose
        h = self.patch_embed[0](x)          # (B, D, P)
        h = h.transpose(1, 2)               # (B, P, D)
        h = self.patch_embed[1](h)          # LayerNorm over last dim
        h = self.embed_drop(h)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        h   = torch.cat([cls, h], dim=1)    # (B, P+1, D)

        # Transformer blocks with RoPE
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
        for block in self.blocks:
            h = block(h, cos, sin)

        # Classification via CLS token
        cls_out = self.norm(h[:, 0])        # (B, D)
        return self.head(cls_out).squeeze(-1)   # (B,)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
