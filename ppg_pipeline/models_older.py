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
