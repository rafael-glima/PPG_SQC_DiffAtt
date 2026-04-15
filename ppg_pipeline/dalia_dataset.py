"""
dalia_dataset.py
----------------
Loader for the SMoLK / PPG-DaLiA processed dataset.

Windowing
~~~~~~~~~
  Each original segment (row of the .npy file) is sliced into fixed-length
  windows with a configurable step (overlap):

    window_size = round(window_sec * fs)   default 3 s → 192 samples @ 64 Hz
    step        = round(step_sec   * fs)   default 1 s →  64 samples @ 64 Hz

  Segments shorter than one full window are discarded.

Label format (majority vote)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  The original per-sample artifact mask is collapsed to a single binary label
  per window using majority voting:

    label = 1  (artifact)   if  mean(mask[window]) > 0.5
    label = 0  (clean)      otherwise

  This gives one scalar {0, 1} per window instead of a length-L mask.

Quality score
~~~~~~~~~~~~~
  quality_score = 1 - mean(mask[window])   ∈ [0, 1]   (fraction clean)

  ACCEPT  : score >= 0.75
  CAUTION : score >= 0.50
  REJECT  : score <  0.50

Sampling frequency
~~~~~~~~~~~~~~~~~~
  PPG-DaLiA wrist PPG (Empatica E4) is recorded at 64 Hz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# Verdict indices
VERDICT_ACCEPT  = 0
VERDICT_CAUTION = 1
VERDICT_REJECT  = 2

# DaLiA E4 wristband sampling frequency (Hz)
DALIA_FS = 64.0


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _find_npy_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No .npy files found in {folder}\n"
            f"  Clone the SMoLK repo: https://github.com/SullyChen/SMoLK\n"
            f"  and point load_dalia() at the processed_dataset sub-folder."
        )
    return files


def _classify_npy(path: Path) -> str:
    """Return 'labels' if array is strictly binary {0,1}, else 'data'."""
    arr = np.load(path)
    unique = set(np.unique(arr).tolist())
    if unique.issubset({0, 1, 0.0, 1.0}) and len(unique) <= 2:
        return "labels"
    return "data"


def _inspect_folder(folder: Path, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Auto-discover PPG data and artifact-mask arrays in *folder*.

    Returns
    -------
    ppg   : np.ndarray  shape (N, L)   float32
    masks : np.ndarray  shape (N, L)   float32  binary {0, 1}
    """
    files = _find_npy_files(folder)

    if verbose:
        print(f"  Found {len(files)} .npy file(s) in {folder}:")
        for f in files:
            a = np.load(f)
            print(f"    {f.name:40s}  shape={a.shape}  dtype={a.dtype}  "
                  f"range=[{a.min():.3f}, {a.max():.3f}]")

    name_map = {f.stem.lower(): f for f in files}

    _data_names  = {"x", "x_train", "x_test", "ppg", "ppg_data",
                    "data", "signals", "ppg_signals", "segments"}
    _label_names = {"y", "y_train", "y_test", "labels", "artifact_labels",
                    "masks", "artifact_masks", "annotations"}

    data_file  = next((name_map[n] for n in _data_names  if n in name_map), None)
    label_file = next((name_map[n] for n in _label_names if n in name_map), None)

    if data_file is None or label_file is None:
        classified = {f: _classify_npy(f) for f in files}
        if data_file is None:
            cands = [f for f, t in classified.items() if t == "data"]
            if not cands:
                raise ValueError(
                    f"Cannot identify PPG data file in {folder}.\n"
                    f"  Files: {[f.name for f in files]}"
                )
            data_file = cands[0]
        if label_file is None:
            cands = [f for f, t in classified.items() if t == "labels"]
            if not cands:
                raise ValueError(
                    f"Cannot identify artifact label file in {folder}.\n"
                    f"  Files: {[f.name for f in files]}"
                )
            label_file = cands[0]

    ppg   = np.load(data_file).astype(np.float32)
    masks = np.load(label_file).astype(np.float32)

    if verbose:
        print(f"  → PPG data : {data_file.name}   shape={ppg.shape}")
        print(f"  → Labels   : {label_file.name}  shape={masks.shape}")

    if ppg.shape != masks.shape:
        raise ValueError(
            f"Shape mismatch: PPG {ppg.shape} vs masks {masks.shape}."
        )
    if ppg.ndim != 2:
        raise ValueError(f"Expected 2D arrays (N, L), got {ppg.shape}")

    return ppg, masks


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def apply_windowing(
    ppg:         np.ndarray,
    masks:       np.ndarray,
    fs:          float = DALIA_FS,
    window_sec:  float = 3.0,
    step_sec:    float = 1.0,
    verbose:     bool  = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice every row of *ppg* and *masks* into overlapping windows and derive
    a single binary label per window via majority voting.

    Parameters
    ----------
    ppg        : (N, L_orig)   raw PPG segments
    masks      : (N, L_orig)   per-sample artifact masks
    fs         : sampling frequency (Hz)
    window_sec : window length in seconds  (default 3 s)
    step_sec   : step between windows in seconds (default 1 s, giving 2 s overlap)
    verbose    : print summary

    Returns
    -------
    ppg_win    : (N_win, win_size)  float32   PPG windows
    labels     : (N_win,)          float32   binary majority-vote label
                                             1 = artifact, 0 = clean
    """
    win_size = int(round(window_sec * fs))
    step     = int(round(step_sec   * fs))
    N, L     = ppg.shape

    if win_size > L:
        raise ValueError(
            f"Window size ({win_size} samples = {window_sec}s @ {fs}Hz) is larger "
            f"than segment length ({L} samples). Reduce window_sec or increase fs."
        )

    ppg_wins, label_list = [], []

    for seg_ppg, seg_mask in zip(ppg, masks):
        start = 0
        while start + win_size <= L:
            w_ppg  = seg_ppg [start : start + win_size]
            w_mask = seg_mask[start : start + win_size]

            # Majority vote: 1 if more than half the samples are artefacts
            label = 1.0 if w_mask.mean() > 0.5 else 0.0

            ppg_wins.append(w_ppg)
            label_list.append(label)
            start += step

    ppg_out    = np.stack(ppg_wins).astype(np.float32)   # (N_win, win_size)
    labels_out = np.array(label_list, dtype=np.float32)  # (N_win,)

    if verbose:
        n_art   = int(labels_out.sum())
        n_clean = len(labels_out) - n_art
        print(f"  Windowing: {N} segments → {len(labels_out)} windows "
              f"(win={window_sec}s  step={step_sec}s  size={win_size}samp)")
        print(f"  Clean={n_clean}  Artifact={n_art}  "
              f"artifact_rate={n_art/max(1,len(labels_out)):.3f}")

    return ppg_out, labels_out


# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------

def labels_to_quality(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive quality scores and 3-class verdicts from binary window labels.

    A clean window (label=0) has quality 1.0; an artifact window (label=1)
    has quality 0.0.  The score is used to assign a coarse verdict.

    Returns
    -------
    scores   : (N,) float32   1 - label
    verdicts : (N,) int64     ACCEPT / CAUTION / REJECT
    """
    scores = (1.0 - labels).astype(np.float32)
    verdicts = np.where(
        scores >= 0.75, VERDICT_ACCEPT,
        np.where(scores >= 0.50, VERDICT_CAUTION, VERDICT_REJECT)
    ).astype(np.int64)
    return scores, verdicts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DaLiADataset(Dataset):
    """
    PyTorch Dataset for windowed PPG-DaLiA segments with binary quality labels.

    Each item:
      waveform  : Tensor (1, win_size)   normalised PPG window
      features  : Tensor (24,)           extracted features (zeros if not computed)
      label     : Tensor scalar float    0.0 = clean  1.0 = artifact  (majority vote)
      score     : Tensor scalar float    quality score  1 - label
      verdict   : Tensor scalar int      0=ACCEPT 1=CAUTION 2=REJECT

    Parameters
    ----------
    ppg_win      : (N, win_size)  windowed PPG
    labels       : (N,)           binary majority-vote labels
    feature_vecs : (N, 24) or None
    normalise    : per-window zero-mean unit-std normalisation
    """

    VERDICT_NAMES = ["ACCEPT", "CAUTION", "REJECT"]

    def __init__(
        self,
        ppg_win:      np.ndarray,
        labels:       np.ndarray,
        feature_vecs: Optional[np.ndarray] = None,
        normalise:    bool = True,
    ) -> None:
        self.labels_np             = labels.astype(np.float32)
        self.scores, self.verdicts_np = labels_to_quality(self.labels_np)

        if normalise:
            mean = ppg_win.mean(axis=1, keepdims=True)
            std  = ppg_win.std(axis=1,  keepdims=True) + 1e-8
            ppg_win = (ppg_win - mean) / std

        self.waveforms  = torch.from_numpy(ppg_win).unsqueeze(1)           # (N,1,W)
        self.labels_t   = torch.from_numpy(self.labels_np)                 # (N,)
        self.scores_t   = torch.from_numpy(self.scores)                    # (N,)
        self.verdicts   = torch.from_numpy(self.verdicts_np)               # (N,)

        if feature_vecs is not None:
            self.feature_vecs = torch.from_numpy(feature_vecs.astype(np.float32))
        else:
            self.feature_vecs = torch.zeros(len(ppg_win), 24)

    def __len__(self) -> int:
        return len(self.labels_t)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "waveform": self.waveforms[idx],
            "features": self.feature_vecs[idx],
            "label":    self.labels_t[idx],
            "score":    self.scores_t[idx],
            "verdict":  self.verdicts[idx],
        }

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for the binary label (clean / artifact)."""
        n_art   = float(self.labels_np.sum())
        n_clean = float(len(self.labels_np)) - n_art
        total   = n_art + n_clean
        # weight[0]=clean weight, weight[1]=artifact weight
        w = torch.tensor([total / (2 * max(n_clean, 1)),
                           total / (2 * max(n_art,   1))], dtype=torch.float32)
        return w

    def split(
        self, val_fraction: float = 0.15, seed: int = 0
    ) -> tuple["DaLiADataset", "DaLiADataset"]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self))
        n_val = max(1, int(len(self) * val_fraction))
        val_i, tr_i = idx[:n_val], idx[n_val:]

        def _sub(indices):
            ds = DaLiADataset.__new__(DaLiADataset)
            ds.waveforms    = self.waveforms[indices]
            ds.labels_t     = self.labels_t[indices]
            ds.labels_np    = self.labels_np[indices]
            ds.scores_t     = self.scores_t[indices]
            ds.scores       = self.scores[indices]
            ds.verdicts     = self.verdicts[indices]
            ds.verdicts_np  = self.verdicts_np[indices]
            ds.feature_vecs = self.feature_vecs[indices]
            return ds

        return _sub(tr_i), _sub(val_i)

    def summary(self) -> str:
        n       = len(self)
        n_art   = int(self.labels_np.sum())
        n_clean = n - n_art
        win_len = self.waveforms.shape[-1]
        return (
            f"DaLiADataset  N={n}  win_size={win_len} "
            f"({win_len/DALIA_FS:.1f}s @ {DALIA_FS:.0f}Hz)\n"
            f"  Clean={n_clean}  Artifact={n_art}  "
            f"artifact_rate={n_art/max(1,n):.3f}"
        )


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

def load_dalia(
    train_dir:       str | Path,
    test_dir:        str | Path,
    window_sec:      float = 3.0,
    step_sec:        float = 1.0,
    extract_features: bool = True,
    verbose:         bool = True,
) -> tuple[DaLiADataset, DaLiADataset]:
    """
    Load, window, and label the PPG-DaLiA processed dataset.

    Parameters
    ----------
    train_dir       : path to new_PPG_DaLiA_train/processed_dataset
    test_dir        : path to new_PPG_DaLiA_test/processed_dataset
    window_sec      : window length in seconds  (default 3.0)
    step_sec        : step between windows in seconds  (default 1.0)
    extract_features: if True, compute 24-dim feature vectors per window
    verbose         : print progress

    Returns
    -------
    train_ds, test_ds : DaLiADataset
    """
    train_dir = Path(train_dir)
    test_dir  = Path(test_dir)

    for d in (train_dir, test_dir):
        if not d.exists():
            raise FileNotFoundError(
                f"Directory not found: {d}\n"
                f"  Clone https://github.com/SullyChen/SMoLK and pass:\n"
                f"    train_dir = 'SMoLK/PPG data/new_PPG_DaLiA_train/processed_dataset'\n"
                f"    test_dir  = 'SMoLK/PPG data/new_PPG_DaLiA_test/processed_dataset'"
            )

    if verbose: print("Loading DaLiA train split …")
    ppg_tr, mask_tr = _inspect_folder(train_dir, verbose=verbose)

    if verbose: print("Loading DaLiA test split …")
    ppg_te, mask_te = _inspect_folder(test_dir,  verbose=verbose)

    if verbose: print("\nApplying windowing …")
    ppg_tr_w, lab_tr = apply_windowing(ppg_tr, mask_tr, DALIA_FS,
                                       window_sec, step_sec, verbose=verbose)
    ppg_te_w, lab_te = apply_windowing(ppg_te, mask_te, DALIA_FS,
                                       window_sec, step_sec, verbose=verbose)

    feat_tr = feat_te = None
    if extract_features:
        feat_tr = _batch_extract_features(ppg_tr_w, verbose=verbose, split="train")
        feat_te = _batch_extract_features(ppg_te_w, verbose=verbose, split="test")

    train_ds = DaLiADataset(ppg_tr_w, lab_tr, feat_tr)
    test_ds  = DaLiADataset(ppg_te_w, lab_te, feat_te)

    if verbose:
        print("\n" + train_ds.summary())
        print(test_ds.summary())

    return train_ds, test_ds


def _batch_extract_features(
    ppg: np.ndarray, verbose: bool, split: str
) -> np.ndarray:
    """Extract 24-dim feature vectors for every windowed segment."""
    from .preprocessor      import PPGPreprocessor
    from .feature_extractor import PPGFeatureExtractor

    pre  = PPGPreprocessor(fs=DALIA_FS)
    feat = PPGFeatureExtractor(fs=DALIA_FS)
    vecs = []

    if verbose:
        print(f"  Extracting features for {split} ({len(ppg)} windows) …")

    for i, seg in enumerate(ppg):
        try:
            filt, peaks = pre.process(seg)
            fv = feat.extract(filt, peaks)
            vecs.append(fv.to_array())
        except Exception:
            vecs.append(np.zeros(24, dtype=np.float32))
        if verbose and (i + 1) % 1000 == 0:
            print(f"    {i+1}/{len(ppg)}")

    return np.stack(vecs).astype(np.float32)
