# feature_extractor.py
"""
Feature Extraction Bridge: DQLCT Pipeline -> Q-CRNN Training Format
====================================================================
Uses the ORIGINAL dqlct_pipeline code WITHOUT any logic changes.
This file only adds:
  1. Resampling (44.1kHz -> 16kHz)
  2. VAD (energy-based silence removal)
  3. Window slicing into fixed T=624 frames
  4. Conversion from list-of-Quaternion -> numpy float32 array (4, T, F)

Import chain (zero logic changes to originals):
  holistic_features.HilbertQuaternionFeatures  -> quaternion signal
  dqlct_transform.QLCT1D.direct_transform      -> DQLCT spectrum per frame
  quaternion_core.Quaternion                   -> quaternion arithmetic

Output contract (required by train.py / precompute.py):
  process_audio_file(filepath, training_mode) -> list of np.ndarray (4, 624, 257)
"""

import sys
import os
import warnings
import numpy as np
import librosa

warnings.filterwarnings('ignore', category=UserWarning)

# ── Path injection ─────────────────────────────────────────────────────────────
# CRITICAL: dqlct_pipeline modules must ONLY be imported as flat modules,
# never as src.dqlct_pipeline.* package imports.
# If both exist in sys.modules, Python sees two different Quaternion types
# and Hamilton product raises: "Cannot multiply Quaternion with Quaternion".
#
# Fix: insert dqlct_pipeline/ at sys.path[0] so flat imports always win,
# then purge any already-cached package-style entries from sys.modules.

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_DQLCT_DIR = os.path.join(_THIS_DIR, 'dqlct_pipeline')

if _DQLCT_DIR not in sys.path:
    sys.path.insert(0, _DQLCT_DIR)

# Purge any package-style cached imports (e.g. src.dqlct_pipeline.quaternion_core)
_to_purge = [k for k in sys.modules if any(
    k == m or k.endswith('.' + m)
    for m in ('quaternion_core', 'holistic_features', 'dqlct_transform')
)]
for _k in _to_purge:
    del sys.modules[_k]

# Import ONLY the three needed modules — flat, never via package path.
# Do NOT import complete_pipeline or spectral_distance (pull in matplotlib).
from quaternion_core   import Quaternion, create_quaternion_array
from holistic_features import HilbertQuaternionFeatures
from dqlct_transform   import QLCT1D, create_standard_matrices


# ═══════════════════════════════════════════════════════════════════════════════
# Constants — must match qcrnn_model.py tensor contract
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_SR   = 16000      # resample all audio to this
FRAME_LEN   = 512        # DQLCT frame length (samples)
HOP_LEN     = 256        # 50% overlap between DQLCT frames
T_FRAMES    = 624        # time frames per 10s segment (624 frames at hop=256, sr=16000)
F_BINS      = 257        # positive frequency bins (FRAME_LEN // 2 + 1 = 257)
VAD_DB      = -40.0      # RMS silence threshold in dB
MATRIX_TYPE = 'Fractional_45deg'


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level singletons (expensive to recreate per file)
# ═══════════════════════════════════════════════════════════════════════════════

_matrices = create_standard_matrices()
_a, _b, _c, _d = _matrices[MATRIX_TYPE]

_hilbert = HilbertQuaternionFeatures(
    sr=TARGET_SR,
    frame_length=FRAME_LEN,
    hop_length=HOP_LEN
)

_qlct = QLCT1D(FRAME_LEN, _a, _b, _c, _d)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Load + resample
# ═══════════════════════════════════════════════════════════════════════════════

def _load_audio(filepath: str) -> np.ndarray:
    """Load WAV, resample to TARGET_SR, convert to mono float32."""
    audio, _ = librosa.load(filepath, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: VAD
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_vad(audio: np.ndarray) -> np.ndarray:
    """
    Energy-based VAD. Keeps frames whose RMS > VAD_DB threshold.
    Returns concatenation of voiced frames. Falls back to full audio if
    all frames are below threshold.
    """
    threshold = 10 ** (VAD_DB / 20.0)
    voiced    = []

    for start in range(0, len(audio) - FRAME_LEN + 1, HOP_LEN):
        frame = audio[start : start + FRAME_LEN]
        if np.sqrt(np.mean(frame ** 2)) > threshold:
            voiced.append(frame)

    return np.concatenate(voiced) if voiced else audio


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Quaternion signal  (original HilbertQuaternionFeatures — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _to_quaternion_signal(audio: np.ndarray) -> list:
    """
    Delegates directly to HilbertQuaternionFeatures.audio_to_quaternion_signal().
    Returns list of Quaternion objects: q[n] = w + xi + 0j + 0k
    Zero logic change.
    """
    return _hilbert.audio_to_quaternion_signal(audio, verbose=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: DQLCT spectrogram  (original QLCT1D — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_dqlct_spectrogram(quat_signal: list) -> np.ndarray:
    """
    Apply QLCT1D.direct_transform() frame-by-frame.

    Extracts 4 real component channels from the quaternion spectrum:
        Ch 0 (W): real scalar component of each spectrum bin
        Ch 1 (X): i-component
        Ch 2 (Y): j-component  (zero for standard Hilbert input)
        Ch 3 (Z): k-component  (zero for standard Hilbert input)

    Args:
        quat_signal : list of Quaternion objects

    Returns:
        np.ndarray  shape (4, n_frames, F_BINS)  float32
    """
    zero_q   = Quaternion(0, 0, 0, 0)
    n_total  = len(quat_signal)
    n_frames = max(1, (n_total - FRAME_LEN) // HOP_LEN + 1)

    W = np.zeros((n_frames, F_BINS), dtype=np.float32)
    X = np.zeros((n_frames, F_BINS), dtype=np.float32)
    Y = np.zeros((n_frames, F_BINS), dtype=np.float32)
    Z = np.zeros((n_frames, F_BINS), dtype=np.float32)

    for fi in range(n_frames):
        start = fi * HOP_LEN
        end   = start + FRAME_LEN

        if end <= n_total:
            frame_list = quat_signal[start:end]
        else:
            frame_list = quat_signal[start:] + [zero_q] * (end - n_total)

        frame_arr = create_quaternion_array(frame_list)

        # ── Original DQLCT call — zero logic change ────────────────────────
        spectrum = _qlct.direct_transform(frame_arr)

        # Unpack positive frequencies into 4 component arrays
        for bi in range(F_BINS):
            q        = spectrum[bi]
            W[fi, bi] = float(q.w)
            X[fi, bi] = float(q.x)
            Y[fi, bi] = float(q.y)
            Z[fi, bi] = float(q.z)

    return np.stack([W, X, Y, Z], axis=0)   # (4, n_frames, F_BINS)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Slice into fixed T_FRAMES windows
# ═══════════════════════════════════════════════════════════════════════════════

def _slice_windows(spectrogram: np.ndarray, training_mode: bool) -> list:
    """
    Slice (4, n_frames, F_BINS) into list of (4, T_FRAMES, F_BINS) windows.

    training_mode=True  : 50% overlap (more data from each recording)
    training_mode=False : non-overlapping (inference / precompute --no_overlap)

    Zero-pads clips shorter than T_FRAMES.
    """
    _, n_frames, _ = spectrogram.shape
    hop            = T_FRAMES // 2 if training_mode else T_FRAMES

    if n_frames < T_FRAMES:
        pad = np.zeros((4, T_FRAMES - n_frames, F_BINS), dtype=np.float32)
        return [np.concatenate([spectrogram, pad], axis=1).astype(np.float32)]

    windows = []
    start   = 0
    while start + T_FRAMES <= n_frames:
        windows.append(spectrogram[:, start:start + T_FRAMES, :].astype(np.float32))
        start += hop

    return windows


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — called by precompute.py
# ═══════════════════════════════════════════════════════════════════════════════

def process_audio_file(filepath: str, training_mode: bool = True) -> list:
    """
    Full pipeline: WAV -> list of (4, 624, 257) float32 numpy arrays.

    Pipeline:
        load (-> 16kHz) -> VAD -> HilbertQuaternionFeatures
        -> QLCT1D.direct_transform (per frame) -> window slicing

    Args:
        filepath      : absolute path to .wav file
        training_mode : True  = 50% overlap windows (training)
                        False = non-overlapping (inference)

    Returns:
        list of np.ndarray, each shape (4, 624, 257)
        Empty list if clip too short after VAD.
    """
    audio = _load_audio(filepath)
    audio = _apply_vad(audio)

    if len(audio) < FRAME_LEN:
        return []

    quat_signal = _to_quaternion_signal(audio)
    spectrogram = _compute_dqlct_spectrogram(quat_signal)
    return _slice_windows(spectrogram, training_mode)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import time

    print("=" * 55)
    print("  feature_extractor.py  —  self-test")
    print("=" * 55)

    # Synthetic PD-like signal: 150Hz F0 + 5Hz tremor modulation
    sr       = TARGET_SR
    t        = np.linspace(0, 3.0, int(sr * 3.0))
    f0       = 150 + 5 * np.sin(2 * np.pi * 5 * t)
    audio    = (0.5 * np.sin(2 * np.pi * f0 * t)
                + 0.3 * np.sin(2 * np.pi * 500 * t)
                + 0.01 * np.random.randn(len(t))).astype(np.float32)

    print(f"\n  Synthetic PD signal: 3.0s  sr={sr}Hz")
    print(f"  Running 3-frame DQLCT test (full run is O(N^2) — slow)...")

    t0          = time.time()
    voiced      = _apply_vad(audio)
    quat_signal = _to_quaternion_signal(voiced)

    # Only process 3 frames in self-test
    test_signal = quat_signal[:FRAME_LEN * 3]
    spec        = _compute_dqlct_spectrogram(test_signal)
    elapsed     = time.time() - t0

    print(f"\n  VAD output    : {len(voiced)} samples ({len(voiced)/sr:.2f}s)")
    print(f"  Quat signal   : {len(quat_signal)} samples, q[0] = {quat_signal[0]}")
    print(f"  Spectrogram   : {spec.shape}  ({elapsed:.1f}s for 3 frames)")
    print(f"  W range       : [{spec[0].min():.4f}, {spec[0].max():.4f}]")
    print(f"  X range       : [{spec[1].min():.4f}, {spec[1].max():.4f}]")

    windows = _slice_windows(spec, training_mode=True)
    print(f"\n  Windows       : {len(windows)}")
    if windows:
        assert windows[0].shape == (4, T_FRAMES, F_BINS), f"Shape: {windows[0].shape}"
        print(f"  Window shape  : {windows[0].shape}  PASS")

    print(f"\n  Import chain:")
    print(f"    quaternion_core.Quaternion                   OK")
    print(f"    holistic_features.HilbertQuaternionFeatures  OK")
    print(f"    dqlct_transform.QLCT1D                       OK")
    print(f"\n  Bridge functional.")
    print("=" * 55)