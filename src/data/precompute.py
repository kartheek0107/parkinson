"""
precompute.py
=============
Reads manifest.csv from segmentation, runs each 10s WAV through the DQLCT
pipeline, and saves one .npz file per segment.

Output .npz format matches train.py DQLCTWindowDataset exactly:
    features   : float32  (4, 624, 257)
    label      : int32    0=HC  1=PD
    subject_id : str
    task       : str      RT | SD
    source_idx : int32    index of parent file

Filename convention (required by DQLCTWindowDataset parser):
    subjID00_label0_taskRT_src003_win0000.npz

Pipeline per segment (delegates to feature_extractor.py — zero logic change):
    WAV (10s, 160000 samples)
    → _load_audio (already 16kHz from segmentation)
    → Hilbert quaternion signal
    → DQLCT frame-by-frame (FRAME_LEN=512, HOP=256)
    → (4, n_frames, 257) tensor
    → pad/trim to (4, 624, 257)
    → save .npz

Usage (run from src/ directory, dqlct_pipeline/ must be reachable):
    cd /path/to/DQLCT_Parkinsons/src
    python data/precompute.py \\
        --manifest  "/path/to/segments_output/manifest.csv" \\
        --seg_root  "/path/to/segments_output" \\
        --npz_dir   "/path/to/npz_cache" \\
        --workers   4

Resume-safe: skips .npz files that already exist.
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

# ── Path setup: must find dqlct_pipeline as flat modules ──────────────────────
_THIS_DIR  = Path(__file__).resolve().parent
_SRC_DIR   = _THIS_DIR.parent if _THIS_DIR.name == 'data' else _THIS_DIR
_DQLCT_DIR = _SRC_DIR / 'dqlct_pipeline'

if not _DQLCT_DIR.exists():
    # Try one level up
    _DQLCT_DIR = _THIS_DIR / 'dqlct_pipeline'

if str(_DQLCT_DIR) not in sys.path:
    sys.path.insert(0, str(_DQLCT_DIR))

# Purge cached package-style imports to prevent dual Quaternion type conflict
_purge = [k for k in sys.modules if any(
    k == m or k.endswith('.' + m)
    for m in ('quaternion_core', 'holistic_features', 'dqlct_transform')
)]
for _k in _purge:
    del sys.modules[_k]

from scipy.signal import hilbert as scipy_hilbert
from quaternion_core   import Quaternion, create_quaternion_array
from holistic_features import HilbertQuaternionFeatures
from dqlct_transform   import QLCT1D, create_standard_matrices
import librosa

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match qcrnn_model.py and feature_extractor.py
# ─────────────────────────────────────────────────────────────────────────────
SR           = 16000
FRAME_LEN    = 512
HOP_LEN      = 256
SAMPLES      = SR * 10                              # 160,000
N_FRAMES     = 1 + (SAMPLES - FRAME_LEN) // HOP_LEN  # 624
N_BINS_HALF  = FRAME_LEN // 2 + 1                  # 257
N_CHANNELS   = 4
FEAT_SHAPE   = (N_CHANNELS, N_FRAMES, N_BINS_HALF)  # (4, 624, 257)
MATRIX_TYPE  = 'Fractional_45deg'

TASK_ABBREV  = {'ReadText': 'RT', 'SpontaneousDialogue': 'SD'}
LABEL_MAP    = {'HC': 0, 'PD': 1}

# ─────────────────────────────────────────────────────────────────────────────
# Core extraction (module-level singletons re-created per worker process)
# ─────────────────────────────────────────────────────────────────────────────

_hilbert_extractor = None
_qlct              = None

def _init_worker():
    """Initialise DQLCT singletons once per worker process."""
    global _hilbert_extractor, _qlct
    _hilbert_extractor = HilbertQuaternionFeatures(
        sr=SR, frame_length=FRAME_LEN, hop_length=HOP_LEN)
    matrices = create_standard_matrices()
    a, b, c, d = matrices[MATRIX_TYPE]
    _qlct = QLCT1D(FRAME_LEN, a, b, c, d)


def _extract(wav_path: str) -> np.ndarray:
    """
    WAV → (4, 624, 257) float32.
    Uses module-level singletons; call _init_worker() before first use.
    """
    global _hilbert_extractor, _qlct
    if _hilbert_extractor is None:
        _init_worker()

    audio, _ = librosa.load(wav_path, sr=SR, mono=True)
    # Pad or trim to exact 10s
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    # Hilbert → quaternion signal
    quat_signal = _hilbert_extractor.audio_to_quaternion_signal(
        audio.astype(np.float32), verbose=False)

    # Frame + DQLCT
    zero_q = Quaternion(0, 0, 0, 0)
    tensor = np.zeros((N_FRAMES, N_BINS_HALF, N_CHANNELS), dtype=np.float32)

    for fi in range(N_FRAMES):
        start = fi * HOP_LEN
        end   = start + FRAME_LEN
        if end <= len(quat_signal):
            frame_list = quat_signal[start:end]
        else:
            frame_list = quat_signal[start:] + [zero_q] * (end - len(quat_signal))

        spectrum = _qlct.direct_transform(create_quaternion_array(frame_list))

        for bi in range(N_BINS_HALF):
            q = spectrum[bi]
            tensor[fi, bi, 0] = q.w
            tensor[fi, bi, 1] = q.x
            tensor[fi, bi, 2] = q.y
            tensor[fi, bi, 3] = q.z

    # (N_FRAMES, N_BINS_HALF, 4) → (4, N_FRAMES, N_BINS_HALF)
    return tensor.transpose(2, 0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args):
    """
    Process one manifest row → save .npz.
    Returns (npz_stem, error_msg or None).
    """
    row, seg_root, npz_dir = args

    label_int  = LABEL_MAP[row['label_str']]
    task_abbr  = TASK_ABBREV.get(row['task'], row['task'][:2].upper())
    subject_id = row['subject_id']
    seg_idx    = int(row['seg_idx'])

    # Build source_idx: enumerate unique parent files per subject+task
    # Encoded directly in npz filename — use parent_file hash mod 1000
    src_idx    = abs(hash(row['parent_file'])) % 1000

    npz_stem = (
        f"subj{subject_id}_label{label_int}_task{task_abbr}"
        f"_src{src_idx:03d}_win{seg_idx:04d}"
    )
    npz_path = Path(npz_dir) / f"{npz_stem}.npz"

    if npz_path.exists():
        return npz_stem, None   # resume

    wav_path = Path(seg_root) / row['segment_path']
    if not wav_path.exists():
        return npz_stem, f"WAV not found: {wav_path}"

    try:
        features = _extract(str(wav_path))
        assert features.shape == FEAT_SHAPE, \
            f"Shape {features.shape} != {FEAT_SHAPE}"

        np.savez_compressed(
            str(npz_path),
            features   = features,
            label      = np.int32(label_int),
            subject_id = subject_id,
            task       = task_abbr,
            source_idx = np.int32(src_idx),
        )
        return npz_stem, None

    except Exception as e:
        return npz_stem, f"{type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_precompute(manifest_csv, seg_root, npz_dir, workers):
    npz_path = Path(npz_dir)
    npz_path.mkdir(parents=True, exist_ok=True)

    with open(manifest_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    est_min = total * 3.5 / 60 / max(workers, 1)

    print(f"\n{'='*62}")
    print(f"  DQLCT PRECOMPUTE  →  .npz cache")
    print(f"{'='*62}")
    print(f"  Segments   : {total}")
    print(f"  Output     : {npz_path}")
    print(f"  Feat shape : {FEAT_SHAPE}  float32")
    print(f"  Workers    : {workers}")
    print(f"  Est. time  : ~{est_min:.0f} min  (DQLCT O(N²) per frame)")
    print(f"{'='*62}\n")

    worker_args = [(row, seg_root, npz_dir) for row in rows]
    errors  = []
    n_ok    = 0
    t0      = time.time()

    if workers == 1:
        _init_worker()
        for args in tqdm(worker_args, desc="Precomputing", unit="seg"):
            stem, err = _worker(args)
            if err:
                errors.append((stem, err))
            else:
                n_ok += 1
    else:
        with Pool(processes=workers, initializer=_init_worker) as pool:
            for stem, err in tqdm(
                pool.imap_unordered(_worker, worker_args, chunksize=2),
                total=total, desc="Precomputing", unit="seg"
            ):
                if err:
                    errors.append((stem, err))
                else:
                    n_ok += 1

    elapsed = time.time() - t0

    if errors:
        err_log = Path(npz_dir) / 'precompute_errors.log'
        with open(err_log, 'w') as f:
            for stem, msg in errors:
                f.write(f"{stem}\t{msg}\n")
        print(f"\n  ⚠  {len(errors)} errors → {err_log}")

    print(f"\n{'='*62}")
    print(f"  DONE")
    print(f"  Succeeded : {n_ok}/{total}")
    print(f"  Errors    : {len(errors)}")
    print(f"  Elapsed   : {elapsed/60:.1f} min  ({elapsed/max(n_ok,1):.1f}s/seg)")
    print(f"  Cache dir : {npz_path}")
    print(f"\n  Next step:")
    print(f"    python src/models/train_5fold.py \\")
    print(f"        --npz_dir {npz_path} \\")
    print(f"        --out_dir src/models/checkpoints/cv_run1")
    print(f"{'='*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Precompute DQLCT .npz cache from segmented WAV files.')
    p.add_argument('--manifest', required=True,
                   help='manifest.csv from audio_segmentation.py')
    p.add_argument('--seg_root', required=True,
                   help='Root of segments_output/ (segment_path is relative to this)')
    p.add_argument('--npz_dir',  required=True,
                   help='Output directory for .npz files')
    p.add_argument('--workers',  type=int, default=1,
                   help='Parallel workers (default 1; use 4 for multi-core CPU)')
    args = p.parse_args()
    run_precompute(args.manifest, args.seg_root, args.npz_dir, args.workers)