"""
precompute_italian.py
=====================
DQLCT precompute for Italian PVS dataset.
Reads manifest.csv from parse_italian_pvs.py → saves .npz per WAV file.

Audio duration handling (Italian PVS files are short unlike KCL 10s segments):
    B1, B2, FB1     (read text, ~10-25s) → segment into 10s windows (like KCL)
    FB2             (words, ~5-10s)       → reflection-pad to 10s → 1 window
    D1, D2          (DDK, 5s)            → reflection-pad to 10s → 1 window
    VA1..VU2        (vowels, 3-5s)       → reflection-pad to 10s → 1 window

Padding strategy: reflection pad → captures real spectral statistics
rather than zeros. Zero-padding would create an artificial silence region
that shifts the DQLCT output toward zero — misleading for short tasks.

Output .npz format: identical to KCL precompute.py
    features:   float32  (4, 624, 257)
    label:      int32    0=HC  1=PD
    subject_id: str
    task:       str      (task_code abbreviation)
    source_idx: int32

Filename: subjID_label0_taskB1_src000_win0000.npz
          (task uses task_code directly for clarity)

Usage:
    python precompute_italian.py \\
        --manifest  path/to/italian_manifest.csv \\
        --seg_root  "path/to/Italian Parkinson's Voice and speech" \\
        --npz_dir   path/to/italian_npz_cache \\
        --workers   4
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

# ── DQLCT path setup ──────────────────────────────────────────────────────────
_THIS_DIR  = Path(__file__).resolve().parent
_SRC_DIR   = _THIS_DIR.parent if _THIS_DIR.name == 'data' else _THIS_DIR
_DQLCT_DIR = _SRC_DIR / 'dqlct_pipeline'
if not _DQLCT_DIR.exists():
    _DQLCT_DIR = _THIS_DIR / 'dqlct_pipeline'
if str(_DQLCT_DIR) not in sys.path:
    sys.path.insert(0, str(_DQLCT_DIR))

_purge = [k for k in sys.modules if any(
    k == m or k.endswith('.' + m)
    for m in ('quaternion_core', 'pd_feature_extractor', 'dqlct_transform')
)]
for _k in _purge:
    del sys.modules[_k]

from quaternion_core      import Quaternion, create_quaternion_array
from pd_feature_extractor import PDQuaternionFeatures
from dqlct_transform      import QLCT1D, create_standard_matrices
import librosa

# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match qcrnn_model.py exactly
# ─────────────────────────────────────────────────────────────────────────────
SR          = 16000
FRAME_LEN   = 512
HOP_LEN     = 256
SAMPLES     = SR * 10                             # 160,000 = 10s
N_FRAMES    = 1 + (SAMPLES - FRAME_LEN) // HOP_LEN  # 624
N_BINS      = FRAME_LEN // 2 + 1                 # 257
FEAT_SHAPE  = (4, N_FRAMES, N_BINS)              # (4, 624, 257)
MATRIX_TYPE = 'Fractional_45deg'
MIN_DUR     = 1.5    # seconds — skip only truly broken files

# Tasks that are long enough to multi-window (like KCL)
MULTIWINDOW_TASKS = {'ReadText', 'B1', 'B2', 'FB1'}
# All others: single window with padding

LABEL_MAP = {'HC': 0, 'PD': 1}

# ─────────────────────────────────────────────────────────────────────────────
# Audio preparation
# ─────────────────────────────────────────────────────────────────────────────

def reflect_pad_to(audio: np.ndarray, target: int) -> np.ndarray:
    """
    Extend audio to target length using reflection padding.
    Repeats the signal in mirror fashion until target is reached.
    Preserves spectral statistics of real speech vs zero-padding.
    """
    if len(audio) >= target:
        return audio[:target]
    result = audio.copy()
    while len(result) < target:
        needed    = target - len(result)
        chunk_len = min(len(audio), needed)
        result    = np.concatenate([result, audio[-chunk_len:][::-1]])
        if len(result) < target:
            needed    = target - len(result)
            chunk_len = min(len(audio), needed)
            result    = np.concatenate([result, audio[:chunk_len]])
    return result[:target]


def prepare_audio_windows(wav_path: str, task: str, task_code: str):
    """
    Load WAV and return list of (audio_chunk, window_idx) tuples,
    each audio_chunk being exactly SAMPLES long.

    ReadText tasks (B1, B2, FB1): segment into 10s windows (non-overlapping)
    All others: single reflection-padded 10s window
    """
    audio, _ = librosa.load(wav_path, sr=SR, mono=True)
    audio     = audio.astype(np.float32)

    # Multi-window for long ReadText recordings
    is_long = (task in MULTIWINDOW_TASKS or task_code in MULTIWINDOW_TASKS)

    if is_long and len(audio) >= SAMPLES:
        windows = []
        start   = 0
        win_idx = 0
        while start + SAMPLES <= len(audio):
            windows.append((audio[start:start+SAMPLES], win_idx))
            start   += SAMPLES
            win_idx += 1
        # Include remaining tail if >= 3s (pad to 10s)
        tail = audio[start:]
        if len(tail) >= MIN_DUR * SR:
            windows.append((reflect_pad_to(tail, SAMPLES), win_idx))
        return windows
    else:
        # Short file or non-ReadText: always single padded window
        return [(reflect_pad_to(audio, SAMPLES), 0)]


# ─────────────────────────────────────────────────────────────────────────────
# DQLCT extraction (module-level singletons, re-created per worker)
# ─────────────────────────────────────────────────────────────────────────────

_pd_features = None
_qlct        = None

def _init_worker():
    global _pd_features, _qlct
    _pd_features = PDQuaternionFeatures(
        sr=SR, frame_length=FRAME_LEN, hop_length=HOP_LEN)
    matrices = create_standard_matrices()
    a, b, c, d = matrices[MATRIX_TYPE]
    _qlct = QLCT1D(FRAME_LEN, a, b, c, d)


def _extract_features(audio: np.ndarray) -> np.ndarray:
    """(160000,) float32 → (4, 624, 257) float32"""
    global _pd_features, _qlct
    if _pd_features is None:
        _init_worker()

    quat_signal = _pd_features.audio_to_quaternion_signal(audio, verbose=False)
    zero_q      = Quaternion(0, 0, 0, 0)
    tensor      = np.zeros((N_FRAMES, N_BINS, 4), dtype=np.float32)

    for fi in range(N_FRAMES):
        start = fi * HOP_LEN
        end   = start + FRAME_LEN
        if end <= len(quat_signal):
            frame_list = quat_signal[start:end]
        else:
            frame_list = quat_signal[start:] + \
                         [zero_q] * (end - len(quat_signal))

        spectrum = _qlct.direct_transform(create_quaternion_array(frame_list))
        for bi in range(N_BINS):
            q = spectrum[bi]
            tensor[fi, bi, 0] = q.w
            tensor[fi, bi, 1] = q.x
            tensor[fi, bi, 2] = q.y
            tensor[fi, bi, 3] = q.z

    return tensor.transpose(2, 0, 1).astype(np.float32)   # (4, 624, 257)


# ─────────────────────────────────────────────────────────────────────────────
# Worker: one manifest row → one or more .npz files
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args):
    row, seg_root, npz_dir = args
    results = []

    label_int  = LABEL_MAP[row['label_str']]
    task_type  = row['task']
    task_code  = row.get('task_code', row['task'])
    subject_id = row['subject_id']
    src_idx    = abs(hash(row['parent_file'])) % 1000
    duration   = float(row.get('duration', 10.0))

    # Skip broken files only
    if duration < MIN_DUR:
        return [(f"subj{subject_id}_SKIP", f"SKIP:too_short({duration:.1f}s)")]

    wav_path = Path(seg_root) / row['segment_path']
    if not wav_path.exists():
        return [(f"subj{subject_id}_{task_code}", f"WAV not found: {wav_path}")]

    try:
        windows = prepare_audio_windows(str(wav_path), task_type, task_code)

        for audio_chunk, win_idx in windows:
            npz_stem = (
                f"subj{subject_id}_label{label_int}_task{task_code}"
                f"_src{src_idx:03d}_win{win_idx:04d}"
            )
            npz_path = Path(npz_dir) / f"{npz_stem}.npz"

            if npz_path.exists():
                results.append((npz_stem, None))
                continue

            features = _extract_features(audio_chunk)
            assert features.shape == FEAT_SHAPE, \
                f"Shape mismatch: {features.shape}"

            np.savez_compressed(
                str(npz_path),
                features   = features,
                label      = np.int32(label_int),
                subject_id = subject_id,
                task       = task_code,
                source_idx = np.int32(src_idx),
            )
            results.append((npz_stem, None))

    except Exception as e:
        results.append((f"subj{subject_id}_{task_code}",
                        f"{type(e).__name__}: {e}"))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_precompute(manifest_csv, seg_root, npz_dir, workers):
    Path(npz_dir).mkdir(parents=True, exist_ok=True)

    with open(manifest_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    # Count expected windows
    rt_rows = sum(1 for r in rows if r['task'] in MULTIWINDOW_TASKS)
    short_rows = len(rows) - rt_rows
    print(f"\n{'='*62}")
    print(f"  Italian PVS  —  DQLCT Precompute")
    print(f"{'='*62}")
    print(f"  WAV files        : {len(rows)}")
    print(f"  ReadText files   : {rt_rows}  (may produce multiple 10s windows)")
    print(f"  Short task files : {short_rows}  (padded to 10s → 1 window each)")
    print(f"  Output dir       : {npz_dir}")
    print(f"  Workers          : {workers}")
    print(f"  Estimated time   : ~{len(rows)*3.5/60/max(workers,1):.0f} min")
    print(f"{'='*62}\n")

    worker_args = [(row, seg_root, npz_dir) for row in rows]
    errors, skipped, n_ok = [], [], 0
    t0 = time.time()

    def process_results(result_list):
        nonlocal n_ok
        for stem, err in result_list:
            if err is None:
                n_ok += 1
            elif 'SKIP' in str(err):
                skipped.append((stem, err))
            else:
                errors.append((stem, err))

    if workers == 1:
        _init_worker()
        for args in tqdm(worker_args, desc="Precomputing", unit="file"):
            process_results(_worker(args))
    else:
        with Pool(processes=workers, initializer=_init_worker) as pool:
            for result_list in tqdm(
                pool.imap_unordered(_worker, worker_args, chunksize=2),
                total=len(rows), desc="Precomputing", unit="file"
            ):
                process_results(result_list)

    elapsed = time.time() - t0

    if errors:
        err_log = Path(npz_dir) / 'precompute_errors.log'
        with open(err_log, 'w') as f:
            for stem, msg in errors:
                f.write(f"{stem}\t{msg}\n")

    print(f"\n{'='*62}")
    print(f"  DONE")
    print(f"  .npz created  : {n_ok}")
    print(f"  Skipped       : {len(skipped)}  (< {MIN_DUR}s)")
    print(f"  Errors        : {len(errors)}")
    print(f"  Elapsed       : {elapsed/60:.1f} min")
    print(f"\n  Next step:")
    print(f"    python models/merge_folds_and_finetune.py \\")
    print(f"        --cv_dir   src/models/checkpoints/cv_run1 \\")
    print(f"        --npz_dir  {npz_dir} \\")
    print(f"        --out_dir  src/models/checkpoints/finetune_italian")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--seg_root', required=True)
    p.add_argument('--npz_dir',  required=True)
    p.add_argument('--workers',  type=int, default=1)
    args = p.parse_args()
    run_precompute(args.manifest, args.seg_root, args.npz_dir, args.workers)