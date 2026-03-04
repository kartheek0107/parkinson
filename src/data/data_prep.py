# data_prep.py
"""
MDVR-KCL Dataset Preparation
=============================
Walks the dataset directory, builds labels.csv and a fixed splits.json.

Expected directory structure:
    <root>/
    └── 26-29_09_2017_KCL/
        ├── ReadText/
        │   ├── HC/
        │   │   └── ID00_hc_0_0_0.wav
        │   └── PD/
        │       └── ID00_pd_0_0_0.wav
        └── SpontaneousDialogue/
            ├── HC/
            └── PD/

Filename convention:  {SubjectID}_{class}_{...}.wav
    SubjectID : ID00, ID01, ...  (first token before '_')
    class     : hc / pd          (second token — also matches parent folder)

Outputs
-------
data/labels.csv
    filepath    : absolute path to .wav
    subject_id  : e.g. ID00
    label       : 0 = HC, 1 = PD
    task        : ReadText | SpontaneousDialogue
    duration_s  : clip duration in seconds

data/splits.json
    {
      "train": ["ID02", "ID05", ...],
      "val":   ["ID01", "ID07", ...],
      "test":  ["ID00", "ID03", ...]
    }
    Pinned at generation time — never regenerated so results are reproducible.

Usage
-----
    python data/data_prep.py --root "C:/path/to/26-29_09_2017_KCL"

    # Dry-run (no files written):
    python data/data_prep.py --root "C:/path/to/26-29_09_2017_KCL" --dry_run

    # Custom output dir:
    python data/data_prep.py --root "..." --out_dir "data/"
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf      # reads duration without loading full audio


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

# Folder name → integer label
FOLDER_TO_LABEL = {'HC': 0, 'PD': 1}

# Task subfolder names to scan (add more if dataset expands)
TASK_FOLDERS = ['ReadText', 'SpontaneousDialogue']

# Minimum clip duration to keep (shorter clips won't produce a full T=128 window)
MIN_DURATION_S = 0.5

# Subject-level split ratios
VAL_RATIO  = 0.15
TEST_RATIO = 0.20

SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Walk directory, collect file records
# ═══════════════════════════════════════════════════════════════════════════════

def collect_records(root: Path) -> list:
    """
    Walk the MDVR-KCL tree and return a list of dicts, one per WAV file.

    Each dict:
        filepath    : str  (absolute)
        subject_id  : str  e.g. 'ID00'
        label       : int  0=HC, 1=PD
        task        : str  'ReadText' | 'SpontaneousDialogue'
        duration_s  : float
    """
    records  = []
    skipped  = []

    for task in TASK_FOLDERS:
        task_dir = root / task
        if not task_dir.exists():
            warnings.warn(f"Task folder not found, skipping: {task_dir}")
            continue

        for class_name, label in FOLDER_TO_LABEL.items():
            class_dir = task_dir / class_name
            if not class_dir.exists():
                warnings.warn(f"Class folder not found, skipping: {class_dir}")
                continue

            wav_files = sorted(class_dir.glob('*.wav'))
            if not wav_files:
                warnings.warn(f"No .wav files in: {class_dir}")
                continue

            for wav_path in wav_files:
                # Parse subject ID from filename:  ID00_hc_0_0_0.wav → ID00
                stem   = wav_path.stem                      # 'ID00_hc_0_0_0'
                tokens = stem.split('_')
                if len(tokens) < 2:
                    skipped.append((str(wav_path), 'cannot parse subject_id from filename'))
                    continue

                subject_id = tokens[0].upper()              # 'ID00'

                # Sanity: class token in filename should match folder
                file_class = tokens[1].upper()              # 'HC' or 'PD'
                if file_class not in ('HC', 'PD'):
                    skipped.append((str(wav_path), f'unexpected class token: {tokens[1]}'))
                    continue
                if file_class != class_name:
                    warnings.warn(
                        f"Class mismatch: file says '{file_class}' "
                        f"but folder says '{class_name}'. "
                        f"Using folder label ({label}). File: {wav_path.name}"
                    )

                # Get duration without loading full audio
                try:
                    info       = sf.info(str(wav_path))
                    duration_s = info.frames / info.samplerate
                except Exception as exc:
                    skipped.append((str(wav_path), f'soundfile error: {exc}'))
                    continue

                if duration_s < MIN_DURATION_S:
                    skipped.append((str(wav_path), f'too short: {duration_s:.2f}s'))
                    continue

                records.append({
                    'filepath':   str(wav_path.resolve()),
                    'subject_id': subject_id,
                    'label':      label,
                    'task':       task,
                    'duration_s': round(duration_s, 3),
                })

    return records, skipped


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Audit
# ═══════════════════════════════════════════════════════════════════════════════

def audit(df: pd.DataFrame) -> None:
    """Print dataset statistics to verify collection is correct."""

    subjects_pd = set(df[df['label'] == 1]['subject_id'].unique())
    subjects_hc = set(df[df['label'] == 0]['subject_id'].unique())
    overlap     = subjects_pd & subjects_hc

    print(f"\n{'='*55}")
    print(f"  DATASET AUDIT")
    print(f"{'='*55}")
    print(f"  Total recordings   : {len(df)}")
    print(f"  Total subjects     : {df['subject_id'].nunique()}")
    print(f"    PD subjects      : {len(subjects_pd)}")
    print(f"    HC subjects      : {len(subjects_hc)}")

    if overlap:
        # This should not happen — a subject is either PD or HC
        print(f"\n  *** WARNING: {len(overlap)} subjects appear in BOTH classes: {overlap}")

    print(f"\n  By task:")
    for task, grp in df.groupby('task'):
        n_pd = (grp['label'] == 1).sum()
        n_hc = (grp['label'] == 0).sum()
        print(f"    {task:<25} {len(grp):>4} files  "
              f"({n_pd} PD / {n_hc} HC)")

    print(f"\n  Duration stats (seconds):")
    print(f"    min    : {df['duration_s'].min():.2f}")
    print(f"    median : {df['duration_s'].median():.2f}")
    print(f"    mean   : {df['duration_s'].mean():.2f}")
    print(f"    max    : {df['duration_s'].max():.2f}")

    print(f"\n  Recordings per subject (top 5):")
    counts = df.groupby('subject_id').size().sort_values(ascending=False)
    for sid, n in counts.head(5).items():
        lbl = 'PD' if df[df['subject_id'] == sid]['label'].iloc[0] == 1 else 'HC'
        print(f"    {sid}  {lbl}  {n} recordings")
    print(f"{'='*55}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Subject-level stratified split
# ═══════════════════════════════════════════════════════════════════════════════

def make_splits(df: pd.DataFrame, seed: int = SEED) -> dict:
    """
    Split at SUBJECT level — no subject appears in more than one split.

    Stratified by class: PD and HC subjects are split independently
    to maintain the class ratio across train/val/test.

    Returns:
        {
          'train': ['ID02', 'ID05', ...],
          'val':   ['ID01', ...],
          'test':  ['ID00', 'ID03', ...]
        }
    """
    rng = np.random.default_rng(seed)

    # Get one row per subject (subject_id, label)
    subjects = (
        df[['subject_id', 'label']]
        .drop_duplicates('subject_id')
        .set_index('subject_id')
    )

    pd_subjects = subjects[subjects['label'] == 1].index.tolist()
    hc_subjects = subjects[subjects['label'] == 0].index.tolist()

    def _split_class(sids: list) -> tuple:
        arr = np.array(sorted(sids))
        rng.shuffle(arr)
        n       = len(arr)
        n_test  = max(1, round(n * TEST_RATIO))
        n_val   = max(1, round(n * VAL_RATIO))
        n_train = n - n_test - n_val
        if n_train < 1:
            raise ValueError(
                f"Too few subjects ({n}) for the requested split ratios. "
                f"Reduce VAL_RATIO / TEST_RATIO in data_prep.py."
            )
        train = arr[: n_train].tolist()
        val   = arr[n_train : n_train + n_val].tolist()
        test  = arr[n_train + n_val :].tolist()
        return train, val, test

    pd_tr, pd_va, pd_te = _split_class(pd_subjects)
    hc_tr, hc_va, hc_te = _split_class(hc_subjects)

    splits = {
        'train': sorted(pd_tr + hc_tr),
        'val':   sorted(pd_va + hc_va),
        'test':  sorted(pd_te + hc_te),
    }

    # Verify no overlap
    tr, va, te = set(splits['train']), set(splits['val']), set(splits['test'])
    assert not (tr & va), f"Leak: train ∩ val = {tr & va}"
    assert not (tr & te), f"Leak: train ∩ test = {tr & te}"
    assert not (va & te), f"Leak: val ∩ test = {va & te}"

    # Count PD/HC per split
    def _class_counts(sids):
        n_pd = sum(1 for s in sids if subjects.loc[s, 'label'] == 1)
        n_hc = len(sids) - n_pd
        return n_pd, n_hc

    print(f"\n  Subject-level splits (seed={seed}):")
    for split_name, sids in splits.items():
        n_pd, n_hc = _class_counts(sids)
        files_in   = df[df['subject_id'].isin(sids)]
        print(f"    {split_name:<6}: {len(sids):>2} subjects "
              f"({n_pd} PD / {n_hc} HC)  |  {len(files_in)} recordings")
    print(f"  No subject overlap across splits ✓")

    return splits


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(root_str: str, out_dir_str: str, dry_run: bool) -> None:

    root    = Path(root_str).resolve()
    out_dir = Path(out_dir_str).resolve()

    if not root.exists():
        print(f"ERROR: Dataset root not found: {root}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  MDVR-KCL Data Preparation")
    print(f"{'='*55}")
    print(f"  Root    : {root}")
    print(f"  Out dir : {out_dir}")
    print(f"  Dry run : {dry_run}")

    # ── Collect ──────────────────────────────────────────────────────────────
    print(f"\n[1] Scanning for .wav files...")
    records, skipped = collect_records(root)

    if skipped:
        print(f"\n  Skipped {len(skipped)} files:")
        for path, reason in skipped[:10]:
            print(f"    {Path(path).name:<40} {reason}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    if not records:
        print("ERROR: No valid recordings found. Check --root path.")
        sys.exit(1)

    df = pd.DataFrame(records)
    print(f"  Collected {len(df)} recordings from {df['subject_id'].nunique()} subjects")

    # ── Audit ────────────────────────────────────────────────────────────────
    print(f"\n[2] Dataset audit...")
    audit(df)

    # ── Splits ───────────────────────────────────────────────────────────────
    print(f"\n[3] Building subject-level splits...")
    splits = make_splits(df, seed=SEED)

    # ── Write ────────────────────────────────────────────────────────────────
    if dry_run:
        print(f"\n  [DRY RUN] No files written.")
        print(f"  Would write:")
        print(f"    {out_dir / 'labels.csv'}")
        print(f"    {out_dir / 'splits.json'}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path    = out_dir / 'labels.csv'
    splits_path = out_dir / 'splits.json'

    df.to_csv(csv_path, index=False)

    # Embed metadata in splits.json for full reproducibility
    splits_doc = {
        'seed':       SEED,
        'val_ratio':  VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'n_subjects': int(df['subject_id'].nunique()),
        'splits':     splits,
    }
    with open(splits_path, 'w') as fh:
        json.dump(splits_doc, fh, indent=2)

    print(f"\n[4] Output written:")
    print(f"    {csv_path}   ({len(df)} rows)")
    print(f"    {splits_path}")
    print(f"\n  Next step:")
    print(f"    python data/precompute.py --labels {csv_path} --out_dir data/npz_cache/")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDVR-KCL dataset preparation')
    parser.add_argument(
        '--root',
        type=str,
        required=True,
        help='Path to 26-29_09_2017_KCL/ directory',
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='data',
        help='Directory to write labels.csv and splits.json (default: data/)',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Scan and audit without writing any files',
    )
    args = parser.parse_args()
    main(args.root, args.out_dir, args.dry_run)