"""
parse_italian_pvs.py
====================
Builds manifest.csv for Italian Parkinson's Voice and Speech dataset
(Dimauro et al. 2017, IEEE Access).

Confirmed dataset structure:
    <root>/
        15 Young Healthy Control/
            <Subject Name>/
                B1.wav  B2.wav  D1.wav  D2.wav
                FB1.wav FB2.wav
                VA1.wav VA2.wav  VE1.wav VE2.wav
                VI1.wav VI2.wav  VO1.wav VO2.wav  VU1.wav VU2.wav
        22 Elderly Healthy Control/
            <Subject Name>/
                (same task files)
        28 Parkinson/
            <Subject Name>/
                (same task files)

Group → Label mapping:
    "15 Young Healthy Control"  → HC  (subtype: YHC)
    "22 Elderly Healthy Control" → HC  (subtype: EHC)
    "28 Parkinson"               → PD

Task code → canonical type:
    B1, B2, FB1, FB2  → ReadText    (10-20s, segments to 10s windows)
    D1, D2            → DDK         (5s, padded to 10s)
    VA1..VU2          → Vowel_X     (3-5s, padded to 10s)

Output manifest.csv columns (identical to KCL manifest.csv):
    segment_path, subject_id, label_str, label, task, parent_file,
    seg_idx, duration

Usage:
    python parse_italian_pvs.py \\
        --root   "path/to/Italian Parkinson's Voice and speech" \\
        --output "path/to/italian_manifest.csv"
"""

import os
import re
import csv
import argparse
from pathlib import Path

try:
    import librosa
except ImportError:
    print("pip install librosa")
    raise

# ─────────────────────────────────────────────────────────────────────────────
# Exact mappings from confirmed structure
# ─────────────────────────────────────────────────────────────────────────────

GROUP_LABEL = {
    '15 young healthy control':   ('HC', 'YHC'),
    '22 elderly healthy control': ('HC', 'EHC'),
    '28 parkinson':               ('PD', 'PD'),
    "28 people with parkinson's disease": ('PD', 'PD'),
}

TASK_TYPE = {
    'B1': 'ReadText',  'B2': 'ReadText',
    'FB1':'ReadText',  'FB2':'ReadText',
    'D1': 'DDK',       'D2': 'DDK',
    'VA1':'Vowel_A',   'VA2':'Vowel_A',
    'VE1':'Vowel_E',   'VE2':'Vowel_E',
    'VI1':'Vowel_I',   'VI2':'Vowel_I',
    'VO1':'Vowel_O',   'VO2':'Vowel_O',
    'VU1':'Vowel_U',   'VU2':'Vowel_U',
    'PR1':'ReadText',  'PR2':'ReadText',   # "prova" reading
    'PR11':'ReadText',
}

# Regex to extract the task code prefix from filenames like B1APGANRET55F170320171104
_TASK_PREFIX_RE = re.compile(
    r'^(FB[12]|PR11|PR[12]|B[12]|D[12]|V[AEIOU][12])',
    re.IGNORECASE,
)


def sanitise_id(name: str) -> str:
    """'Alberto R' → 'Alberto_R' (safe for filenames and npz stems)."""
    return re.sub(r'[^A-Za-z0-9]', '_', name.strip()).strip('_')


def build_manifest(root: str, output: str) -> None:
    root_path = Path(root)
    out_path  = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Italian PVS  —  Manifest Builder")
    print(f"{'='*62}")
    print(f"  Root: {root_path}")

    rows         = []
    unknown_grps = []
    unknown_task = []
    subj_seg_idx = {}   # subject_id → running file counter (for seg_idx)

    # Walk: root / group_folder / subject_folder / task.wav
    for group_dir in sorted(root_path.iterdir()):
        if not group_dir.is_dir():
            continue

        group_key = group_dir.name.strip().lower()
        # Fuzzy match (handles minor spelling variants)
        label_str, subtype = None, None
        for key, (lbl, sub) in GROUP_LABEL.items():
            if key in group_key or group_key in key:
                label_str, subtype = lbl, sub
                break
        if label_str is None:
            # Try partial keyword match
            if 'parkinson' in group_key or 'pd' in group_key:
                label_str, subtype = 'PD', 'PD'
            elif 'healthy' in group_key or 'control' in group_key or 'hc' in group_key:
                label_str, subtype = 'HC', 'HC'
            else:
                unknown_grps.append(group_dir.name)
                continue

        label_int = 1 if label_str == 'PD' else 0

        # Collect subject dirs — handle PD's nested range sub-folders
        subj_dirs = []
        for child in sorted(group_dir.iterdir()):
            if not child.is_dir():
                continue
            # If the child looks like a range folder (e.g. "1-5", "6-10")
            if re.match(r'^\d+-\d+$', child.name):
                # Go one level deeper to find actual subject folders
                for subchild in sorted(child.iterdir()):
                    if subchild.is_dir():
                        subj_dirs.append(subchild)
            else:
                subj_dirs.append(child)

        for subj_dir in subj_dirs:
            subject_id = sanitise_id(subj_dir.name)

            for wav_path in sorted(subj_dir.glob('*.wav')) + \
                            sorted(subj_dir.glob('*.WAV')):

                # Extract task code prefix from filename
                stem = wav_path.stem.strip()
                m = _TASK_PREFIX_RE.match(stem)
                if m:
                    task_code = m.group(1).upper()
                else:
                    task_code = stem.upper()
                task_type = TASK_TYPE.get(task_code)

                if task_type is None:
                    unknown_task.append(f"{subj_dir.name}/{wav_path.name}")
                    # Still include with Unknown type rather than discard
                    task_type = 'Unknown'

                try:
                    duration = librosa.get_duration(path=str(wav_path))
                except Exception:
                    duration = 0.0

                seg_idx = subj_seg_idx.get(subject_id, 0)
                subj_seg_idx[subject_id] = seg_idx + 1

                rows.append({
                    'segment_path': str(wav_path.relative_to(root_path)),
                    'subject_id':   subject_id,
                    'label_str':    label_str,
                    'label':        label_int,
                    'task':         task_type,
                    'task_code':    task_code,   # extra column: B1, VA1, etc.
                    'subtype':      subtype,     # extra column: YHC / EHC / PD
                    'parent_file':  wav_path.stem,
                    'seg_idx':      seg_idx,
                    'duration':     round(duration, 3),
                })

    # Write manifest
    fieldnames = ['segment_path', 'subject_id', 'label_str', 'label',
                  'task', 'task_code', 'subtype', 'parent_file',
                  'seg_idx', 'duration']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ────────────────────────────────────────────────────────────────
    hc_rows = [r for r in rows if r['label_str'] == 'HC']
    pd_rows = [r for r in rows if r['label_str'] == 'PD']
    subjs_hc = sorted(set(r['subject_id'] for r in hc_rows))
    subjs_pd = sorted(set(r['subject_id'] for r in pd_rows))

    task_counts = {}
    for r in rows:
        k = r['task_code']
        task_counts[k] = task_counts.get(k, 0) + 1

    print(f"\n  ── Manifest Summary ──────────────────────────────────")
    print(f"  Total WAV files  : {len(rows)}")
    print(f"  HC files         : {len(hc_rows)}  ({len(subjs_hc)} subjects)")
    print(f"  PD files         : {len(pd_rows)}  ({len(subjs_pd)} subjects)")
    print(f"  Unknown groups   : {len(unknown_grps)}")
    print(f"  Unknown task codes: {len(unknown_task)}")
    print(f"\n  Task code distribution:")
    for code in sorted(task_counts):
        ttype = TASK_TYPE.get(code, 'Unknown')
        print(f"    {code:<5}  {ttype:<12}  {task_counts[code]:3d} files")

    import numpy as np
    durs = [r['duration'] for r in rows if r['duration'] > 0]
    print(f"\n  Duration stats:")
    print(f"    Mean   : {np.mean(durs):.1f}s")
    print(f"    Min    : {np.min(durs):.1f}s")
    print(f"    Max    : {np.max(durs):.1f}s")
    print(f"    < 3s   : {sum(1 for d in durs if d < 3)}")
    print(f"    < 10s  : {sum(1 for d in durs if d < 10)}  (will be padded)")
    print(f"    >= 10s : {sum(1 for d in durs if d >= 10)}  (will be windowed)")

    if unknown_grps:
        print(f"\n  ⚠ Unrecognised group folders: {unknown_grps}")
    if unknown_task:
        print(f"\n  ⚠ Unknown task codes (first 5): {unknown_task[:5]}")

    print(f"\n  Manifest saved → {out_path}")
    print(f"\n  Next step:")
    print(f"    python precompute_italian.py \\")
    print(f"        --manifest  {out_path} \\")
    print(f"        --seg_root  {root_path} \\")
    print(f"        --npz_dir   /path/to/italian_npz_cache")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Build manifest.csv for Italian PVS dataset")
    p.add_argument('--root',   required=True,
                   help="Root of Italian PVS dataset "
                        "(contains '15 Young...', '22 Elderly...', '28 Parkinson' folders)")
    p.add_argument('--output', required=True,
                   help="Output path for manifest.csv")
    args = p.parse_args()
    build_manifest(args.root, args.output)