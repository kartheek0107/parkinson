"""
audio_segmentation.py
=====================
Segments 2-minute KCL PD/HC audio files into overlapping 10-second windows
with PD-safe silence rejection.

Dataset structure expected:
    26-29_09_2017_KCL/
    ├── ReadText/
    │   ├── HC/  *.wav
    │   └── PD/  *.wav
    └── SpontaneousDialogue/
        ├── HC/  *.wav
        └── PD/  *.wav

Output:
    segments_output/
    ├── ReadText/HC/*.wav   ReadText/PD/*.wav
    ├── SpontaneousDialogue/HC/*.wav  SpontaneousDialogue/PD/*.wav
    ├── manifest.csv        (all segments + metadata)
    ├── train.csv / val.csv / test.csv   (if --split used)
    └── silence_report.csv  (rejected segments for audit)

Usage:
    python audio_segmentation.py \
        --data_root  "/mnt/c/Users/.../data/26-29_09_2017_KCL" \
        --output_dir "/mnt/c/Users/.../data/segments_output" \
        --split
"""

import os
import re
import argparse
import math
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
SEGMENT_SEC = 10
HOP_SEC     = 5
SR          = 16000
TASKS       = ["ReadText", "SpontaneousDialogue"]
CLASSES     = {"HC": 0, "PD": 1}

# ── Silence rejection parameters ─────────────────────────────────────────────
# CRITICAL for PD: patients exhibit hypophonia (soft speech, RMS ~0.005–0.01).
# Thresholds are conservative to avoid discarding valid soft PD speech.
#
# Rejection logic — segment is REJECTED only if BOTH are true:
#   1. Segment RMS  < RMS_SILENCE_THRESHOLD   (whole segment energy too low)
#   2. Speech ratio < MIN_SPEECH_RATIO        (<30% of 20ms frames are active)
#
# Dual-gate prevents:
#   - Rejecting PD hypophonic speech (low RMS but speech IS present)
#   - Keeping segments that are 95% silence with a single word
RMS_SILENCE_THRESHOLD  = 0.002   # -54 dBFS — below = silence / mic noise only
FRAME_SPEECH_THRESHOLD = 0.003   # per-20ms-frame RMS to count as "speech active"
FRAME_SIZE_SEC         = 0.02    # 20ms frames (standard VAD frame size)
MIN_SPEECH_RATIO       = 0.30    # ≥30% of frames must be speech-active to keep


# ─────────────────────────────────────────────────────────────────────────────
# SILENCE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def is_silent_segment(audio: np.ndarray, sr: int,
                      rms_thresh: float = RMS_SILENCE_THRESHOLD,
                      frame_thresh: float = FRAME_SPEECH_THRESHOLD,
                      frame_size_sec: float = FRAME_SIZE_SEC,
                      min_speech_ratio: float = MIN_SPEECH_RATIO):
    """
    Two-gate silence detector, conservative for PD hypophonia.

    Returns:
        silent (bool), diagnostics (dict)
    """
    # Gate 1: whole-segment RMS energy
    seg_rms = float(np.sqrt(np.mean(audio ** 2)))

    # Gate 2: fraction of 20ms frames with speech activity
    frame_len = max(1, int(frame_size_sec * sr))
    n_frames  = len(audio) // frame_len
    if n_frames == 0:
        return True, {"seg_rms": seg_rms, "speech_ratio": 0.0, "seg_rms_dbfs": -240.0}

    frames       = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms    = np.sqrt(np.mean(frames ** 2, axis=1))
    speech_ratio = float(np.mean(frame_rms > frame_thresh))

    # Reject only when BOTH gates agree
    silent = (seg_rms < rms_thresh) and (speech_ratio < min_speech_ratio)

    return silent, {
        "seg_rms":      round(seg_rms, 6),
        "speech_ratio": round(speech_ratio, 4),
        "seg_rms_dbfs": round(20 * np.log10(seg_rms + 1e-12), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_subject_id(filename: str) -> str:
    """
    Extract subject ID from filename.
    e.g., 'ID00_hc_0_0_0.wav' → 'ID00'
    Falls back to full stem if pattern not found.
    """
    match = re.match(r'^(ID\d+)', filename, re.IGNORECASE)
    return match.group(1).upper() if match else Path(filename).stem


def collect_files(data_root: Path) -> list:
    """
    Walk the KCL directory tree and collect all wav files with metadata.
    Returns list of dicts: {path, task, label_str, label_int, subject_id}
    """
    records = []
    for task in TASKS:
        for label_str, label_int in CLASSES.items():
            folder = data_root / task / label_str
            if not folder.exists():
                print(f"  ⚠ Folder not found, skipping: {folder}")
                continue
            wavs = sorted(folder.glob("*.wav"))
            for wav in wavs:
                records.append({
                    "path":       wav,
                    "task":       task,
                    "label_str":  label_str,
                    "label_int":  label_int,
                    "subject_id": parse_subject_id(wav.name),
                })
    return records


def segment_audio(audio: np.ndarray, sr: int,
                  segment_sec: float, hop_sec: float) -> list:
    """
    Slice audio into overlapping windows.
    Drops any trailing window shorter than segment_sec (no padding artifacts).
    """
    seg_len  = int(segment_sec * sr)
    hop_len  = int(hop_sec * sr)
    n_audio  = len(audio)
    segments = []
    start = 0
    idx   = 0
    while start + seg_len <= n_audio:
        end = start + seg_len
        segments.append({
            "idx":       idx,
            "start_sec": round(start / sr, 3),
            "end_sec":   round(end   / sr, 3),
            "data":      audio[start:end],
        })
        start += hop_len
        idx   += 1
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SEGMENTATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation(data_root: Path, output_dir: Path,
                     segment_sec: float, hop_sec: float, sr: int) -> None:

    print("\n" + "=" * 65)
    print("  KCL AUDIO SEGMENTATION PIPELINE")
    print("=" * 65)
    print(f"  Data root    : {data_root}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Segment len  : {segment_sec}s")
    print(f"  Hop size     : {hop_sec}s  ({int((1 - hop_sec/segment_sec)*100)}% overlap)")
    print(f"  Sample rate  : {sr} Hz")
    print(f"  Expected segs: ~{math.floor((120 - segment_sec) / hop_sec) + 1} per 2-min file")
    print(f"  VAD thresholds:")
    print(f"    RMS silence  < {RMS_SILENCE_THRESHOLD} ({20*np.log10(RMS_SILENCE_THRESHOLD):.0f} dBFS)")
    print(f"    Speech ratio < {MIN_SPEECH_RATIO*100:.0f}% of 20ms frames")
    print("=" * 65 + "\n")

    # ── 1. Collect source files ───────────────────────────────────────────────
    records = collect_files(data_root)
    if not records:
        raise RuntimeError(f"No wav files found under {data_root}. Check your path.")

    print(f"  Found {len(records)} source files:")
    for task in TASKS:
        for cls in CLASSES:
            n = sum(1 for r in records if r["task"] == task and r["label_str"] == cls)
            print(f"    {task}/{cls}: {n} files")

    # ── 2. Prepare output directories ─────────────────────────────────────────
    for task in TASKS:
        for cls in CLASSES:
            (output_dir / task / cls).mkdir(parents=True, exist_ok=True)

    # ── 3. Segment + silence filter ───────────────────────────────────────────
    manifest_rows = []
    silence_rows  = []
    stats         = defaultdict(int)

    for rec in tqdm(records, desc="Segmenting", unit="file"):
        try:
            audio, _ = librosa.load(rec["path"], sr=sr, mono=True)
        except Exception as e:
            print(f"\n  ✗ Failed to load {rec['path'].name}: {e}")
            stats["load_errors"] += 1
            continue

        actual_dur = len(audio) / sr
        if actual_dur < segment_sec:
            print(f"\n  ⚠ Skipping {rec['path'].name} — too short ({actual_dur:.1f}s)")
            stats["too_short"] += 1
            continue

        segs       = segment_audio(audio, sr, segment_sec, hop_sec)
        out_folder = output_dir / rec["task"] / rec["label_str"]
        stem       = rec["path"].stem

        for seg in segs:
            silent, diag = is_silent_segment(seg["data"], sr)

            if silent:
                # Log to silence report but do NOT write wav
                silence_rows.append({
                    "parent_file":  rec["path"].name,
                    "seg_idx":      seg["idx"],
                    "start_sec":    seg["start_sec"],
                    "end_sec":      seg["end_sec"],
                    "label_str":    rec["label_str"],
                    "subject_id":   rec["subject_id"],
                    "task":         rec["task"],
                    "seg_rms":      diag["seg_rms"],
                    "seg_rms_dbfs": diag["seg_rms_dbfs"],
                    "speech_ratio": diag["speech_ratio"],
                })
                stats["silent_rejected"] += 1
                stats[f"silent_{rec['label_str']}"] += 1
                continue

            # Write valid segment
            seg_filename = f"{stem}_seg{seg['idx']:03d}.wav"
            seg_path     = out_folder / seg_filename
            sf.write(str(seg_path), seg["data"], sr, subtype="PCM_16")

            manifest_rows.append({
                "segment_path": str(seg_path.relative_to(output_dir)),
                "label":        rec["label_int"],
                "label_str":    rec["label_str"],
                "subject_id":   rec["subject_id"],
                "task":         rec["task"],
                "parent_file":  rec["path"].name,
                "seg_idx":      seg["idx"],
                "start_sec":    seg["start_sec"],
                "end_sec":      seg["end_sec"],
                "seg_rms":      diag["seg_rms"],
                "speech_ratio": diag["speech_ratio"],
            })
            stats[f"{rec['task']}_{rec['label_str']}"] += 1
            stats["total_kept"] += 1

        stats["total_files"] += 1

    # ── 4. Write manifest CSV ──────────────────────────────────────────────────
    manifest_path = output_dir / "manifest.csv"
    manifest_fields = ["segment_path", "label", "label_str", "subject_id",
                       "task", "parent_file", "seg_idx",
                       "start_sec", "end_sec", "seg_rms", "speech_ratio"]
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        w.writerows(manifest_rows)

    # ── 5. Write silence report ────────────────────────────────────────────────
    silence_path = output_dir / "silence_report.csv"
    silence_fields = ["parent_file", "seg_idx", "start_sec", "end_sec",
                      "label_str", "subject_id", "task",
                      "seg_rms", "seg_rms_dbfs", "speech_ratio"]
    with open(silence_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=silence_fields)
        w.writeheader()
        w.writerows(silence_rows)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    total_gen = stats["total_kept"] + stats["silent_rejected"]
    rej_pct   = 100 * stats["silent_rejected"] / total_gen if total_gen else 0

    print("\n" + "=" * 65)
    print("  SEGMENTATION COMPLETE — SUMMARY")
    print("=" * 65)
    print(f"  Source files processed : {stats['total_files']}")
    print(f"  Load errors            : {stats['load_errors']}")
    print(f"  Too-short skipped      : {stats['too_short']}")
    print(f"  Segments generated     : {total_gen}")
    print(f"  Silent rejected        : {stats['silent_rejected']}  ({rej_pct:.1f}%)")
    print(f"    └─ HC silent         : {stats['silent_HC']}")
    print(f"    └─ PD silent         : {stats['silent_PD']}")
    print(f"  Segments KEPT          : {stats['total_kept']}")
    print()
    for task in TASKS:
        for cls in CLASSES:
            key = f"{task}_{cls}"
            print(f"    {task}/{cls}: {stats[key]} kept")
    print(f"\n  Manifest     → {manifest_path}")
    print(f"  Silence log  → {silence_path}")

    # ── 7. Bias warning ───────────────────────────────────────────────────────
    silent_hc = stats["silent_HC"]
    silent_pd = stats["silent_PD"]
    if total_gen > 0 and silent_pd > 0:
        pd_total = sum(stats[f"{t}_PD"] for t in TASKS) + silent_pd
        pd_rej_pct = 100 * silent_pd / pd_total if pd_total else 0
        hc_total = sum(stats[f"{t}_HC"] for t in TASKS) + silent_hc
        hc_rej_pct = 100 * silent_hc / hc_total if hc_total else 0
        if abs(pd_rej_pct - hc_rej_pct) > 10:
            print(f"\n  ⚠ BIAS WARNING:")
            print(f"    HC silent rejection rate: {hc_rej_pct:.1f}%")
            print(f"    PD silent rejection rate: {pd_rej_pct:.1f}%")
            print(f"    Difference > 10% — consider raising MIN_SPEECH_RATIO")
            print(f"    or inspect silence_report.csv for PD entries.")

    # ── 8. Leakage check ──────────────────────────────────────────────────────
    subjects_hc = set(r["subject_id"] for r in manifest_rows if r["label_str"] == "HC")
    subjects_pd = set(r["subject_id"] for r in manifest_rows if r["label_str"] == "PD")
    overlap     = subjects_hc & subjects_pd
    print(f"\n  LEAKAGE CHECK:")
    print(f"    Unique HC subjects: {len(subjects_hc)}")
    print(f"    Unique PD subjects: {len(subjects_pd)}")
    if overlap:
        print(f"    ⚠ OVERLAP: {overlap} — inspect your dataset labels")
    else:
        print(f"    ✓ No subject overlap between classes")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SUBJECT-LEVEL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def subject_level_split(manifest_csv: str,
                        train_ratio: float = 0.70,
                        val_ratio:   float = 0.15,
                        test_ratio:  float = 0.15,
                        seed: int = 42) -> None:
    """
    Produces train/val/test CSVs with zero subject overlap across splits.
    Uses GroupShuffleSplit on subject_id — non-negotiable for medical ML.
    """
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit

    df       = pd.read_csv(manifest_csv)
    subjects = df.drop_duplicates("subject_id")[["subject_id", "label_str"]].reset_index(drop=True)

    gss1 = GroupShuffleSplit(n_splits=1,
                             test_size=(val_ratio + test_ratio),
                             random_state=seed)
    train_idx, temp_idx = next(gss1.split(subjects, groups=subjects["subject_id"]))
    train_subs = set(subjects.iloc[train_idx]["subject_id"])
    temp_df    = subjects.iloc[temp_idx].reset_index(drop=True)

    val_frac = val_ratio / (val_ratio + test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=(1 - val_frac), random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["subject_id"]))
    val_subs  = set(temp_df.iloc[val_idx]["subject_id"])
    test_subs = set(temp_df.iloc[test_idx]["subject_id"])

    def assign(sid):
        if sid in train_subs: return "train"
        if sid in val_subs:   return "val"
        return "test"

    df["split"] = df["subject_id"].map(assign)
    out_dir = Path(manifest_csv).parent

    print("\n  SUBJECT-LEVEL SPLITS")
    print("  " + "-" * 55)
    for split_name in ["train", "val", "test"]:
        sdf = df[df["split"] == split_name]
        hc  = (sdf["label"] == 0).sum()
        pd_ = (sdf["label"] == 1).sum()
        n_s = sdf["subject_id"].nunique()
        out = out_dir / f"{split_name}.csv"
        sdf.to_csv(out, index=False)
        print(f"  {split_name:5s}: {len(sdf):6,} segs | HC={hc:,}  PD={pd_:,} | {n_s} subjects")

    print("\n  ✓ Zero subject overlap guaranteed across all splits.")
    print("  ✓ Use train.csv / val.csv / test.csv in your DataLoader.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment KCL PD/HC audio into overlapping 10s windows with silence rejection.")
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Absolute path to 26-29_09_2017_KCL/")
    parser.add_argument("--output_dir",  type=str, default="segments_output",
                        help="Where to write segments + manifest.csv")
    parser.add_argument("--segment_sec", type=float, default=SEGMENT_SEC)
    parser.add_argument("--hop_sec",     type=float, default=HOP_SEC)
    parser.add_argument("--sr",          type=int,   default=SR)
    parser.add_argument("--split",       action="store_true",
                        help="Generate train/val/test split CSVs after segmentation")
    # Allow tuning silence thresholds from CLI for experimentation
    parser.add_argument("--rms_threshold",   type=float, default=RMS_SILENCE_THRESHOLD,
                        help=f"RMS silence threshold (default: {RMS_SILENCE_THRESHOLD})")
    parser.add_argument("--speech_ratio",    type=float, default=MIN_SPEECH_RATIO,
                        help=f"Min speech frame ratio to keep segment (default: {MIN_SPEECH_RATIO})")
    args = parser.parse_args()

    # Override globals if CLI args passed
    RMS_SILENCE_THRESHOLD = args.rms_threshold
    MIN_SPEECH_RATIO      = args.speech_ratio

    data_root  = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    print(f"\n  Resolved data_root : {data_root}")
    print(f"  Resolved output_dir: {output_dir}")
    if not data_root.exists():
        raise FileNotFoundError(
            f"\n  ✗ data_root not found: {data_root}\n"
            f"    Use the full absolute path:\n"
            f"    --data_root \"/mnt/c/Users/Kartheek Budime/DQLCT_Parkinsons/data/26-29_09_2017_KCL\""
        )

    run_segmentation(data_root, output_dir, args.segment_sec, args.hop_sec, args.sr)

    if args.split:
        print("  Generating subject-level splits...")
        subject_level_split(str(output_dir / "manifest.csv"))