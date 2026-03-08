"""
evaluate_subjects.py
====================
Subject-level voting evaluation for Q-CRNN PD detection.

Loads saved fold checkpoints, runs inference on test windows,
aggregates predictions per subject, and reports subject-level metrics.

NO RETRAINING REQUIRED — uses existing best_model.pt from each fold.

Aggregation strategies:
    1. Simple mean     — average probability across all windows
    2. Confidence-wtd  — weight = |prob - 0.5|  (certain windows count more)
    3. Median          — robust to outlier windows

Usage:
    python src/models/evaluate_subjects.py \
        --npz_dir  src/npz_cache \
        --cv_dir   src/models/checkpoints/cv_run1
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

# ── Path setup ────────────────────────────────────────────────────────────────
_MODELS_DIR = Path(__file__).resolve().parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from qcrnn_model import build_model


# ═══════════════════════════════════════════════════════════════════════════════
# Lightweight dataset — loads all test windows for a given subject list
# ═══════════════════════════════════════════════════════════════════════════════

class SubjectWindowDataset(torch.utils.data.Dataset):
    """
    Loads .npz windows for specified subjects.
    Returns (features, label, subject_id) per window.
    """

    def __init__(self, npz_dir: str, subject_ids: list):
        self.files      = []
        self.labels     = []
        self.subject_ids = []
        subject_set     = set(subject_ids)
        npz_dir         = Path(npz_dir)

        for f in sorted(npz_dir.glob('*.npz')):
            name = f.stem
            try:
                sid   = name.split('_label')[0].replace('subj', '')
                label = int(name.split('_label')[1].split('_')[0])
            except (IndexError, ValueError):
                continue
            if sid not in subject_set:
                continue
            self.files.append(f)
            self.labels.append(label)
            self.subject_ids.append(sid)

        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        x    = torch.from_numpy(data['features'].astype(np.float32))
        return x, self.labels[idx], self.subject_ids[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Subject-level aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_subject_predictions(subject_probs: dict, subject_labels: dict):
    """
    Aggregate window-level probabilities per subject.

    Args:
        subject_probs:  {subject_id: [prob1, prob2, ...]}
        subject_labels: {subject_id: true_label (0 or 1)}

    Returns:
        dict with per-subject predictions under each strategy
    """
    results = {}

    for sid in sorted(subject_probs.keys()):
        probs = np.array(subject_probs[sid])
        true_label = subject_labels[sid]

        # Strategy 1: Simple mean
        mean_prob = float(np.mean(probs))

        # Strategy 2: Confidence-weighted mean
        #   weight = |prob - 0.5| → uncertain windows near 0.5 get low weight
        #   Ensures weights sum to 1 for fair comparison
        confidences = np.abs(probs - 0.5)
        if confidences.sum() > 0:
            weights = confidences / confidences.sum()
            conf_wtd_prob = float(np.sum(probs * weights))
        else:
            conf_wtd_prob = mean_prob  # fallback if all at 0.5

        # Strategy 3: Median
        median_prob = float(np.median(probs))

        results[sid] = {
            'true_label':     int(true_label),
            'n_windows':      len(probs),
            'mean_prob':      round(mean_prob, 4),
            'conf_wtd_prob':  round(conf_wtd_prob, 4),
            'median_prob':    round(median_prob, 4),
            'pred_mean':      int(mean_prob >= 0.5),
            'pred_conf_wtd':  int(conf_wtd_prob >= 0.5),
            'pred_median':    int(median_prob >= 0.5),
        }

    return results


def compute_subject_metrics(subject_results: dict, prob_key: str,
                            pred_key: str):
    """Compute accuracy, sensitivity, specificity, F1 from subject-level preds."""
    true_labels = []
    pred_labels = []
    pred_probs  = []

    for sid, r in subject_results.items():
        true_labels.append(r['true_label'])
        pred_labels.append(r[pred_key])
        pred_probs.append(r[prob_key])

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs  = np.array(pred_probs)

    tp = int(((pred_labels == 1) & (true_labels == 1)).sum())
    tn = int(((pred_labels == 0) & (true_labels == 0)).sum())
    fp = int(((pred_labels == 1) & (true_labels == 0)).sum())
    fn = int(((pred_labels == 0) & (true_labels == 1)).sum())

    n     = tp + tn + fp + fn
    acc   = (tp + tn) / max(n, 1)
    sens  = tp / max(tp + fn, 1)
    spec  = tn / max(tn + fp, 1)
    prec  = tp / max(tp + fp, 1)
    f1    = 2 * prec * sens / max(prec + sens, 1e-9)

    # Subject-level AUC (if enough unique labels)
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(true_labels, pred_probs)
    except ValueError:
        auc = float('nan')

    return {
        'accuracy':    round(acc, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'f1':          round(f1, 4),
        'auc':         round(auc, 4),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'n_subjects':  n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-fold inference
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_fold_inference(fold_dir: Path, npz_dir: str, device: torch.device):
    """
    Load a fold's best model, run inference on test subjects,
    return per-window (subject_id, probability, true_label).
    """
    # Load fold metadata
    results_path = fold_dir / 'test_results.json'
    if not results_path.exists():
        print(f"  ⚠  Skipping {fold_dir.name}: no test_results.json")
        return None, None

    with open(results_path) as f:
        fold_meta = json.load(f)

    test_sids = fold_meta['test_subjs']

    # Load model
    ckpt_path = fold_dir / 'best_model.pt'
    if not ckpt_path.exists():
        print(f"  ⚠  Skipping {fold_dir.name}: no best_model.pt")
        return None, None

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = build_model(device=str(device))
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Dataset
    ds = SubjectWindowDataset(npz_dir, test_sids)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                        pin_memory=(device.type == 'cuda'))

    # Inference
    subject_probs  = defaultdict(list)
    subject_labels = {}

    for x, labels, sids in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu',
                      dtype=torch.float16 if device.type == 'cuda' else torch.float32):
            logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

        for i in range(len(sids)):
            sid = sids[i]
            subject_probs[sid].append(float(probs[i]))
            subject_labels[sid] = int(labels[i])

    return dict(subject_probs), dict(subject_labels)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Subject-level voting evaluation for Q-CRNN PD detection.')
    parser.add_argument('--npz_dir', required=True,
                        help='Path to npz_cache/ directory')
    parser.add_argument('--cv_dir', required=True,
                        help='Path to CV output dir (e.g. checkpoints/cv_run1)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold (default 0.5)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cv_dir = Path(args.cv_dir)

    print(f"\n{'='*65}")
    print(f"  Subject-Level Voting Evaluation")
    print(f"{'='*65}")
    print(f"  Device  : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ''))
    print(f"  NPZ dir : {args.npz_dir}")
    print(f"  CV dir  : {cv_dir}")
    print(f"  Thresh  : {args.threshold}")

    # ── Collect predictions across all folds ──────────────────────────────────
    all_subject_probs  = defaultdict(list)
    all_subject_labels = {}
    fold_subject_results = {}
    n_folds_found = 0

    fold_dirs = sorted(cv_dir.glob('fold*'))
    if not fold_dirs:
        print(f"\n  ✗  No fold directories found in {cv_dir}")
        return

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        print(f"\n  ── {fold_name} ──")

        subject_probs, subject_labels = run_fold_inference(
            fold_dir, args.npz_dir, device)

        if subject_probs is None:
            continue
        n_folds_found += 1

        # Store per-fold subject results
        fold_subj_results = aggregate_subject_predictions(
            subject_probs, subject_labels)
        fold_subject_results[fold_name] = fold_subj_results

        # Accumulate for cross-fold summary
        for sid, probs in subject_probs.items():
            all_subject_probs[sid].extend(probs)
            all_subject_labels[sid] = subject_labels[sid]

        # Print per-fold per-subject table
        print(f"    {'Subject':<10} {'True':>5} {'#Win':>5} "
              f"{'Mean':>7} {'ConfWtd':>8} {'Median':>7} "
              f"{'Pred_M':>7} {'Pred_C':>7} {'Pred_D':>7}")
        print(f"    {'─'*10} {'─'*5} {'─'*5} "
              f"{'─'*7} {'─'*8} {'─'*7} "
              f"{'─'*7} {'─'*7} {'─'*7}")

        for sid in sorted(fold_subj_results.keys()):
            r = fold_subj_results[sid]
            label_str = 'PD' if r['true_label'] == 1 else 'HC'
            pred_m = '✓' if r['pred_mean']     == r['true_label'] else '✗'
            pred_c = '✓' if r['pred_conf_wtd']  == r['true_label'] else '✗'
            pred_d = '✓' if r['pred_median']    == r['true_label'] else '✗'
            print(f"    {sid:<10} {label_str:>5} {r['n_windows']:>5} "
                  f"{r['mean_prob']:>7.4f} {r['conf_wtd_prob']:>8.4f} "
                  f"{r['median_prob']:>7.4f} "
                  f"{pred_m:>7} {pred_c:>7} {pred_d:>7}")

        # Per-fold subject-level metrics
        for strategy, prob_key, pred_key in [
            ('Simple Mean',      'mean_prob',     'pred_mean'),
            ('Confidence-Wtd',   'conf_wtd_prob', 'pred_conf_wtd'),
            ('Median',           'median_prob',   'pred_median'),
        ]:
            m = compute_subject_metrics(fold_subj_results, prob_key, pred_key)
            print(f"\n    {strategy}: "
                  f"Acc={m['accuracy']:.3f}  "
                  f"Sen={m['sensitivity']:.3f}  "
                  f"Spe={m['specificity']:.3f}  "
                  f"F1={m['f1']:.3f}  "
                  f"AUC={m['auc']:.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Cross-fold aggregate (all subjects across all folds)
    # ═══════════════════════════════════════════════════════════════════════════

    if n_folds_found == 0:
        print("\n  ✗  No folds with results found.")
        return

    print(f"\n{'='*65}")
    print(f"  CROSS-FOLD SUBJECT-LEVEL RESULTS  ({n_folds_found} folds)")
    print(f"{'='*65}")
    print(f"  Total subjects evaluated: {len(all_subject_probs)}")

    overall_results = aggregate_subject_predictions(
        all_subject_probs, all_subject_labels)

    # Print full subject table
    print(f"\n  {'Subject':<10} {'True':>5} {'#Win':>5} "
          f"{'Mean':>7} {'ConfWtd':>8} {'Median':>7} "
          f"{'Pred_M':>7} {'Pred_C':>7} {'Pred_D':>7}")
    print(f"  {'─'*10} {'─'*5} {'─'*5} "
          f"{'─'*7} {'─'*8} {'─'*7} "
          f"{'─'*7} {'─'*7} {'─'*7}")

    for sid in sorted(overall_results.keys()):
        r = overall_results[sid]
        label_str = 'PD' if r['true_label'] == 1 else 'HC'
        pred_m = '✓' if r['pred_mean']     == r['true_label'] else '✗'
        pred_c = '✓' if r['pred_conf_wtd']  == r['true_label'] else '✗'
        pred_d = '✓' if r['pred_median']    == r['true_label'] else '✗'
        print(f"  {sid:<10} {label_str:>5} {r['n_windows']:>5} "
              f"{r['mean_prob']:>7.4f} {r['conf_wtd_prob']:>8.4f} "
              f"{r['median_prob']:>7.4f} "
              f"{pred_m:>7} {pred_c:>7} {pred_d:>7}")

    # Compute overall metrics
    print(f"\n  {'Strategy':<18} {'Acc':>7} {'Sen':>7} {'Spe':>7} "
          f"{'F1':>7} {'AUC':>7}")
    print(f"  {'─'*18} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    all_metrics = {}
    for strategy, prob_key, pred_key in [
        ('Simple Mean',      'mean_prob',     'pred_mean'),
        ('Confidence-Wtd',   'conf_wtd_prob', 'pred_conf_wtd'),
        ('Median',           'median_prob',   'pred_median'),
    ]:
        m = compute_subject_metrics(overall_results, prob_key, pred_key)
        all_metrics[strategy] = m
        print(f"  {strategy:<18} {m['accuracy']:>7.4f} {m['sensitivity']:>7.4f} "
              f"{m['specificity']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f}")

    # Compare with window-level results
    print(f"\n  ── Comparison: Window-Level vs Subject-Level ──")
    print(f"  (Window-level metrics from test_results.json, averaged across folds)")

    window_aucs = []
    window_accs = []
    for fold_dir in fold_dirs:
        rp = fold_dir / 'test_results.json'
        if rp.exists():
            with open(rp) as f:
                fr = json.load(f)
            window_aucs.append(fr['threshold_05']['auc'])
            window_accs.append(fr['threshold_05']['acc'])

    if window_aucs:
        print(f"  Window-level :  Acc={np.mean(window_accs):.4f}  "
              f"AUC={np.mean(window_aucs):.4f}")
        best_strat = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
        print(f"  Subject-level:  Acc={best_strat[1]['accuracy']:.4f}  "
              f"AUC={best_strat[1]['auc']:.4f}  "
              f"(best: {best_strat[0]})")
        improvement = best_strat[1]['accuracy'] - np.mean(window_accs)
        print(f"  Improvement  :  +{improvement:.4f} accuracy")

    # ── Save results ──────────────────────────────────────────────────────────
    save_path = cv_dir / 'subject_level_results.json'
    save_data = {
        'n_folds':          n_folds_found,
        'n_subjects':       len(all_subject_probs),
        'threshold':        args.threshold,
        'overall_metrics':  all_metrics,
        'per_subject':      overall_results,
        'per_fold':         {fn: {
            sid: fold_subject_results[fn][sid]
            for sid in fold_subject_results[fn]
        } for fn in fold_subject_results},
    }

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved → {save_path}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
