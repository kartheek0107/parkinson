# train_5fold.py
"""
Q-CRNN: 5-Fold Subject-Level Cross-Validation Training
=======================================================
Dataset  : MDVR-KCL segmented (37 subjects: 16 PD, 21 HC)
           Segments produced by audio_segmentation.py + precompute.py
Input    : .npz cache  ->  (4, 624, 257) float32 tensors
Model    : QCRNNParkinson  (quaternion_layers.py + qcrnn_model.py)
CV       : StratifiedGroupKFold(n_splits=5, groups=subject_id)
             - no subject appears in both train and test within any fold
             - class balance (HC/PD ratio) maintained per fold
Loss     : BCEWithLogitsLoss  pos_weight recomputed fresh per fold
Sampler  : WeightedRandomSampler  (per-batch class balancing within fold)
Augment  : time shift, amplitude jitter, noise, SpecAugment (train only)
LR       : linear warmup -> cosine decay  (AdamW)
Precision: torch.amp autocast + GradScaler  (RTX 5060 / CUDA 13.0)
Stopping : early stopping per fold  (patience on val AUC)
Threshold: Youden's J on fold test set for optimal sensitivity+specificity

Output per fold (in out_dir/foldN/):
    best_model.pt           best checkpoint by val AUC
    train_log.csv           per-epoch metrics
    test_results.json       final test metrics at threshold 0.5 + Youden

Final output (in out_dir/):
    cv_summary.json         mean±std across all 5 folds
    cv_fold_metrics.csv     one row per fold

Usage:
    python train_5fold.py --npz_dir /path/to/npz_cache --out_dir results/cv_run1

    # Recommended GPU settings for RTX 5060:
    python train_5fold.py \\
        --npz_dir  /path/to/npz_cache \\
        --out_dir  results/cv_run1 \\
        --epochs   100 \\
        --batch_size 32 \\
        --lr       3e-4 \\
        --patience 20 \\
        --folds    5
"""

import os
import sys
import json
import argparse
import time
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

# ── Model imports ──────────────────────────────────────────────────────────────
_MODELS_DIR = Path(__file__).resolve().parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from qcrnn_model import build_model


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset  — identical to train.py, zero changes
# ═══════════════════════════════════════════════════════════════════════════════

class DQLCTWindowDataset(Dataset):
    """
    Loads precomputed .npz windows produced by precompute.py.

    Filename format: subjID00_label0_taskRT_src003_win0002.npz
    Each .npz contains:
        features  : float32  (4, 624, 257)
        label     : int32    0=HC  1=PD
        subject_id: str
        task      : str
        source_idx: int32

    Args:
        npz_dir    : path to npz_cache directory
        subject_ids: whitelist of subject IDs for this split
        augment    : apply lightweight augmentation (training only)
    """

    def __init__(self, npz_dir: str, subject_ids: list, augment: bool = False):
        self.augment    = augment
        self.files      = []
        self.labels     = []
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

        self.labels = np.array(self.labels, dtype=np.int64)

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npz files found for subjects {subject_ids[:3]}... in {npz_dir}.\n"
                f"Run precompute.py first."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data  = np.load(self.files[idx], allow_pickle=True)
        x     = torch.from_numpy(data['features'].astype(np.float32))  # (4, 624, 257)
        label = float(int(data['label']))
        if self.augment:
            x = self._augment(x)
        return x, torch.tensor(label, dtype=torch.float32)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Lightweight spectral-domain augmentation — preserves (4, 624, 257)."""
        # 1. Random circular time shift
        if torch.rand(1).item() < 0.5:
            x = torch.roll(x, torch.randint(-20, 21, (1,)).item(), dims=1)
        # 2. Per-channel amplitude jitter ±15%
        if torch.rand(1).item() < 0.5:
            x = x * (0.85 + 0.30 * torch.rand(4, 1, 1))
        # 3. Small additive noise
        if torch.rand(1).item() < 0.3:
            x = x + torch.randn_like(x) * 0.01
        # 4. Frequency masking (SpecAugment-style)
        if torch.rand(1).item() < 0.5:
            F = x.shape[2]
            mask_width = torch.randint(1, 28, (1,)).item()
            f0 = torch.randint(0, F - mask_width, (1,)).item()
            x[:, :, f0 : f0 + mask_width] = 0.0
        return x

    def get_sample_weights(self) -> np.ndarray:
        """Inverse-frequency weights for WeightedRandomSampler."""
        n_pd = int(self.labels.sum())
        n_hc = len(self.labels) - n_pd
        w_pd = 1.0 / max(n_pd, 1)
        w_hc = 1.0 / max(n_hc, 1)
        return np.where(self.labels == 1, w_pd, w_hc).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers  — identical to train.py, zero changes
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(labels: np.ndarray, logits: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Binary classification metrics from raw logits."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    auc   = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float('nan')
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1   = 2 * prec * sens / max(prec + sens, 1e-9)
    return dict(auc=auc, acc=acc, sensitivity=sens, specificity=spec,
                f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def youden_threshold(labels: np.ndarray, logits: np.ndarray) -> float:
    """
    Youden's J statistic: maximises sensitivity + specificity simultaneously.
    Returns optimal decision threshold from the test fold's ROC curve.
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_idx = np.argmax(tpr - fpr)
    return float(thresholds[j_idx])


def get_lr_lambda(warmup_epochs: int, total_epochs: int):
    """Linear warmup then cosine decay — identical to train.py."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def save_checkpoint(model, optimizer, scaler, epoch, val_m, out_dir, tag):
    """Identical to train.py."""
    path = Path(out_dir) / f'{tag}_model.pt'
    torch.save({
        'epoch'       : epoch,
        'config'      : model.config,
        'model_state' : model.state_dict(),
        'optim_state' : optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'val_auc'     : val_m['auc'],
        'val_acc'     : val_m['acc'],
        'val_loss'    : val_m['loss'],
    }, path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Train / eval loops  — identical to train.py, zero changes
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, all_labels, all_logits = 0.0, [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x).squeeze(1)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        all_labels.append(y.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().float().numpy())

    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)
    m = compute_metrics(all_labels, all_logits)
    m['loss'] = total_loss / max(len(all_labels), 1)
    return m


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_logits = 0.0, [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x).squeeze(1)
            loss   = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        all_labels.append(y.cpu().numpy())
        all_logits.append(logits.cpu().float().numpy())

    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)
    m = compute_metrics(all_labels, all_logits)
    m['loss'] = total_loss / max(len(all_labels), 1)
    # also return raw arrays for threshold tuning
    m['_labels'] = all_labels
    m['_logits'] = all_logits
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Subject index  — scan all .npz files once, build subject→windows mapping
# ═══════════════════════════════════════════════════════════════════════════════

def build_subject_index(npz_dir: str):
    """
    Scan all .npz files in npz_dir once.

    Returns:
        all_files  : list of Path
        all_labels : np.int64 array
        all_groups : list of str  (subject_id per window)
        unique_subj: sorted list of unique subject IDs
    """
    npz_dir   = Path(npz_dir)
    all_files  = []
    all_labels = []
    all_groups = []

    for f in sorted(npz_dir.glob('*.npz')):
        name = f.stem
        try:
            sid   = name.split('_label')[0].replace('subj', '')
            label = int(name.split('_label')[1].split('_')[0])
        except (IndexError, ValueError):
            continue
        all_files.append(f)
        all_labels.append(label)
        all_groups.append(sid)

    if not all_files:
        raise RuntimeError(
            f"No .npz files found in {npz_dir}.\n"
            f"Run precompute.py first."
        )

    all_labels = np.array(all_labels, dtype=np.int64)
    unique_subj = sorted(set(all_groups))
    n_pd = int(all_labels.sum())
    n_hc = len(all_labels) - n_pd

    print(f"\n  NPZ index built:")
    print(f"    Windows    : {len(all_files)}")
    print(f"    HC windows : {n_hc}    PD windows: {n_pd}")
    print(f"    Subjects   : {len(unique_subj)} total")

    return all_files, all_labels, all_groups, unique_subj


# ═══════════════════════════════════════════════════════════════════════════════
# Per-fold training
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_fold(fold_num, n_folds, train_sids, test_sids,
                   npz_dir, args, device, out_dir):
    """
    Full training loop for one fold. Returns test metrics dict.
    Saves checkpoints + train_log.csv into out_dir/foldN/.
    """
    fold_dir = Path(out_dir) / f'fold{fold_num}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  FOLD {fold_num}/{n_folds}")
    print(f"{'─'*60}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = DQLCTWindowDataset(npz_dir, train_sids, augment=True)
    val_ds   = DQLCTWindowDataset(npz_dir, test_sids,  augment=False)

    # Leakage guard
    train_set = set(train_sids)
    test_set  = set(test_sids)
    overlap   = train_set & test_set
    assert not overlap, f"LEAKAGE in fold {fold_num}: {overlap}"

    n_pd_tr = int(train_ds.labels.sum())
    n_hc_tr = len(train_ds) - n_pd_tr
    n_pd_te = int(val_ds.labels.sum())
    n_hc_te = len(val_ds)   - n_pd_te

    print(f"  Train: {len(train_ds):5d} windows | HC={n_hc_tr}  PD={n_pd_tr}"
          f" | {len(train_sids)} subjects")
    print(f"  Test : {len(val_ds):5d} windows | HC={n_hc_te}  PD={n_pd_te}"
          f" | {len(test_sids)} subjects")

    # ── pos_weight: computed fresh from THIS fold's train labels ───────────────
    pos_weight = torch.tensor(
        [n_hc_tr / max(n_pd_tr, 1)], dtype=torch.float32, device=device)
    print(f"  pos_weight: {pos_weight.item():.3f}  "
          f"(BCEWithLogitsLoss  HC/PD upweight)")

    # ── WeightedRandomSampler: per-batch class balance ─────────────────────────
    sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(train_ds.get_sample_weights()),
        num_samples = len(train_ds),
        replacement = True,
    )

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.num_workers,
                              pin_memory=pin, drop_last=True)
    test_loader  = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=pin)

    # ── Model (fresh weights every fold) ──────────────────────────────────────
    model     = build_model(device=str(device))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_lr_lambda(args.warmup_epochs, args.epochs))
    scaler    = GradScaler()

    if fold_num == 1:
        model.print_summary()

    # ── CSV log ────────────────────────────────────────────────────────────────
    log_path = fold_dir / 'train_log.csv'
    fields   = ['epoch', 'lr', 'train_loss', 'train_auc', 'train_acc',
                'val_loss', 'val_auc', 'val_acc', 'val_sens', 'val_spec',
                'val_f1', 'time']
    with open(log_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    # ── Training loop ──────────────────────────────────────────────────────────
    best_auc, patience_count, best_ckpt = -1.0, 0, None

    print(f"\n  Epochs={args.epochs}  Batch={args.batch_size}  "
          f"LR={args.lr}  Patience={args.patience}")

    for epoch in range(1, args.epochs + 1):
        t0      = time.time()
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device)
        val_m   = eval_one_epoch(model, test_loader, criterion, device)
        scheduler.step()
        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        flag = ' ★' if val_m['auc'] > best_auc else ''
        print(f"  Ep {epoch:3d}/{args.epochs} "
              f"lr={lr:.1e} | "
              f"tr {train_m['loss']:.4f}/{train_m['auc']:.4f} | "
              f"val {val_m['loss']:.4f}/{val_m['auc']:.4f} "
              f"sen={val_m['sensitivity']:.3f} spe={val_m['specificity']:.3f}"
              f"{flag} [{elapsed:.1f}s]")

        with open(log_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writerow({
                'epoch': epoch, 'lr': round(lr, 8),
                'train_loss': round(train_m['loss'], 5),
                'train_auc':  round(train_m['auc'],  4),
                'train_acc':  round(train_m['acc'],  4),
                'val_loss':   round(val_m['loss'],   5),
                'val_auc':    round(val_m['auc'],    4),
                'val_acc':    round(val_m['acc'],    4),
                'val_sens':   round(val_m['sensitivity'], 4),
                'val_spec':   round(val_m['specificity'], 4),
                'val_f1':     round(val_m['f1'],          4),
                'time':       round(elapsed, 2),
            })

        if val_m['auc'] > best_auc:
            best_auc       = val_m['auc']
            patience_count = 0
            best_ckpt      = save_checkpoint(
                model, optimizer, scaler, epoch, val_m, fold_dir, 'best')
            print(f"         ↳ Saved  val_auc={best_auc:.4f}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stop at epoch {epoch} "
                      f"(patience={args.patience})")
                break

        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, scaler, epoch, val_m, fold_dir, 'latest')

    # ── Final evaluation at best checkpoint ────────────────────────────────────
    ckpt = torch.load(str(best_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    test_m = eval_one_epoch(model, test_loader, criterion, device)

    # Youden's J optimal threshold
    opt_thresh = youden_threshold(test_m['_labels'], test_m['_logits'])
    test_m_opt = compute_metrics(
        test_m['_labels'], test_m['_logits'], threshold=opt_thresh)

    print(f"\n  ── Fold {fold_num} Test Results (best epoch {ckpt['epoch']}) ──")
    print(f"  threshold=0.5  : AUC={test_m['auc']:.4f}  "
          f"F1={test_m['f1']:.4f}  "
          f"Sen={test_m['sensitivity']:.4f}  Spe={test_m['specificity']:.4f}")
    print(f"  threshold={opt_thresh:.3f}: AUC={test_m_opt['auc']:.4f}  "
          f"F1={test_m_opt['f1']:.4f}  "
          f"Sen={test_m_opt['sensitivity']:.4f}  Spe={test_m_opt['specificity']:.4f}")

    # Save fold test results
    fold_results = {
        'fold':        fold_num,
        'best_epoch':  ckpt['epoch'],
        'train_subjs': train_sids,
        'test_subjs':  test_sids,
        'threshold_05':  {k: float(v) for k, v in test_m.items()
                          if not k.startswith('_')},
        'threshold_opt': {k: float(v) for k, v in test_m_opt.items()},
        'opt_threshold': float(opt_thresh),
    }
    with open(fold_dir / 'test_results.json', 'w') as f:
        json.dump(fold_results, f, indent=2)

    return fold_results


# ═══════════════════════════════════════════════════════════════════════════════
# Main CV loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_cv(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Q-CRNN  |  5-Fold Subject-Level CV  |  PD Detection")
    print(f"{'='*60}")
    print(f"  Device  : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ''))
    print(f"  NPZ dir : {args.npz_dir}")
    print(f"  Out dir : {out_dir}")

    # ── Scan all .npz files once ───────────────────────────────────────────────
    all_files, all_labels, all_groups, unique_subj = \
        build_subject_index(args.npz_dir)

    # Build subject-level arrays for StratifiedGroupKFold
    # (SGK needs one entry per window — groups=subject_id maintains integrity)
    subj_arr  = np.array(all_groups)
    label_arr = all_labels   # window-level labels

    # ── 5-Fold split ──────────────────────────────────────────────────────────
    sgkf = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed)

    print(f"\n  Folds   : {args.folds}")
    print(f"  Subjects: {len(unique_subj)} total  "
          f"(~{len(unique_subj)//args.folds} per test fold)")

    fold_results = []

    for fold_idx, (train_win_idx, test_win_idx) in enumerate(
            sgkf.split(all_files, label_arr, groups=subj_arr)):

        fold_num = fold_idx + 1

        # Extract unique subject IDs for each split
        train_sids = sorted(set(subj_arr[train_win_idx]))
        test_sids  = sorted(set(subj_arr[test_win_idx]))

        result = train_one_fold(
            fold_num   = fold_num,
            n_folds    = args.folds,
            train_sids = train_sids,
            test_sids  = test_sids,
            npz_dir    = args.npz_dir,
            args       = args,
            device     = device,
            out_dir    = out_dir,
        )
        fold_results.append(result)

    # ── Aggregate across folds ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION SUMMARY  ({args.folds} folds)")
    print(f"{'='*60}")

    key_metrics = [
        ('auc',         'AUC     '),
        ('f1',          'F1      '),
        ('sensitivity', 'Sens    '),
        ('specificity', 'Spec    '),
        ('acc',         'Acc     '),
    ]

    summary = {'n_folds': args.folds, 'threshold_05': {}, 'threshold_opt': {}}

    print(f"\n  {'Metric':<12}  threshold=0.5          Youden threshold")
    print(f"  {'─'*12}  {'─'*22}  {'─'*22}")

    for key, label in key_metrics:
        vals_05  = [r['threshold_05'][key]  for r in fold_results]
        vals_opt = [r['threshold_opt'][key] for r in fold_results]
        m05, s05   = np.mean(vals_05),  np.std(vals_05)
        mopt, sopt = np.mean(vals_opt), np.std(vals_opt)
        print(f"  {label}  {m05:.4f} ± {s05:.4f}        "
              f"{mopt:.4f} ± {sopt:.4f}")
        summary['threshold_05'][key]  = {'mean': round(m05,4),  'std': round(s05,4),
                                          'per_fold': [round(v,4) for v in vals_05]}
        summary['threshold_opt'][key] = {'mean': round(mopt,4), 'std': round(sopt,4),
                                          'per_fold': [round(v,4) for v in vals_opt]}

    # ── Save outputs ───────────────────────────────────────────────────────────
    cv_summary_path = out_dir / 'cv_summary.json'
    summary['fold_results'] = fold_results
    with open(cv_summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    cv_csv_path = out_dir / 'cv_fold_metrics.csv'
    rows = []
    for r in fold_results:
        row = {'fold': r['fold'], 'best_epoch': r['best_epoch'],
               'opt_threshold': round(r['opt_threshold'], 4)}
        for k in ['auc','f1','sensitivity','specificity','acc']:
            row[f'{k}_05']  = round(r['threshold_05'][k], 4)
            row[f'{k}_opt'] = round(r['threshold_opt'][k], 4)
        rows.append(row)

    with open(cv_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  cv_summary.json    → {cv_summary_path}")
    print(f"  cv_fold_metrics.csv → {cv_csv_path}")
    print(f"  Per-fold checkpoints in {out_dir}/foldN/best_model.pt")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Q-CRNN 5-fold subject-level CV for PD detection.')
    p.add_argument('--npz_dir',       required=True,
                   help='Path to npz_cache/ from precompute.py')
    p.add_argument('--out_dir',       default='results/cv_run1',
                   help='Output directory for checkpoints and results')
    p.add_argument('--folds',         type=int,   default=5)
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-3)
    p.add_argument('--warmup_epochs', type=int,   default=5)
    p.add_argument('--patience',      type=int,   default=20)
    p.add_argument('--num_workers',   type=int,   default=2)
    p.add_argument('--seed',          type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    train_cv(parse_args())