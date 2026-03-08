"""
merge_folds_and_finetune.py
===========================
Phase 1 — Average 5 KCL fold checkpoints → ft_init.pt
Phase 2 — Fine-tune on Italian PVS with subject-level 5-fold CV

Checkpoint question answer:
    We do NOT pick one fold — we AVERAGE all 5 fold weights.
    Each fold trained on ~30/37 KCL subjects. Averaging gives one model
    that has seen ALL 37 subjects, reduces fold-specific overfitting,
    and is a strictly better initialisation for cross-dataset fine-tuning
    than any single fold.

Layer freeze strategy for Italian PVS (65 subjects, short recordings):
    Frozen  : qcnn[0]  — low-level acoustic edges (language-independent)
    Trainable: qcnn[1], qcnn[2], bigru, head
    Rationale: The first QCNN block learns basic time-frequency patterns
    shared across English and Italian speech. Freezing prevents catastrophic
    forgetting. Upper layers adapt to Italian phonation patterns + PD markers.

Fine-tuning hyperparameters vs KCL training:
    LR          : 1e-4  (3× lower than KCL 3e-4)
    Epochs      : 30    (dataset is smaller)
    Patience    : 10    (reduced from 20)
    Batch size  : 16    (reduced from 32, fewer samples per subject)
    Warmup      : 3     (reduced from 5)

Usage:
    python merge_folds_and_finetune.py \\
        --cv_dir   src/models/checkpoints/cv_run1 \\
        --npz_dir  path/to/italian_npz_cache \\
        --out_dir  src/models/checkpoints/finetune_italian
"""

import os
import sys
import json
import argparse
import time
import csv
import math
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from qcrnn_model import QCRNNParkinson


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Weight Averaging
# ═══════════════════════════════════════════════════════════════════════════════

def average_fold_weights(cv_dir: str, out_dir: str) -> str:
    """
    Load all 5 best_model.pt, average floating-point weights,
    save ft_init.pt. Returns path to ft_init.pt.
    """
    cv_path  = Path(cv_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  PHASE 1  —  Averaging 5-Fold KCL Checkpoints")
    print(f"{'='*62}")

    fold_ckpts = sorted(cv_path.glob('fold*/best_model.pt'))
    if not fold_ckpts:
        fold_ckpts = sorted(cv_path.glob('fold*/*_model.pt'))
    if not fold_ckpts:
        raise FileNotFoundError(
            f"No fold checkpoints found in:\n  {cv_path}\n"
            f"Expected pattern: {cv_path}/foldN/best_model.pt")

    print(f"  Found {len(fold_ckpts)} checkpoints:")
    for p in fold_ckpts:
        ck  = torch.load(str(p), map_location='cpu', weights_only=False)
        auc = ck.get('val_auc', float('nan'))
        ep  = ck.get('epoch', '?')
        print(f"    {p.parent.name}  epoch={ep}  val_auc={auc:.4f}")

    ref_ck = torch.load(str(fold_ckpts[0]), map_location='cpu', weights_only=False)
    config = ref_ck['config']
    n      = len(fold_ckpts)

    avg_state = OrderedDict()
    for ckpt_path in fold_ckpts:
        ck    = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        state = ck['model_state']
        for key, val in state.items():
            if val.dtype.is_floating_point:
                if key not in avg_state:
                    avg_state[key] = val.clone().float() / n
                else:
                    avg_state[key] += val.float() / n
            else:
                if key not in avg_state:
                    avg_state[key] = val.clone()

    # Verify
    model = QCRNNParkinson(**config)
    model.load_state_dict(avg_state)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  ✓ Averaged weights verified — {total_params:,} params")

    ft_init_path = out_path / 'ft_init.pt'
    torch.save({
        'config'      : config,
        'model_state' : avg_state,
        'source_folds': [str(p) for p in fold_ckpts],
        'method'      : 'fold_weight_average',
        'n_folds'     : n,
    }, str(ft_init_path))
    print(f"  ft_init.pt → {ft_init_path}")
    return str(ft_init_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset (identical to train_5fold.py)
# ═══════════════════════════════════════════════════════════════════════════════

class DQLCTWindowDataset(Dataset):
    """
    Same as train_5fold.py — loads .npz, supports augmentation.
    Parses subject_id from filename: subjID_label0_taskXX_src000_win0000.npz
    NOTE: Italian subject IDs may contain underscores (Alberto_R)
          The split is on '_label' so this is safe.
    """
    def __init__(self, npz_dir, subject_ids, augment=False):
        self.augment  = augment
        self.files    = []
        self.labels   = []
        subject_set   = set(subject_ids)

        for f in sorted(Path(npz_dir).glob('*.npz')):
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
                f"No .npz found for {list(subject_ids)[:3]}... in {npz_dir}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        data  = np.load(self.files[idx], allow_pickle=True)
        x     = torch.from_numpy(data['features'].astype(np.float32))
        label = float(int(data['label']))
        if self.augment:
            x = self._augment(x)
        return x, torch.tensor(label, dtype=torch.float32)

    def _augment(self, x):
        # Time shift — moderate for short padded files
        if torch.rand(1).item() < 0.5:
            x = torch.roll(x, torch.randint(-15, 16, (1,)).item(), dims=1)
        # Amplitude jitter
        if torch.rand(1).item() < 0.5:
            x = x * (0.85 + 0.30 * torch.rand(4, 1, 1))
        # Noise
        if torch.rand(1).item() < 0.3:
            x = x + torch.randn_like(x) * 0.01
        # Freq masking
        if torch.rand(1).item() < 0.5:
            F  = x.shape[2]
            mw = torch.randint(1, 28, (1,)).item()
            f0 = torch.randint(0, F - mw, (1,)).item()
            x[:, :, f0:f0+mw] = 0.0
        return x

    def get_sample_weights(self):
        n_pd = int(self.labels.sum())
        n_hc = len(self.labels) - n_pd
        return np.where(self.labels==1,
                        1.0/max(n_pd,1),
                        1.0/max(n_hc,1)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(labels, logits, threshold=0.5):
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    auc   = roc_auc_score(labels, probs) \
            if len(np.unique(labels)) > 1 else float('nan')
    tp = int(((preds==1)&(labels==1)).sum())
    tn = int(((preds==0)&(labels==0)).sum())
    fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    sens = tp / max(tp+fn, 1)
    spec = tn / max(tn+fp, 1)
    prec = tp / max(tp+fp, 1)
    f1   = 2*prec*sens / max(prec+sens, 1e-9)
    return dict(auc=auc, acc=(tp+tn)/max(tp+tn+fp+fn,1),
                sensitivity=sens, specificity=spec, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)

def youden_threshold(labels, logits):
    probs = 1.0 / (1.0 + np.exp(-logits))
    fpr, tpr, thresholds = roc_curve(labels, probs)
    return float(thresholds[np.argmax(tpr - fpr)])

def get_lr_lambda(warmup, total):
    def lr_lambda(e):
        if e < warmup:
            return float(e+1) / float(warmup)
        p = (e-warmup) / max(total-warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return lr_lambda

def build_subject_index(npz_dir):
    all_files, all_labels, all_groups = [], [], []
    for f in sorted(Path(npz_dir).glob('*.npz')):
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
            f"No .npz in {npz_dir}. Run precompute_italian.py first.")

    all_labels = np.array(all_labels, dtype=np.int64)
    n_pd = int(all_labels.sum())
    n_hc = len(all_labels) - n_pd
    unique_subj = sorted(set(all_groups))
    print(f"\n  Italian NPZ index:")
    print(f"    {len(all_files)} windows  |  HC={n_hc}  PD={n_pd}"
          f"  |  {len(unique_subj)} subjects")
    return all_files, all_labels, all_groups, unique_subj

def freeze_layers(model):
    """Freeze qcnn[0] (first QCNNBlock). All others trainable."""
    for param in model.qcnn[0].parameters():
        param.requires_grad = False
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen  (qcnn[0]) : {frozen:>8,} params")
    print(f"  Trainable         : {trainable:>8,} params")


# ═══════════════════════════════════════════════════════════════════════════════
# Train / eval loops (identical to train_5fold.py)
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
    m['_labels'] = all_labels
    m['_logits'] = all_logits
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Fine-tune CV
# ═══════════════════════════════════════════════════════════════════════════════

def finetune_cv(ft_init_path, npz_dir, out_dir, args):
    out_path = Path(out_dir)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*62}")
    print(f"  PHASE 2  —  Fine-tune on Italian PVS")
    print(f"{'='*62}")
    print(f"  Init   : {ft_init_path}")
    print(f"  Device : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type=='cuda' else ''))
    print(f"  LR     : {args.ft_lr}  (vs KCL 3e-4)")
    print(f"  Epochs : {args.ft_epochs}  |  Patience: {args.patience}")

    all_files, all_labels, all_groups, unique_subj = \
        build_subject_index(npz_dir)
    subj_arr  = np.array(all_groups)
    label_arr = all_labels

    sgkf = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_results = []

    for fold_idx, (tr_idx, te_idx) in enumerate(
            sgkf.split(all_files, label_arr, groups=subj_arr)):

        fold_num   = fold_idx + 1
        fold_dir   = out_path / f'finetune_fold{fold_num}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_sids = sorted(set(subj_arr[tr_idx]))
        test_sids  = sorted(set(subj_arr[te_idx]))
        assert not (set(train_sids) & set(test_sids)), \
            f"LEAKAGE in fold {fold_num}"

        print(f"\n  {'─'*58}")
        print(f"  Fine-tune Fold {fold_num}/{args.folds}")
        print(f"  {'─'*58}")

        train_ds = DQLCTWindowDataset(npz_dir, train_sids, augment=True)
        test_ds  = DQLCTWindowDataset(npz_dir, test_sids,  augment=False)

        n_pd_tr = int(train_ds.labels.sum())
        n_hc_tr = len(train_ds) - n_pd_tr
        n_pd_te = int(test_ds.labels.sum())
        n_hc_te = len(test_ds)  - n_pd_te

        print(f"  Train : {len(train_ds):4d} windows | "
              f"HC={n_hc_tr} PD={n_pd_tr} | {len(train_sids)} subj")
        print(f"  Test  : {len(test_ds):4d} windows | "
              f"HC={n_hc_te} PD={n_pd_te} | {len(test_sids)} subj")

        pos_weight = torch.tensor(
            [n_hc_tr / max(n_pd_tr, 1)], dtype=torch.float32, device=device)
        print(f"  pos_weight: {pos_weight.item():.3f}")

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(train_ds.get_sample_weights()),
            num_samples=len(train_ds), replacement=True)

        pin = (device.type == 'cuda')
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=2,
                                  pin_memory=pin, drop_last=True)
        test_loader  = DataLoader(test_ds, batch_size=args.batch_size*2,
                                  shuffle=False, num_workers=2,
                                  pin_memory=pin)

        # Fresh load of averaged weights for every fold
        init_ck = torch.load(ft_init_path, map_location=device, weights_only=False)
        model   = QCRNNParkinson(**init_ck['config']).to(device)
        model.load_state_dict(init_ck['model_state'])
        freeze_layers(model)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(trainable_params,
                                      lr=args.ft_lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=get_lr_lambda(args.warmup_epochs, args.ft_epochs))
        scaler    = GradScaler()

        log_path = fold_dir / 'train_log.csv'
        fields   = ['epoch','lr','train_loss','train_auc',
                    'val_loss','val_auc','val_sens','val_spec','val_f1','time']
        with open(log_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

        best_auc, patience_count, best_ckpt_path = -1.0, 0, None

        print(f"\n  Epoch   LR         TrLoss/AUC    ValLoss/AUC   Sen   Spe")

        for epoch in range(1, args.ft_epochs + 1):
            t0      = time.time()
            train_m = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device)
            val_m   = eval_one_epoch(model, test_loader, criterion, device)
            scheduler.step()
            lr      = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0

            flag = ' ★' if val_m['auc'] > best_auc else ''
            print(f"  {epoch:3d}/{args.ft_epochs}"
                  f"  {lr:.1e}"
                  f"  {train_m['loss']:.4f}/{train_m['auc']:.4f}"
                  f"  {val_m['loss']:.4f}/{val_m['auc']:.4f}"
                  f"  {val_m['sensitivity']:.3f}"
                  f"  {val_m['specificity']:.3f}"
                  f"{flag}  [{elapsed:.1f}s]")

            with open(log_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=fields).writerow({
                    'epoch': epoch, 'lr': round(lr, 8),
                    'train_loss': round(train_m['loss'],5),
                    'train_auc':  round(train_m['auc'],4),
                    'val_loss':   round(val_m['loss'],5),
                    'val_auc':    round(val_m['auc'],4),
                    'val_sens':   round(val_m['sensitivity'],4),
                    'val_spec':   round(val_m['specificity'],4),
                    'val_f1':     round(val_m['f1'],4),
                    'time':       round(elapsed,2),
                })

            if val_m['auc'] > best_auc:
                best_auc       = val_m['auc']
                patience_count = 0
                best_ckpt_path = fold_dir / 'best_model.pt'
                torch.save({
                    'epoch'         : epoch,
                    'config'        : init_ck['config'],
                    'model_state'   : model.state_dict(),
                    'val_auc'       : best_auc,
                    'freeze_strategy': 'qcnn0_frozen',
                    'ft_init'       : ft_init_path,
                }, str(best_ckpt_path))
                print(f"         ↳ Checkpoint saved  val_auc={best_auc:.4f}")
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    print(f"  Early stop (patience={args.patience})")
                    break

        # Final eval at best checkpoint
        ck   = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state'])
        test_m     = eval_one_epoch(model, test_loader, criterion, device)
        opt_thresh = youden_threshold(test_m['_labels'], test_m['_logits'])
        test_opt   = compute_metrics(test_m['_labels'], test_m['_logits'],
                                      threshold=opt_thresh)

        print(f"\n  ── Fold {fold_num} Results (best epoch {ck['epoch']}) ──")
        print(f"  t=0.50 : AUC={test_m['auc']:.4f}  "
              f"F1={test_m['f1']:.4f}  "
              f"Sen={test_m['sensitivity']:.4f}  "
              f"Spe={test_m['specificity']:.4f}")
        print(f"  t={opt_thresh:.3f}: AUC={test_opt['auc']:.4f}  "
              f"F1={test_opt['f1']:.4f}  "
              f"Sen={test_opt['sensitivity']:.4f}  "
              f"Spe={test_opt['specificity']:.4f}")

        fold_results.append({
            'fold'         : fold_num,
            'best_epoch'   : ck['epoch'],
            'train_sids'   : train_sids,
            'test_sids'    : test_sids,
            'threshold_05' : {k: float(v) for k, v in test_m.items()
                               if not k.startswith('_')},
            'threshold_opt': {k: float(v) for k, v in test_opt.items()},
            'opt_threshold': float(opt_thresh),
        })

    # ── Cross-fold summary ─────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  FINE-TUNE SUMMARY  —  Italian PVS  ({args.folds} folds)")
    print(f"{'='*62}")
    print(f"\n  {'Metric':<12}  thresh=0.5            Youden thresh")
    print(f"  {'─'*12}  {'─'*22}  {'─'*22}")

    summary = {'n_folds': args.folds, 'dataset': 'Italian_PVS',
               'threshold_05': {}, 'threshold_opt': {}}

    for key, label in [('auc','AUC'), ('f1','F1'),
                        ('sensitivity','Sen'), ('specificity','Spe'), ('acc','Acc')]:
        v05  = [r['threshold_05'][key]  for r in fold_results]
        vopt = [r['threshold_opt'][key] for r in fold_results]
        m05, s05   = np.mean(v05),  np.std(v05)
        mopt, sopt = np.mean(vopt), np.std(vopt)
        print(f"  {label:<12}  {m05:.4f} ± {s05:.4f}       "
              f"{mopt:.4f} ± {sopt:.4f}")
        summary['threshold_05'][key]  = {'mean': round(m05,4), 'std': round(s05,4),
                                          'per_fold': [round(v,4) for v in v05]}
        summary['threshold_opt'][key] = {'mean': round(mopt,4),'std': round(sopt,4),
                                          'per_fold': [round(v,4) for v in vopt]}

    summary['fold_results'] = fold_results
    out_json = out_path / 'ft_cv_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    # CSV
    rows = []
    for r in fold_results:
        row = {'fold': r['fold'], 'best_epoch': r['best_epoch'],
               'opt_threshold': round(r['opt_threshold'], 4)}
        for k in ['auc','f1','sensitivity','specificity','acc']:
            row[f'{k}_05']  = round(r['threshold_05'][k], 4)
            row[f'{k}_opt'] = round(r['threshold_opt'][k], 4)
        rows.append(row)
    fold_csv = out_path / 'ft_fold_metrics.csv'
    with open(fold_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print(f"\n  ft_cv_summary.json   → {out_json}")
    print(f"  ft_fold_metrics.csv  → {fold_csv}")
    print(f"{'='*62}\n")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ft_init_path = average_fold_weights(args.cv_dir, args.out_dir)
    finetune_cv(ft_init_path, args.npz_dir, args.out_dir, args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Average KCL fold weights → fine-tune on Italian PVS')
    p.add_argument('--cv_dir',        required=True,
                   help='KCL cv_run1/ directory (contains fold1../best_model.pt)')
    p.add_argument('--npz_dir',       required=True,
                   help='Italian PVS npz_cache/ from precompute_italian.py')
    p.add_argument('--out_dir',       default='results/finetune_italian')
    p.add_argument('--folds',         type=int,   default=5)
    p.add_argument('--ft_epochs',     type=int,   default=30)
    p.add_argument('--ft_lr',         type=float, default=1e-4)
    p.add_argument('--batch_size',    type=int,   default=16)
    p.add_argument('--weight_decay',  type=float, default=1e-3)
    p.add_argument('--warmup_epochs', type=int,   default=3)
    p.add_argument('--patience',      type=int,   default=10)
    p.add_argument('--seed',          type=int,   default=42)
    args = p.parse_args()
    main(args)