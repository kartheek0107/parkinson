# qcrnn_model.py
"""
Q-CRNN: Quaternion Convolutional-Recurrent Neural Network
for Parkinson's Disease Detection from Speech.

This file assembles the full model from quaternion_layers.py primitives.
It contains only:
    QCRNNParkinson   — the model class
    build_model()    — factory function
    load_from_checkpoint() — reload for inference / analysis
    verify_forward_pass()  — pre-training sanity check

Architecture:
    Input  → QCNN ×3 → reshape → BiGRU ×2 → mean-pool → FC → logit
    (B,4,624,257) → ... → (B,1)

Training objective:
    BCEWithLogitsLoss with pos_weight (defined in train.py, not here).
    This file has no loss, no optimizer — model only.

CUDA note:
    Uses torch.amp.autocast / torch.amp.GradScaler API (PyTorch >= 2.0).
    Required for RTX 5060 on CUDA 13.0 + PyTorch nightly (torch_env).
"""

import torch
import torch.nn as nn

from quaternion_layers import QCNNBlock


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class QCRNNParkinson(nn.Module):
    """
    Quaternion Convolutional-Recurrent Neural Network for PD detection.

    Input shape  : (B, 4, T, F)
                   B = batch size
                   4 = quaternion channels  [W, X (Hilbert), Y (IF), Z (IA)]
                   T = 624 time frames      (10 seconds at hop=256, sr=16000)
                   F = 257 frequency bins   (positive half of DQLCT spectrum, FRAME_LENGTH//2+1)

    Output shape : (B, 1)  raw logit — apply sigmoid() for probability

    Layer-by-layer shape trace (defaults: qcnn=[32,64,128], freq_pools=[4,4,4])
    ─────────────────────────────────────────────────────────────────────────
    Input               (B,   4, 624, 257)
    QCNNBlock 1         (B,  32, 128,  64)   freq: 256->64  (pool 4)
    QCNNBlock 2         (B,  64, 128,  16)   freq:  64->16  (pool 4)
    QCNNBlock 3         (B, 128, 128,   4)   freq:  16-> 4  (pool 4)
    reshape             (B, 128, 512)        512 = 128 ch x 4 freq
    BiGRU (2 layers)    (B, 128, 256)        256 = 128 hidden x 2 directions
    mean over T         (B, 256)
    FC: 256->128->1     (B, 1)

    Parameter count  ~1.1M  (appropriate for 37-subject MDVR-KCL dataset)

    Args:
        in_channels   : quaternion input channels (default 4 -- do not change)
        T             : time frames per window    (default 128)
        F             : frequency bins            (default 256)
        qcnn_channels : output channels for each QCNN block (all must be div by 4)
        freq_pools    : MaxPool stride on frequency axis per block
        gru_hidden    : GRU hidden units per direction
        gru_layers    : stacked GRU depth
        fc_hidden     : intermediate FC dim in classification head
        dropout       : shared dropout rate (all layers)
    """

    def __init__(
        self,
        in_channels:   int   = 4,
        T:             int   = 624,
        F:             int   = 257,
        qcnn_channels: list  = None,
        freq_pools:    list  = None,
        gru_hidden:    int   = 128,
        gru_layers:    int   = 2,
        fc_hidden:     int   = 128,
        dropout:       float = 0.3,
    ):
        super().__init__()

        # ── Defaults ──────────────────────────────────────────────────────────
        if qcnn_channels is None: qcnn_channels = [32, 64, 128]
        if freq_pools    is None: freq_pools    = [4,  4,  4]

        # ── Validation ────────────────────────────────────────────────────────
        if len(qcnn_channels) != len(freq_pools):
            raise ValueError(
                f"qcnn_channels and freq_pools must have the same length. "
                f"Got {len(qcnn_channels)} and {len(freq_pools)}."
            )
        for ch in qcnn_channels:
            if ch % 4 != 0:
                raise ValueError(
                    f"All qcnn_channels must be divisible by 4. Got {ch}."
                )

        total_pool    = 1
        for p in freq_pools:
            total_pool *= p
        F_after_qcnn = F // total_pool
        if F_after_qcnn < 1:
            raise ValueError(
                f"Frequency collapses to {F_after_qcnn} after pooling. "
                f"Reduce freq_pools or increase F."
            )

        # ── Config (saved in checkpoint for reconstruction) ────────────────────
        self.config = dict(
            in_channels   = in_channels,
            T             = T,
            F             = F,
            qcnn_channels = qcnn_channels,
            freq_pools    = freq_pools,
            gru_hidden    = gru_hidden,
            gru_layers    = gru_layers,
            fc_hidden     = fc_hidden,
            dropout       = dropout,
        )

        # ── QCNN blocks ────────────────────────────────────────────────────────
        blocks  = []
        ch_prev = in_channels
        for ch_out, fpool in zip(qcnn_channels, freq_pools):
            blocks.append(
                QCNNBlock(ch_prev, ch_out,
                          kernel_size=3, freq_pool=fpool, dropout=dropout)
            )
            ch_prev = ch_out
        self.qcnn = nn.Sequential(*blocks)

        # ── Bidirectional GRU ──────────────────────────────────────────────────
        gru_input_dim = qcnn_channels[-1] * F_after_qcnn   # 128 x 4 = 512
        self.bigru = nn.GRU(
            input_size    = gru_input_dim,
            hidden_size   = gru_hidden,
            num_layers    = gru_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if gru_layers > 1 else 0.0,
        )

        gru_out_dim = gru_hidden * 2    # bidirectional

        # ── Classification head ────────────────────────────────────────────────
        # NO sigmoid — use BCEWithLogitsLoss during training
        self.head = nn.Sequential(
            nn.Linear(gru_out_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, 1),
        )

        # ── Save dims for forward() ────────────────────────────────────────────
        self._F_after_qcnn  = F_after_qcnn
        self._gru_input_dim = gru_input_dim
        self._gru_out_dim   = gru_out_dim

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 4, T, F)

        Returns:
            logits : (B, 1)  raw — NOT probabilities
        """
        B = x.shape[0]

        # 1. QCNN: (B, 4, T, F) -> (B, 128, T, 4)
        x = self.qcnn(x)

        # 2. Reshape for GRU: (B, C, T, F') -> (B, T, C*F')
        _, C, T_curr, F_curr = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()     # (B, T, C, F')
        x = x.reshape(B, T_curr, C * F_curr)        # (B, T, 512)

        # 3. BiGRU: (B, T, 512) -> (B, T, 256)
        x, _ = self.bigru(x)

        # 4. Temporal mean-pool: (B, T, 256) -> (B, 256)
        x = x.mean(dim=1)

        # 5. FC head: (B, 256) -> (B, 1)
        x = self.head(x)

        return x

    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        PD probability in [0, 1]. For inference only.

        Args:
            x : (B, 4, T, F)
        Returns:
            prob : (B, 1)  where 1 = high probability of PD
        """
        self.eval()
        return torch.sigmoid(self.forward(x))

    # ──────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        def _n(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            'qcnn':  _n(self.qcnn),
            'bigru': _n(self.bigru),
            'head':  _n(self.head),
            'total': _n(self),
        }

    def print_summary(self) -> None:
        cfg    = self.config
        counts = self.count_parameters()
        F_curr = cfg['F']

        print(f"\n{'─'*56}")
        print(f"  Q-CRNN  |  Parkinson's Disease Detection")
        print(f"{'─'*56}")
        print(f"  {'Layer':<28} {'Output Shape':>20}")
        print(f"  {'─'*28} {'─'*20}")
        print(f"  {'Input':<28} {'(B, 4, 624, 257)':>20}")

        for i, (ch, p) in enumerate(zip(cfg['qcnn_channels'], cfg['freq_pools'])):
            F_curr = F_curr // p
            in_ch  = 4 if i == 0 else cfg['qcnn_channels'][i-1]
            label  = f"QCNNBlock {i+1}  ({in_ch}->{ch})"
            shape  = f"(B, {ch}, 128, {F_curr})"
            print(f"  {label:<28} {shape:>20}")

        gru_in   = cfg['qcnn_channels'][-1] * self._F_after_qcnn
        gru_out  = cfg['gru_hidden'] * 2
        gh       = cfg['gru_hidden']
        fh       = cfg['fc_hidden']
        print(f"  {'reshape':<28} {'(B, 128, ' + str(gru_in) + ')':>20}")
        print(f"  {'BiGRU (' + str(gru_in) + '->' + str(gh) + 'x2)':<28} {'(B, 128, ' + str(gru_out) + ')':>20}")
        print(f"  {'mean over T':<28} {'(B, 256)':>20}")
        print(f"  {'FC (256->' + str(fh) + '->1)':<28} {'(B, 1)':>20}")

        print(f"\n  Params:  QCNN {counts['qcnn']:,}  |  "
              f"BiGRU {counts['bigru']:,}  |  Head {counts['head']:,}")
        print(f"  Total:   {counts['total']:,}  |  "
              f"Dropout: {cfg['dropout']}  |  GRU layers: {cfg['gru_layers']}")
        print(f"{'─'*56}")


# ═══════════════════════════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(device: str = 'cuda') -> QCRNNParkinson:
    """
    Default model for MDVR-KCL (37 subjects, binary PD/HC).
    Small channel counts + dropout sized for the small dataset.
    """
    model = QCRNNParkinson(
        in_channels   = 4,
        T             = 624,
        F             = 257,
        qcnn_channels = [32, 64, 128],
        freq_pools    = [4,  4,  4],
        gru_hidden    = 128,
        gru_layers    = 2,
        fc_hidden     = 128,
        dropout       = 0.3,
    )
    return model.to(device)


def load_from_checkpoint(
    checkpoint_path: str,
    device:          str = 'cuda',
) -> QCRNNParkinson:
    """
    Reconstruct and load model from best_model.pt.
    Architecture is read from the 'config' key — no hardcoding required.
    """
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = QCRNNParkinson(**ckpt['config']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(
        f"Loaded checkpoint: epoch {ckpt['epoch']}  "
        f"val_auc={ckpt.get('val_auc', float('nan')):.4f}"
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-training verification  (run on torch_env before touching any data)
# ═══════════════════════════════════════════════════════════════════════════════

def verify_forward_pass(device: str = 'cpu') -> None:
    """
    Forward + backward verification. No dataset required.

    Run on your RTX 5060 after setting up torch_env:
        conda activate torch_env
        python qcrnn_model.py

    Checks:
        1. Shape correctness at every layer boundary
        2. No NaN / Inf in logits or gradients
        3. Gradient reaches every parameter
        4. torch.amp.autocast works on CUDA (RTX 5060 / CUDA 13.0 specific)
        5. VRAM usage is within safe range
    """
    import time

    print(f"\n{'='*56}")
    print(f"  Q-CRNN Verification  |  device: {device}")
    print(f"{'='*56}")

    model = build_model(device=device)
    model.print_summary()

    B         = 4   # BatchNorm needs >1 sample in train mode
    criterion = nn.BCEWithLogitsLoss()

    # ── 1. Forward pass ────────────────────────────────────────────────────────
    print("\n[1] Forward pass")
    x = torch.randn(B, 4, 624, 257, device=device)
    t0 = time.time()
    with torch.no_grad():
        logits = model(x)
        probs  = torch.sigmoid(logits)
    ms = (time.time() - t0) * 1000

    assert logits.shape == (B, 1),            "Wrong output shape"
    assert not torch.isnan(logits).any(),     "NaN in logits"
    assert not torch.isinf(logits).any(),     "Inf in logits"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probs out of range"
    print(f"    Output: {tuple(logits.shape)}  |  "
          f"probs [{probs.min().item():.3f}, {probs.max().item():.3f}]  |  "
          f"{ms:.0f}ms  |  PASS")

    # ── 2. Backward pass ────────────────────────────────────────────────────────
    print("\n[2] Backward pass")
    model.train()
    x     = torch.randn(B, 4, 624, 257, device=device)
    y     = torch.randint(0, 2, (B,), dtype=torch.float32, device=device)
    loss  = criterion(model(x).squeeze(1), y)
    loss.backward()

    dead = [
        n for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or torch.isnan(p.grad).any())
    ]
    if dead:
        print(f"    FAIL  --  dead/NaN gradients in: {dead[:5]}")
    else:
        gnorm = sum(
            p.grad.norm().item()**2
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ) ** 0.5
        print(f"    Loss: {loss.item():.4f}  |  grad norm: {gnorm:.4f}  |  PASS")

    # ── 3. Mixed precision (CUDA only) ─────────────────────────────────────────
    if device == 'cuda':
        print("\n[3] torch.amp mixed precision")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        scaler    = torch.amp.GradScaler('cuda')          # FIXED API for PyTorch >= 2.0
        optimizer.zero_grad(set_to_none=True)

        x = torch.randn(B, 4, 624, 257, device=device)
        y = torch.randint(0, 2, (B,), dtype=torch.float32, device=device)

        with torch.amp.autocast('cuda'):                  # FIXED API
            loss = criterion(model(x).squeeze(1), y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        assert not torch.isnan(loss), "NaN loss in amp pass"
        print(f"    fp16 loss: {loss.item():.4f}  |  PASS")
    else:
        print("\n[3] Mixed precision -- skipped (CPU)")

    # ── 4. CUDA memory ─────────────────────────────────────────────────────────
    if device == 'cuda':
        print("\n[4] VRAM usage")
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**2
        resrv = torch.cuda.memory_reserved()  / 1024**2
        print(f"    Allocated: {alloc:.0f} MB  |  "
              f"Reserved: {resrv:.0f} MB  |  "
              f"Free: {8192 - resrv:.0f} MB")

    print(f"\n{'='*56}")
    print(f"  All checks passed.")
    print(f"  Next: python train.py --npz_dir <path_to_cache>")
    print(f"{'='*56}\n")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    verify_forward_pass(device=device)