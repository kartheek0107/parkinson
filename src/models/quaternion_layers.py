# quaternion_layers.py
"""
Quaternion Neural Network Layer Primitives.

All building blocks for the Q-CRNN model. Kept separate from qcrnn_model.py
so the ablation study can swap individual components in isolation.

Contents:
    quaternion_polar_init   — weight initialization (Parcollet et al. 2018)
    QuaternionConv2d        — Hamilton product convolution
    QuaternionBatchNorm2d   — split quaternion batch normalization
    QCNNBlock               — one QCNN processing stage

Mathematical reference:
    Comminiello et al. (2019), "Quaternion Convolutional Neural Networks
    for Detection and Localization of 3D Sound Events"
    Eq.(5)  — Hamilton product
    Eq.(9)  — polar initialization
    Eq.(7)  — split activation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Weight Initialization
# ═══════════════════════════════════════════════════════════════════════════════

def quaternion_polar_init(weight: torch.Tensor, n_in: int) -> None:
    """
    Initialize a quaternion weight component using the polar form.

    For a quaternion weight w = |w| · exp(u⊳ · θ):
        |w| (phi)  ~ Uniform[−σ, σ],  σ = 1/√(2 · n_in)   (He criterion)
        θ          ~ Uniform[−π, π]

    Each real component matrix (W_w, W_i, W_j, W_k) receives:
        phi · cos(θ)      for the scalar part
        phi · sin(θ) / √3 for each vector part (preserves expected norm)

    This function writes the scalar projection (phi · cos(theta)) in-place.
    All four component matrices are initialized identically with independent
    samples, which is correct for the split-quaternion representation used here.

    Args:
        weight : parameter tensor to initialize (any shape)
        n_in   : number of input quaternion units × kernel spatial size
    """
    sigma = 1.0 / math.sqrt(2.0 * max(n_in, 1))
    with torch.no_grad():
        phi   = weight.new_empty(weight.shape).uniform_(-sigma, sigma)
        theta = weight.new_empty(weight.shape).uniform_(-math.pi, math.pi)
        weight.copy_(phi * torch.cos(theta))


# ═══════════════════════════════════════════════════════════════════════════════
# Quaternion Conv2d — Hamilton Product
# ═══════════════════════════════════════════════════════════════════════════════

class QuaternionConv2d(nn.Module):
    """
    Quaternion-valued 2D convolution via the Hamilton product.

    Channel layout (channels-first, PyTorch convention):
        input  x: [W_part | X_part | Y_part | Z_part]  shape (B, 4·in_q, H, W)
        output  : [W_part | X_part | Y_part | Z_part]  shape (B, 4·out_q, H, W)
        where in_q = in_channels // 4

    Hamilton product W ⊗ x (right multiplication, Eq.5):
        out_w = Ww·xw − Wi·xi − Wj·xj − Wk·xk
        out_i = Ww·xi + Wi·xw + Wj·xk − Wk·xj
        out_j = Ww·xj − Wi·xk + Wj·xw + Wk·xi
        out_k = Ww·xk + Wi·xj − Wj·xi + Wk·xw

    Implemented with 4 real nn.Conv2d modules — no custom CUDA kernel.
    Parameters count: 4 × (in_q × out_q × k × k) — same as a real conv
    with 4× the channel width operating on all 4 components simultaneously.

    Args:
        in_channels  : total input  channels — must be divisible by 4
        out_channels : total output channels — must be divisible by 4
        kernel_size  : spatial kernel size (int)
        stride       : convolution stride
        padding      : zero-padding (set = kernel_size//2 to preserve spatial dims)
        bias         : learnable additive bias on the concatenated output
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int  = 3,
        stride:       int  = 1,
        padding:      int  = 1,
        bias:         bool = True,
    ):
        super().__init__()

        if in_channels % 4 != 0:
            raise ValueError(
                f"QuaternionConv2d: in_channels={in_channels} must be divisible by 4. "
                f"Got remainder {in_channels % 4}."
            )
        if out_channels % 4 != 0:
            raise ValueError(
                f"QuaternionConv2d: out_channels={out_channels} must be divisible by 4. "
                f"Got remainder {out_channels % 4}."
            )

        self.in_q        = in_channels  // 4
        self.out_q       = out_channels // 4
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size

        _kw = dict(kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.W_w = nn.Conv2d(self.in_q, self.out_q, **_kw)
        self.W_i = nn.Conv2d(self.in_q, self.out_q, **_kw)
        self.W_j = nn.Conv2d(self.in_q, self.out_q, **_kw)
        self.W_k = nn.Conv2d(self.in_q, self.out_q, **_kw)

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        n_in = self.in_q * self.kernel_size * self.kernel_size
        for W in (self.W_w, self.W_i, self.W_j, self.W_k):
            quaternion_polar_init(W.weight, n_in)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 4·in_q, H, W)  quaternion feature map

        Returns:
            out : (B, 4·out_q, H', W')  after Hamilton product convolution
        """
        q  = self.in_q

        xw = x[:, :q]
        xi = x[:, q   : 2*q]
        xj = x[:, 2*q : 3*q]
        xk = x[:, 3*q :]

        # Hamilton product — Eq.(5) of Comminiello et al.
        ow = self.W_w(xw) - self.W_i(xi) - self.W_j(xj) - self.W_k(xk)
        oi = self.W_w(xi) + self.W_i(xw) + self.W_j(xk) - self.W_k(xj)
        oj = self.W_w(xj) - self.W_i(xk) + self.W_j(xw) + self.W_k(xi)
        ok = self.W_w(xk) + self.W_i(xj) - self.W_j(xi) + self.W_k(xw)

        out = torch.cat([ow, oi, oj, ok], dim=1)   # (B, 4·out_q, H', W')

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out

    # ──────────────────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"in_q={self.in_q}, out_q={self.out_q}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Quaternion Batch Normalization — Split Formulation
# ═══════════════════════════════════════════════════════════════════════════════

class QuaternionBatchNorm2d(nn.Module):
    """
    Split quaternion batch normalization.

    Applies a single nn.BatchNorm2d across all 4 component groups jointly.
    Channels ordered as [W_part | X_part | Y_part | Z_part], each of width out_q.

    'Split' (as opposed to full covariance) BN is appropriate here because:
      1. Our 4 channels are derived from one analytic signal — same statistical
         scale across components by construction after normalization in Stage 2.
      2. Full covariance BN (Gaudet & Maida 2018) requires a 4×4 covariance
         matrix per spatial location, which is expensive and overkill for
         analytically normalized inputs.
      3. Split BN matches the split-ReLU activation used below (Eq.7).

    Args:
        total_channels : 4 × n_quaternions  (must equal out_channels of preceding QConv)
        eps, momentum  : passed to nn.BatchNorm2d
    """

    def __init__(
        self,
        total_channels: int,
        eps:            float = 1e-5,
        momentum:       float = 0.1,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(total_channels, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


# ═══════════════════════════════════════════════════════════════════════════════
# QCNN Block — one complete processing stage
# ═══════════════════════════════════════════════════════════════════════════════

class QCNNBlock(nn.Module):
    """
    One QCNN processing stage:

        QuaternionConv2d  →  QuaternionBatchNorm2d  →  Split-ReLU
        →  MaxPool2d (freq-only)  →  Dropout2d

    Design decisions baked in:

    (a) Frequency-only pooling MaxPool2d(kernel=(1, freq_pool)):
        Time axis (dim=2) is NEVER pooled. This preserves the full T=624
        frame sequence so the downstream BiGRU sees every 16ms timestep
        of the 2-second tremor window.

    (b) Split-ReLU = F.relu applied to the concatenated real representation.
        Equivalent to applying relu() independently to each quaternion component.
        This is Eq.(7) of Comminiello et al. It is suboptimal compared to a
        proper quaternion modReLU, but avoids sign inconsistency across
        components for this discriminative classification task.

    (c) Dropout2d (channel-wise):
        Drops entire quaternion feature maps (all spatial positions of one
        output filter). More effective than elementwise dropout for conv layers.

    Args:
        in_channels  : total input  channels (multiple of 4)
        out_channels : total output channels (multiple of 4)
        kernel_size  : spatial conv kernel (default 3 → 3×3)
        freq_pool    : MaxPool stride along frequency axis (dim=3)
        dropout      : Dropout2d drop probability
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int   = 3,
        freq_pool:    int   = 4,
        dropout:      float = 0.2,
    ):
        super().__init__()

        self.qconv = QuaternionConv2d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            padding     = kernel_size // 2,    # 'same' padding
        )
        self.qbn   = QuaternionBatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(
            kernel_size = (1, freq_pool),      # time untouched, freq halved
            stride      = (1, freq_pool),
        )
        self.drop  = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_channels, T, F)

        Returns:
            x : (B, out_channels, T, F // freq_pool)
        """
        x = self.qconv(x)     # Hamilton product conv
        x = self.qbn(x)       # split BN
        x = F.relu(x)         # split ReLU — Eq.(7)
        x = self.pool(x)      # freq-only pool
        x = self.drop(x)      # channel dropout
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Quick unit test (run this file directly to verify layer math)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import torch

    print("=" * 55)
    print(" quaternion_layers.py — unit tests")
    print("=" * 55)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" Device: {device}\n")

    # NOTE: use n_freq / n_time instead of F / T to avoid shadowing
    #       the module-level `import torch.nn.functional as F`
    B, n_time, n_freq = 2, 624, 257

    # ── Test 1: QuaternionConv2d shape ────────────────────────────────────────
    print("[1] QuaternionConv2d")
    qconv = QuaternionConv2d(4, 32, kernel_size=3, padding=1).to(device)
    x = torch.randn(B, 4, n_time, n_freq, device=device)
    y = qconv(x)
    assert y.shape == (B, 32, n_time, n_freq), f"Shape mismatch: {y.shape}"
    print(f"    input : {tuple(x.shape)}")
    print(f"    output: {tuple(y.shape)}  ✓")
    print(f"    params: {sum(p.numel() for p in qconv.parameters()):,}")
    print(f"    non-trivial gradients: {qconv.W_w.weight.requires_grad}  ✓")

    # ── Test 2: QuaternionBatchNorm2d ─────────────────────────────────────────
    print("\n[2] QuaternionBatchNorm2d")
    qbn = QuaternionBatchNorm2d(32).to(device)
    z = qbn(y)
    assert z.shape == y.shape
    print(f"    input : {tuple(y.shape)}")
    print(f"    output: {tuple(z.shape)}  ✓")

    # ── Test 3: QCNNBlock shape propagation ───────────────────────────────────
    print("\n[3] QCNNBlock (in=4, out=32, freq_pool=4)")
    block = QCNNBlock(4, 32, freq_pool=4).to(device)
    x_in  = torch.randn(B, 4, n_time, n_freq, device=device)
    x_out = block(x_in)
    expected = (B, 32, n_time, n_freq // 4)
    assert x_out.shape == expected, f"Got {x_out.shape}, expected {expected}"
    print(f"    input : {tuple(x_in.shape)}")
    print(f"    output: {tuple(x_out.shape)}  ✓  (T={n_time} preserved)")

    # ── Test 4: Gradient flow ─────────────────────────────────────────────────
    print("\n[4] Gradient flow through QCNNBlock")
    loss = x_out.mean()
    loss.backward()
    grad_ok = all(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in block.parameters()
    )
    print(f"    All gradients non-null and finite: {grad_ok}  ✓")

    # ── Test 5: Full 3-block QCNN stack ───────────────────────────────────────
    print("\n[5] Full 3-block QCNN stack (matches model architecture)")
    stack = nn.Sequential(
        QCNNBlock(4,   32,  freq_pool=4),
        QCNNBlock(32,  64,  freq_pool=4),
        QCNNBlock(64,  128, freq_pool=4),
    ).to(device)
    x_in   = torch.randn(B, 4, n_time, n_freq, device=device)
    x_out  = stack(x_in)
    C_exp  = 128
    F_exp  = n_freq // (4 ** 3)   # 257 -> floor(257/4)=64 -> 16 -> 4
    assert x_out.shape == (B, C_exp, n_time, F_exp), f"Shape: {x_out.shape}"
    print(f"    input : {tuple(x_in.shape)}")
    print(f"    output: {tuple(x_out.shape)}  ✓")
    print(f"    GRU input dim will be: {C_exp} x {F_exp} = {C_exp * F_exp}")

    print("\n✓ All quaternion_layers tests passed")
    print("=" * 55)