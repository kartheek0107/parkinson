import numpy as np
import torch
import sys
import os

# Import the vectorized core
from src.dqlct.core import QLCT1D


def verify_math():
    print("=" * 60)
    print("🧪 DQLCT INTEGRITY CHECK (Vectorized Engine)")
    print("=" * 60)

    # 1. Setup
    N = 64  # Small size for easy checking
    np.random.seed(42)

    # Create a random COMPLEX quaternion signal (Real, i, j, k all active)
    w = np.random.randn(N)
    x = np.random.randn(N)
    y = np.random.randn(N)
    z = np.random.randn(N)
    signal = np.stack([w, x, y, z])

    # Initialize with QFT parameters (Standard Fourier Case)
    cfg = {'a': 0.0, 'b': 1.0, 'c': -1.0, 'd': 0.0}
    qlct = QLCT1D(N, cfg)

    # =========================================================
    # TEST 4: PERFECT RECONSTRUCTION (The Ultimate Test)
    # Theory: Inverse(Forward(x)) == x
    # =========================================================
    print("\n[4/4] Testing Inverse Reconstruction...")

    # 1. Forward
    spectrum = qlct.forward(signal)

    # 2. Inverse
    reconstructed = qlct.inverse(spectrum)

    # 3. Compare
    error = np.abs(signal - reconstructed)
    max_error = np.max(error)
    mse = np.mean(error ** 2)

    print(f"   Max Reconstruction Error: {max_error:.9f}")

    if max_error < 1e-12:
        print("✅ PASS: Perfect Reconstruction (Exact Match)")
    elif max_error < 1e-5:
        print("⚠️ PASS: Good Reconstruction (Minor Float Precision noise)")
    else:
        print(f"❌ FAIL: Reconstruction broken. Error: {max_error}")
        return

    print("\n" + "=" * 60)
    print("🎉 INTEGRITY VERIFIED: Your Inverse Logic Works.")
    print("=" * 60)


if __name__ == "__main__":
    verify_math()