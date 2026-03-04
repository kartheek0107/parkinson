# spectral_distance.py
"""
Itakura-Saito Cosh (IS-CosH) Spectral Distance Measure
Enhanced for publication-quality plots with LARGE, BOLD text.
All enhancements included inline - no additional files needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft


# ============================================================================
# PUBLICATION-QUALITY MATPLOTLIB CONFIGURATION
# ============================================================================

def setup_publication_quality_plots():
    """
    Configure matplotlib for publication-quality figures with LARGE, BOLD text.
    """
    # Font configuration
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'

    # Axes labels and titles - MUCH LARGER
    plt.rcParams['axes.labelsize'] = 20  # +67% larger
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 22  # +83% larger
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 2.5  # +150% thicker

    # Tick labels - MUCH LARGER
    plt.rcParams['xtick.labelsize'] = 18  # +80% larger
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['xtick.major.width'] = 2.5  # +150% thicker
    plt.rcParams['ytick.major.width'] = 2.5
    plt.rcParams['xtick.major.size'] = 10  # +150% larger
    plt.rcParams['ytick.major.size'] = 10

    # Legend - LARGER AND BOLDER
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['legend.title_fontsize'] = 18
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['legend.framealpha'] = 0.95

    # Lines and markers - THICKER AND LARGER
    plt.rcParams['lines.linewidth'] = 3.0  # +100% thicker
    plt.rcParams['lines.markersize'] = 10  # +67% larger

    # Grid - BETTER VISIBILITY
    plt.rcParams['grid.linewidth'] = 1.5
    plt.rcParams['grid.alpha'] = 0.4

    # DPI settings - PUBLICATION STANDARD
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def compute_power_spectrum(signal, n_fft=512):
    """
    Compute power spectrum of signal.

    Args:
        signal: Time-domain signal
        n_fft: FFT size

    Returns:
        Power spectrum (positive frequencies only)
    """
    spectrum = fft(signal, n=n_fft)
    power = np.abs(spectrum[:n_fft // 2 + 1]) ** 2
    power = np.maximum(power, 1e-10)
    return power


def is_cosh_distance(original_signal, reconstructed_signal, n_fft=512):
    """
    Compute Itakura-Saito Cosh (IS-CosH) spectral distance.

    IS-CosH = (1/K) * Σ [P_orig(k)/P_recon(k) + P_recon(k)/P_orig(k) - 2]

    where K is the number of frequency bins.

    Args:
        original_signal: Original time-domain signal
        reconstructed_signal: Reconstructed time-domain signal
        n_fft: FFT size

    Returns:
        IS-CosH distance value (lower is better)
    """
    P_orig = compute_power_spectrum(original_signal, n_fft)
    P_recon = compute_power_spectrum(reconstructed_signal, n_fft)

    ratio1 = P_orig / P_recon
    ratio2 = P_recon / P_orig

    is_cosh = np.mean(ratio1 + ratio2 - 2)

    return is_cosh


def compute_frame_distances(original_frames, reconstructed_frames, n_fft=512):
    """
    Compute IS-CosH distance for each frame.

    Args:
        original_frames: List of original frames
        reconstructed_frames: List of reconstructed frames
        n_fft: FFT size

    Returns:
        List of IS-CosH distances per frame
    """
    distances = []

    for orig, recon in zip(original_frames, reconstructed_frames):
        dist = is_cosh_distance(orig, recon, n_fft)
        distances.append(dist)

    return np.array(distances)


def frame_signal(signal, frame_length, hop_length):
    """
    Frame a signal into overlapping frames.

    Args:
        signal: Input signal
        frame_length: Frame size
        hop_length: Hop size

    Returns:
        List of frames
    """
    frames = []
    start = 0

    while start + frame_length <= len(signal):
        frame = signal[start:start + frame_length]
        frames.append(frame)
        start += hop_length

    return frames


def plot_spectral_distance(original_signal, reconstructed_signal,
                           frame_length=512, hop_length=256,
                           n_fft=512, sr=16000, save_path=None,
                           title="IS-CosH Spectral Distance"):
    """
    Plot IS-CosH spectral distance over time with LARGE, BOLD text.

    Args:
        original_signal: Original audio
        reconstructed_signal: Reconstructed audio
        frame_length: Frame size
        hop_length: Hop size
        n_fft: FFT size
        sr: Sample rate
        save_path: Optional save path
        title: Plot title
    """
    setup_publication_quality_plots()

    # Frame both signals
    orig_frames = frame_signal(original_signal, frame_length, hop_length)
    recon_frames = frame_signal(reconstructed_signal, frame_length, hop_length)

    # Ensure same number of frames
    n_frames = min(len(orig_frames), len(recon_frames))
    orig_frames = orig_frames[:n_frames]
    recon_frames = recon_frames[:n_frames]

    # Compute distances
    distances = compute_frame_distances(orig_frames, recon_frames, n_fft)

    # Time axis
    time_axis = np.arange(n_frames) * hop_length / sr

    # Create plot with LARGE text
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f'{title} - Frame-by-Frame Analysis',
                 fontsize=24, fontweight='bold', y=0.995)

    # Panel 1: IS-CosH distance over time
    axes[0].plot(time_axis, distances, 'o-', linewidth=3.5, markersize=7,
                 color='#FF0000', markerfacecolor='#FF6600', markeredgewidth=2)
    axes[0].set_ylabel('IS-CosH Distance', fontsize=20, fontweight='bold')
    axes[0].set_title('(a) Temporal Evolution of IS-CosH Distance',
                      fontsize=20, fontweight='bold', loc='left')
    axes[0].grid(True, alpha=0.3, linewidth=1.5)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].set_xlim([time_axis[0], time_axis[-1]])

    # Panel 2: Histogram with statistics
    mean_val = np.mean(distances)
    median_val = np.median(distances)

    axes[1].hist(distances, bins=40, alpha=0.75,
                 color='#0066CC', edgecolor='black', linewidth=2.5)
    axes[1].axvline(mean_val, color='#FF0000', linestyle='--', linewidth=3.5,
                    label=f'Mean: {mean_val:.4f}')
    axes[1].axvline(median_val, color='#00AA00', linestyle='-.', linewidth=3.5,
                    label=f'Median: {median_val:.4f}')

    axes[1].set_xlabel('IS-CosH Distance', fontsize=20, fontweight='bold')
    axes[1].set_ylabel('Frequency (Counts)', fontsize=20, fontweight='bold')
    axes[1].set_title('(b) Distribution of IS-CosH Distances',
                      fontsize=20, fontweight='bold', loc='left')
    axes[1].legend(fontsize=18, loc='upper right', frameon=True, fancybox=False)
    axes[1].grid(True, alpha=0.3, axis='y', linewidth=1.5)
    axes[1].tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved IS-CosH plot to {save_path}")

    # Print statistics with LARGE fonts conceptually
    print("\n" + "=" * 70)
    print(f"{title.upper()} - STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"Total Frames Analyzed:  {len(distances)}")
    print(f"Mean IS-CosH Distance:   {mean_val:.6f}")
    print(f"Median IS-CosH Distance: {median_val:.6f}")
    print(f"Std Dev IS-CosH Distance:{np.std(distances):.6f}")
    print(f"Min IS-CosH Distance:    {np.min(distances):.6f}")
    print(f"Max IS-CosH Distance:    {np.max(distances):.6f}")
    print("=" * 70)
    print("NOTE: Lower IS-CosH values indicate better reconstruction quality")
    print("=" * 70)

    plt.show()
    return distances


def compare_spectra(original_signal, reconstructed_signal, n_fft=512, sr=16000,
                    save_path=None, title="Power Spectrum Comparison"):
    """
    Compare power spectra of original and reconstructed signals with LARGE text.

    Args:
        original_signal: Original audio
        reconstructed_signal: Reconstructed audio
        n_fft: FFT size
        sr: Sample rate
        save_path: Optional save path
        title: Plot title
    """
    setup_publication_quality_plots()

    # Compute power spectra
    P_orig = compute_power_spectrum(original_signal, n_fft)
    P_recon = compute_power_spectrum(reconstructed_signal, n_fft)

    # Frequency axis
    freqs = np.linspace(0, sr / 2, len(P_orig))

    # Convert to dB
    P_orig_db = 10 * np.log10(P_orig + 1e-10)
    P_recon_db = 10 * np.log10(P_recon + 1e-10)

    # Create plot with LARGE text
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.995)

    # Panel 1: Overlaid spectra
    axes[0].plot(freqs, P_orig_db, linewidth=3.5, label='Original Signal',
                 color='#0066CC', alpha=0.85)
    axes[0].plot(freqs, P_recon_db, linewidth=3.5, label='Reconstructed Signal',
                 color='#00AA00', alpha=0.85, linestyle='--')
    axes[0].fill_between(freqs, P_orig_db, alpha=0.15, color='#0066CC')
    axes[0].set_ylabel('Power (dB)', fontsize=20, fontweight='bold')
    axes[0].set_title('(a) Power Spectrum: Original vs Reconstructed',
                      fontsize=20, fontweight='bold', loc='left')
    axes[0].grid(True, alpha=0.3, linewidth=1.5)
    axes[0].legend(fontsize=18, loc='upper right', frameon=True, fancybox=False)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].set_xlim([0, sr / 2])

    # Panel 2: Spectral difference
    diff = P_orig_db - P_recon_db
    axes[1].plot(freqs, diff, linewidth=3.5, color='#FF6600')
    axes[1].axhline(0, color='black', linestyle='-', linewidth=2.5, alpha=0.7)
    axes[1].fill_between(freqs, 0, diff, where=(diff >= 0), alpha=0.3,
                         color='#FF0000', label='Original > Reconstructed')
    axes[1].fill_between(freqs, 0, diff, where=(diff < 0), alpha=0.3,
                         color='#0066CC', label='Reconstructed > Original')
    axes[1].set_xlabel('Frequency (Hz)', fontsize=20, fontweight='bold')
    axes[1].set_ylabel('Difference (dB)', fontsize=20, fontweight='bold')
    axes[1].set_title('(b) Spectral Difference: Original - Reconstructed',
                      fontsize=20, fontweight='bold', loc='left')
    axes[1].grid(True, alpha=0.3, linewidth=1.5)
    axes[1].legend(fontsize=18, loc='upper right', frameon=True, fancybox=False)
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    axes[1].set_xlim([0, sr / 2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved spectrum comparison to {save_path}")

    # Compute metrics
    mse_spectrum = np.mean((P_orig_db - P_recon_db) ** 2)

    print("\n" + "=" * 70)
    print(f"{title.upper()} - SPECTRAL METRICS")
    print("=" * 70)
    print(f"MSE (Spectrum):       {mse_spectrum:.6f} dB²")
    print(f"Mean Difference:      {np.mean(diff):.6f} dB")
    print(f"Max Difference:       {np.max(np.abs(diff)):.6f} dB")
    print(f"Spectral Correlation: {np.corrcoef(P_orig_db, P_recon_db)[0, 1]:.6f}")
    print("=" * 70)

    plt.show()
    return P_orig_db, P_recon_db, diff


def plot_comparative_spectra(signals_dict, n_fft=512, sr=16000, save_path=None):
    """
    Compare multiple signal spectra on a single plot with LARGE text.

    Args:
        signals_dict: Dictionary of {label: signal} pairs
        n_fft: FFT size
        sr: Sample rate
        save_path: Optional save path
    """
    setup_publication_quality_plots()

    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    colors = ['#0066CC', '#00AA00', '#FF0000', '#FF6600', '#9900CC']

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle('Comparative Power Spectrum Analysis', fontsize=24, fontweight='bold', y=0.98)

    for idx, (label, signal) in enumerate(signals_dict.items()):
        P = compute_power_spectrum(signal, n_fft)
        P_db = 10 * np.log10(P + 1e-10)
        ax.plot(freqs, P_db, linewidth=3.5, label=label, color=colors[idx % len(colors)])

    ax.set_xlabel('Frequency (Hz)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Power (dB)', fontsize=20, fontweight='bold')
    ax.set_title('Multiple Signal Spectra Comparison', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.legend(fontsize=16, loc='upper right', frameon=True, fancybox=False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim([0, sr / 2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved comparative spectrum plot to {save_path}")

    plt.show()


# Test functions
if __name__ == "__main__":
    print("=" * 70)
    print("IS-COSH SPECTRAL DISTANCE TEST (PUBLICATION QUALITY)")
    print("=" * 70)

    # Generate test signals
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Original signal
    original = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)

    # Reconstructed with slight error
    reconstructed = original + 0.05 * np.random.randn(len(original))

    # Compute IS-CosH distance
    print("\nComputing IS-CosH distance...")
    dist = is_cosh_distance(original, reconstructed)
    print(f"IS-CosH Distance: {dist:.6f}")

    # Plot comparison
    print("\nGenerating publication-quality plots...")
    plot_spectral_distance(original, reconstructed, sr=sr)
    compare_spectra(original, reconstructed, sr=sr)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)