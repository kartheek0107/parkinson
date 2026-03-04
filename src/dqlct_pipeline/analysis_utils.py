# analysis_utils.py
"""
Additional analysis utilities and export functions for DQLCT processing.
Includes spectrum extraction, phonetic analysis, and data export.
"""

import numpy as np
import matplotlib.pyplot as plt


def get_dqlct_spectrum_matrix(frame_results):
    """
    Extract quaternion spectrum components for all frames.

    Args:
        frame_results: List of frame processing results

    Returns:
        numpy array of shape (n_frames, n_bins, 4) with [w, x, y, z] components
    """
    n_frames = len(frame_results)
    if n_frames == 0:
        return np.zeros((0, 0, 4), dtype=np.float64)

    first_spec = frame_results[0]['spectrum']
    n_bins = len(first_spec)

    # Create component matrix
    comp_matrix = np.zeros((n_frames, n_bins, 4), dtype=np.float64)

    for f, fr in enumerate(frame_results):
        spectrum = fr['spectrum']
        for b, q in enumerate(spectrum):
            comp_matrix[f, b, 0] = q.w
            comp_matrix[f, b, 1] = q.x
            comp_matrix[f, b, 2] = q.y
            comp_matrix[f, b, 3] = q.z

    return comp_matrix


def get_dqlct_magnitude_db(frame_results, ref=1.0, eps=1e-12):
    """
    Compute DQLCT magnitude in dB for all frames.

    Args:
        frame_results: List of frame processing results
        ref: Reference magnitude for dB calculation
        eps: Small constant to avoid log(0)

    Returns:
        numpy array of shape (n_frames, n_bins) with magnitude in dB
    """
    comp_matrix = get_dqlct_spectrum_matrix(frame_results)

    # Compute quaternion norm: sqrt(w² + x² + y² + z²)
    norms = np.sqrt(np.sum(comp_matrix ** 2, axis=2))

    # Convert to dB
    mag_db = 20.0 * np.log10(norms / ref + eps)

    return mag_db


def save_spectrum_data(frame_results, filepath="dqlct_spectra.npz"):
    """
    Save spectrum data to compressed numpy file.

    Args:
        frame_results: List of frame processing results
        filepath: Output file path
    """
    comps = get_dqlct_spectrum_matrix(frame_results)
    mag_db = get_dqlct_magnitude_db(frame_results)

    np.savez_compressed(filepath,
                        components=comps,
                        magnitude_db=mag_db)

    print(f"✓ Saved spectrum data to {filepath}")
    print(f"  Components shape: {comps.shape}")
    print(f"  Magnitude shape: {mag_db.shape}")

    return filepath


def load_spectrum_data(filepath="dqlct_spectra.npz"):
    """
    Load spectrum data from numpy file.

    Returns:
        dict with 'components' and 'magnitude_db' arrays
    """
    data = np.load(filepath)
    print(f"✓ Loaded spectrum data from {filepath}")
    print(f"  Components shape: {data['components'].shape}")
    print(f"  Magnitude shape: {data['magnitude_db'].shape}")

    return {
        'components': data['components'],
        'magnitude_db': data['magnitude_db']
    }


def analyze_phonetic_sensitivity(processor, vowel_configs=None):
    """
    Analyze how different phonemes produce different quaternion signatures.

    Args:
        processor: DQLCTSpeechProcessor instance
        vowel_configs: Dict of vowel: (F1, F2) pairs
    """
    if vowel_configs is None:
        vowel_configs = {
            '/a/': (700, 1200),  # Low back vowel
            '/i/': (300, 2500),  # High front vowel
            '/u/': (300, 800),  # High back vowel
            '/e/': (400, 2000),  # Mid front vowel
            '/o/': (500, 900),  # Mid back vowel
        }

    print("\n" + "=" * 60)
    print("PHONETIC SENSITIVITY ANALYSIS")
    print("=" * 60)

    sr = processor.sr
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    fundamental = 150  # Hz

    results = {}

    for vowel, (f1, f2) in vowel_configs.items():
        # Generate vowel-like sound
        audio = (np.sin(2 * np.pi * fundamental * t) *
                 (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)))
        audio = audio * np.hanning(len(audio)) * 0.3

        # Extract quaternion
        quat_signal = processor.feature_extractor.audio_to_quaternion_signal(
            audio, verbose=False
        )

        # Compute average quaternion
        w_avg = np.mean([q.w for q in quat_signal])
        x_avg = np.mean([q.x for q in quat_signal])
        y_avg = np.mean([q.y for q in quat_signal])
        z_avg = np.mean([q.z for q in quat_signal])

        norm = np.sqrt(w_avg ** 2 + x_avg ** 2 + y_avg ** 2 + z_avg ** 2)

        results[vowel] = {
            'f1': f1,
            'f2': f2,
            'w': w_avg,
            'x': x_avg,
            'y': y_avg,
            'z': z_avg,
            'norm': norm
        }

        print(f"\n{vowel} (F1={f1}Hz, F2={f2}Hz):")
        print(f"  Q_avg = {w_avg:.3f} + {x_avg:.3f}i + {y_avg:.3f}j + {z_avg:.3f}k")
        print(f"  Norm = {norm:.3f}")

    # Visualize phonetic space
    _plot_phonetic_space(results)

    print("\n" + "=" * 60)
    print("Different vowels produce distinct quaternion signatures!")
    print("=" * 60)

    return results


def _plot_phonetic_space(phonetic_results):
    """Plot vowels in quaternion component space."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    vowels = list(phonetic_results.keys())
    w_vals = [phonetic_results[v]['w'] for v in vowels]
    x_vals = [phonetic_results[v]['x'] for v in vowels]
    y_vals = [phonetic_results[v]['y'] for v in vowels]
    z_vals = [phonetic_results[v]['z'] for v in vowels]

    # W vs X
    axes[0, 0].scatter(w_vals, x_vals, s=100, alpha=0.7)
    for i, v in enumerate(vowels):
        axes[0, 0].annotate(v, (w_vals[i], x_vals[i]), fontsize=12)
    axes[0, 0].set_xlabel('W (Energy)', fontsize=12)
    axes[0, 0].set_ylabel('X (Centroid)', fontsize=12)
    axes[0, 0].set_title('Energy vs Spectral Centroid', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Y vs Z
    axes[0, 1].scatter(y_vals, z_vals, s=100, alpha=0.7, color='red')
    for i, v in enumerate(vowels):
        axes[0, 1].annotate(v, (y_vals[i], z_vals[i]), fontsize=12)
    axes[0, 1].set_xlabel('Y (Pitch)', fontsize=12)
    axes[0, 1].set_ylabel('Z (Formant)', fontsize=12)
    axes[0, 1].set_title('Pitch vs Formant Dispersion', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # W vs Y
    axes[1, 0].scatter(w_vals, y_vals, s=100, alpha=0.7, color='green')
    for i, v in enumerate(vowels):
        axes[1, 0].annotate(v, (w_vals[i], y_vals[i]), fontsize=12)
    axes[1, 0].set_xlabel('W (Energy)', fontsize=12)
    axes[1, 0].set_ylabel('Y (Pitch)', fontsize=12)
    axes[1, 0].set_title('Energy vs Pitch', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # X vs Z
    axes[1, 1].scatter(x_vals, z_vals, s=100, alpha=0.7, color='purple')
    for i, v in enumerate(vowels):
        axes[1, 1].annotate(v, (x_vals[i], z_vals[i]), fontsize=12)
    axes[1, 1].set_xlabel('X (Centroid)', fontsize=12)
    axes[1, 1].set_ylabel('Z (Formant)', fontsize=12)
    axes[1, 1].set_title('Centroid vs Formant Dispersion', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_matrix_types(audio, sr=16000, frame_length=512, hop_length=256):
    """
    Compare different DQLCT matrix configurations.

    Args:
        audio: Input audio signal
        sr: Sample rate
        frame_length: Frame size
        hop_length: Hop size
    """
    from src.dqlct_pipeline.complete_pipeline import DQLCTSpeechProcessor

    matrices_to_test = ['QFT', 'Fractional_45deg', 'Fractional_30deg', 'Custom']

    print("\n" + "=" * 70)
    print("COMPARING DQLCT MATRIX CONFIGURATIONS")
    print("=" * 70)

    comparison_results = {}

    for matrix_type in matrices_to_test:
        print(f"\nTesting matrix: {matrix_type}")
        print("-" * 60)

        try:
            processor = DQLCTSpeechProcessor(
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
                matrix_type=matrix_type
            )

            results = processor.process_audio(audio, validate=False)

            comparison_results[matrix_type] = {
                'mean_error': results['stats']['mean_error'],
                'max_error': results['stats']['max_error'],
                'mean_time': results['stats']['mean_time'],
                'total_time': results['stats']['total_time']
            }

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            comparison_results[matrix_type] = None

    # Plot comparison
    _plot_matrix_comparison(comparison_results)

    return comparison_results


def _plot_matrix_comparison(comparison_results):
    """Plot comparison of different matrix types."""
    valid_matrices = {k: v for k, v in comparison_results.items() if v is not None}

    if not valid_matrices:
        print("No valid results to plot")
        return

    matrices = list(valid_matrices.keys())
    mean_errors = [valid_matrices[m]['mean_error'] for m in matrices]
    max_errors = [valid_matrices[m]['max_error'] for m in matrices]
    mean_times = [valid_matrices[m]['mean_time'] for m in matrices]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Mean errors
    axes[0].bar(matrices, mean_errors, alpha=0.7, color='blue')
    axes[0].set_ylabel('Mean Error')
    axes[0].set_title('Mean Reconstruction Error', fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)

    # Max errors
    axes[1].bar(matrices, max_errors, alpha=0.7, color='red')
    axes[1].set_ylabel('Max Error')
    axes[1].set_title('Max Reconstruction Error', fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)

    # Processing times
    axes[2].bar(matrices, mean_times, alpha=0.7, color='green')
    axes[2].set_ylabel('Time (seconds)')
    axes[2].set_title('Mean Processing Time per Frame', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def export_results_summary(results, filepath="dqlct_summary.txt"):
    """
    Export processing results to text file.

    Args:
        results: Processing results dictionary
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DQLCT SPEECH PROCESSING RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Sample rate: {results['sr']} Hz\n")
        f.write(f"Frame length: {results['frame_length']} samples\n")
        f.write(f"Hop length: {results['hop_length']} samples\n")
        f.write(f"Matrix type: {results['matrix_type']}\n")
        f.write(f"Audio duration: {len(results['audio']) / results['sr']:.2f} seconds\n")
        f.write(f"Number of frames: {len(results['frame_results'])}\n\n")

        # Statistics
        f.write("PROCESSING STATISTICS\n")
        f.write("-" * 70 + "\n")
        stats = results['stats']
        f.write(f"Mean reconstruction error: {stats['mean_error']:.6e}\n")
        f.write(f"Max reconstruction error: {stats['max_error']:.6e}\n")
        f.write(f"Mean processing time: {stats['mean_time']:.4f} seconds/frame\n")
        f.write(f"Total processing time: {stats['total_time']:.2f} seconds\n")
        f.write(f"Real-time factor: {len(results['audio']) / results['sr'] / stats['total_time']:.2f}x\n\n")

        # Validation
        if 'validation' in results:
            f.write("VALIDATION RESULTS\n")
            f.write("-" * 70 + "\n")
            val = results['validation']
            f.write(f"Energy conservation error: {val['energy_error']:.6e}\n")
            f.write(f"Reconstruction mean error: {val['reconstruction_mean_error']:.6e}\n")
            f.write(f"Reconstruction max error: {val['reconstruction_max_error']:.6e}\n")
            f.write(f"Linearity error: {val['linearity_error']:.6e}\n\n")

        f.write("=" * 70 + "\n")

    print(f"✓ Saved results summary to {filepath}")


# Test functions
if __name__ == "__main__":
    print("=" * 70)
    print("ANALYSIS UTILITIES MODULE")
    print("=" * 70)

    print("\nThis module provides utilities for:")
    print("  - Spectrum extraction and export")
    print("  - Phonetic sensitivity analysis")
    print("  - Matrix configuration comparison")
    print("  - Results summary export")

    print("\nImport this module in your main script to use these functions.")
    print("=" * 70)