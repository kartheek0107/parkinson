# complete_pipeline.py
"""
Complete DQLCT Speech Processing Pipeline - Publication Quality Version
Enhanced for IEEE with larger, bolder, more visible text throughout all plots.
No additional files needed - all enhancements included here.
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import librosa
import time
from src.dqlct_pipeline.quaternion_core import Quaternion, create_quaternion_array
from src.dqlct_pipeline.holistic_features import HilbertQuaternionFeatures
from src.dqlct_pipeline.dqlct_transform import QLCT1D, validate_dqlct, create_standard_matrices


# ============================================================================
# PUBLICATION-QUALITY MATPLOTLIB CONFIGURATION
# ============================================================================

def setup_publication_quality_plots():
    """
    Configure matplotlib for publication-quality figures.
    Call this once at the start for all subsequent plots.
    """
    # Font configuration - BOLD and LARGE
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'

    # Axes labels and titles - MUCH LARGER
    plt.rcParams['axes.labelsize'] = 20  # X and Y labels: +67% larger
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 22  # Subplot titles: +83% larger
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 2.5  # Spine thickness: +150%

    # Tick labels - MUCH LARGER
    plt.rcParams['xtick.labelsize'] = 18  # +80% larger than original
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['xtick.major.width'] = 2.5  # +150% thicker
    plt.rcParams['ytick.major.width'] = 2.5
    plt.rcParams['xtick.major.size'] = 10  # +150% larger
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['xtick.minor.size'] = 6
    plt.rcParams['ytick.minor.size'] = 6

    # Legend - LARGER AND BOLDER
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['legend.title_fontsize'] = 18
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['legend.framealpha'] = 0.95
    plt.rcParams['legend.borderpad'] = 0.8

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
# MAIN PROCESSOR CLASS WITH ENHANCED PLOTS
# ============================================================================

class DQLCTSpeechProcessor:
    """
    Complete speech processing pipeline using DQLCT.
    All plots now use publication-quality formatting with large, bold text.
    """

    def __init__(self, sr=16000, frame_length=512, hop_length=256,
                 matrix_type='Fractional_45deg'):
        # Apply publication settings once at initialization
        setup_publication_quality_plots()

        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.matrix_type = matrix_type

        self.feature_extractor = HilbertQuaternionFeatures(
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length
        )

        matrices = create_standard_matrices()
        if matrix_type not in matrices:
            raise ValueError(f"Unknown matrix type: {matrix_type}")

        self.a, self.b, self.c, self.d = matrices[matrix_type]

        print(f"\n✓ Initialized DQLCTSpeechProcessor (Publication-Quality Plots)")
        print(f"  Matrix type: {matrix_type}")
        print(f"  ABCD: [{self.a:.4f}, {self.b:.4f}, {self.c:.4f}, {self.d:.4f}]")

    def load_audio(self, audio_file, max_duration=None):
        """Load audio file."""
        try:
            audio, sr = librosa.load(audio_file, sr=self.sr, duration=max_duration)
            print(f"\n✓ Loaded: {audio_file}")
            print(f"  Duration: {len(audio) / sr:.2f} seconds")
            print(f"  Samples: {len(audio)}")
            return audio
        except Exception as e:
            raise ValueError(f"Could not load audio: {e}")

    def process_audio(self, audio, validate=True):
        """Complete processing pipeline with overlap-add reconstruction."""
        print("\n" + "=" * 70)
        print("DQLCT SPEECH PROCESSING PIPELINE")
        print("=" * 70)

        results = {
            'audio': audio,
            'sr': self.sr,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'matrix_type': self.matrix_type
        }

        # Step 1: Extract quaternion features
        print("\nStep 1: Extracting Holistic Quaternion Features")
        print("-" * 60)
        quat_signal = self.feature_extractor.audio_to_quaternion_signal(audio)
        results['quaternion_signal'] = quat_signal

        # Step 2: Frame the signal
        print("\nStep 2: Framing Quaternion Signal")
        print("-" * 60)
        frames, frame_positions = self._frame_signal_with_positions(quat_signal)
        results['frames'] = frames
        results['frame_positions'] = frame_positions
        print(f"  Created {len(frames)} frames of size {self.frame_length}")

        # Step 3: Apply DQLCT
        print("\nStep 3: Applying DQLCT Transform")
        print("-" * 60)

        qlct = QLCT1D(self.frame_length, self.a, self.b, self.c, self.d)
        results['qlct'] = qlct

        frame_results = []
        processing_times = []

        for i, frame in enumerate(frames):
            if i % 10 == 0:
                print(f"  Processing frame {i + 1}/{len(frames)}...")

            start_time = time.time()
            spectrum = qlct.direct_transform(frame)
            reconstructed = qlct.inverse_transform(spectrum)
            frame_error = sum((frame[j] - reconstructed[j]).norm()
                              for j in range(len(frame))) / len(frame)
            processing_time = time.time() - start_time

            frame_results.append({
                'frame_idx': i,
                'original': frame,
                'spectrum': spectrum,
                'reconstructed': reconstructed,
                'error': frame_error,
                'time': processing_time,
                'position': frame_positions[i]
            })
            processing_times.append(processing_time)

        results['frame_results'] = frame_results
        results['processing_times'] = processing_times

        # Step 4: Reconstruct with overlap-add
        print("\nStep 4: Reconstructing Full Signal (Overlap-Add)")
        print("-" * 60)
        reconstructed_full = self._overlap_add_reconstruction(frame_results, len(quat_signal))
        results['reconstructed_signal'] = reconstructed_full
        print(f"  ✓ Reconstructed {len(reconstructed_full)} samples")

        # Step 5: Validation
        if validate and len(frames) > 0:
            print("\nStep 5: Validation")
            print("-" * 60)
            validation_results = validate_dqlct(qlct, frames[0])
            results['validation'] = validation_results

        # Step 6: Statistics
        print("\nStep 6: Processing Statistics")
        print("-" * 60)
        errors = [fr['error'] for fr in frame_results]
        times = [fr['time'] for fr in frame_results]

        print(f"  Frames processed: {len(frame_results)}")
        print(f"  Mean reconstruction error: {np.mean(errors):.6e}")
        print(f"  Max reconstruction error: {np.max(errors):.6e}")
        print(f"  Mean processing time per frame: {np.mean(times):.4f} seconds")
        print(f"  Total processing time: {np.sum(times):.2f} seconds")

        results['stats'] = {
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'mean_time': np.mean(times),
            'total_time': np.sum(times)
        }

        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)

        return results

    def _frame_signal_with_positions(self, quat_signal):
        """Frame signal and track positions."""
        frames = []
        positions = []
        zero_q = Quaternion(0, 0, 0, 0)

        start = 0
        while start < len(quat_signal):
            end = start + self.frame_length
            if end <= len(quat_signal):
                frame = quat_signal[start:end]
            else:
                pad_len = end - len(quat_signal)
                frame = quat_signal[start:] + [zero_q] * pad_len

            frames.append(create_quaternion_array(frame))
            positions.append(start)
            start += self.hop_length

        return frames, positions

    def _overlap_add_reconstruction(self, frame_results, target_length):
        """Reconstruct signal using overlap-add with Hanning window."""
        accumulator = [Quaternion(0, 0, 0, 0) for _ in range(target_length)]
        window_sum = np.zeros(target_length)
        window = np.hanning(self.frame_length)

        for fr in frame_results:
            start_pos = fr['position']
            reconstructed = fr['reconstructed']

            for i, q in enumerate(reconstructed):
                pos = start_pos + i
                if pos < target_length:
                    w_val = window[i]
                    accumulator[pos] = accumulator[pos] + (q * w_val)
                    window_sum[pos] += w_val

        reconstructed_signal = []
        for i, q in enumerate(accumulator):
            if window_sum[i] > 0:
                normalized = Quaternion(
                    q.w / window_sum[i],
                    q.x / window_sum[i],
                    q.y / window_sum[i],
                    q.z / window_sum[i]
                )
                reconstructed_signal.append(normalized)
            else:
                reconstructed_signal.append(q)

        return reconstructed_signal

    def visualize_results(self, results, save_prefix=None):
        """Create and save all publication-quality visualization plots."""
        print("\n" + "=" * 70)
        print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
        print("=" * 70)

        # Plot 1: Hilbert Features
        print("\n1. Hilbert Transform Components")
        try:
            save_path = f"{save_prefix}_features.png" if save_prefix else None
            self._plot_hilbert_features(
                results['audio'],
                results['quaternion_signal'],
                save_path
            )
            plt.close('all')
        except Exception as e:
            print(f"  ⚠ Error in plot 1: {e}")

        # Plot 2: IS-CosH Before DQLCT
        print("\n2. IS-CosH Distance (Before DQLCT)")
        try:
            save_path = f"{save_prefix}_iscosh_before.png" if save_prefix else None
            self._plot_iscosh_before_dqlct(results, save_path)
            plt.close('all')
        except Exception as e:
            print(f"  ⚠ Error in plot 2: {e}")

        # Plot 3: DQLCT Magnitude Spectrum
        print("\n3. DQLCT Magnitude Spectrum")
        try:
            save_path = f"{save_prefix}_dqlct_spectrum.png" if save_prefix else None
            self._plot_dqlct_magnitude_spectrum(results, save_path)
            plt.close('all')
        except Exception as e:
            print(f"  ⚠ Error in plot 3: {e}")

        # Plot 4: Spectrogram
        print("\n4. DQLCT Spectrogram")
        try:
            save_path = f"{save_prefix}_spectrogram.png" if save_prefix else None
            self._plot_dqlct_spectrogram(results, save_path)
            plt.close('all')
        except Exception as e:
            print(f"  ⚠ Error in plot 4: {e}")

        # Plot 5: IS-CosH After DQLCT
        print("\n5. IS-CosH Distance (After Inverse DQLCT)")
        try:
            save_path = f"{save_prefix}_iscosh_after.png" if save_prefix else None
            self._plot_iscosh_after_dqlct(results, save_path)
            plt.close('all')
        except Exception as e:
            print(f"  ⚠ Error in plot 5: {e}")

        # Plot 6: Reconstruction
        print("\n6. Reconstruction Comparison")
        try:
            save_path = f"{save_prefix}_reconstruction.png" if save_prefix else None
            self._plot_reconstruction_analysis(results, save_path)
            plt.close('all')
        except Exception as e:
            print(f"  ⚠ Error in plot 6: {e}")

        # Plot 7: Errors
        print("\n7. Error Analysis")
        try:
            save_path = f"{save_prefix}_errors.png" if save_prefix else None
            self._plot_error_analysis(results, save_path)
            plt.close('all')
        except Exception as e:
            print(f"  ⚠ Error in plot 7: {e}")

        # Plot 8: Waveform
        print("\n8. Reconstructed Waveform")
        try:
            save_path = f"{save_prefix}_waveform.png" if save_prefix else None
            self._plot_reconstructed_waveform(results, save_path)
            plt.close('all')
        except Exception as e:
            print(f"  ⚠ Error in plot 8: {e}")

        print("\n✓ All visualizations complete")

    def _plot_hilbert_features(self, audio, quat_signal, save_path):
        """Plot Hilbert features with LARGE, BOLD text."""
        w_vals = [q.w for q in quat_signal]
        x_vals = [q.x for q in quat_signal]
        time_axis = np.arange(len(quat_signal)) / self.sr

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Quaternion Feature Extraction: Standard Hilbert Transform',
                     fontsize=24, fontweight='bold', y=0.995)

        # Original waveform
        axes[0].plot(time_axis, audio, linewidth=3, color='black')
        axes[0].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
        axes[0].set_title('(a) Original Audio Waveform', fontsize=20, fontweight='bold', loc='left')
        axes[0].grid(True, alpha=0.3, linewidth=1.5)
        axes[0].tick_params(axis='both', which='major', labelsize=18)

        # W-component
        axes[1].plot(time_axis, w_vals, color='#0066CC', linewidth=3)
        axes[1].fill_between(time_axis, w_vals, alpha=0.25, color='#0066CC')
        axes[1].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
        axes[1].set_title('(b) W-Component: Real Part (Original Signal)',
                          fontsize=20, fontweight='bold', loc='left')
        axes[1].grid(True, alpha=0.3, linewidth=1.5)
        axes[1].tick_params(axis='both', which='major', labelsize=18)

        # X-component
        axes[2].plot(time_axis, x_vals, color='#CC0000', linewidth=3)
        axes[2].fill_between(time_axis, x_vals, alpha=0.25, color='#CC0000')
        axes[2].set_xlabel('Time (seconds)', fontsize=20, fontweight='bold')
        axes[2].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
        axes[2].set_title('(c) X-Component: Imaginary Part (Hilbert Transform)',
                          fontsize=20, fontweight='bold', loc='left')
        axes[2].grid(True, alpha=0.3, linewidth=1.5)
        axes[2].tick_params(axis='both', which='major', labelsize=18)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {save_path}")

    def _plot_iscosh_before_dqlct(self, results, save_path):
        """Plot IS-CosH before DQLCT with LARGE text."""
        from src.dqlct_pipeline.spectral_distance import compute_frame_distances, frame_signal

        audio = results['audio']
        quat_signal = results['quaternion_signal']
        quat_audio = np.array([q.w for q in quat_signal])
        min_len = min(len(audio), len(quat_audio))

        orig_frames = frame_signal(audio[:min_len], self.frame_length, self.hop_length)
        quat_frames = frame_signal(quat_audio[:min_len], self.frame_length, self.hop_length)
        distances = compute_frame_distances(orig_frames, quat_frames, self.frame_length)
        time_axis = np.arange(len(distances)) * self.hop_length / self.sr

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('IS-CosH Spectral Distance Analysis: Before DQLCT Transform',
                     fontsize=24, fontweight='bold', y=0.995)

        # Time series
        axes[0].plot(time_axis, distances, linewidth=3.5, color='#FF0000',
                     marker='o', markersize=6, markerfacecolor='#FF6600', markeredgewidth=2)
        axes[0].set_ylabel('IS-CosH Distance', fontsize=20, fontweight='bold')
        axes[0].set_title('(a) IS-CosH Distance Over Time', fontsize=20, fontweight='bold', loc='left')
        axes[0].grid(True, alpha=0.3, linewidth=1.5)
        axes[0].tick_params(axis='both', which='major', labelsize=18)

        # Histogram
        mean_val = np.mean(distances)
        axes[1].hist(distances, bins=40, alpha=0.75, color='#0066CC',
                     edgecolor='black', linewidth=2)
        axes[1].axvline(mean_val, color='#FF0000', linestyle='--', linewidth=3.5,
                        label=f'Mean: {mean_val:.4f}')
        axes[1].set_xlabel('IS-CosH Distance', fontsize=20, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=20, fontweight='bold')
        axes[1].set_title('(b) IS-CosH Distribution', fontsize=20, fontweight='bold', loc='left')
        axes[1].legend(fontsize=18, loc='upper right')
        axes[1].grid(True, alpha=0.3, axis='y', linewidth=1.5)
        axes[1].tick_params(axis='both', which='major', labelsize=18)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {save_path}")

    def _plot_iscosh_after_dqlct(self, results, save_path):
        """Plot IS-CosH after DQLCT with LARGE text."""
        from src.dqlct_pipeline.spectral_distance import compute_frame_distances, frame_signal

        audio = results['audio']
        reconstructed_signal = results['reconstructed_signal']
        reconstructed_audio = np.array([q.w for q in reconstructed_signal])
        min_len = min(len(audio), len(reconstructed_audio))

        orig_frames = frame_signal(audio[:min_len], self.frame_length, self.hop_length)
        recon_frames = frame_signal(reconstructed_audio[:min_len], self.frame_length, self.hop_length)
        distances = compute_frame_distances(orig_frames, recon_frames, self.frame_length)
        time_axis = np.arange(len(distances)) * self.hop_length / self.sr

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('IS-CosH Spectral Distance Analysis: After Inverse DQLCT Transform',
                     fontsize=24, fontweight='bold', y=0.995)

        # Time series
        axes[0].plot(time_axis, distances, linewidth=3.5, color='#00AA00',
                     marker='s', markersize=6, markerfacecolor='#00DD00', markeredgewidth=2)
        axes[0].set_ylabel('IS-CosH Distance', fontsize=20, fontweight='bold')
        axes[0].set_title('(a) IS-CosH Distance Over Time', fontsize=20, fontweight='bold', loc='left')
        axes[0].grid(True, alpha=0.3, linewidth=1.5)
        axes[0].tick_params(axis='both', which='major', labelsize=18)

        # Histogram
        mean_val = np.mean(distances)
        axes[1].hist(distances, bins=40, alpha=0.75, color='#00AA00',
                     edgecolor='black', linewidth=2)
        axes[1].axvline(mean_val, color='#FF0000', linestyle='--', linewidth=3.5,
                        label=f'Mean: {mean_val:.4f}')
        axes[1].set_xlabel('IS-CosH Distance', fontsize=20, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=20, fontweight='bold')
        axes[1].set_title('(b) IS-CosH Distribution', fontsize=20, fontweight='bold', loc='left')
        axes[1].legend(fontsize=18, loc='upper right')
        axes[1].grid(True, alpha=0.3, axis='y', linewidth=1.5)
        axes[1].tick_params(axis='both', which='major', labelsize=18)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {save_path}")

    def _plot_dqlct_magnitude_spectrum(self, results, save_path):
        """Plot DQLCT magnitude spectrum with LARGE text."""
        frame_results = results['frame_results']
        n_frames = len(frame_results)

        frame_indices = [0, n_frames // 2, n_frames - 1] if n_frames > 2 else [0]
        freqs = np.fft.rfftfreq(self.frame_length, 1 / self.sr)

        fig, axes = plt.subplots(len(frame_indices), 1, figsize=(16, 5 * len(frame_indices)))
        if len(frame_indices) == 1:
            axes = [axes]
        fig.suptitle('DQLCT Magnitude Spectrum Analysis', fontsize=24, fontweight='bold', y=0.995)

        labels = ['(a)', '(b)', '(c)']
        for idx, frame_idx in enumerate(frame_indices):
            spectrum = frame_results[frame_idx]['spectrum']
            magnitudes = np.array([q.norm() for q in spectrum])
            mag_db = 20 * np.log10(magnitudes + 1e-12)
            n_bins = len(magnitudes) // 2 + 1

            axes[idx].plot(freqs, mag_db[:n_bins], linewidth=3.5, color='#0066CC')
            axes[idx].fill_between(freqs, mag_db[:n_bins], alpha=0.25, color='#0066CC')
            axes[idx].set_ylabel('Magnitude (dB)', fontsize=20, fontweight='bold')
            time_sec = frame_idx * self.hop_length / self.sr
            axes[idx].set_title(f'{labels[idx]} Frame {frame_idx} (t = {time_sec:.3f} s)',
                                fontsize=20, fontweight='bold', loc='left')
            axes[idx].grid(True, alpha=0.3, linewidth=1.5)
            axes[idx].tick_params(axis='both', which='major', labelsize=18)
            axes[idx].set_xlim([0, self.sr / 2])

            if idx == len(frame_indices) - 1:
                axes[idx].set_xlabel('Frequency (Hz)', fontsize=20, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {save_path}")

    def _plot_dqlct_spectrogram(self, results, save_path):
        """Plot DQLCT spectrogram with LARGE text."""
        frame_results = results['frame_results']
        n_frames = len(frame_results)

        mag_matrix = np.zeros((n_frames, self.frame_length))
        for i, fr in enumerate(frame_results):
            mag_matrix[i, :] = [q.norm() for q in fr['spectrum']]

        mag_db = 20 * np.log10(mag_matrix + 1e-12)
        frame_times = [(i * self.hop_length) / self.sr for i in range(n_frames)]
        freq_bins = np.fft.rfftfreq(self.frame_length, 1 / self.sr)

        fig, ax = plt.subplots(figsize=(16, 8))
        fig.suptitle('DQLCT Magnitude Spectrogram', fontsize=24, fontweight='bold', y=0.98)

        im = ax.imshow(mag_db[:, :len(freq_bins)].T, aspect='auto', origin='lower',
                       extent=[frame_times[0], frame_times[-1], freq_bins[0], freq_bins[-1]],
                       cmap='viridis', interpolation='bilinear')

        ax.set_xlabel('Time (seconds)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, alpha=0.2, linewidth=1)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitude (dB)', fontsize=20, fontweight='bold')
        cbar.ax.tick_params(labelsize=18)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {save_path}")

    def _plot_reconstruction_analysis(self, results, save_path):
        """Plot reconstruction comparison with LARGE text."""
        quat_signal = results['quaternion_signal']
        reconstructed_full = results['reconstructed_signal']
        min_len = min(len(quat_signal), len(reconstructed_full))
        time = np.arange(min_len) / self.sr

        w_orig = [q.w for q in quat_signal[:min_len]]
        x_orig = [q.x for q in quat_signal[:min_len]]
        w_recon = [q.w for q in reconstructed_full[:min_len]]
        x_recon = [q.x for q in reconstructed_full[:min_len]]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Signal Reconstruction Analysis: Quaternion Components',
                     fontsize=24, fontweight='bold', y=0.995)

        # W original
        axes[0, 0].plot(time, w_orig, color='#0066CC', linewidth=3)
        axes[0, 0].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
        axes[0, 0].set_title('(a) Original W-Component', fontsize=20, fontweight='bold', loc='left')
        axes[0, 0].grid(True, alpha=0.3, linewidth=1.5)
        axes[0, 0].tick_params(axis='both', which='major', labelsize=18)

        # W reconstructed
        axes[0, 1].plot(time, w_recon, color='#0066CC', linewidth=3)
        axes[0, 1].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
        axes[0, 1].set_title('(b) Reconstructed W-Component', fontsize=20, fontweight='bold', loc='left')
        axes[0, 1].grid(True, alpha=0.3, linewidth=1.5)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=18)

        # X original
        axes[1, 0].plot(time, x_orig, color='#CC0000', linewidth=3)
        axes[1, 0].set_xlabel('Time (seconds)', fontsize=20, fontweight='bold')
        axes[1, 0].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
        axes[1, 0].set_title('(c) Original X-Component', fontsize=20, fontweight='bold', loc='left')
        axes[1, 0].grid(True, alpha=0.3, linewidth=1.5)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=18)

        # X reconstructed
        axes[1, 1].plot(time, x_recon, color='#CC0000', linewidth=3)
        axes[1, 1].set_xlabel('Time (seconds)', fontsize=20, fontweight='bold')
        axes[1, 1].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
        axes[1, 1].set_title('(d) Reconstructed X-Component', fontsize=20, fontweight='bold', loc='left')
        axes[1, 1].grid(True, alpha=0.3, linewidth=1.5)
        axes[1, 1].tick_params(axis='both', which='major', labelsize=18)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {save_path}")

    def _plot_error_analysis(self, results, save_path):
        """Plot error analysis with LARGE text and statistics."""
        errors = [fr['error'] for fr in results['frame_results']]
        times = [fr['time'] for fr in results['frame_results']]
        frame_times = [(i * self.hop_length) / self.sr for i in range(len(errors))]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Processing Performance and Error Analysis', fontsize=24, fontweight='bold', y=0.995)

        # Reconstruction error over time
        axes[0, 0].semilogy(frame_times, errors, 'o-', linewidth=3.5, markersize=6, color='#FF0000')
        axes[0, 0].set_ylabel('Reconstruction Error (log scale)', fontsize=20, fontweight='bold')
        axes[0, 0].set_title('(a) Reconstruction Error vs Time', fontsize=20, fontweight='bold', loc='left')
        axes[0, 0].grid(True, alpha=0.3, linewidth=1.5, which='both')
        axes[0, 0].tick_params(axis='both', which='major', labelsize=18)

        # Error histogram
        axes[0, 1].hist(errors, bins=40, alpha=0.75, color='#0066CC', edgecolor='black', linewidth=2)
        axes[0, 1].axvline(np.mean(errors), color='#FF0000', linestyle='--', linewidth=3.5,
                           label=f'Mean: {np.mean(errors):.2e}')
        axes[0, 1].set_xlabel('Reconstruction Error', fontsize=20, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=20, fontweight='bold')
        axes[0, 1].set_title('(b) Error Distribution', fontsize=20, fontweight='bold', loc='left')
        axes[0, 1].legend(fontsize=18, loc='upper right')
        axes[0, 1].grid(True, alpha=0.3, axis='y', linewidth=1.5)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=18)

        # Processing time per frame
        axes[1, 0].plot(frame_times, times, 'o-', linewidth=3.5, markersize=6, color='#00AA00')
        axes[1, 0].set_xlabel('Time (seconds)', fontsize=20, fontweight='bold')
        axes[1, 0].set_ylabel('Processing Time (s)', fontsize=20, fontweight='bold')
        axes[1, 0].set_title('(c) Processing Time per Frame', fontsize=20, fontweight='bold', loc='left')
        axes[1, 0].grid(True, alpha=0.3, linewidth=1.5)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=18)

        # Statistics text box
        axes[1, 1].axis('off')
        stats_text = f"""PROCESSING STATISTICS

Total Frames: {len(errors)}

Reconstruction Error:
  Mean:  {np.mean(errors):.6e}
  Max:   {np.max(errors):.6e}
  Min:   {np.min(errors):.6e}

Processing Time:
  Mean:  {np.mean(times):.4f} s/frame
  Total: {np.sum(times):.2f} s

Audio Information:
  Duration: {len(results['audio']) / self.sr:.2f} s
  Sample Rate: {self.sr} Hz
"""
        axes[1, 1].text(0.05, 0.95, stats_text, fontsize=16, family='monospace',
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        weight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {save_path}")

    def _plot_reconstructed_waveform(self, results, save_path):
        """Plot reconstructed waveform with LARGE text."""
        audio = results['audio']
        reconstructed_signal = results['reconstructed_signal']
        reconstructed_audio = np.array([q.w for q in reconstructed_signal])

        min_len = min(len(audio), len(reconstructed_audio))
        audio = audio[:min_len]
        reconstructed_audio = reconstructed_audio[:min_len]
        time_axis = np.arange(min_len) / self.sr
        error = np.abs(audio - reconstructed_audio)

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Full Audio Reconstruction: Original vs Reconstructed Waveforms',
                     fontsize=24, fontweight='bold', y=0.995)

        # Original
        axes[0].plot(time_axis, audio, color='#0066CC', linewidth=2.5)
        axes[0].fill_between(time_axis, audio, alpha=0.2, color='#0066CC')
        axes[0].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
        axes[0].set_title('(a) Original Audio Waveform', fontsize=20, fontweight='bold', loc='left')
        axes[0].grid(True, alpha=0.3, linewidth=1.5)
        axes[0].tick_params(axis='both', which='major', labelsize=18)

        # Reconstructed
        axes[1].plot(time_axis, reconstructed_audio, color='#00AA00', linewidth=2.5)
        axes[1].fill_between(time_axis, reconstructed_audio, alpha=0.2, color='#00AA00')
        axes[1].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
        axes[1].set_title('(b) Reconstructed Waveform', fontsize=20, fontweight='bold', loc='left')
        axes[1].grid(True, alpha=0.3, linewidth=1.5)
        axes[1].tick_params(axis='both', which='major', labelsize=18)

        # Comparison with error
        axes[2].plot(time_axis, audio, color='#0066CC', linewidth=2.5, label='Original', alpha=0.8)
        axes[2].plot(time_axis, reconstructed_audio, color='#00AA00', linewidth=2.5,
                     label='Reconstructed', alpha=0.8, linestyle='--')
        axes[2].fill_between(time_axis, 0, error, color='#FF6600', alpha=0.3, label='Error')
        axes[2].set_xlabel('Time (seconds)', fontsize=20, fontweight='bold')
        axes[2].set_ylabel('Amplitude', fontsize=20, fontweight='bold')
        axes[2].set_title('(c) Comparison with Error Envelope', fontsize=20, fontweight='bold', loc='left')
        axes[2].grid(True, alpha=0.3, linewidth=1.5)
        axes[2].legend(fontsize=18, loc='upper right')
        axes[2].tick_params(axis='both', which='major', labelsize=18)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to {save_path}")

        mse = np.mean(error ** 2)
        snr = 10 * np.log10(np.mean(audio ** 2) / (mse + 1e-12))
        print(f"\n  Performance Metrics:")
        print(f"    MSE: {mse:.6e}")
        print(f"    SNR: {snr:.2f} dB")
        print(f"    Mean Error: {np.mean(error):.6e}")
        print(f"    Max Error: {np.max(error):.6e}")


def main():
    """Run complete pipeline."""
    print("=" * 70)
    print("DQLCT SPEECH PROCESSING WITH PUBLICATION-QUALITY PLOTS")
    print("=" * 70)

    audio_file = "test_2_wav.wav"
    processor = DQLCTSpeechProcessor(sr=16000, frame_length=512, hop_length=256)

    try:
        audio = processor.load_audio(audio_file)
    except Exception as e:
        print(f"⚠ Could not load {audio_file}: {e}")
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 150 * t) * np.sin(2 * np.pi * 500 * t) * np.hanning(len(t))

    results = processor.process_audio(audio, validate=True)
    processor.visualize_results(results, save_prefix="dqlct_output")

    print("\n" + "=" * 70)
    print("COMPLETE - All plots saved with publication-quality formatting")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()