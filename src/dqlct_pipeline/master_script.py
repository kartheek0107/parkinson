# master_script.py
"""
Master script demonstrating complete DQLCT speech processing pipeline.
With standard Hilbert transform, IS-CosH distance, and DQLCT magnitude spectrum.

Required files:
  1. quaternion_core.py - Quaternion arithmetic
  2. holistic_features.py - Feature extraction (standard Hilbert)
  3. dqlct_transform.py - DQLCT implementation
  4. spectral_distance.py - IS-CosH distance computation
  5. complete_pipeline.py - Full processing pipeline
  6. analysis_utils.py - Additional utilities
"""

import numpy as np

# Import all modules
from src.dqlct_pipeline.complete_pipeline import DQLCTSpeechProcessor
from src.dqlct_pipeline.spectral_distance import is_cosh_distance
from src.dqlct_pipeline.analysis_utils import (
    save_spectrum_data,
    analyze_phonetic_sensitivity,
    export_results_summary
)


def run_complete_analysis(audio_file="test_2_wav.wav"):
    """
    Run complete DQLCT analysis on audio file with all features.
    """

    print("=" * 80)
    print(" " * 20 + "DQLCT SPEECH PROCESSING")
    print(" " * 10 + "Complete Analysis with IS-CosH & Spectrum Visualization")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Initialize Processor
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: INITIALIZING PROCESSOR")
    print("=" * 80)

    processor = DQLCTSpeechProcessor(
        sr=16000,
        frame_length=512,
        hop_length=256,
        matrix_type='Fractional_45deg'
    )

    # ========================================================================
    # STEP 2: Load Audio
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: LOADING AUDIO")
    print("=" * 80)

    try:
        audio = processor.load_audio(audio_file, max_duration=None)
    except Exception as e:
        print(f"⚠ Could not load {audio_file}: {e}")
        print("  Generating synthetic speech for demonstration...")

        # Generate realistic synthetic speech
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))

        # Multiple formants with vibrato
        f0 = 150 + 10 * np.sin(2 * np.pi * 5 * t)  # F0 with vibrato
        f1, f2, f3 = 500, 1500, 2500

        audio = (np.sin(2 * np.pi * f0 * t) *
                 (np.sin(2 * np.pi * f1 * t) +
                  0.5 * np.sin(2 * np.pi * f2 * t) +
                  0.3 * np.sin(2 * np.pi * f3 * t)))

        # Add amplitude envelope (speech-like)
        envelope = np.exp(-1.5 * np.abs(t - duration / 2)) + 0.1
        audio = audio * envelope * 0.3

        # Add some noise
        audio += np.random.randn(len(audio)) * 0.01

    # ========================================================================
    # STEP 3: Process Audio Through Complete Pipeline
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: PROCESSING AUDIO THROUGH DQLCT PIPELINE")
    print("=" * 80)

    results = processor.process_audio(audio, validate=True)

    # ========================================================================
    # STEP 4: Generate All Visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING ALL VISUALIZATIONS")
    print("=" * 80)
    print("This includes:")
    print("  - Hilbert transform components")
    print("  - IS-CosH distance (before DQLCT)")
    print("  - DQLCT magnitude spectrum")
    print("  - DQLCT spectrogram")
    print("  - IS-CosH distance (after inverse DQLCT)")
    print("  - Reconstruction comparison")
    print("  - Error analysis")
    print("  - Reconstructed waveform")

    processor.visualize_results(results, save_prefix="output/dqlct")

    # ========================================================================
    # STEP 5: Save Spectrum Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: SAVING SPECTRUM DATA")
    print("=" * 80)

    save_spectrum_data(results['frame_results'], "output/dqlct_spectra.npz")

    # ========================================================================
    # STEP 6: Export Results Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: EXPORTING RESULTS SUMMARY")
    print("=" * 80)

    export_results_summary(results, "output/dqlct_summary.txt")

    # ========================================================================
    # STEP 7: Compute Overall IS-CosH Distance
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: COMPUTING OVERALL IS-COSH DISTANCE")
    print("=" * 80)

    # Use properly reconstructed signal (overlap-add)
    reconstructed_quat = results['reconstructed_signal']
    reconstructed_audio = np.array([q.w for q in reconstructed_quat])

    # Trim to same length
    min_len = min(len(audio), len(reconstructed_audio))
    original_trimmed = audio[:min_len]
    reconstructed_trimmed = reconstructed_audio[:min_len]

    # Compute overall IS-CosH distance
    overall_iscosh = is_cosh_distance(original_trimmed, reconstructed_trimmed, n_fft=512)

    print(f"\nOverall IS-CosH Distance: {overall_iscosh:.6f}")
    print("(Lower values indicate better reconstruction quality)")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 80)

    print(f"\n✓ Audio processed: {len(audio) / processor.sr:.2f} seconds")
    print(f"✓ Frames analyzed: {len(results['frame_results'])}")
    print(f"✓ Mean reconstruction error: {results['stats']['mean_error']:.6e}")
    print(f"✓ Processing time: {results['stats']['total_time']:.2f} seconds")
    print(f"✓ Real-time factor: {len(audio) / processor.sr / results['stats']['total_time']:.2f}x")
    print(f"✓ IS-CosH Distance: {overall_iscosh:.6f}")

    print("\n✓ Generated files:")
    print("  - output/dqlct_features.png              (Hilbert components)")
    print("  - output/dqlct_iscosh_before.png         (IS-CosH before DQLCT)")
    print("  - output/dqlct_dqlct_spectrum.png        (DQLCT magnitude spectrum)")
    print("  - output/dqlct_spectrogram.png           (DQLCT spectrogram)")
    print("  - output/dqlct_iscosh_after.png          (IS-CosH after inverse DQLCT)")
    print("  - output/dqlct_reconstruction.png        (Reconstruction comparison)")
    print("  - output/dqlct_errors.png                (Error analysis)")
    print("  - output/dqlct_waveform.png              (Reconstructed waveform)")
    print("  - output/dqlct_spectra.npz               (Spectrum data)")
    print("  - output/dqlct_summary.txt               (Text summary)")

    print("\n" + "=" * 80)
    print(" " * 25 + "ALL DONE!")
    print("=" * 80)

    return results


def quick_test():
    """
    Quick test with minimal output for verification.
    """
    print("=" * 70)
    print("QUICK TEST - DQLCT PIPELINE")
    print("=" * 70)

    # Generate short test signal
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 150 * t) * np.sin(2 * np.pi * 500 * t) * np.hanning(len(t))

    # Initialize and process
    processor = DQLCTSpeechProcessor(sr=sr, frame_length=256, hop_length=128)
    results = processor.process_audio(audio, validate=True)

    # Quick check
    max_error = results['stats']['max_error']

    if max_error < 1e-12:
        print("\n✓ PASS: Pipeline working correctly!")
    else:
        print(f"\n⚠ WARNING: Reconstruction error = {max_error:.6e}")

    return results


def demonstrate_single_frame():
    """
    Demonstrate DQLCT on a single frame for educational purposes.
    """
    print("=" * 70)
    print("SINGLE FRAME DQLCT DEMONSTRATION")
    print("=" * 70)

    from src.dqlct_pipeline.quaternion_core import Quaternion, create_quaternion_array
    from src.dqlct_pipeline.dqlct_transform import QLCT1D

    # Create simple test signal
    N = 16
    signal = [
        Quaternion(1, 0.5, 0, 0) if i == 0 else Quaternion(0, 0, 0, 0)
        for i in range(N)
    ]
    signal = create_quaternion_array(signal)

    print(f"\nInput signal (N={N}, standard Hilbert: w + xi + 0j + 0k):")
    print(f"  signal[0] = {signal[0]}")
    print(f"  Others = 0")

    # Initialize DQLCT
    qlct = QLCT1D(N, a=0.0, b=1.0, c=-1.0, d=0.0)

    # Forward transform
    print("\nApplying DQLCT forward transform...")
    spectrum = qlct.direct_transform(signal)

    print("\nDQLCT Spectrum (first 5 values):")
    for i in range(5):
        print(f"  X[{i}] = {spectrum[i]}")

    # Inverse transform
    print("\nApplying DQLCT inverse transform...")
    reconstructed = qlct.inverse_transform(spectrum)

    print("\nReconstructed signal:")
    print(f"  reconstructed[0] = {reconstructed[0]}")

    # Check error
    error = (signal[0] - reconstructed[0]).norm()
    print(f"\nReconstruction error: {error:.6e}")

    if error < 1e-12:
        print("✓ Perfect reconstruction!")
    else:
        print("⚠ Reconstruction error detected")

    # Energy check
    E_in = sum(q.norm() ** 2 for q in signal)
    E_out = sum(q.norm() ** 2 for q in spectrum)
    print(f"\nEnergy conservation:")
    print(f"  Input energy:  {E_in:.6f}")
    print(f"  Output energy: {E_out:.6f}")
    print(f"  Difference:    {abs(E_in - E_out):.6e}")

    return signal, spectrum, reconstructed


def interactive_menu():
    """
    Interactive menu for running different analyses.
    """
    while True:
        print("\n" + "=" * 70)
        print("DQLCT SPEECH PROCESSING - INTERACTIVE MENU")
        print("=" * 70)
        print("\n1. Run complete analysis on audio file")
        print("2. Quick test with synthetic signal")
        print("3. Single frame demonstration")
        print("4. Phonetic sensitivity analysis")
        print("5. Exit")

        choice = input("\nSelect option (1-5): ")

        if choice == '1':
            audio_file = input("Enter audio file path (or press Enter for default): ")
            if not audio_file.strip():
                audio_file = "test_2_wav.wav"
            run_complete_analysis(audio_file)

        elif choice == '2':
            quick_test()

        elif choice == '3':
            demonstrate_single_frame()

        elif choice == '4':
            processor = DQLCTSpeechProcessor()
            analyze_phonetic_sensitivity(processor)

        elif choice == '5':
            print("\nExiting. Goodbye!")
            break

        else:
            print("\n⚠ Invalid choice. Please select 1-5.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    import os

    # Create output directory
    os.makedirs("output", exist_ok=True)

    print("\n" + "=" * 80)
    print(" " * 20 + "DQLCT SPEECH PROCESSING SYSTEM")
    print(" " * 20 + "With Standard Hilbert & IS-CosH")
    print("=" * 80)

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            # Quick test mode
            print("\nRunning in QUICK TEST mode...")
            quick_test()

        elif sys.argv[1] == '--demo':
            # Single frame demo
            print("\nRunning SINGLE FRAME DEMONSTRATION...")
            demonstrate_single_frame()

        elif sys.argv[1] == '--interactive':
            # Interactive menu
            interactive_menu()

        elif sys.argv[1] == '--help':
            # Help message
            print("\nUsage:")
            print("  python master_script.py                    # Run complete analysis")
            print("  python master_script.py --quick            # Quick test")
            print("  python master_script.py --demo             # Single frame demo")
            print("  python master_script.py --interactive      # Interactive menu")
            print("  python master_script.py --file <path>      # Process specific file")
            print("  python master_script.py --help             # Show this help")

        elif sys.argv[1] == '--file' and len(sys.argv) > 2:
            # Process specific file
            audio_file = sys.argv[2]
            print(f"\nProcessing file: {audio_file}")
            run_complete_analysis(audio_file)

        else:
            print(f"\n⚠ Unknown option: {sys.argv[1]}")
            print("  Use --help for usage information")

    else:
        # Default: Run complete analysis
        print("\nRunning COMPLETE ANALYSIS...")
        print("(Use --help to see other options)\n")
        run_complete_analysis()