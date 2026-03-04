# holistic_features.py
"""
Standard Hilbert Transform Quaternion Representation for Speech.
Original logic unchanged: q[n] = Re{fa[n]} + Im{fa[n]}i + 0j + 0k

matplotlib is imported lazily (inside visualize_features only) so this
module can be used in torch_env without matplotlib installed.
"""

import numpy as np
from scipy.signal import hilbert
from quaternion_core import Quaternion, create_quaternion_array


class HilbertQuaternionFeatures:
    def __init__(self, sr=16000, frame_length=512, hop_length=256):
        self.sr           = sr
        self.frame_length = frame_length
        self.hop_length   = hop_length

    def audio_to_quaternion_signal(self, audio, verbose=True):
        """
        Convert raw audio to quaternion signal using standard Hilbert transform.
        q[n] = Re{fa[n]} + Im{fa[n]}i + 0j + 0k
        Original method — logic unchanged.
        """
        if verbose:
            print("  Extracting standard Hilbert-based quaternion features...")

        analytic_signal = hilbert(audio)
        w_component     = np.real(analytic_signal)
        x_component     = np.imag(analytic_signal)
        y_component     = np.zeros_like(audio)
        z_component     = np.zeros_like(audio)

        def normalize_component(comp):
            max_val = np.max(np.abs(comp))
            return comp / max_val if max_val > 0 else comp

        w_component = normalize_component(w_component)
        x_component = normalize_component(x_component)

        quat_signal = []
        for i in range(len(audio)):
            quat_signal.append(Quaternion(
                w=float(w_component[i]),
                x=float(x_component[i]),
                y=0.0,
                z=0.0
            ))

        if verbose:
            print(f"  Created {len(quat_signal)} quaternion samples")

        return quat_signal