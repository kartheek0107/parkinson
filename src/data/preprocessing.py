import librosa
import numpy as np


class AudioPreprocessor:
    def __init__(self, sample_rate=16000, duration=4.0):
        self.sr = sample_rate
        self.target_len = int(sample_rate * duration)

    def remove_silence(self, signal):
        """
        Strips silence from the beginning and end, and splits/joins 
        active speech segments.
        """
        # Split returns a list of [start, end] intervals where speech is present
        intervals = librosa.effects.split(signal, top_db=20)
        
        # Concatenate only the non-silent parts
        clean_signal = np.concatenate([signal[start:end] for start, end in intervals])
        
        # Fallback: if signal is all silence, return original to avoid crash
        if len(clean_signal) == 0:
            return signal
        return clean_signal

    def pad_truncate(self, signal):
        """
        Ensures every audio file is exactly 'duration' seconds long.
        """
        if len(signal) > self.target_len:
            # Truncate (take the center part usually, or just beginning)
            return signal[:self.target_len]
        else:
            # Pad with zeros (silence) at the end
            padding = self.target_len - len(signal)
            return np.pad(signal, (0, padding), 'constant')

    def process_file(self, file_path):
        """
        Master function to go from File Path -> Clean Numpy Array
        """
        # 1. Load Audio (librosa handles resampling automatically)
        signal, _ = librosa.load(file_path, sr=self.sr)
        
        # 2. Remove Silence
        signal = self.remove_silence(signal)
        
        # 3. Fix Length
        signal = self.pad_truncate(signal)
        
        # 4. Normalization (Min-Max to -1, 1 range)
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val
            
        return signal