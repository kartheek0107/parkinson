import sys
import os

# Add project root to path so "src" can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import numpy as np
import librosa
from tqdm import tqdm
from src.data.preprocessing import AudioPreprocessor
from src.dqlct.core import QLCT1D


class DatasetBuilder:
    def __init__(self, config):
        self.raw_dir = config['data']['raw_path']
        self.processed_dir = config['data']['processed_path']
        self.window_size = config['dqlct']['window_size']
        self.hop_length = config['dqlct']['hop_length']
        self.preprocessor = AudioPreprocessor(
            sample_rate=config['data']['sampling_rate'],
            duration=config['data']['duration']
        )
        self.qlct = QLCT1D(N=self.window_size, cfg=config['dqlct'])

    def compute_short_time_dqlct(self, signal):
        # 1. Frame the signal: Shape (Window_Size, Num_Frames)
        frames = librosa.util.frame(signal, frame_length=self.window_size, hop_length=self.hop_length)
        num_frames = frames.shape[1]

        # 2. Prepare Input: We treat audio as "Real" quaternion components (w)
        # Shape: (4, Window_Size, Num_Frames) -> We want to process batch of windows?
        # For simplicity, let's just loop over frames but use Vectorized QLCT inside

        spectrogram = np.zeros((4, num_frames, self.window_size), dtype=np.float32)

        # Apply Hanning window to all frames at once
        win_func = np.hanning(self.window_size)[:, None]  # Shape (N, 1)
        frames = frames * win_func

        for t in range(num_frames):
            # Input is just Real part (Audio), x,y,z are 0
            # signal_chunk shape: (512,)
            signal_chunk = frames[:, t]

            # Fast Transform -> returns (4, 512)
            spectrum = self.qlct.forward(signal_chunk)

            # Store
            spectrogram[:, t, :] = spectrum

        return torch.tensor(spectrogram)

    def process_folder(self, folder_name, label):
        input_path = os.path.join(self.raw_dir, folder_name)
        output_path = os.path.join(self.processed_dir, folder_name)
        os.makedirs(output_path, exist_ok=True)

        if not os.path.exists(input_path):
            print(f"Skipping {folder_name}, folder not found.")
            return

        files = [f for f in os.listdir(input_path) if f.endswith('.wav')]
        print(f"Processing {len(files)} files in {folder_name}...")

        for filename in tqdm(files):
            file_path = os.path.join(input_path, filename)
            clean_sig = self.preprocessor.process_file(file_path)
            dqlct_tensor = self.compute_short_time_dqlct(clean_sig)

            # Save tensor
            save_name = filename.replace('.wav', '.pt')
            torch.save(dqlct_tensor, os.path.join(output_path, save_name))

    def run(self):
        self.process_folder('pd', label=1)
        self.process_folder('healthy', label=0)


if __name__ == "__main__":
    config = {
        'data': {
            'raw_path': './data/synthetic',
            'processed_path': './data/processed',
            'sampling_rate': 16000,
            'duration': 4.0
        },
        'dqlct': {
            'window_size': 512,
            'hop_length': 256,
            'a': 0.0, 'b': 1.0, 'c': -1.0, 'd': 0.0
        }
    }
    builder = DatasetBuilder(config)
    builder.run()