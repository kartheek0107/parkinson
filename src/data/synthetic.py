import numpy as np
import soundfile as sf
import os


def generate_signal(duration=4.0, sr=16000, frequency=150, jitter_level=0.0):
    """
    Generates a sine wave. 
    If jitter_level > 0, injects random frequency fluctuations 
    (Parkinson's).
    """
    t = np.linspace(0, duration, int(sr * duration))
    
    # 1. Base Frequency with Jitter (Frequency Modulation)
    # Healthy = 0 jitter. PD = High jitter.
    noise = np.random.normal(0, jitter_level, size=t.shape)
    mod_frequency = frequency * (1 + noise)
    
    # 2. Integrate frequency to get phase
    phase = 2 * np.pi * np.cumsum(mod_frequency) / sr
    
    # 3. Generate Signal
    signal = np.sin(phase)
    
    # 4. Add Shimmer (Amplitude Modulation) - Optional complexity
    if jitter_level > 0:
        amp_mod = 1.0 + np.random.normal(0, jitter_level * 0.5, size=t.shape)
        signal = signal * amp_mod
        
    return signal.astype(np.float32)


def create_mock_dataset(output_dir, num_samples=20):
    """
    Creates a folder structure with synthetic Healthy and PD wav files.
    """
    classes = {'healthy': 0.0001, 'pd': 0.02}  # Jitter values
    
    for label, jitter in classes.items():
        save_path = os.path.join(output_dir, label)
        os.makedirs(save_path, exist_ok=True)
        
        for i in range(num_samples):
            # Randomize pitch slightly so not every sample is identical 150Hz
            pitch = np.random.randint(130, 180) 
            signal = generate_signal(frequency=pitch, jitter_level=jitter)
            
            filename = os.path.join(save_path, f"mock_{label}_{i}.wav")
            sf.write(filename, signal, 16000)
            
    print(f"✅ Generated {num_samples*2} mock samples in {output_dir}")


if __name__ == "__main__":
    # Test run
    create_mock_dataset("./data/synthetic")