from src.data.preprocessing import AudioPreprocessor
import matplotlib.pyplot as plt

processor = AudioPreprocessor(duration=4.0)

# Load a generated PD file
path = "./data/synthetic/pd/mock_pd_0.wav"
clean_audio = processor.process_file(path)

print(f"Shape: {clean_audio.shape}") # Should be (64000,) if 4s @ 16kHz

# Quick plot
plt.plot(clean_audio)
plt.title("Processed Mock Parkinson's Audio")
plt.show()