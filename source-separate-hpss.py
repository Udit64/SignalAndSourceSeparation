import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import os

class MyHPSS:
    def __init__(self, y, sr, kernel_size=31, mask_type='soft', mask_threshold=0.5):
        self.y = y
        self.sr = sr
        self.kernel_size = kernel_size
        self.mask_type = mask_type
        self.mask_threshold = mask_threshold

    def run(self):
        D = librosa.stft(self.y)
        harmonic_mask, percussive_mask = librosa.decompose.hpss(D, kernel_size=self.kernel_size, mask=True)
        
        harmonic = librosa.istft(D * harmonic_mask)
        percussive = librosa.istft(D * percussive_mask)
        return harmonic, percussive

# Load the audio file
audio_file = 'SignalSourceDataset/Black Bloc - If You Want Success.wav'
y, sr = librosa.load(audio_file, sr=None)

# Run MyHPSS
my_hpss = MyHPSS(y, sr)
harmonic, percussive = my_hpss.run()

# Save the separated outputs
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

harmonic_output = os.path.join(output_dir, 'harmonic_output.wav')
percussive_output = os.path.join(output_dir, 'percussive_output.wav')

sf.write(harmonic_output, harmonic, sr)
sf.write(percussive_output, percussive, sr)

print(f"Saved harmonic component to {harmonic_output}")
print(f"Saved percussive component to {percussive_output}")

# Visualize Spectrogram
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max), y_axis='log', x_axis='time')
plt.title('Full spectrogram')

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(harmonic)), ref=np.max), y_axis='log', x_axis='time')
plt.title('Harmonic spectrogram')

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(percussive)), ref=np.max), y_axis='log', x_axis='time')
plt.title('Percussive spectrogram')
plt.tight_layout()
plt.show()