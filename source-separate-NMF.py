import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import soundfile as sf
from tqdm import tqdm

def sinebell(length):
    return np.sin(np.pi * np.arange(length) / length)

def nmf_kl_admm(V, W_init, H_init, rho=1, max_iter=1000):
    m, n = V.shape
    _, k = W_init.shape

    W = W_init.copy()
    H = H_init.copy()
    X = W @ H
    Wplus = W.copy()
    Hplus = H.copy()
    alphaX = np.zeros_like(X)
    alphaW = np.zeros_like(W)
    alphaH = np.zeros_like(H)

    for iteration in range(max_iter):
        # Update H
        H = np.linalg.solve(W.T @ W + np.eye(k), W.T @ X + Hplus + 1 / rho * (W.T @ alphaX - alphaH))

        # Update W
        P = H @ H.T + np.eye(k)
        Q = H @ (X + 1 / rho * alphaX).T + Wplus.T - 1 / rho * alphaW.T
        W = np.linalg.solve(P, Q).T

        # Update X
        X_ap = W @ H
        b = rho * X_ap - alphaX - 1
        X = (b + np.sqrt(b ** 2 + 4 * rho * V)) / (2 * rho)

        # Update Hplus and Wplus
        Hplus = np.maximum(H + 1 / rho * alphaH, 0)
        Wplus = np.maximum(W + 1 / rho * alphaW, 0)

        # Update dual variables
        alphaX = alphaX + rho * (X - X_ap)
        alphaH = alphaH + rho * (H - Hplus)
        alphaW = alphaW + rho * (W - Wplus)

    return W, H

def save_components(components, sr, base_filename, output_dir, hop_length):
    for i, component in enumerate(components):
        # Inverse STFT
        reconstructed_audio = librosa.istft(component, hop_length=hop_length)
        # Save audio file
        output_file_path = os.path.join(output_dir, f"{base_filename}_component_{i}.wav")
        sf.write(output_file_path, reconstructed_audio, sr)

sr_target = 11025  # Target sampling rate (downsampled)
window_length_ms = 23  # Window length in milliseconds
n_components = 2  # Number of NMF components
hop_length_percent = 0.5  # 50% overlap

directory = "SignalSourceDataset"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

for audio_file in tqdm(os.listdir(directory)):
    file_path = os.path.join(directory, audio_file)
    audio, sr = librosa.load(file_path, sr=sr_target)  # Downsample to 11025 Hz

    window_length = int(window_length_ms / 1000 * sr_target)
    window = sinebell(window_length)
    hop_length = int(window_length * hop_length_percent)

    # STFT
    S = np.abs(librosa.stft(audio, n_fft=window_length, hop_length=hop_length, window=window))

    # Initialize W and H
    W_init = np.random.rand(S.shape[0], n_components)
    H_init = np.random.rand(n_components, S.shape[1])

    # Apply NMF
    W, H = nmf_kl_admm(S, W_init, H_init, rho=1, max_iter=50)

    # Reconstruct audio from components
    components = [np.dot(W[:, [i]], H[[i], :]) for i in range(n_components)]

    base_filename = os.path.splitext(audio_file)[0]
    save_components(components, sr_target, base_filename, output_dir, hop_length)

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(components[0]), ref=np.max), y_axis='log', x_axis='time')
plt.title('Component 1 spectrogram')

plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(components[1]), ref=np.max), y_axis='log', x_axis='time')
plt.title('Component 2 spectrogram')
plt.tight_layout()
plt.show()