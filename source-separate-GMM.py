import os
import numpy as np
import librosa
import soundfile as sf

from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

directory = "SignalSourceDataset"
sample_length = 5  # length of audio samples in seconds
n_components = 5   # number of components for GMM

def sample_audio(audio, sr, length):
    num_samples = sr * length
    start = np.random.randint(0, len(audio) - num_samples)
    return audio[start:start + num_samples]

def extract_features(audio, sr):
    mfcc = mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=4)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features = np.concatenate((mfcc, spectral_contrast, chroma), axis=0)
    return features.flatten()

features = []
for audio_file in tqdm(os.listdir(directory)):

    output_dir = "input-GMM"

    os.makedirs(output_dir, exist_ok=True)
    
    audio, sr = librosa.load(os.path.join(directory, audio_file))
    for i in range(10):
        sampled_audio = sample_audio(audio, sr, sample_length)
        output_file_path = os.path.join(output_dir, f"{audio_file}_sampled_audio_{i}.wav")
        sf.write(output_file_path, sampled_audio, sr)
        features.append(extract_features(sampled_audio, sr))

input_gmm_directory = "input-GMM"

features = []
for audio_file in tqdm(os.listdir(input_gmm_directory)):
    audio, sr = librosa.load(os.path.join(input_gmm_directory, audio_file))
    features.append(extract_features(audio, sr))

features = np.array(features)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

gmm = GaussianMixture(n_components=n_components)
gmm.fit(features_scaled)

def separate_sources(audio_file, gmm, scaler, directory, n_components):
    audio, sr = librosa.load(os.path.join(directory, audio_file))
    chunk_size = sample_length * sr

    separated_sources = [[] for _ in range(n_components)]

    for start in range(0, len(audio), chunk_size):
        end = start + chunk_size
        audio_chunk = audio[start:end]

        features = extract_features(audio_chunk, sr)
        if(len(features) < 4968): continue
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_scaled = scaler.transform(features_scaled)

        labels = gmm.predict(features_scaled)

        for i in range(n_components):
            mask = labels == i
            source_chunk = audio_chunk * mask
            separated_sources[i].append(source_chunk)

    separated_sources = [np.concatenate(sources, axis=0) for sources in separated_sources]

    return separated_sources

for audio_file in tqdm(os.listdir(directory)):
    output_dir = "output-GMM"
    base_filename = os.path.splitext(os.path.basename(audio_file))[0]
    file_output_dir = os.path.join(output_dir, base_filename)

    os.makedirs(file_output_dir, exist_ok=True)
    separated_sources = separate_sources(audio_file, gmm, scaler, directory, n_components)

    for i, source in enumerate(separated_sources):
        output_file_path = os.path.join(file_output_dir, f"component_{i}.wav")
        sf.write(output_file_path, source, sr)