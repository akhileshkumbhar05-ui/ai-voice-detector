
import io
import os
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000

def load_audio(file_or_bytes, target_sr: int = TARGET_SR):
    """Load audio from path or bytes, mono, target SR."""
    if isinstance(file_or_bytes, (str, os.PathLike)):
        y, sr = librosa.load(file_or_bytes, sr=target_sr, mono=True)
    else:
        # assume bytes-like
        y, sr0 = sf.read(io.BytesIO(file_or_bytes))
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr0 != target_sr:
            y = librosa.resample(y, orig_sr=sr0, target_sr=target_sr)
        sr = target_sr
    # normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32), sr

def pad_or_trim(y: np.ndarray, duration_s: float = 3.0, sr: int = TARGET_SR):
    """Pad with zeros or trim to a fixed duration (seconds)."""
    n = int(duration_s * sr)
    if len(y) < n:
        pad = n - len(y)
        y = np.pad(y, (0, pad))
    elif len(y) > n:
        y = y[:n]
    return y

def logmel(y: np.ndarray, sr: int = TARGET_SR, n_mels: int = 64):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels, fmin=20, fmax=sr//2)
    S = np.log1p(S).astype(np.float32)
    return S  # (n_mels, T)

def heuristic_features(y: np.ndarray, sr: int = TARGET_SR):
    """Lightweight features for a quick heuristic classifier."""
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=256).mean()
    cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    flat = librosa.feature.spectral_flatness(y=y).mean()
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    feats = np.array([zcr, cent, flat, roll, rms, *mfcc], dtype=np.float32)
    return feats
