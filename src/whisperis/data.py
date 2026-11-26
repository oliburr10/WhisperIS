
# Audio loading and noise Augementation for WhisperIS

# Module for Data DVC
from __future__ import  annotations
from pathlib import Path

import numpy as np

SAMPLE_RATE = 160000 # Whisper expects 16 khz audio
SNR_LEVELS_DB = (20, 10, 5, 0) # Shared by data pipeline, training and eval

def load_audio(path: str | Path, sr: int = SAMPLE_RATE):
    # load any audio files as mono float32 at sr Hz
    import librosa # imported here so the rest works without it

    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)

def power(x): # HOW LOUD is the signal
    # Average power of a signal
    return float(np.mean(np.square(x, dtype=np.float64)))

def snr_db(signal, noise): # mesasuring the SNR
    return 10 * np.log10(power(signal) / power(noise))

def fit_length(noise, length, rng):
    # Return a noise clip exactly " length" samples long

    if len(noise) == 0:
        raise ValueError("Noise is empty")
    if len(noise) < length:
        repeats = int(np.ceil(length/len(noise)))
        noise = np.tile(noise, repeats) # repeats the array end to end
    start = rng.integers(0, len(noise) - length + 1)
    return noise[start : start + length]


def add_noise(
        speech,
        noise,
        snr,
        rng: np.random.Generator | None = None,
        
):
    # Mix noise into speech so the results has the given SNR (in dB)
    " Both inputs must be Mono 1D - pass a seeded rng to get thr same mix every time"

    if speech.ndim != 1 or noise.ndim != 1:
        raise ValueError("Speech and noise must be mono (1D arrays)")
    
    rng = rng if rng is not None else np.random.default_rng()
    noise = fit_length(noise, len(speech), rng)

    speech_power = power(speech)
    noise_power = power(noise)

    if speech_power == 0 or noise_power == 0:
        return speech.astype(np.float32) # Silence: nothing sensible to mix
    
    target_noise_power = speech_power / (10 ** (snr / 10))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)

    mixed = speech  + scaled_noise

    # keep samples inside [-1, 1]
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)
    














