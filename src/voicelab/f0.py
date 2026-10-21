"""Small FFT YIN pitch tracker for the low-memory RVC profile (16 kHz input)."""

import numpy as np
from scipy.fft import irfft, rfft


def fast_pitch(audio):
    """Return one F0 value per 10 ms; silence and unvoiced frames remain zero."""
    audio = np.asarray(audio, np.float32).reshape(-1)
    count = len(audio) // 160
    if not count:
        return np.zeros(0, np.float32)
    width, minimum, maximum = 1024, 16, 320
    padded = np.pad(audio, (width // 2, width // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, width)[::160][:count].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    energy = np.concatenate((np.zeros((count, 1), np.float32),
                             np.cumsum(frames * frames, axis=1)), axis=1)
    spectrum = rfft(frames, n=2 * width, axis=1, workers=1)
    correlation = irfft(spectrum * spectrum.conj(), axis=1, workers=1)[:, :maximum + 1]
    lags = np.arange(1, maximum + 1)
    difference = np.maximum(energy[:, width - lags] + energy[:, -1:] - energy[:, lags]
                            - 2 * correlation[:, 1:], 0)
    normalized = difference * lags / np.maximum(np.cumsum(difference, axis=1), 1e-12)
    curve = np.concatenate((np.ones((count, 1)), normalized), axis=1)
    frequency = np.zeros(count, np.float32)
    for index in range(count):
        if energy[index, -1] / width < 1e-8:
            continue
        candidates = np.flatnonzero(curve[index, minimum:maximum + 1] < 0.1)
        if not candidates.size:
            continue
        lag = int(candidates[0]) + minimum
        while lag < maximum and curve[index, lag + 1] < curve[index, lag]:
            lag += 1
        refined = float(lag)
        if 1 <= lag < maximum:
            left, center, right = curve[index, lag - 1:lag + 2]
            denominator = left - 2 * center + right
            if abs(denominator) > 1e-12:
                refined += float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
        frequency[index] = 16000 / refined
    return frequency
