"""Streaming phase-vocoder pitch shift with bounded buffers and no model files.

Stretch by the pitch ratio, then resample by the same ratio to keep duration.
Analysis uses a 2048-sample Hann window and a 256-sample hop. The small initial
FIFO absorbs the resampler's lookahead and gives a constant-size live output.
"""

import numpy as np


class StreamingPitchShifter:
    WINDOW = 2048
    HOP = 256

    def __init__(self, semitones: float):
        if not -12 <= semitones <= 12:
            raise ValueError("Pitch must be between -12 and +12 semitones")
        self.ratio = 2 ** (semitones / 12)
        self._window = np.hanning(self.WINDOW)
        self._omega = 2 * np.pi * np.arange(self.WINDOW // 2 + 1) / self.WINDOW
        self._input = np.zeros(self.WINDOW - self.HOP, dtype=np.float32)
        self._ola = np.zeros(self.WINDOW)
        self._weights = np.zeros(self.WINDOW)
        self._previous_phase = None
        self._phase = None
        self._synthesis_cursor = 0
        self._previous_hop = 0
        self._frames = 0
        self._resample = np.empty(0)
        self._resample_position = 0.0
        self._ready = np.zeros(self.WINDOW, dtype=np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        self._input = np.concatenate((self._input, samples))
        generated = []
        while len(self._input) >= self.WINDOW:
            spectrum = np.fft.rfft(self._input[:self.WINDOW] * self._window)
            angle = np.angle(spectrum)
            if self._previous_phase is None:
                self._phase = angle.copy()
            else:
                delta = angle - self._previous_phase - self._omega * self.HOP
                delta -= 2 * np.pi * np.round(delta / (2 * np.pi))
                frequency = self._omega + delta / self.HOP
                self._phase += frequency * self._previous_hop
                self._phase %= 2 * np.pi
            self._previous_phase = angle
            self._frames += 1
            next_cursor = round(self._frames * self.HOP * self.ratio)
            hop = next_cursor - self._synthesis_cursor
            self._synthesis_cursor = next_cursor
            self._previous_hop = hop
            frame = np.fft.irfft(np.abs(spectrum) * np.exp(1j * self._phase), n=self.WINDOW)
            self._ola += frame * self._window
            self._weights += self._window ** 2
            normalized = self._ola[:hop] / np.maximum(self._weights[:hop], 1e-8)
            generated.append(normalized.copy())
            self._ola[:-hop] = self._ola[hop:]
            self._ola[-hop:] = 0
            self._weights[:-hop] = self._weights[hop:]
            self._weights[-hop:] = 0
            self._input = self._input[self.HOP:]
        if generated:
            self._resample = np.concatenate((self._resample, *generated))
        available = len(self._resample) - 1 - self._resample_position
        if available >= 0:
            count = int(np.floor(available / self.ratio)) + 1
            positions = self._resample_position + np.arange(count) * self.ratio
            result = np.interp(positions, np.arange(len(self._resample)), self._resample)
            self._resample_position += count * self.ratio
            consumed = min(int(self._resample_position), len(self._resample))
            self._resample = self._resample[consumed:]
            self._resample_position -= consumed
            self._ready = np.concatenate((self._ready, result.astype(np.float32)))
        count = len(samples)
        if len(self._ready) < count:
            raise RuntimeError("Pitch FIFO underrun")
        result = self._ready[:count].copy()
        self._ready = self._ready[count:]
        return result
