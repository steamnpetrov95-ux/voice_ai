"""Stateful DSP shared by the live engine and offline rendering."""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import (
    Chorus, Compressor, Delay, Distortion, Gain, HighpassFilter, HighShelfFilter,
    LowpassFilter, LowShelfFilter, PeakFilter, Pedalboard, Reverb,
)

from .pitch import StreamingPitchShifter
from .presets import VoiceSettings

BLOCK_SIZE = 1024


class VoiceProcessor:
    """Use from one audio worker only; retain plugin state between blocks."""

    def __init__(self, sample_rate: float, settings: VoiceSettings):
        self.sample_rate = sample_rate
        self.settings = settings
        self.phase = 0
        self.pitch_shifter = StreamingPitchShifter(settings.pitch) if settings.pitch else None
        plugins = []
        if settings.timbre:
            plugins.extend([
                LowShelfFilter(cutoff_frequency_hz=350, gain_db=settings.timbre * 6),
                HighShelfFilter(cutoff_frequency_hz=2200, gain_db=-settings.timbre * 4),
            ])
        if settings.clarity:
            # Presence EQ highlights consonants without speech recognition or translation.
            plugins.extend([
                HighpassFilter(cutoff_frequency_hz=75),
                PeakFilter(cutoff_frequency_hz=2600, gain_db=settings.clarity, q=0.8),
                Compressor(threshold_db=-16, ratio=2, attack_ms=8, release_ms=100),
            ])
        if settings.radio:
            plugins.extend([HighpassFilter(cutoff_frequency_hz=450), LowpassFilter(cutoff_frequency_hz=3000)])
        if settings.chorus:
            plugins.append(Chorus(rate_hz=1.2, depth=0.3, mix=0.25))
        if settings.drive:
            plugins.append(Distortion(drive_db=settings.drive))
        if settings.echo:
            plugins.append(Delay(delay_seconds=0.22, feedback=0.25, mix=settings.echo))
        if settings.room:
            plugins.append(Reverb(room_size=0.7, wet_level=settings.room * 0.65, dry_level=1 - settings.room * 0.3))
        if settings.gain:
            plugins.append(Gain(gain_db=settings.gain))
        self.board = Pedalboard(plugins)

    def process(self, samples: np.ndarray) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        if not self.settings.enabled:
            return np.clip(np.nan_to_num(audio), -1, 1)
        if self.pitch_shifter is not None:
            audio = self.pitch_shifter.process(audio)
        if self.settings.robot:
            t = (np.arange(audio.size) + self.phase) / self.sample_rate
            modulation = np.sin(2 * math.pi * 75 * t)
            mix = self.settings.robot
            audio *= (1 - mix) + mix * modulation
        self.phase = (self.phase + audio.size) % int(self.sample_rate)
        if len(self.board):
            audio = self.board(audio, self.sample_rate, reset=False)
        return np.clip(np.nan_to_num(audio), -1, 1).astype(np.float32, copy=False)


def render_file(source: str | Path, destination: str | Path, settings: VoiceSettings, voice=None, provider="auto") -> dict:
    """Render WAV/FLAC/etc. as mono WAV. Existing files are never overwritten."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Входной и выходной файлы должны отличаться.")
    # Exclusive creation also protects against a destination appearing after selection.
    with sf.SoundFile(source) as incoming:
        neural = None
        block_size = BLOCK_SIZE
        if voice is not None and settings.enabled:
            from .neural import AI_BLOCK_SIZE, NeuralConverter
            neural = NeuralConverter(voice, incoming.samplerate, provider)
            block_size = AI_BLOCK_SIZE
        processor = VoiceProcessor(incoming.samplerate, replace(settings, pitch=0) if neural else settings)
        with destination.open("xb") as handle:
            with sf.SoundFile(handle, mode="w", samplerate=incoming.samplerate,
                              channels=1, format="WAV", subtype="PCM_16") as outgoing:
                frames = 0
                for block in incoming.blocks(blocksize=block_size, dtype="float32", always_2d=True):
                    mono = block.mean(axis=1)
                    if neural:
                        converted = neural.process(np.pad(mono, (0, block_size - len(mono))), settings.pitch)
                        mono = converted[:len(mono)]
                    result = processor.process(mono)
                    outgoing.write(result)
                    frames += len(result)
                # Keep the effect's buffered audio and reverb tail for processed presets.
                tail_frames = incoming.samplerate if settings.enabled and (len(processor.board) or settings.pitch) and not neural else 0
                for offset in range(0, tail_frames, BLOCK_SIZE):
                    outgoing.write(processor.process(np.zeros(min(BLOCK_SIZE, tail_frames - offset), dtype=np.float32)))
    return {"frames": frames + tail_frames, "sample_rate": incoming.samplerate, "channels": 1}
