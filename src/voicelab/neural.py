"""Local RVC inference with ONNX Runtime; no cloud inference or pickle loading.

Supports VC Client ONNX exports (feats/p_len/pitch/pitchf/sid) and
RVC-ONNX exports (phone/phone_lengths/pitch/pitchf/ds/rnd).
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


AI_BLOCK_SIZE = 15360


def resample(audio, source_rate, target_rate):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return audio.copy()
    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)


@dataclass(frozen=True)
class NeuralVoice:
    id: str
    name: str
    model: Path
    encoder: Path
    pitch: Path
    sample_rate: int
    channels: int
    credit: str = ""
    terms_url: str = ""
    decoder: Path | None = None


def _session(path, provider):
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider == "auto":
        preferred = "DmlExecutionProvider" if "DmlExecutionProvider" in available else "CPUExecutionProvider"
    elif provider == "cpu":
        preferred = "CPUExecutionProvider"
    elif provider == "dml":
        preferred = "DmlExecutionProvider"
    else:
        raise ValueError("Неизвестный режим AI-ускорения.")
    if preferred not in available:
        raise RuntimeError("Ускорение видеокартой недоступно. Выберите CPU или обновите драйвер видеокарты.")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.enable_mem_pattern = preferred != "DmlExecutionProvider"
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = [preferred]
    if preferred != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(path), sess_options=options, providers=providers)


def _dtype(info):
    return {"tensor(float)": np.float32, "tensor(float16)": np.float16,
            "tensor(int64)": np.int64, "tensor(int32)": np.int32,
            "tensor(bool)": np.bool_}.get(info.type, np.float32)


def voice_metadata(session):
    metadata = session.get_modelmeta().custom_metadata_map
    values = json.loads(metadata.get("metadata", "{}"))
    if not isinstance(values, dict):
        raise ValueError("Некорректные метаданные RVC.")
    sample_rate = values.get("samplingRate")
    if not sample_rate and "config" in metadata:
        sample_rate = json.loads(metadata["config"])[-1]
    feature = next((item for item in session.get_inputs() if item.name in ("feats", "phone", "hubert")), None)
    if feature is None or len(feature.shape) != 3 or feature.shape[-1] not in (256, 768):
        raise ValueError("Этот файл не является поддерживаемой моделью RVC ONNX (256/768 признаков).")
    names = {item.name for item in session.get_inputs()}
    supported = {"feats", "phone", "hubert", "p_len", "phone_lengths", "hubert_length", "lengths",
                 "pitch", "pitchf", "nsff0", "sid", "ds", "rnd", "noise"}
    if names - supported or not names.intersection({"p_len", "phone_lengths", "hubert_length", "lengths"}):
        raise ValueError("Неподдерживаемый формат входов RVC ONNX.")
    if sample_rate not in (32000, 40000, 48000):
        raise ValueError("В модели нет частоты 32/40/48 кГц. Нужен ONNX с метаданными RVC.")
    if values.get("embedder") not in (None, "hubert_base", "contentvec"):
        raise ValueError("Для этой модели нужен другой кодировщик речи; поддерживаются ContentVec/HuBERT.")
    return int(sample_rate), int(feature.shape[-1])


class NeuralConverter:
    """One loaded voice, a 640 ms history and SOLA stitching; owned by the audio worker."""

    def __init__(self, voice: NeuralVoice, sample_rate: int, provider="auto"):
        self.voice = voice
        self.sample_rate = sample_rate
        self.synth = _session(voice.model, provider)
        self.decoder = _session(voice.decoder, provider) if voice.decoder else None
        rate, channels = voice_metadata(self.synth)
        if (rate, channels) != (voice.sample_rate, voice.channels):
            raise ValueError("Модель голоса изменилась. Импортируйте её заново.")
        self.encoder = _session(voice.encoder, provider)
        self.pitch = _session(voice.pitch, provider)
        encoder_shape = self.encoder.get_inputs()[0].shape
        if len(encoder_shape) != 3:
            raise ValueError("Нужен ContentVec ONNX формата [1, 1, samples].")
        if {item.name for item in self.pitch.get_inputs()} != {"waveform", "threshold"}:
            raise ValueError("Нужен RMVPE ONNX с входами waveform и threshold.")
        self.provider = self.synth.get_providers()[0]
        self.history = np.zeros(10240, np.float32)
        self.overlap = round(sample_rate * 0.04)
        self.search = round(sample_rate * 0.012)
        self.tail = np.zeros(self.overlap, np.float32)
        self.rng = np.random.default_rng()

    def reset(self):
        self.history.fill(0)
        self.tail.fill(0)

    def _convert(self, audio, pitch_shift):
        encoder_input = self.encoder.get_inputs()[0]
        # A small symmetric pad compensates for ContentVec's convolution edges.
        padded = np.pad(audio, (160, 160), mode="reflect")
        encoded = self.encoder.run(None, {encoder_input.name: padded[None, None, :].astype(_dtype(encoder_input))})[0]
        if encoded.ndim != 3 or encoded.shape[1] != self.voice.channels:
            raise ValueError("Размерность кодировщика не совпадает с голосовой моделью.")
        features = np.repeat(encoded.transpose(0, 2, 1), 2, axis=1)
        pitch_inputs = {item.name: item for item in self.pitch.get_inputs()}
        frequency = self.pitch.run(None, {
            "waveform": padded[None, :].astype(_dtype(pitch_inputs["waveform"])),
            "threshold": np.array([0.3], dtype=_dtype(pitch_inputs["threshold"])),
        })[0].reshape(-1).astype(np.float32)
        length = min(features.shape[1], len(frequency), len(audio) // 160)
        if length < 1:
            raise RuntimeError("Модель не вернула кадров речи.")
        frequency = frequency[:length] * (2 ** (pitch_shift / 12))
        mel = 1127 * np.log1p(np.maximum(frequency, 0) / 700)
        lower, upper = 1127 * np.log1p(np.array([50, 1100]) / 700)
        bins = np.rint(np.clip((mel - lower) * 254 / (upper - lower) + 1, 1, 255)).astype(np.int64)
        values = {
            "feats": features[:, :length], "p_len": np.array([length]),
            "pitch": bins[None, :], "pitchf": frequency[None, :], "sid": np.array([0]),
        }
        aliases = {"phone": "feats", "hubert": "feats", "phone_lengths": "p_len",
                   "hubert_length": "p_len", "lengths": "p_len", "ds": "sid", "nsff0": "pitchf"}
        feeds = {}
        for item in self.synth.get_inputs():
            if item.name in ("rnd", "noise"):
                channels = item.shape[1] if isinstance(item.shape[1], int) else 192
                value = self.rng.standard_normal((1, channels, length)).astype(np.float32)
            else:
                value = values[aliases.get(item.name, item.name)]
            feeds[item.name] = value.astype(_dtype(item), copy=False)
        outputs = self.synth.run(None, feeds)
        if self.decoder is not None:
            latent = outputs[0] * outputs[1]
            # Decode only the new block plus overlap, keeping attention context
            # in the encoder. This reduces work on older 4 GB GPUs.
            frames = min(length, math.ceil(self._needed_samples * 100 / self.sample_rate) + 1)
            decoder_values = {"z": latent[:, :, -frames:], "nsff0": frequency[None, -frames:],
                              "sid": np.array([0], dtype=np.int64)}
            output = self.decoder.run(None, {item.name: decoder_values[item.name].astype(_dtype(item))
                                            for item in self.decoder.get_inputs()})[0].reshape(-1)
        else:
            output = outputs[0].reshape(-1)
        if not np.isfinite(output).all():
            raise RuntimeError("AI-модель вернула некорректный звук.")
        return resample(output, self.voice.sample_rate, self.sample_rate)

    def process(self, block, pitch_shift=0):
        block = np.asarray(block, np.float32).reshape(-1)
        incoming = resample(block, self.sample_rate, 16000)
        # Retain some context even across silence, but do not synthesize noise.
        audio = np.concatenate((self.history, incoming))
        self.history = audio[-len(self.history):].copy()
        if np.max(np.abs(audio), initial=0) < 1e-5:
            self.tail.fill(0)
            return np.zeros_like(block)
        required = len(block) + self.overlap + self.search
        self._needed_samples = required
        output = self._convert(audio, pitch_shift)
        if len(output) < required:
            raise RuntimeError("Выход AI-модели короче входа. Проверьте частоту модели.")
        segment = output[-required:]
        # SOLA aligns successive independently synthesized windows before crossfade.
        windows = np.lib.stride_tricks.sliding_window_view(segment[:self.overlap + self.search], self.overlap)
        numerator = windows @ self.tail
        energy = np.sqrt(np.sum(windows * windows, axis=1) + 1e-8)
        offset = int(np.argmax(numerator / energy)) if np.any(self.tail) else 0
        segment = segment[offset:offset + len(block) + self.overlap].copy()
        fade = np.linspace(0, 1, self.overlap, dtype=np.float32)
        segment[:self.overlap] = segment[:self.overlap] * fade + self.tail * (1 - fade)
        self.tail = segment[len(block):len(block) + self.overlap].copy()
        return np.clip(segment[:len(block)], -1, 1).astype(np.float32)
