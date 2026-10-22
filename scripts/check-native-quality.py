"""Compare the native wrapper's block sizes and explicit NORMAL mode in fresh processes.

No production SDK settings are changed. PRO mode, account and calibration calls
are not used. Acoustic comparisons are not a naturalness score.
"""

import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.signal import correlate, correlation_lags
import soundfile as sf

from voicelab.native import NativeConverter, installed_native_voices
from voicelab.neural import resample


def compare(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    correlations = correlate(a, b, mode="full", method="fft")
    lags = correlation_lags(len(a), len(b))
    valid = np.abs(lags) <= 4800
    lag = int(lags[valid][np.argmax(correlations[valid])])
    if lag >= 0:
        a, b = a[lag:], b[:len(a) - lag]
    else:
        a, b = a[:len(b) + lag], b[-lag:]
    size = min(len(a), len(b))
    a, b = a[:size], b[:size]
    error = np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(a * a)), 1e-9)
    cosine = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
    return {"lag_samples": lag, "relative_rms_error": round(float(error), 6),
            "waveform_correlation": round(float(cosine), 6)}


def read_configuration(runtime):
    class Configuration(ctypes.Structure):
        _fields_ = [("mode", ctypes.c_int), ("intonation", ctypes.c_float), ("pitch", ctypes.c_float)]
    # DubbingAI.exe RVA 0x1249af invokes slot 0x68, then reads offsets 4 and 8.
    pointer = runtime._method(0x68, ctypes.c_void_p)(runtime.instance)
    if not pointer:
        return None
    value = ctypes.cast(pointer, ctypes.POINTER(Configuration)).contents
    return {"mode": value.mode, "intonation": value.intonation, "pitch": value.pitch}


def run_case(args, source, voices):
    results = []
    count = 7680 if args.case == "default160" else 3840
    incoming = np.pad(source, (0, 30720))
    for voice in voices:
        converter = NativeConverter(voice, 48000)
        try:
            before = read_configuration(converter.runtime)
            if args.case == "normal80":
                # RVA 0x11f739: self, int mode, float intonation, float pitch.
                # The public SDK header defines NORMAL_MODE=0. Do not invoke PRO.
                converter.runtime._method(0x30, None, ctypes.c_int, ctypes.c_float, ctypes.c_float)(
                    converter.runtime.instance, 0, 0.0, 0.0)
            after = read_configuration(converter.runtime)
            blocks = []
            for start in range(0, len(incoming), count):
                block = incoming[start:start + count]
                blocks.append(converter.process(np.pad(block, (0, count - len(block))), 0))
            output = np.concatenate(blocks)[:len(incoming)]
            assert np.isfinite(output).all() and np.max(np.abs(output)) > 1e-4
            path = args.output / f"{voice.speaker_id}-{args.case}.wav"
            sf.write(path, output, 48000, subtype="PCM_16")
            results.append({"speaker_id": voice.speaker_id, "before": before, "after": after})
        finally:
            converter.close()
    (args.output / f"{args.case}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist/native-quality"))
    parser.add_argument("--case", choices=("default80", "default160", "normal80"))
    args = parser.parse_args()
    os.environ["VOICELAB_NATIVE_DIR"] = str(args.models.resolve())
    voices = [v for v in installed_native_voices() if v.speaker_id in (502, 713, 396)]
    if len(voices) != 3:
        raise ValueError("Expected comparison profiles are missing")
    source, rate = sf.read(args.speech, dtype="float32", always_2d=True)
    source = resample(source.mean(axis=1), rate, 48000)
    source *= 0.35 / max(float(np.max(np.abs(source))), 1e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case:
        run_case(args, source, voices)
        return
    sf.write(args.output / "original-human.wav", source, 48000, subtype="PCM_16")
    cases = ("default80", "default160", "normal80")
    for case in cases:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--models", str(args.models),
                        "--speech", str(args.speech), "--output", str(args.output), "--case", case],
                       check=True, timeout=180)
    report = {"note": "Waveform comparisons, not a naturalness score.", "profiles": []}
    for voice in voices:
        arrays = {case: sf.read(args.output / f"{voice.speaker_id}-{case}.wav")[0] for case in cases}
        row = {"speaker_id": voice.speaker_id,
               "80_vs_160": compare(arrays["default80"], arrays["default160"]),
               "default_vs_normal": compare(arrays["default80"], arrays["normal80"])}
        report["profiles"].append(row)
    report["configuration"] = {case: json.loads((args.output / f"{case}.json").read_text()) for case in cases}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("auditions", Path(__file__).with_name("make-dnd-auditions.py"))
    auditions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditions)
    records = [{"id": voice.id, "name": voice.name, "speaker_id": voice.speaker_id,
                "audio": auditions.wav_url(sf.read(args.output / f"{voice.speaker_id}-default80.wav", dtype="float32")[0])}
               for voice in voices]
    auditions.write_page(args.output.parent / "DnD-human-voice-check.html", source, records,
                         "Запись живой русской речи Дмитрия из Nabu Casa / OHF Voice Datasets.",
                         "Исходник — живая запись, CC0. AI работает без дополнительных эффектов; "
                         "громкость выровнена. Это проверка звучания, а не готовые голоса персонажей.")
    print("::notice title=Native quality comparison::" + json.dumps(report))


if __name__ == "__main__":
    main()
