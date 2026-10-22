"""Validate the personal native catalogue, real speech, and optional installed EXE."""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import urllib.request

import numpy as np
import soundfile as sf

from voicelab.native import NativeConverter, installed_native_voices
from voicelab.neural import resample


def threaded_route(voices, samples):
    """Real SDK and audio callbacks, using a host stub instead of sound hardware."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from unittest.mock import patch
    from PySide6.QtWidgets import QApplication
    from voicelab import audio
    from voicelab.presets import VoiceSettings

    class Stream:
        def __init__(self, **kwargs):
            self.kwargs, self.active = kwargs, False

        def start(self):
            self.active = True

        def close(self):
            self.active = False

    application = QApplication.instance() or QApplication([])
    for sample_rate, milliseconds in ((48000, 80), (48000, 40), (44100, 80), (44100, 40)):
        def input_settings(**kwargs):
            if kwargs["samplerate"] != sample_rate:
                raise ValueError("Use the requested test sample rate")

        with (patch.object(audio.sd, "InputStream", Stream), patch.object(audio.sd, "OutputStream", Stream),
              patch.object(audio.sd, "query_devices", lambda *args: {"max_output_channels": 2}),
              patch.object(audio.sd, "check_input_settings", input_settings),
              patch.object(audio.sd, "check_output_settings", lambda **kwargs: None)):
            engine = audio.AudioEngine()
            engine.configure_neural(voices[0], buffer_ms=milliseconds)
            engine.configure(VoiceSettings(clarity=0))
            engine.start(0, 1)
            peak = 0
            try:
                deadline = time.monotonic() + 15
                while engine.loading and time.monotonic() < deadline:
                    application.processEvents()
                    time.sleep(0.01)
                assert not engine.loading and not engine.error, engine.error
                engine.start_monitoring(2)
                assert engine._input.kwargs["blocksize"] == engine._output.kwargs["blocksize"] == 0
                incoming = resample(samples, 48000, sample_rate)
                count = engine.block_size
                for index, offset in enumerate(range(0, len(incoming), count)):
                    if index in (5, 10, 15):
                        engine.switch_native_voice(voices[index * 60])
                    block = np.pad(incoming[offset:offset + count], (0, max(0, offset + count - len(incoming))))
                    for position in range(0, count, 229):
                        part = block[position:position + 229, None]
                        engine._capture(part, len(part), None, False)
                    deadline = time.monotonic() + 5
                    while (engine._out_queue.empty() or engine._monitor_queue.empty()) and time.monotonic() < deadline:
                        application.processEvents()
                        time.sleep(0.001)
                    assert not engine.error, engine.error
                    for position in range(0, count, 257):
                        frames = min(257, count - position)
                        primary, monitor = np.empty((frames, 2), np.float32), np.empty((frames, 2), np.float32)
                        engine._play(primary, frames, None, False)
                        engine._monitor.kwargs["callback"](monitor, frames, None, False)
                        np.testing.assert_array_equal(primary, monitor)
                        assert np.isfinite(primary).all()
                        peak = max(peak, float(np.max(np.abs(primary))))
                assert peak > 1e-4 and not engine.error, engine.error
                assert engine.neural_voice.id == voices[900].id and engine.dropouts == 0
            finally:
                engine.stop()
                assert not engine.busy and not engine.error, engine.error
    print("::notice title=Native audio worker::Real SDK, Qt, 40/80 ms, 48/44.1 kHz, variable callbacks and both outputs passed.", flush=True)


def memory_mb():
    from ctypes import wintypes as w

    class Counters(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("faults", w.DWORD)] + [(key, ctypes.c_size_t) for key in
                    ("peak", "working", "paged_peak", "paged", "nonpaged_peak", "nonpaged", "pagefile", "pagefile_peak")]

    kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel.GetCurrentProcess.restype = w.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(Counters), w.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise RuntimeError("Could not measure native process memory")
    return {"working_mb": round(counters.working / 1048576, 1), "peak_mb": round(counters.peak / 1048576, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--speech", type=Path)
    parser.add_argument("--app", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dist/native-benchmark.json"))
    parser.add_argument("--skip-catalogue", action="store_true")
    parser.add_argument("--stream-seconds", type=int, default=0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["VOICELAB_NATIVE_DIR"] = str(args.models.resolve())
    voices = installed_native_voices()
    assert len(voices) == 1241, f"Expected 1241 profiles, got {len(voices)}"
    with tempfile.TemporaryDirectory(prefix="npetrov-native-smoke-") as directory:
        root = Path(directory)
        speech = args.speech
        if speech is None:
            speech = root / "speech.wav"
            url = "https://raw.githubusercontent.com/dev6699/rvc-onnx/9d20b0ff54df72e83c2826e76c2e1eff5f1ab824/assets/input_1.wav"
            with urllib.request.urlopen(url, timeout=30) as source:
                speech.write_bytes(source.read(4 * 1024 * 1024))
        samples, rate = sf.read(speech, dtype="float32", always_2d=True)
        samples = resample(samples.mean(axis=1), rate, 48000)
        threaded_route(voices, samples)
        if not args.skip_catalogue:
            converter = NativeConverter(voices[0], 48000)
            try:
                for voice in voices:
                    code = converter.runtime.select(voice.speaker_id)
                    assert code == 10000, f"Profile {voice.speaker_id} failed with {code}"
                print("Validated selection of all 1241 native profiles", flush=True)
            finally:
                converter.close()
        timings, details = [], []
        for sample_rate, count in ((48000, 1920), (48000, 3840), (48000, 7680), (48000, 15360), (48000, 23040), (48000, 30720), (44100, 1792), (44100, 3584), (44100, 7040)):
            print(f"Checking {sample_rate} Hz, {count} frames", flush=True)
            converter = NativeConverter(voices[0], sample_rate)
            incoming = resample(samples, 48000, sample_rate)
            outputs, durations = [], []
            try:
                for offset in range(0, len(incoming), count):
                    block = incoming[offset:offset + count]
                    block = np.pad(block, (0, count - len(block)))
                    started = time.perf_counter()
                    result = converter.process(block)
                    durations.append((time.perf_counter() - started) * 1000)
                    assert result.shape == block.shape and np.isfinite(result).all()
                    outputs.append(result)
                converted = np.concatenate(outputs)[:len(incoming)]
                assert np.max(np.abs(converted)) > 1e-4
                assert np.mean(np.abs(converted - incoming)) > 1e-4
                details.append({"rate": sample_rate, "block_ms": round(count * 1000 / sample_rate),
                                "median_ms": round(statistics.median(durations[2:]), 2)})
                if sample_rate == 48000 and count == 7680:
                    timings = durations[2:]
                    sf.write(args.output.with_name("native-reference.wav"), converted, sample_rate)
            finally:
                converter.close()
        stream = None
        if args.stream_seconds:
            converter = NativeConverter(voices[0], 48000)
            stream_count, stream_duration = 3840, 0.080
            started = time.perf_counter()
            before = memory_mb()
            durations = []
            try:
                for index in range(math.ceil(args.stream_seconds / stream_duration)):
                    if index and index % 100 == 0:
                        converter.select_voice(voices[(index * 3) % len(voices)])
                    offset = (index * stream_count) % len(samples)
                    block = np.pad(samples[offset:offset + stream_count], (0, max(0, offset + stream_count - len(samples))))
                    block_start = time.perf_counter()
                    output = converter.process(block)
                    durations.append((time.perf_counter() - block_start) * 1000)
                    assert len(output) == stream_count and np.isfinite(output).all()
                    deadline = started + (index + 1) * stream_duration
                    time.sleep(max(0, deadline - time.perf_counter()))
                stream = {"seconds": args.stream_seconds, "block_ms": 80, "blocks": len(durations),
                          "p95_ms": round(float(np.percentile(durations, 95)), 2),
                          "start_working_mb": before["working_mb"], "end_working_mb": memory_mb()["working_mb"]}
            finally:
                converter.close()
        if args.app:
            # Ensure a user-side source override cannot hide a packaged path bug.
            environment = dict(os.environ)
            environment.pop("VOICELAB_NATIVE_DIR", None)
            source = root / "input.wav"
            sf.write(source, samples, 48000)
            for index in (0, 200, 600, 1000, 1240):
                voice = voices[index]
                output = root / f"installed-{index}.wav"
                subprocess.run([str(args.app), "--render", str(source), str(output), "--ai-voice", voice.id],
                               env=environment, check=True, timeout=60)
                data, rate = sf.read(output)
                assert rate == 48000 and len(data) == len(samples) and np.max(np.abs(data)) > 1e-4
        report = {"note": "Native SDK Windows CI CPU measurements, not the user's i7-2600.",
                  "profiles": len(voices), "named_profiles": sum(not v.name.startswith("AI-голос ") for v in voices), "provider": "Native CPU", "block_ms": 160,
                  "median_ms": round(statistics.median(timings), 2), "blocks": details,
                  "live_stream": stream, **memory_mb()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("::notice title=Native voice validation::" + json.dumps(report), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("::error title=Native voice validation failed::" + repr(exc), flush=True)
        import traceback
        traceback.print_exc()
        raise
