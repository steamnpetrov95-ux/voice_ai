"""Compare full/light RVC on real speech in separate processes, without a GPU."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
import soundfile as sf

from voicelab.neural import NeuralConverter, NeuralVoice, resample


def memory_mb():
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"working_mb": round(counters.WorkingSetSize / 1048576, 1),
                "peak_mb": round(counters.PeakWorkingSetSize / 1048576, 1)}
    import resource
    return {"peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}


def measure(args):
    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parents[2]
    voice = NeuralVoice(value["id"], value["name"], root / value["model"], root / value["encoder"],
                        root / value["pitch"], value["sample_rate"], value["channels"],
                        decoder=root / value["decoder"] if value.get("decoder") else None)
    samples, rate = sf.read(args.speech, dtype="float32", always_2d=True)
    samples = resample(samples.mean(axis=1), rate, 48000)
    count = 15360
    pieces = [samples[offset:offset + count] for offset in range(0, len(samples), count)]
    block = max(pieces, key=lambda piece: float(np.mean(piece * piece)))
    block = np.pad(block, (0, count - len(block)))
    started = time.perf_counter()
    converter = NeuralConverter(voice, 48000, "cpu", args.profile)
    load_ms = (time.perf_counter() - started) * 1000
    durations, stages = [], {}
    for index in range(16):
        started = time.perf_counter()
        result = converter.process(block)
        elapsed = (time.perf_counter() - started) * 1000
        assert result.shape == block.shape and np.isfinite(result).all() and np.max(np.abs(result)) > 1e-5
        if index >= 4:
            durations.append(elapsed)
            for name, duration in converter.stage_ms.items():
                stages.setdefault(name, []).append(duration)
    return {"profile": args.profile, "provider": converter.provider, "block_ms": 320,
            "load_ms": round(load_ms), "median_ms": round(statistics.median(durations), 1),
            "p95_ms": round(float(np.percentile(durations, 95)), 1),
            "stage_median_ms": {name: round(statistics.median(values), 1) for name, values in stages.items()},
            **memory_mb()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("dist/neural-models"))
    parser.add_argument("--output", type=Path, default=Path("dist/neural-benchmark.json"))
    parser.add_argument("--profile", choices=("light", "full"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--speech", type=Path)
    args = parser.parse_args()
    if args.profile:
        print(json.dumps(measure(args)))
        return
    manifest = next(args.models.glob("voices/*/voice.json")).resolve()
    with tempfile.TemporaryDirectory(prefix="npetrov-ai-benchmark-") as directory:
        speech = Path(directory) / "speech.wav"
        url = "https://raw.githubusercontent.com/dev6699/rvc-onnx/9d20b0ff54df72e83c2826e76c2e1eff5f1ab824/assets/input_1.wav"
        with urllib.request.urlopen(url, timeout=45) as source:
            speech.write_bytes(source.read(4 * 1024 * 1024))
        results = []
        for profile in ("full", "light"):
            child = subprocess.run([sys.executable, __file__, "--manifest", str(manifest), "--speech", str(speech),
                                    "--profile", profile], capture_output=True, text=True, check=True, timeout=300)
            results.append(json.loads(child.stdout))
    report = {"note": "Windows CI CPU measurements; not a measurement of the user's i7-2600 or GTX 1050 Ti.",
              "profiles": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for result in results:
        print("::notice title=AI CPU benchmark::" + json.dumps(result), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
