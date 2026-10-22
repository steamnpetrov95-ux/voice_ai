"""Compare short SDK blocks on real speech without audio devices or networking."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from voicelab.native import NativeConverter, installed_native_voices
from voicelab.neural import resample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist/native-latency.json"))
    args = parser.parse_args()
    os.environ["VOICELAB_NATIVE_DIR"] = str(args.models.resolve())
    voices = installed_native_voices()
    source, rate = sf.read(args.speech, dtype="float32", always_2d=True)
    original = resample(source.mean(axis=1), rate, 48000)
    results = []
    for index in (0, 200, 600, 1000, 1240):
        for milliseconds in (160, 80, 40):
            count = 48 * milliseconds
            converter = NativeConverter(voices[index], 48000)
            outputs, elapsed = [], []
            try:
                for offset in range(0, len(original), count):
                    block = np.pad(original[offset:offset + count], (0, max(0, offset + count - len(original))))
                    started = time.perf_counter()
                    converted = converter.process(block)
                    elapsed.append((time.perf_counter() - started) * 1000)
                    assert converted.shape == block.shape and np.isfinite(converted).all()
                    outputs.append(converted)
                output = np.concatenate(outputs)[:len(original)]
                assert np.max(np.abs(output)) > 1e-4, "Output is silent"
                assert np.mean(np.abs(output - original)) > 1e-4, "Output is unchanged"
                warmup = math.ceil(320 / milliseconds)
                durations = elapsed[warmup:]
                result = {"profile": index + 1, "block_ms": milliseconds,
                          "median_ms": round(float(np.median(durations)), 2),
                          "p95_ms": round(float(np.percentile(durations, 95)), 2),
                          "max_ms": round(max(durations), 2),
                          "rms": round(float(np.sqrt(np.mean(output * output))), 5)}
                results.append(result)
                print("::notice title=Native latency::" + json.dumps(result), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if index == 0:
                    sf.write(args.output.with_name(f"native-{milliseconds}ms.wav"), output, 48000)
            finally:
                converter.close()
    args.output.write_text(json.dumps({"note": "Windows CI CPU, not i7-2600; excludes devices and cable latency.",
                                      "results": results}, indent=2), encoding="utf-8")
    print("::warning title=Native latency summary::" + json.dumps(results), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("::error title=Native latency failed::" + repr(exc), flush=True)
        raise
