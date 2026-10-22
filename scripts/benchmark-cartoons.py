"""Measure short-block cartoon DSP, optionally verify every style in a frozen EXE."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf

from voicelab.effects import VoiceProcessor
from voicelab.presets import CARTOON_PRESETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dist/cartoon-benchmark.json"))
    args = parser.parse_args()
    rate, block = 48000, 1024
    t = np.arange(rate * 3) / rate
    # Harmonic signal with consonant-like noise, not a subjective voice demo.
    rng = np.random.default_rng(42)
    samples = (0.12 * np.sin(2 * np.pi * 180 * t) + 0.04 * np.sin(2 * np.pi * 540 * t)
               + 0.01 * rng.normal(size=len(t))).astype(np.float32)
    results = []
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source.wav"
        sf.write(source, samples, rate)
        samples, _ = sf.read(source, dtype="float32")
        for preset in CARTOON_PRESETS:
            processor = VoiceProcessor(rate, preset.settings)
            times, outputs = [], []
            for i in range(0, len(samples), block):
                started = time.perf_counter()
                output = processor.process(samples[i:i + block])
                times.append((time.perf_counter() - started) * 1000)
                outputs.append(output)
            result = np.concatenate(outputs)
            assert len(result) == len(samples) and np.isfinite(result).all()
            assert np.max(np.abs(result)) > 0.01
            assert not np.allclose(result, samples)
            if args.app:
                target = Path(directory) / f"{preset.id}.wav"
                subprocess.run([str(args.app.resolve()), "--render", str(source), str(target),
                                "--preset", preset.id], check=True, timeout=60)
                rendered, rendered_rate = sf.read(target)
                assert rendered_rate == rate and len(rendered) == len(samples) + rate
                assert np.isfinite(rendered).all() and np.max(np.abs(rendered)) > 0.01
                # Installed DSP must match the source processor, allowing PCM16 rounding.
                np.testing.assert_allclose(rendered[:len(result)], result, atol=0.0002)
            results.append({"id": preset.id, "name": preset.name,
                            "processing_p95_ms": round(float(np.percentile(times[3:], 95)), 3),
                            "block_ms": round(block * 1000 / rate, 3)})
    report = {"note": "Compute time on this machine, not microphone-to-output latency.",
              "sample_rate": rate, "styles": results, "installed_exe_checked": bool(args.app)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("::notice::Cartoon DSP " + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
