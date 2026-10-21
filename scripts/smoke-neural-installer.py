"""Verify that the installed EXE can find and run every bundled ONNX voice."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf


parser = argparse.ArgumentParser()
parser.add_argument("--app", type=Path, required=True)
parser.add_argument("--models", type=Path, required=True)
args = parser.parse_args()
manifests = list(args.models.glob("voices/*/voice.json"))
assert len(manifests) == 6, f"Expected six installed voices, found {len(manifests)}"
with tempfile.TemporaryDirectory(prefix="npetrov-neural-smoke-") as directory:
    root = Path(directory)
    source = root / "input.wav"
    sf.write(source, np.sin(2 * np.pi * 180 * np.arange(15360) / 48000) * 0.2, 48000)
    for path in manifests:
        value = json.loads(path.read_text(encoding="utf-8"))
        for profile in ("light", "full"):
            output = root / (value["id"] + "-" + profile + ".wav")
            subprocess.run([str(args.app), "--render", str(source), str(output), "--ai-voice", value["id"],
                            "--ai-provider", "cpu", "--ai-profile", profile], check=True, timeout=120)
            result, rate = sf.read(output)
            assert rate == 48000 and len(result) == 15360
            assert np.isfinite(result).all() and np.max(np.abs(result)) > 1e-5
            print("Installed neural voice validated: " + value["name"] + " / " + profile, flush=True)
