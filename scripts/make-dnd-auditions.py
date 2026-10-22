"""Render real native profiles on Russian speech for human listening, not MOS scoring."""

import argparse
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from voicelab.f0 import fast_pitch
from voicelab.native import NativeConverter, installed_native_voices
from voicelab.neural import resample


def render_voice(source, voice, factory=NativeConverter):
    """Use the app's 80 ms native path with no post-effects; flush the SDK tail."""
    incoming = np.pad(source, (0, 30720))
    converter = factory(voice, 48000)
    blocks = []
    try:
        for start in range(0, len(incoming), 3840):
            block = incoming[start:start + 3840]
            blocks.append(converter.process(np.pad(block, (0, 3840 - len(block))), 0))
    finally:
        converter.close()
    result = np.concatenate(blocks)[:len(incoming)]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite output")
    return result


def fingerprint(samples):
    """Acoustic diversity only: these measurements cannot judge a good NPC voice."""
    audio = resample(samples, 48000, 16000)
    frames = np.lib.stride_tricks.sliding_window_view(audio, 512)[::160]
    energy = np.mean(frames * frames, axis=1)
    active = frames[energy > max(float(energy.max()) * 0.01, 1e-7)]
    if len(active) < 10:
        raise ValueError("Not enough speech")
    spectrum = np.abs(np.fft.rfft(active * np.hanning(512), axis=1)) ** 2
    frequencies = np.fft.rfftfreq(512, 1 / 16000)
    edges = np.geomspace(70, 7600, 17)
    bands = np.array([np.mean(spectrum[:, (frequencies >= left) & (frequencies < right)])
                      for left, right in zip(edges[:-1], edges[1:])])
    bands = np.log(np.maximum(bands, 1e-12))
    bands -= bands.mean()
    f0 = fast_pitch(audio)
    voiced = f0[f0 > 0]
    if len(voiced) < 10:
        raise ValueError("Not enough periodic speech for comparison")
    return np.concatenate((bands, np.log(np.percentile(voiced, [20, 50, 80])) * 4))


def choose_diverse(features, count):
    """Pick actual examples near acoustic cluster centres, not quality winners."""
    values = np.asarray(features, dtype=np.float64)
    count = min(count, len(values))
    if not count:
        return []
    scale = np.maximum(values.std(axis=0), 0.1)
    values = np.clip((values - np.median(values, axis=0)) / scale, -4, 4)
    selected = [int(np.argmin(np.sum(values * values, axis=1)))]
    while len(selected) < count:
        distances = np.min(np.sum((values[:, None] - values[selected]) ** 2, axis=2), axis=1)
        distances[selected] = -1
        selected.append(int(np.argmax(distances)))
    centres = values[selected].copy()
    for _ in range(10):
        groups = np.argmin(np.sum((values[:, None] - centres) ** 2, axis=2), axis=1)
        for index in range(count):
            if np.any(groups == index):
                centres[index] = values[groups == index].mean(axis=0)
    selected = []
    for centre in centres:
        distance = np.sum((values - centre) ** 2, axis=1)
        distance[selected] = np.inf
        selected.append(int(np.argmin(distance)))
    return selected


def wav_url(samples):
    """Volume match for comparison, preserving the model's pitch and timbre."""
    audio = resample(samples, 48000, 16000)
    rms = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))
    if rms < 1e-5:
        raise ValueError("Silent preview")
    audio *= min(0.085 / rms, 0.88 / max(peak, 1e-6))
    output = io.BytesIO()
    sf.write(output, audio, 16000, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def write_page(destination, source, records, text, reference_note):
    template = Path(__file__).with_name("dnd-auditions.html").read_text(encoding="utf-8")
    data = {"text": text, "reference_note": reference_note, "original": wav_url(source), "voices": records}
    # No HTML from catalogue names; the page uses textContent for all labels.
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    destination.write_text(template.replace("__AUDITION_DATA__", encoded), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--reference-text", type=Path, required=True)
    parser.add_argument("--reference-note", required=True)
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("dist/DnD-voice-auditions.html"))
    args = parser.parse_args()
    if not 1 <= args.count <= args.candidates <= 128:
        raise ValueError("Use 1 <= count <= candidates <= 128")
    os.environ["VOICELAB_NATIVE_DIR"] = str(args.models.resolve())
    voices = installed_native_voices()
    if not voices:
        raise ValueError("No native profiles")
    source, rate = sf.read(args.speech, dtype="float32", always_2d=True)
    source = resample(source.mean(axis=1), rate, 48000)
    if not 3 <= len(source) / 48000 <= 20:
        raise ValueError("Reference must be 3–20 seconds")
    source *= 0.35 / max(float(np.max(np.abs(source))), 1e-6)
    candidates, rejected = [], []
    for index in np.unique(np.linspace(0, len(voices) - 1, args.candidates).astype(int)):
        voice = voices[index]
        output = render_voice(source, voice)
        try:
            features = fingerprint(output)
        except ValueError as exc:
            rejected.append({"id": voice.id, "reason": str(exc)})
            continue
        candidates.append((voice, output, features))
        print(f"Rendered {voice.name}", flush=True)
    if len(candidates) < args.count:
        raise ValueError("Too few comparable candidates; do not publish an incomplete page")
    selected = choose_diverse([item[2] for item in candidates], args.count)
    records = [{"id": candidates[i][0].id, "name": candidates[i][0].name,
                "speaker_id": candidates[i][0].speaker_id, "audio": wav_url(candidates[i][1])}
               for i in selected]
    text = args.reference_text.read_text(encoding="utf-8").strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_page(args.output, source, records, text, args.reference_note)
    report = {"reference": args.reference_note,
              "note": "Selected for acoustic diversity. Naturalness and NPC suitability need human listening.",
              "candidates_rendered": len(candidates), "rejected": rejected,
              "voices": [{k: v for k, v in item.items() if k != "audio"} for item in records]}
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("::notice title=DnD auditions::" + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
