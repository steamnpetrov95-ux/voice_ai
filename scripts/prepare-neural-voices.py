"""Prepare the owner's offline Russian voice pack; Torch is a build dependency only.

Extracts selected checkpoints with bounded ranges (not the 39 GB archive),
loads weights-only, exports split ONNX graphs, and tests actual inference.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

from remote_zip import HTTPRangeFile


SOURCE_COMMIT = "67694e5665d221210d166f1e64cd34568dacd012"
SOURCE_URL = f"https://huggingface.co/daswer123/RVC-big-ru-voice-pack/resolve/{SOURCE_COMMIT}/MODELS.zip"
SOURCE_SIZE = 38891306078
EXPORT_COMMIT = "4de2fd8f895321c506db14df0c102a2f74820005"
VOICES = (
    ("Владимир Зайцев — русский дубляж", "Актеры дублежа/Владимир Зайцев/Владимир Зайцев.pth"),
    ("Иван Жарков — русский дубляж", "Актеры дублежа/Иван Жарков/Иван Жарков.pth"),
    ("Всеволод Кузнецов — русский дубляж", "Актеры дублежа/Всеволод Кузнецов/Всеволод Кузнецов.pth"),
    ("Андрей Ярославцев — русский дубляж", "Актеры дублежа/Андрей Ярославцев/Yaroslavtsev.pth"),
    ("Никита Прозоровский — русский дубляж", "Актеры дублежа/Никита Прозоровский/Никита Прозоровский.pth"),
    ("Сергей Чихачёв — русский дубляж", "Актеры дублежа/Сергей Чихачев/Сергей Чихачев.pth"),
)


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as incoming, path.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)


def prepare_exporter(root):
    base = f"https://raw.githubusercontent.com/SUC-DriverOld/rvc.onnx/{EXPORT_COMMIT}"
    for name in ("attentions", "commons", "models", "modules", "transforms"):
        fetch(f"{base}/infer/module/{name}.py", root / "infer" / "module" / f"{name}.py")
    (root / "infer" / "__init__.py").write_text("")
    (root / "infer" / "module" / "__init__.py").write_text("")
    fetch(f"{base}/scripts/convert_rvc_to_onnx.py", root / "converter.py")
    fetch(f"{base}/LICENSE", root / "LICENSE-rvc.onnx.txt")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("npetrov_rvc_export_helpers", root / "converter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.patch_attention_for_onnx()
    return module


def export_voice(checkpoint_path, destination, exporter):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from torch import nn

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = list(checkpoint["config"])
    config[-3] = checkpoint["weight"]["emb_g.weight"].shape[0]
    version = checkpoint.get("version", "v1")
    f0 = checkpoint.get("f0", 1)
    if f0 != 1 or version not in ("v1", "v2"):
        raise ValueError("The bundled pack currently requires pitch-conditioned RVC v1/v2")
    channels = 256 if version == "v1" else 768
    rate = {"32k": 32000, "40k": 40000, "48k": 48000}.get(config[-1], config[-1])
    net = exporter.SYNTHESIZER_CLASS[(version, f0)](*config, is_half=False)
    del net.enc_q
    result = net.load_state_dict(checkpoint["weight"], strict=False)
    if result.missing_keys:
        raise ValueError(f"Checkpoint has missing inference weights: {result.missing_keys}")
    net.eval()
    net.remove_weight_norm()
    del checkpoint

    class Encoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, phone, lengths, pitch, sid, noise):
            speaker = self.model.emb_g(sid).unsqueeze(-1)
            mean, logs, mask = self.model.enc_p(phone, pitch, lengths)
            latent = (mean + torch.exp(logs) * noise * 0.66666) * mask
            latent = self.model.flow(latent, mask, g=speaker, reverse=True)
            return latent, mask

    class Decoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, z, nsff0, sid):
            speaker = self.model.emb_g(sid).unsqueeze(-1)
            return self.model.dec(z, nsff0, g=speaker)

    destination.mkdir(parents=True, exist_ok=True)
    encoder_path, decoder_path = destination / "voice.enc.onnx", destination / "voice.dec.onnx"
    length = 100
    arguments = (torch.randn(1, length, channels), torch.tensor([length]),
                 torch.full((1, length), 100, dtype=torch.int64), torch.tensor([0]),
                 torch.randn(1, config[2], length))
    with torch.inference_mode():
        torch.onnx.export(Encoder(net).eval(), arguments, str(encoder_path), opset_version=17,
                          input_names=["phone", "lengths", "pitch", "sid", "noise"], output_names=["z", "x_mask"],
                          dynamic_axes={"phone": {1: "frames"}, "pitch": {1: "frames"}, "noise": {2: "frames"},
                                        "z": {2: "frames"}, "x_mask": {2: "frames"}}, do_constant_folding=True)
        torch.onnx.export(Decoder(net).eval(), (torch.randn(1, config[2], 40), torch.full((1, 40), 180.0), torch.tensor([0])),
                          str(decoder_path), opset_version=17, input_names=["z", "nsff0", "sid"], output_names=["audio"],
                          dynamic_axes={"z": {2: "frames"}, "nsff0": {1: "frames"}, "audio": {2: "samples"}},
                          do_constant_folding=True)
    for path in (encoder_path, decoder_path):
        model = onnx.load(path)
        onnx.helper.set_model_props(model, {"metadata": json.dumps({"samplingRate": int(rate), "embedder": "contentvec",
                                                                   "embChannels": channels})})
        onnx.checker.check_model(model)
        onnx.save(model, path)
    # Exercise a different length than the export trace, to reject baked shapes.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    enc = ort.InferenceSession(str(encoder_path), options, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(decoder_path), options, providers=["CPUExecutionProvider"])
    for frames in (64, 96):
        latent, mask = enc.run(None, {"phone": np.zeros((1, frames, channels), np.float32),
                                     "lengths": np.array([frames], np.int64), "pitch": np.full((1, frames), 100, np.int64),
                                     "sid": np.array([0], np.int64), "noise": np.zeros((1, config[2], frames), np.float32)})
        output = dec.run(None, {"z": (latent * mask)[:, :, -32:], "nsff0": np.full((1, 32), 180, np.float32),
                               "sid": np.array([0], np.int64)})[0]
        if output.size != 32 * int(rate) // 100 or not np.isfinite(output).all():
            raise ValueError("Exported model failed dynamic-length inference")
    return int(rate), channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/neural-models"))
    parser.add_argument("--only", type=int, default=len(VOICES))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    from voicelab.models import COMMON, download
    from voicelab.neural import AI_BLOCK_SIZE, NeuralConverter, NeuralVoice, resample
    import numpy as np
    import soundfile as sf
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifests = []
    with tempfile.TemporaryDirectory(prefix="npetrov-neural-build-") as directory:
        temporary = Path(directory)
        exporter = prepare_exporter(temporary / "exporter")
        with HTTPRangeFile(SOURCE_URL, SOURCE_SIZE) as remote, zipfile.ZipFile(remote) as archive:
            for name, member in VOICES[:args.only]:
                print(f"Preparing Russian model: {name}", flush=True)
                checkpoint = temporary / "voice.pth"
                with archive.open(member) as incoming, checkpoint.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                with checkpoint.open("rb") as source:
                    original_hash = hashlib.file_digest(source, "sha256").hexdigest()
                identifier = original_hash[:24]
                destination = root / "voices" / identifier
                rate, channels = export_voice(checkpoint, destination, exporter)
                manifest = {"id": identifier, "name": name, "model": f"voices/{identifier}/voice.enc.onnx",
                            "decoder": f"voices/{identifier}/voice.dec.onnx", "encoder": "common/" + COMMON[channels].filename,
                            "pitch": "common/" + COMMON["pitch"].filename, "sample_rate": rate, "channels": channels,
                            "credit": "Модель сообщества: daswer123/RVC-big-ru-voice-pack; " + name,
                            "terms_url": "https://huggingface.co/daswer123/RVC-big-ru-voice-pack",
                            "source_checkpoint_sha256": original_hash, "source_url": SOURCE_URL, "source_member": member}
                (destination / "voice.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                manifests.append(manifest)
                gc.collect()
        for key in sorted({item["channels"] for item in manifests}):
            download(COMMON[key], root / "common" / COMMON[key].filename,
                     lambda text, percent: print(text, flush=True) if percent % 10 == 0 else None)
        download(COMMON["pitch"], root / "common" / COMMON["pitch"].filename,
                 lambda text, percent: print(text, flush=True) if percent % 10 == 0 else None)
        fetch("https://raw.githubusercontent.com/dev6699/rvc-onnx/9d20b0ff54df72e83c2826e76c2e1eff5f1ab824/assets/input_1.wav", temporary / "speech.wav")
        samples, input_rate = sf.read(temporary / "speech.wav", dtype="float32", always_2d=True)
        samples = resample(samples.mean(axis=1), input_rate, 48000)
        pieces = [samples[offset:offset + AI_BLOCK_SIZE] for offset in range(0, len(samples), AI_BLOCK_SIZE)]
        block = max(pieces, key=lambda value: float(np.mean(value * value)))
        block = np.pad(block, (0, AI_BLOCK_SIZE - len(block)))
        for value in manifests:
            voice = NeuralVoice(value["id"], value["name"], root / value["model"], root / value["encoder"],
                                root / value["pitch"], value["sample_rate"], value["channels"], decoder=root / value["decoder"])
            converter = NeuralConverter(voice, 48000, "cpu")
            converted = converter.process(block)
            if converted.shape != block.shape or not np.isfinite(converted).all() or np.max(np.abs(converted)) < 1e-4:
                raise ValueError("Real speech inference failed for " + voice.name)
            print("Validated real neural speech: " + voice.name, flush=True)
            del converter
            gc.collect()
        shutil.copyfile(temporary / "exporter" / "LICENSE-rvc.onnx.txt", root / "LICENSE-rvc.onnx.txt")
    (root / "SOURCES.json").write_text(json.dumps({"models": manifests, "exporter_commit": EXPORT_COMMIT,
        "common_assets": {str(key): asset.__dict__ for key, asset in COMMON.items()}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifests)} verified offline voices in {root}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        message = str(exc).replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")
        print(f"::error title=Neural model preparation failed::{message}", flush=True)
        raise SystemExit(1)
