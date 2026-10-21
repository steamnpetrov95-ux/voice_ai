import json
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from voicelab import models, neural


def make_model(path, inputs, output_name, output, metadata=None):
    node = helper.make_node("Constant", [], [output_name], value=numpy_helper.from_array(output))
    graph = helper.make_graph([node], "test-only-constant", inputs,
                              [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, list(output.shape))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    if metadata:
        helper.set_model_props(model, metadata)
    onnx.save(model, str(path))


@pytest.fixture
def voice(tmp_path):
    """Deliberately synthetic ONNX graphs test wiring, not trained voice quality."""
    encoder, pitch, synth = (tmp_path / name for name in ("encoder.onnx", "pitch.onnx", "voice.onnx"))
    make_model(encoder, [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, "samples"])],
               "features", np.ones((1, 768, 128), np.float32))
    make_model(pitch, [helper.make_tensor_value_info("waveform", TensorProto.FLOAT, [1, "samples"]),
                       helper.make_tensor_value_info("threshold", TensorProto.FLOAT, [1])],
               "pitchf", np.full((1, 256), 180, np.float32))
    inputs = [helper.make_tensor_value_info(name, kind, shape) for name, kind, shape in (
        ("feats", TensorProto.FLOAT, [1, "frames", 768]), ("p_len", TensorProto.INT64, [1]),
        ("pitch", TensorProto.INT64, [1, "frames"]), ("pitchf", TensorProto.FLOAT, [1, "frames"]),
        ("sid", TensorProto.INT64, [1]),
    )]
    waveform = (np.sin(2 * np.pi * 220 * np.arange(48000) / 40000) * 0.2).astype(np.float32)
    make_model(synth, inputs, "audio", waveform, {"metadata": json.dumps({"samplingRate": 40000, "embedder": "contentvec"})})
    return neural.NeuralVoice("test", "Тестовая модель", synth, encoder, pitch, 40000, 768)


def test_actual_onnx_runtime_processes_and_stitches_exact_blocks(voice):
    converter = neural.NeuralConverter(voice, 48000, "cpu")
    samples = np.full(neural.AI_BLOCK_SIZE, 0.1, np.float32)
    first, second = converter.process(samples), converter.process(samples, -3)
    assert first.shape == second.shape == samples.shape
    assert first.dtype == second.dtype == np.float32
    assert np.isfinite(second).all() and np.max(np.abs(second)) > 0.1
    assert converter.provider == "CPUExecutionProvider"
    assert len(converter.history) == 10240
    converter.reset()
    assert not converter.history.any() and not converter.tail.any()
    assert not converter.process(np.zeros_like(samples)).any()


def test_rvc_pitch_and_feature_protocol_is_supplied_to_synthesizer(voice):
    converter = neural.NeuralConverter(voice, 48000, "cpu")
    feeds = []
    real = converter.synth
    converter.synth = SimpleNamespace(get_inputs=real.get_inputs,
                                     run=lambda names, values: (feeds.append(values) or real.run(names, values)))
    converter.process(np.ones(neural.AI_BLOCK_SIZE, np.float32) * 0.1, pitch_shift=12)
    values = feeds[0]
    assert values["feats"].shape == (1, 96, 768)
    np.testing.assert_allclose(values["pitchf"], 360)
    assert values["pitch"].dtype == np.int64
    assert values["p_len"].tolist() == [96]


def test_model_metadata_must_match_installed_voice(voice):
    from dataclasses import replace
    with pytest.raises(ValueError, match="изменилась"):
        neural.NeuralConverter(replace(voice, sample_rate=48000), 48000, "cpu")


def test_unsupported_provider_is_explicit_error(voice, monkeypatch):
    import onnxruntime as ort
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="недоступно"):
        neural.NeuralConverter(voice, 48000, "dml")


@pytest.mark.parametrize("frequency", [65, 100, 180, 260, 440, 800])
def test_fast_pitch_tracks_fundamental_with_harmonics(frequency):
    from voicelab.f0 import fast_pitch
    time = np.arange(16000) / 16000
    audio = 0.1 * np.sin(2 * np.pi * frequency * time) + 0.07 * np.sin(4 * np.pi * frequency * time)
    pitch = fast_pitch(audio)[5:-5]
    assert len(pitch) == 90
    assert np.count_nonzero(pitch) >= 88
    assert np.median(pitch) == pytest.approx(frequency, rel=0.015)


def test_fast_pitch_does_not_voice_silence_dc_or_noise():
    from voicelab.f0 import fast_pitch
    for audio in (np.zeros(16000), np.full(16000, 0.3), np.random.default_rng(7).normal(0, 0.1, 16000)):
        assert not fast_pitch(audio).any()
    assert fast_pitch(np.zeros(80)).shape == (0,)


def test_light_profile_runs_without_loading_pitch_network(voice, monkeypatch):
    loaded = []
    original = neural._session

    def session(path, provider):
        loaded.append(path)
        assert path != voice.pitch, "Light AI must not allocate the RMVPE network"
        return original(path, provider)

    monkeypatch.setattr(neural, "_session", session)
    converter = neural.NeuralConverter(voice, 48000, "cpu", "light")
    assert len(loaded) == 2
    assert len(converter.history) == 2560
    samples = np.sin(2 * np.pi * 180 * np.arange(15360) / 48000).astype(np.float32) * 0.1
    first, second = converter.process(samples), converter.process(samples)
    assert first.shape == second.shape == samples.shape
    assert np.isfinite(second).all() and np.max(np.abs(second)) > 0.1
    assert set(converter.stage_ms) == {"encoder", "pitch", "voice"}


def test_light_profile_pitch_preserves_voicing_and_semitone_shift(voice):
    converter = neural.NeuralConverter(voice, 48000, "cpu", "light")
    feeds = []
    real = converter.synth
    converter.synth = SimpleNamespace(get_inputs=real.get_inputs,
                                     run=lambda names, values: (feeds.append(values) or real.run(names, values)))
    samples = np.sin(2 * np.pi * 180 * np.arange(15360) / 48000).astype(np.float32) * 0.1
    converter.process(samples, 12)
    values = feeds[0]
    assert values["feats"].shape == (1, 48, 768)
    assert not values["pitchf"][0, :10].any()
    assert np.median(values["pitchf"][0, 24:44]) == pytest.approx(360, rel=0.02)


def test_resampling_preserves_duration_and_frequency():
    source = np.sin(2 * np.pi * 220 * np.arange(44100) / 44100).astype(np.float32)
    result = neural.resample(source, 44100, 16000)
    assert len(result) == 16000
    frequency = np.argmax(np.abs(np.fft.rfft(result)))
    assert frequency == 220


def test_imported_voice_is_only_published_when_all_assets_ready(voice, monkeypatch, tmp_path):
    monkeypatch.setenv("VOICELAB_DATA_DIR", str(tmp_path / "data"))
    downloaded = []

    def copy_common(asset, destination, progress, cancel):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"test-only")
        downloaded.append(asset.filename)

    monkeypatch.setattr(models, "download", copy_common)
    installed = models.install_voice(voice.model, "Мой голос")
    assert installed.name == "Мой голос" and installed.channels == 768
    assert installed.model.read_bytes() == voice.model.read_bytes()
    assert downloaded == ["vec-768.onnx", "rmvpe.onnx"]
    assert [item.id for item in models.installed_voices()] == [installed.id]
    assert not list(models.voice_directory().glob(".install-*"))


def test_failed_common_download_does_not_publish_a_voice(voice, monkeypatch, tmp_path):
    monkeypatch.setenv("VOICELAB_DATA_DIR", str(tmp_path / "data"))

    def fail(*args):
        raise OSError("network unavailable")

    monkeypatch.setattr(models, "download", fail)
    with pytest.raises(OSError, match="network unavailable"):
        models.install_voice(voice.model, "Мой голос")
    assert models.installed_voices() == []
    assert not list(models.voice_directory().glob(".install-*"))


def test_manifest_cannot_reference_files_outside_library(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICELAB_DATA_DIR", str(tmp_path / "data"))
    directory = models.voice_directory() / "voices" / "bad"
    directory.mkdir(parents=True)
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"model")
    (directory / "voice.json").write_text(json.dumps({"id": "a" * 24, "name": "bad", "model": str(outside),
        "encoder": str(outside), "pitch": str(outside), "sample_rate": 40000, "channels": 768}))
    assert models.installed_voices() == []


@pytest.mark.parametrize("url", ["http://huggingface.co/a/b/resolve/main/voice.onnx",
                                     "https://example.com/a/b/resolve/main/voice.onnx",
                                     "https://huggingface.co/a/b/resolve/main/voice.pth"])
def test_download_source_requires_huggingface_onnx(url):
    with pytest.raises(ValueError):
        models.resolve_huggingface(url)


def test_download_verifies_hash_and_keeps_existing_file_on_failure(tmp_path, monkeypatch):
    import hashlib
    import io
    payload = b"test model bytes"
    destination = tmp_path / "voice.onnx"
    asset = models.ModelAsset("https://huggingface.co/a/b/resolve/main/voice.onnx",
                              hashlib.sha256(payload).hexdigest(), len(payload), "voice.onnx")
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    models.download(asset, destination)
    assert destination.read_bytes() == payload
    destination.write_bytes(b"previous working model")
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="SHA256"):
        models.download(asset, destination)
    assert destination.read_bytes() == b"previous working model"
    assert not list(tmp_path.glob("*.part-*"))


def test_cancelled_download_is_not_published(tmp_path, monkeypatch):
    import hashlib
    import io
    import threading
    payload = b"test model"
    asset = models.ModelAsset("https://huggingface.co/a/b/resolve/main/voice.onnx",
                              hashlib.sha256(payload).hexdigest(), len(payload), "voice.onnx")
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(models.DownloadCancelled):
        models.download(asset, tmp_path / "voice.onnx", cancel=cancel)
    assert not list(tmp_path.iterdir())


def test_import_reuses_bundled_common_model_without_network(tmp_path, monkeypatch):
    import hashlib
    payload = b"shared verified model"
    asset = models.ModelAsset("https://huggingface.co/a/b/resolve/main/voice.onnx",
                              hashlib.sha256(payload).hexdigest(), len(payload), "shared.onnx")
    builtin = tmp_path / "application" / "models"
    (builtin / "common").mkdir(parents=True)
    original = builtin / "common" / asset.filename
    original.write_bytes(payload)
    monkeypatch.setattr(models, "_builtin_directory", lambda: builtin)
    monkeypatch.setattr(models, "download", lambda *args: pytest.fail("Bundled model must not be downloaded again"))
    root = tmp_path / "user-models"
    models.ensure_common(asset, root, lambda *args: None, None)
    original.unlink()
    assert (root / "common" / asset.filename).read_bytes() == payload
