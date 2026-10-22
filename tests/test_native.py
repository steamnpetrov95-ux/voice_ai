import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from voicelab import native
from voicelab.effects import render_file
from voicelab.presets import VoiceSettings


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / "native"
    root.mkdir()
    records, binaries, voices = [], [], []
    for number in range(1, 4):
        name = f"model{number}"
        (root / name).write_bytes(b"test-model")
        records.append({"file": name, "sha256": hashlib.sha256(b"test-model").hexdigest()})
        binaries.append({"id": number, "bid": number, "file": name})
    for number in range(1, 3):
        key = f"{number:032x}"
        (root / key).write_bytes(b"test-profile")
        records.append({"file": key, "sha256": hashlib.sha256(b"test-profile").hexdigest()})
        voices.append({"id": number, "key": key, "file": key, "name": f"Голос {number}"})
    (root / "manifest.json").write_text(json.dumps({"schema": 1, "binaries": binaries, "voices": voices,
                                                    "checksums": records}), encoding="utf-8")
    monkeypatch.setenv("VOICELAB_NATIVE_DIR", str(root))
    monkeypatch.setenv("VOICELAB_DATA_DIR", str(tmp_path / "settings"))
    return root


@pytest.fixture
def runtime(monkeypatch):
    class Runtime:
        instances = []
        select_code = 10000

        def __init__(self, root, manifest):
            self.closed = False
            self.processed = []
            self.instances.append(self)

        def initialize(self, binaries, speakers, rate):
            self.binaries, self.speakers, self.rate = binaries, speakers, rate
            return 10000

        def select(self, identifier):
            self.selected = identifier
            return self.select_code

        def enable(self):
            self.enabled = True

        def process(self, pcm):
            assert pcm.dtype == np.int16 and pcm.ndim == 1 and pcm.flags.c_contiguous
            self.processed.append(pcm.copy())
            pcm *= -1

        def reset(self):
            self.reset_called = True

        def close(self):
            self.closed = True

    monkeypatch.setattr(native, "NativeRuntime", Runtime)
    return Runtime


def test_native_catalog_preserves_ids_and_personal_names(library):
    first, second = native.installed_native_voices()
    native.rename_native_voice(second, "Мой персонаж")
    reloaded = native.installed_native_voices()
    assert [v.id for v in reloaded] == [first.id, second.id]
    assert reloaded[1].name == "Мой персонаж"
    assert reloaded[0].speaker_id == 1


def test_native_catalog_rejects_path_outside_pack(library):
    path = library / "manifest.json"
    value = json.loads(path.read_text())
    value["voices"][0]["file"] = "../outside"
    (library.parent / "outside").write_bytes(b"profile")
    path.write_text(json.dumps(value))
    assert native.installed_native_voices() == []


@pytest.mark.parametrize("rate,count", [(48000, 480), (48000, 7680), (44100, 7040)])
def test_native_pcm_protocol_and_output_length(library, runtime, rate, count):
    from voicelab.neural import create_converter
    voice = native.installed_native_voices()[1]
    converter = create_converter(voice, rate)
    try:
        samples = np.linspace(-0.8, 0.8, count, dtype=np.float32)
        result = converter.process(samples)
        np.testing.assert_allclose(result, -samples, atol=5e-5)
        assert result.shape == samples.shape and result.dtype == np.float32
        assert runtime.instances[-1].rate == rate
        assert runtime.instances[-1].selected == 2
        assert [item["bid"] for item in runtime.instances[-1].binaries] == [1, 2, 3]
        assert runtime.instances[-1].speakers[0]["binFileIds"] == [1, 2, 3]
        assert converter.provider == "NativeCPUExecutionProvider"
        converter.reset()
        assert runtime.instances[-1].reset_called
    finally:
        converter.close()
    assert runtime.instances[-1].closed
    converter.close()


def test_native_failure_releases_singleton_for_retry(library, runtime):
    voice = native.installed_native_voices()[0]
    runtime.select_code = -1
    with pytest.raises(RuntimeError, match="не загружен"):
        native.NativeConverter(voice, 48000)
    assert runtime.instances[-1].closed
    runtime.select_code = 10000
    converter = native.NativeConverter(voice, 48000)
    converter.close()


def test_native_instances_cannot_race_the_sdk_singleton(library, runtime):
    voice = native.installed_native_voices()[0]
    converter = native.NativeConverter(voice, 48000)
    try:
        with pytest.raises(RuntimeError, match="другой обработки"):
            native.NativeConverter(voice, 48000)
    finally:
        converter.close()


def test_native_voice_switch_reuses_the_initialized_runtime(library, runtime):
    first, second = native.installed_native_voices()
    converter = native.NativeConverter(first, 48000)
    try:
        converter.select_voice(second)
        assert len(runtime.instances) == 1
        assert runtime.instances[0].selected == 2 and converter.voice.id == second.id
        assert converter.process(np.ones(480, np.float32) * 0.1).shape == (480,)
    finally:
        converter.close()


def test_live_native_switch_and_both_output_queues(library, runtime):
    import queue
    import threading
    from voicelab.audio import AudioEngine

    first, second = native.installed_native_voices()
    engine = AudioEngine()
    engine.configure_neural(first)
    engine.configure(VoiceSettings(clarity=0))
    engine.running = True
    engine._output = object()
    engine._monitor_queue = queue.Queue(2)
    worker = threading.Thread(target=engine._process)
    worker.start()
    try:
        # Wait for loading to finish by sending until a processed block arrives.
        samples = np.ones(480, np.float32) * 0.2
        for _ in range(20):
            engine._in_queue.put(samples)
            try:
                engine._out_queue.get(timeout=0.05)
                break
            except queue.Empty:
                pass
        while not engine._monitor_queue.empty():
            engine._monitor_queue.get_nowait()
        engine.switch_native_voice(second)
        engine._in_queue.put(samples)
        primary = engine._out_queue.get(timeout=2)
        monitor = engine._monitor_queue.get(timeout=2)
        np.testing.assert_array_equal(primary, monitor)
        np.testing.assert_allclose(primary, -samples, atol=5e-5)
        assert engine.neural_voice.id == second.id
        assert len(runtime.instances) == 1
    finally:
        engine._stop.set()
        worker.join(timeout=2)
    assert not worker.is_alive() and runtime.instances[0].closed


def test_native_profile_integrity_is_checked_before_loading(library, runtime):
    voice = native.installed_native_voices()[0]
    (library / voice.id.removeprefix("native-")).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="Повреждён профиль"):
        native.NativeConverter(voice, 48000)
    assert not runtime.instances


def test_native_file_render_closes_sdk_even_when_output_exists(library, runtime, tmp_path):
    source, target = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(source, np.zeros(4800), 48000)
    target.write_bytes(b"existing recording")
    with pytest.raises(FileExistsError):
        render_file(source, target, VoiceSettings(), native.installed_native_voices()[0])
    assert target.read_bytes() == b"existing recording"
    assert runtime.instances[-1].closed
