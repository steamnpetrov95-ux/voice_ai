import queue
import time

import numpy as np
import pytest
import soundfile as sf

from voicelab import audio
from voicelab.presets import VoiceSettings


class FakeStream:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.active = False
        self.closed = False
        self.instances.append(self)

    def start(self):
        self.active = True

    def close(self):
        self.active = False
        self.closed = True


@pytest.fixture
def devices(monkeypatch):
    FakeStream.instances = []
    monkeypatch.setattr(audio.sd, "query_devices", lambda *args: {"max_output_channels": 2})
    monkeypatch.setattr(audio.sd, "check_input_settings", lambda **kwargs: None)
    monkeypatch.setattr(audio.sd, "check_output_settings", lambda **kwargs: None)
    monkeypatch.setattr(audio.sd, "InputStream", FakeStream)
    monkeypatch.setattr(audio.sd, "OutputStream", FakeStream)


def test_live_pipeline_processes_routes_and_records(tmp_path, devices):
    engine = audio.AudioEngine()
    engine.configure(VoiceSettings(gain=-6, clarity=0))
    engine.start(0, 1)
    path = tmp_path / "record.wav"
    engine.start_recording(path)
    source = np.full((1024, 1), 0.2, dtype=np.float32)
    engine._capture(source, 1024, None, False)
    deadline = time.monotonic() + 3
    while engine._out_queue.empty() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not engine._out_queue.empty(), engine.error
    output = np.zeros((1024, 2), dtype=np.float32)
    engine._play(output, 1024, None, False)
    np.testing.assert_allclose(output, 0.2 * 10 ** (-6 / 20), atol=1e-5)
    engine.stop()
    recording, rate = sf.read(path)
    assert rate == 48000
    assert len(recording) == 1024
    np.testing.assert_allclose(recording, output[:, 0], atol=1 / 32768)
    assert all(stream.closed for stream in FakeStream.instances)
    assert not engine.running


def test_start_failure_rolls_back_open_streams(devices, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("Device busy")
    monkeypatch.setattr(audio.sd, "OutputStream", fail)
    engine = audio.AudioEngine()
    with pytest.raises(RuntimeError, match="Device busy"):
        engine.start(0, 1)
    assert not engine.running
    assert FakeStream.instances[0].closed


def test_sample_rate_fallback(devices, monkeypatch):
    def check(**kwargs):
        if kwargs["samplerate"] == 48000:
            raise ValueError("unsupported")
    monkeypatch.setattr(audio.sd, "check_input_settings", check)
    engine = audio.AudioEngine()
    try:
        engine.start(0, 1)
        assert engine.sample_rate == 44100
    finally:
        engine.stop()


def test_queue_overload_discards_old_audio_and_underrun_plays_silence():
    engine = audio.AudioEngine()
    target = queue.Queue(maxsize=1)
    target.put("old")
    engine._offer(target, "new")
    assert target.get_nowait() == "new"
    output = np.ones((1024, 2), dtype=np.float32)
    engine._play(output, 1024, None, False)
    assert not output.any()
    assert engine.dropouts == 2


def test_recording_requires_running_engine(tmp_path):
    with pytest.raises(RuntimeError):
        audio.AudioEngine().start_recording(tmp_path / "test.wav")


def test_device_disconnect_is_reported(devices):
    engine = audio.AudioEngine()
    engine.start(0, 1)
    try:
        engine._output.active = False
        engine.check_health()
        assert engine.failed
        assert "отключено" in engine.error
    finally:
        engine.stop()
