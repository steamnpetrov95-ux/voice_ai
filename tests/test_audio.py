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


def wait_for_audio(engine):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not engine._out_queue.empty() and (engine._monitor_queue is None or not engine._monitor_queue.empty()):
            return
        time.sleep(0.005)
    pytest.fail(f"No processed audio: {engine.error}")


def test_monitor_duplicates_processed_voice_and_toggles_without_interrupting_main(devices):
    engine = audio.AudioEngine()
    engine.configure(VoiceSettings(gain=-6, clarity=0))
    engine.start(0, 1)
    main = engine._output
    try:
        engine.start_monitoring(2)
        monitor = engine._monitor
        source = np.full((1024, 1), 0.2, dtype=np.float32)
        engine._capture(source, 1024, None, False)
        wait_for_audio(engine)
        primary, headphones = np.zeros((1024, 2), np.float32), np.zeros((1024, 2), np.float32)
        # Reading one endpoint must not consume the other's copy.
        monitor.kwargs["callback"](headphones, 1024, None, False)
        engine._play(primary, 1024, None, False)
        np.testing.assert_allclose(primary, 0.2 * 10 ** (-6 / 20), atol=1e-5)
        np.testing.assert_array_equal(headphones, primary)
        engine.stop_monitoring()
        assert monitor.closed and not engine.monitoring
        assert engine.running and main.active and engine._output is main
        engine.start_monitoring(3)
        # A re-enabled output starts empty; an old callback cannot steal its audio.
        engine._capture(source * 2, 1024, None, False)
        wait_for_audio(engine)
        monitor.kwargs["callback"](headphones, 1024, None, False)
        assert not headphones.any()
        engine._monitor.kwargs["callback"](headphones, 1024, None, False)
        engine._play(primary, 1024, None, False)
        np.testing.assert_array_equal(headphones, primary)
        assert np.max(primary) > 0.19
    finally:
        engine.stop()
    assert all(stream.closed for stream in FakeStream.instances)


def test_monitor_disconnect_keeps_virtual_microphone_running(devices):
    engine = audio.AudioEngine()
    engine.start(0, 1)
    try:
        engine.start_monitoring(2)
        monitor = engine._monitor
        monitor.active = False
        engine.check_health()
        assert not engine.monitoring and monitor.closed
        assert not engine.failed and engine._output.active and engine._input.active
        assert "Прослушивание выключено" in engine.monitor_error
    finally:
        engine.stop()


def test_monitor_start_failure_closes_only_headphone_stream(devices, monkeypatch):
    engine = audio.AudioEngine()
    engine.start(0, 1)

    class BrokenHeadphones(FakeStream):
        def start(self):
            raise RuntimeError("Headphones busy")

    monkeypatch.setattr(audio.sd, "OutputStream", BrokenHeadphones)
    try:
        with pytest.raises(RuntimeError, match="Headphones busy"):
            engine.start_monitoring(2)
        assert FakeStream.instances[-1].closed
        assert not engine.monitoring and engine._monitor_queue is None
        assert engine.running and engine._output.active and engine._input.active
    finally:
        engine.stop()


def test_monitor_checks_current_sample_rate_without_restarting_main(devices, monkeypatch):
    engine = audio.AudioEngine()
    engine.start(0, 1)

    def unsupported(**kwargs):
        assert kwargs["device"] == 2 and kwargs["samplerate"] == engine.sample_rate
        raise ValueError("Unsupported headphone rate")

    monkeypatch.setattr(audio.sd, "check_output_settings", unsupported)
    try:
        with pytest.raises(ValueError, match="Unsupported headphone rate"):
            engine.start_monitoring(2)
        with pytest.raises(ValueError, match="отдельный выход"):
            engine.start_monitoring(1)
        assert engine.running and engine._output.active and not engine.monitoring
    finally:
        engine.stop()


def test_monitor_requires_running_microphone():
    with pytest.raises(RuntimeError, match="Сначала включите микрофон"):
        audio.AudioEngine().start_monitoring(2)


def test_standalone_monitor_needs_no_virtual_output_and_remains_bounded(devices):
    engine = audio.AudioEngine()
    engine.configure(VoiceSettings(enabled=False))
    engine.start(0)
    try:
        assert engine._output is None
        engine.start_monitoring(2)
        source = np.full((1024, 1), 0.2, dtype=np.float32)
        for _ in range(10):
            engine._capture(source, 1024, None, False)
            deadline = time.monotonic() + 3
            while engine.input_level == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
            assert engine.input_level > 0
            engine.input_level = 0
        assert engine._monitor_queue.qsize() <= 3
        assert engine._out_queue.empty()
        headphones = np.zeros((1024, 2), np.float32)
        engine._monitor.kwargs["callback"](headphones, 1024, None, False)
        np.testing.assert_array_equal(headphones, np.repeat(source, 2, axis=1))
        engine.stop_monitoring()
        engine.check_health()
        assert not engine.failed
        engine.start_monitoring(2)
        assert engine._monitor_queue.empty()
    finally:
        engine.stop()
