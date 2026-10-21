"""PortAudio device I/O. DSP and file writing run outside real-time callbacks."""

from dataclasses import dataclass
from pathlib import Path
import queue
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

from .effects import BLOCK_SIZE, VoiceProcessor
from .presets import VoiceSettings


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    host_api: str
    channels: int

    @property
    def label(self) -> str:
        return f"{self.name} · {self.host_api}"


def list_devices() -> tuple[list[Device], list[Device]]:
    apis = sd.query_hostapis()
    inputs, outputs = [], []
    for index, info in enumerate(sd.query_devices()):
        api = apis[info["hostapi"]]["name"]
        if info["max_input_channels"]:
            inputs.append(Device(index, info["name"], api, info["max_input_channels"]))
        if info["max_output_channels"]:
            outputs.append(Device(index, info["name"], api, info["max_output_channels"]))
    # WASAPI usually gives better interactive latency on Windows than MME.
    inputs.sort(key=lambda device: "WASAPI" not in device.host_api)
    outputs.sort(key=lambda device: "WASAPI" not in device.host_api)
    return inputs, outputs


class AudioEngine:
    def __init__(self):
        self.settings = VoiceSettings()
        self.running = False
        self.sample_rate = 48000
        self.input_level = 0.0
        self.output_level = 0.0
        self.waveform = np.zeros(128, dtype=np.float32)
        self.dropouts = 0
        self.error = ""
        self.recording_path: str | None = None
        self.recorded_frames = 0
        self._recording = None
        self._record_lock = threading.Lock()
        self._input = None
        self._output = None
        self._output_device = None
        self._monitor = None
        self._monitor_queue = None
        self.monitor_error = ""
        self._worker = None
        self._stop = threading.Event()
        self._in_queue = queue.Queue(maxsize=6)
        self._out_queue = queue.Queue(maxsize=6)

    def configure(self, settings: VoiceSettings):
        # Frozen settings are replaced atomically. The worker owns all DSP objects.
        self.settings = settings

    def start(self, input_device: int, output_device: int | None = None):
        if self.running:
            return
        self.error = ""
        self.dropouts = 0
        self._in_queue = queue.Queue(maxsize=6)
        self._out_queue = queue.Queue(maxsize=6)
        self._stop.clear()
        output_channels = min(2, sd.query_devices(output_device, "output")["max_output_channels"]) if output_device is not None else 0
        failures = []
        for rate in (48000, 44100):
            try:
                sd.check_input_settings(device=input_device, channels=1, dtype="float32", samplerate=rate)
                if output_device is not None:
                    sd.check_output_settings(device=output_device, channels=output_channels, dtype="float32", samplerate=rate)
                self.sample_rate = rate
                break
            except (sd.PortAudioError, ValueError) as exc:
                failures.append(str(exc))
        else:
            raise RuntimeError("Устройства не поддерживают общую частоту 48 или 44,1 кГц. "
                               "Проверьте настройки звука Windows.\n" + "\n".join(failures))
        try:
            self._input = sd.InputStream(device=input_device, channels=1, samplerate=self.sample_rate,
                                         blocksize=BLOCK_SIZE, latency="low", dtype="float32", callback=self._capture)
            if output_device is not None:
                self._output = sd.OutputStream(device=output_device, channels=output_channels,
                                               samplerate=self.sample_rate, blocksize=BLOCK_SIZE,
                                               latency="low", dtype="float32", callback=self._play)
            self.running = True
            self._worker = threading.Thread(target=self._process, name="voicelab-dsp", daemon=True)
            self._worker.start()
            self._input.start()
            if self._output is not None:
                self._output.start()
            self._output_device = output_device
        except Exception:
            self.stop()
            raise

    @property
    def monitoring(self) -> bool:
        return self._monitor is not None

    def start_monitoring(self, device: int):
        """Duplicate processed audio to headphones without restarting the main route."""
        if not self.running or self._stop.is_set():
            raise RuntimeError("Сначала включите микрофон.")
        if device == self._output_device:
            raise ValueError("Для прослушивания выберите отдельный выход на наушники.")
        if self.monitoring:
            raise RuntimeError("Прослушивание уже включено.")
        self.monitor_error = ""
        channels = min(2, sd.query_devices(device, "output")["max_output_channels"])
        sd.check_output_settings(device=device, channels=channels, dtype="float32",
                                 samplerate=self.sample_rate)
        # Each stream owns its queue: a slow headset cannot consume or block
        # audio destined for the virtual microphone. Old callbacks retain only their old queue.
        pending = queue.Queue(maxsize=3)

        def play_monitor(outdata, frames, time_info, status):
            self._fill_output(pending, outdata, frames)

        stream = sd.OutputStream(device=device, channels=channels, samplerate=self.sample_rate,
                                 blocksize=BLOCK_SIZE, latency="low", dtype="float32",
                                 callback=play_monitor)
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self._monitor = stream
        self._monitor_queue = pending

    def stop_monitoring(self):
        self._monitor_queue = None
        stream, self._monitor = self._monitor, None
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:
                self.monitor_error = f"Не удалось закрыть выход прослушивания: {exc}"

    def _capture(self, indata, frames, time_info, status):
        if status:
            self.dropouts += 1
        self._offer(self._in_queue, indata[:, 0].copy())

    def _play(self, outdata, frames, time_info, status):
        if status:
            self.dropouts += 1
        if not self._fill_output(self._out_queue, outdata, frames):
            self.dropouts += 1

    @staticmethod
    def _fill_output(pending, outdata, frames):
        outdata.fill(0)
        try:
            block = pending.get_nowait()
        except queue.Empty:
            return False
        count = min(frames, len(block))
        outdata[:count, :] = block[:count, None]
        return True

    def _offer(self, target: queue.Queue, block, count_dropouts=True):
        try:
            target.put_nowait(block)
        except queue.Full:
            # Bound latency if two audio clocks drift or the DSP cannot keep up.
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            try:
                target.put_nowait(block)
            except queue.Full:
                pass
            if count_dropouts:
                self.dropouts += 1

    def _process(self):
        processor = None
        current = None
        try:
            while not self._stop.is_set():
                try:
                    block = self._in_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                settings = self.settings
                if settings != current:
                    processor = VoiceProcessor(self.sample_rate, settings)
                    current = settings
                result = processor.process(block)
                self.input_level = float(np.max(np.abs(block))) if block.size else 0
                self.output_level = float(np.max(np.abs(result))) if result.size else 0
                if result.size:
                    self.waveform = result[np.linspace(0, len(result) - 1, 128, dtype=int)].copy()
                with self._record_lock:
                    if self._recording is not None:
                        try:
                            self._recording.write(result)
                            self.recorded_frames += len(result)
                        except Exception as exc:
                            self.error = f"Запись остановлена: {exc}"
                            self._close_recording()
                if self._output is not None:
                    self._offer(self._out_queue, result)
                monitor_queue = self._monitor_queue
                if monitor_queue is not None:
                    self._offer(monitor_queue, result, count_dropouts=False)
        except Exception as exc:
            self.error = f"Обработка звука остановлена: {exc}"
            self._stop.set()

    def start_recording(self, path: str | Path):
        if not self.running or self._stop.is_set():
            raise RuntimeError("Сначала включите микрофон.")
        with self._record_lock:
            if self._recording is not None:
                raise RuntimeError("Запись уже идёт.")
            handle = Path(path).open("xb")
            try:
                recording = sf.SoundFile(handle, mode="w", samplerate=self.sample_rate,
                                         channels=1, format="WAV", subtype="PCM_16")
            except Exception:
                handle.close()
                raise
            self._record_handle = handle
            self._recording = recording
            self.recording_path = str(path)
            self.recorded_frames = 0

    def _close_recording(self):
        if self._recording is not None:
            self._recording.close()
            self._record_handle.close()
            self._recording = None
        self.recording_path = None

    def stop_recording(self):
        with self._record_lock:
            self._close_recording()

    def check_health(self):
        if not self.running or self._stop.is_set():
            return
        if self._monitor is not None:
            try:
                if not self._monitor.active:
                    raise RuntimeError("Наушники отключены или остановлены.")
            except Exception as exc:
                self.stop_monitoring()
                self.monitor_error = f"Прослушивание выключено: {exc} Проверьте подключение и выберите наушники заново."
        try:
            if not self._input.active or (self._output is not None and not self._output.active):
                self.error = "Аудиоустройство отключено или остановлено. Подключите его и обновите список устройств."
                self._stop.set()
        except Exception as exc:
            self.error = f"Аудиоустройство недоступно: {exc}"
            self._stop.set()

    @property
    def failed(self) -> bool:
        return self.running and self._stop.is_set()

    def stop(self):
        self._stop.set()
        self.stop_monitoring()
        self._output_device = None
        close_errors = []
        for attr in ("_input", "_output"):
            stream = getattr(self, attr)
            if stream is not None:
                try:
                    stream.close()
                except Exception as exc:
                    close_errors.append(str(exc))
                setattr(self, attr, None)
        # Workers use bounded waits; join before closing the recording file.
        if self._worker is not None:
            self._worker.join()
            self._worker = None
        self.stop_recording()
        self.running = False
        self.input_level = self.output_level = 0.0
        if close_errors:
            self.error = "; ".join(close_errors)
