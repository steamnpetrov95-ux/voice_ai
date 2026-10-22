"""PortAudio device I/O. DSP and file writing run outside real-time callbacks."""

from dataclasses import dataclass, replace
from pathlib import Path
import queue
import threading
import time

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


class OutputState:
    """A single output callback owns the remainder of its current DSP block."""

    def __init__(self):
        self.block = None
        self.offset = 0


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
        self.neural_voice = None
        self.neural_provider = "auto"
        self.neural_buffer_ms = 320
        self.neural_profile = "light"
        self.stage_ms = {}
        self._native_switch_lock = threading.Lock()
        self._pending_native_voice = None
        self.loading = False
        self.processing_ms = 0.0
        self.provider_label = ""
        self.block_size = BLOCK_SIZE
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
        self._out_state = OutputState()
        self._capture_buffer = None
        self._capture_offset = 0

    @property
    def native(self):
        return getattr(self.neural_voice, "backend", None) == "native"

    @property
    def output_queue_blocks(self):
        # The SDK performs inference in bursts. Keep enough short blocks to
        # bridge one burst without making a slow headset block the main route.
        return max(2, (160 + self.neural_buffer_ms - 1) // self.neural_buffer_ms) if self.native else 2

    def configure(self, settings: VoiceSettings):
        # Frozen settings are replaced atomically. The worker owns all DSP objects.
        self.settings = settings

    def configure_neural(self, voice=None, provider="auto", buffer_ms=320, profile="light"):
        if self.running:
            raise RuntimeError("Остановите микрофон перед сменой AI-голоса.")
        if buffer_ms not in (40, 80, 160, 320, 480, 640):
            raise ValueError("Неподдерживаемый размер буфера AI.")
        if voice is not None and getattr(voice, "backend", None) != "native" and buffer_ms < 160:
            raise ValueError("Буферы 40 и 80 мс доступны для нативного AI.")
        if profile not in ("light", "full"):
            raise ValueError("Неизвестный профиль AI.")
        self.neural_voice, self.neural_provider = voice, provider
        self.neural_buffer_ms = buffer_ms
        self.neural_profile = profile
        with self._native_switch_lock:
            self._pending_native_voice = None

    def switch_native_voice(self, voice):
        if (not self.running or getattr(self.neural_voice, "backend", None) != "native"
                or getattr(voice, "backend", None) != "native" or voice.root != self.neural_voice.root):
            raise RuntimeError("Остановите микрофон перед сменой движка.")
        with self._native_switch_lock:
            self._pending_native_voice = voice

    def start(self, input_device: int, output_device: int | None = None):
        if self.running:
            return
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError("Дождитесь завершения предыдущей обработки AI и повторите запуск.")
        self.loading = self.neural_voice is not None
        self.processing_ms = 0.0
        self.stage_ms = {}
        self.provider_label = ""
        self.error = ""
        self.dropouts = 0
        input_blocks = max(1, (80 + self.neural_buffer_ms - 1) // self.neural_buffer_ms) if self.native else 2
        self._in_queue = queue.Queue(maxsize=input_blocks if self.neural_voice else 6)
        self._out_queue = queue.Queue(maxsize=self.output_queue_blocks if self.neural_voice else 6)
        self._out_state = OutputState()
        self._capture_offset = 0
        self._stop.clear()
        output_channels = min(2, sd.query_devices(output_device, "output")["max_output_channels"]) if output_device is not None else 0
        failures = []
        for rate in (48000, 44100):
            try:
                sd.check_input_settings(device=input_device, channels=1, dtype="float32", samplerate=rate)
                if output_device is not None:
                    sd.check_output_settings(device=output_device, channels=output_channels, dtype="float32", samplerate=rate)
                self.sample_rate = rate
                self.block_size = round(rate * self.neural_buffer_ms / 1000 / 128) * 128 if self.neural_voice else BLOCK_SIZE
                break
            except (sd.PortAudioError, ValueError) as exc:
                failures.append(str(exc))
        else:
            raise RuntimeError("Устройства не поддерживают общую частоту 48 или 44,1 кГц. "
                               "Проверьте настройки звука Windows.\n" + "\n".join(failures))
        try:
            self._capture_buffer = np.empty(self.block_size, dtype=np.float32) if self.native else None
            io_block_size = 0 if self.native else self.block_size
            self._input = sd.InputStream(device=input_device, channels=1, samplerate=self.sample_rate,
                                         blocksize=io_block_size, latency="low", dtype="float32", callback=self._capture)
            if output_device is not None:
                self._output = sd.OutputStream(device=output_device, channels=output_channels,
                                               samplerate=self.sample_rate, blocksize=io_block_size,
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
        pending = queue.Queue(maxsize=self.output_queue_blocks if self.neural_voice else 3)
        output_state = OutputState()

        def play_monitor(outdata, frames, time_info, status):
            self._fill_output(pending, outdata, frames, output_state)

        stream = sd.OutputStream(device=device, channels=channels, samplerate=self.sample_rate,
                                 blocksize=0 if self.native else self.block_size, latency="low", dtype="float32",
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
        if self.native:
            # The host chooses its own small callback size. Accumulate exactly
            # one AI block here; never replay a partial block captured at load.
            if self.loading:
                self._capture_offset = 0
                return
            offset = 0
            while offset < frames:
                count = min(frames - offset, self.block_size - self._capture_offset)
                self._capture_buffer[self._capture_offset:self._capture_offset + count] = indata[offset:offset + count, 0]
                self._capture_offset += count
                offset += count
                if self._capture_offset == self.block_size:
                    self._offer(self._in_queue, self._capture_buffer.copy())
                    self._capture_offset = 0
            return
        self._offer(self._in_queue, indata[:, 0].copy())

    def _play(self, outdata, frames, time_info, status):
        if status:
            self.dropouts += 1
        if not self._fill_output(self._out_queue, outdata, frames, self._out_state) and not self.loading:
            self.dropouts += 1

    @staticmethod
    def _fill_output(pending, outdata, frames, state=None):
        outdata.fill(0)
        state = state if state is not None else OutputState()
        position = 0
        while position < frames:
            if state.block is None:
                try:
                    state.block = pending.get_nowait()
                    state.offset = 0
                except queue.Empty:
                    return False
            count = min(frames - position, len(state.block) - state.offset)
            outdata[position:position + count, :] = state.block[state.offset:state.offset + count, None]
            state.offset += count
            position += count
            if state.offset == len(state.block):
                state.block = None
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
        neural = None
        try:
            if self.neural_voice:
                from .neural import create_converter
                neural = create_converter(self.neural_voice, self.sample_rate, self.neural_provider, self.neural_profile)
                self.provider_label = {"DmlExecutionProvider": "DirectML (GPU)",
                                       "NativeCPUExecutionProvider": "Native CPU"}.get(neural.provider, "CPU")
                # Audio captured while loading a model must never replay later.
                while not self._in_queue.empty():
                    try:
                        self._in_queue.get_nowait()
                    except queue.Empty:
                        break
                self.dropouts = 0
            self.loading = False
            while not self._stop.is_set():
                try:
                    block = self._in_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                settings = self.settings
                if settings != current:
                    processor = VoiceProcessor(self.sample_rate, replace(settings, pitch=0) if neural else settings)
                    current = settings
                    if neural and not settings.enabled:
                        neural.reset()
                started = time.perf_counter()
                with self._native_switch_lock:
                    pending = self._pending_native_voice
                    self._pending_native_voice = None
                if pending is not None:
                    neural.select_voice(pending)
                    self.neural_voice = pending
                converted = neural.process(block, settings.pitch) if neural and settings.enabled else block
                result = processor.process(converted)
                self.processing_ms = (time.perf_counter() - started) * 1000
                self.stage_ms = dict(getattr(neural, "stage_ms", {})) if neural else {}
                if self._stop.is_set():
                    break
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
            if not self._stop.is_set():
                self.error = f"Обработка звука остановлена: {exc}"
            self._stop.set()
        finally:
            self.loading = False
            if neural is not None and hasattr(neural, "close"):
                try:
                    neural.close()
                except Exception as exc:
                    if not self.error:
                        self.error = f"Остановка нативного движка: {exc}"

    def start_recording(self, path: str | Path):
        if not self.running or self._stop.is_set() or self.loading:
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

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

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
            self._worker.join(timeout=0.5)
            if not self._worker.is_alive():
                self._worker = None
        self.stop_recording()
        self.running = False
        self.input_level = self.output_level = 0.0
        if close_errors:
            self.error = "; ".join(close_errors)
