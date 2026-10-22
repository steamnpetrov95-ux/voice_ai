"""Personal compatibility backend for the user-supplied native Dubbing AI SDK.

Only documented-by-observation public calls are used. No account data, network
API, driver installation, binary patching, or authorization changes are made.
"""

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np

from .settings import data_dir


@dataclass(frozen=True)
class NativeVoice:
    id: str
    name: str
    root: Path
    speaker_id: int
    credit: str = "Dubbing AI SDK 1.8.2 — личная совместимая сборка"
    backend: str = "native"


def native_directory():
    override = os.environ.get("VOICELAB_NATIVE_DIR")
    if override:
        return Path(override)
    return (Path(sys.executable).resolve().parent / "native-sdk" if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parents[2] / "dist" / "native-sdk")


def _file(root, relative):
    if not isinstance(relative, str):
        raise ValueError("Некорректный файл нативного голоса.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Файл нативного движка отсутствует или находится вне библиотеки.")
    return path


def _manifest(root):
    value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != 1 or not isinstance(value.get("voices"), list) or not isinstance(value.get("binaries"), list):
        raise ValueError("Неподдерживаемый каталог нативных голосов.")
    if len(value["voices"]) > 5000 or len(value["binaries"]) != 3:
        raise ValueError("Некорректный каталог нативных голосов.")
    return value


def _aliases():
    try:
        value = json.loads((data_dir() / "native-voice-names.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def installed_native_voices():
    root = native_directory()
    try:
        value = _manifest(root)
        aliases = _aliases()
        result = []
        identifiers = set()
        for item in value["voices"]:
            key, number = item["key"], item["id"]
            if (not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{32}", key)
                    or type(number) is not int or number < 1 or number in identifiers):
                raise ValueError("Некорректный идентификатор нативного голоса.")
            identifiers.add(number)
            _file(root, item["file"])
            name = aliases.get(key, item.get("name", ""))
            if not isinstance(name, str) or not name.strip():
                name = f"AI-голос {number:04d}"
            result.append(NativeVoice("native-" + key, name.strip()[:120], root, number))
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return []


def rename_native_voice(voice, name):
    name = str(name).strip()[:120]
    if not name or not voice.id.startswith("native-"):
        raise ValueError("Введите название нативного голоса.")
    value = _aliases()
    value[voice.id.removeprefix("native-")] = name
    directory = data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "native-voice-names.json.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / "native-voice-names.json")


_engine_lock = threading.Lock()
_libraries = {}


def _trace(message):
    if os.environ.get("VOICELAB_NATIVE_TRACE") == "1":
        print("::warning title=Native SDK stage::" + message, flush=True)


class NativeRuntime:
    """Win64 virtual interface recovered from the supplied client, version pinned."""

    def __init__(self, root, manifest):
        if sys.platform != "win32":
            raise RuntimeError("Нативный движок доступен в Windows x64.")
        from ctypes import wintypes as w

        key = str(root.resolve())
        if key not in _libraries:
            common_files = {item["file"] for item in manifest["binaries"]}
            for item in manifest["checksums"]:
                if not item["file"].startswith("sdk/") and item["file"] not in common_files:
                    continue
                path = _file(root, item["file"])
                with path.open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != item["sha256"]:
                        raise ValueError("Повреждён файл нативного движка. Переустановите приложение.")
            directory = os.add_dll_directory(str(root / "sdk"))
            _trace("Loading SDK")
            library = ctypes.WinDLL(str(_file(root, "sdk/dubbing-sdk-windows.dll")), winmode=0x100 | 0x1000)
            factory = getattr(library, "?V_@IVoiceTranscribe@DUBBING@@SAPEAV12@XZ")
            factory.argtypes, factory.restype = [], ctypes.c_void_p
            _libraries[key] = (directory, library, factory)
        self.library = _libraries[key]
        _trace("Calling factory")
        self.instance = self.library[2]()
        if not self.instance:
            raise RuntimeError("Нативный движок не создал экземпляр.")
        self.table = ctypes.cast(self.instance, ctypes.POINTER(ctypes.c_void_p))[0]
        self.process_handle_type = w.HANDLE

    def _method(self, offset, result, *arguments):
        address = ctypes.cast(self.table + offset, ctypes.POINTER(ctypes.c_void_p))[0]
        return ctypes.CFUNCTYPE(result, ctypes.c_void_p, *arguments)(address)

    def initialize(self, binaries, speakers, rate):
        return self._method(8, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int)(
            self.instance, json.dumps(binaries).encode(), json.dumps(speakers).encode(), rate, rate)

    def select(self, identifier):
        return self._method(0x18, ctypes.c_int, ctypes.c_int64)(self.instance, identifier)

    def enable(self):
        self._method(0x28, None, ctypes.c_int)(self.instance, 1)

    def process(self, pcm):
        self._method(0x60, None, ctypes.c_void_p, ctypes.c_int64)(self.instance, pcm.ctypes.data, len(pcm))

    def reset(self):
        self._method(0x20, None)(self.instance)

    def close(self):
        self._method(0x10, None)(self.instance)


class NativeConverter:
    provider = "NativeCPUExecutionProvider"

    def __init__(self, voice, sample_rate):
        self.runtime = None
        self._owns_lock = False
        self.stage_ms = {}
        self.pitch_shifter = None
        self.pitch_shift = 0
        self.sample_rate = sample_rate
        self.root = voice.root.resolve()
        self.voice = voice
        if not _engine_lock.acquire(blocking=False):
            raise RuntimeError("Дождитесь завершения другой обработки нативного голоса.")
        self._owns_lock = True
        try:
            value = _manifest(voice.root)
            self.manifest = value
            binaries = [{"id": item["id"], "bid": item["bid"], "localPath": str(_file(voice.root, item["file"]))}
                        for item in value["binaries"]]
            speakers = [{"id": item["id"], "binFileIds": [1, 2, 3], "localPath": str(_file(voice.root, item["file"]))}
                        for item in value["voices"]]
            if voice.speaker_id not in {item["id"] for item in speakers}:
                raise ValueError("Выбранный нативный голос отсутствует.")
            self._check_voice(voice)
            self.runtime = NativeRuntime(voice.root, value)
            _trace(f"Initializing {len(speakers)} voices at {sample_rate} Hz")
            code = self.runtime.initialize(binaries, speakers, sample_rate)
            _trace(f"Initialized: {code}")
            if code != 10000:
                raise RuntimeError(f"Нативный движок не инициализирован: {code}.")
            self.select_voice(voice)
        except Exception:
            self.close()
            raise

    def _check_voice(self, voice):
        if voice.root.resolve() != self.root or getattr(voice, "backend", None) != "native":
            raise ValueError("Нельзя сменить библиотеку во время обработки.")
        selected = next((item for item in self.manifest["voices"] if item["id"] == voice.speaker_id
                         and voice.id == "native-" + item["key"]), None)
        if selected is None:
            raise ValueError("Нативный голос отсутствует в каталоге.")
        expected = next(item["sha256"] for item in self.manifest["checksums"] if item["file"] == selected["file"])
        if hashlib.sha256(_file(voice.root, selected["file"]).read_bytes()).hexdigest() != expected:
            raise ValueError("Повреждён профиль нативного голоса.")

    def select_voice(self, voice):
        self._check_voice(voice)
        _trace(f"Selecting {voice.speaker_id}")
        code = self.runtime.select(voice.speaker_id)
        if code != 10000:
            raise RuntimeError(f"Нативный голос не загружен: {code}.")
        self.runtime.enable()
        self.voice = voice
        self.pitch_shifter = None
        self.pitch_shift = 0

    def process(self, block, pitch_shift=0):
        if self.runtime is None:
            raise RuntimeError("Нативный движок уже остановлен.")
        incoming = np.asarray(block, np.float32).reshape(-1)
        if not np.isfinite(incoming).all():
            raise ValueError("Некорректный входной звук.")
        pcm = np.clip(np.rint(incoming * 32768), -32768, 32767).astype(np.int16)
        started = time.perf_counter()
        self.runtime.process(pcm)
        self.stage_ms = {"native": (time.perf_counter() - started) * 1000}
        output = pcm.astype(np.float32) / 32768
        if pitch_shift != self.pitch_shift:
            from .pitch import StreamingPitchShifter
            self.pitch_shifter = StreamingPitchShifter(pitch_shift) if pitch_shift else None
            self.pitch_shift = pitch_shift
        if self.pitch_shifter is not None:
            output = self.pitch_shifter.process(output)
        return output

    def reset(self):
        if self.runtime is not None:
            self.runtime.reset()
        self.pitch_shifter = None
        self.pitch_shift = 0

    def close(self):
        try:
            if self.runtime is not None:
                runtime, self.runtime = self.runtime, None
                _trace("Releasing SDK")
                runtime.close()
                _trace("Released SDK")
        finally:
            if self._owns_lock:
                self._owns_lock = False
                _engine_lock.release()
