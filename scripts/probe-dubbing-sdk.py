"""Isolated Windows loader probe for the user-supplied Dubbing AI SDK.

No installer/driver is run, no login or audio is supplied. Output is metadata,
not a memory dump. The CI workflow blocks network access for the Python host.
"""

import argparse
import ctypes
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time


def inspect_loaded(path, initialization=None):
    from ctypes import wintypes as w

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.LoadLibraryExW.argtypes = [w.LPCWSTR, w.HANDLE, w.DWORD]
    kernel.LoadLibraryExW.restype = w.HMODULE
    kernel.ReadProcessMemory.argtypes = [w.HANDLE, w.LPCVOID, w.LPVOID, ctypes.c_size_t,
                                        ctypes.POINTER(ctypes.c_size_t)]
    kernel.ReadProcessMemory.restype = w.BOOL
    process = kernel.GetCurrentProcess()
    directory = os.add_dll_directory(str(path.parent))
    module = kernel.LoadLibraryExW(str(path), None, 0x100 | 0x1000)
    if not module:
        error = ctypes.get_last_error()
        return {"name": path.name, "status": "load_failed", "winerror": error}

    class ModuleInfo(ctypes.Structure):
        _fields_ = [("base", w.LPVOID), ("size", w.DWORD), ("entry", w.LPVOID)]

    psapi.GetModuleInformation.argtypes = [w.HANDLE, w.HMODULE, ctypes.POINTER(ModuleInfo), w.DWORD]
    info = ModuleInfo()
    if not psapi.GetModuleInformation(process, module, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not 0 < info.size < 128 * 1024 * 1024:
        raise ValueError("Unexpected module image size")
    snapshot = bytearray(info.size)
    readable = 0
    for offset in range(0, info.size, 4096):
        size = min(4096, info.size - offset)
        buffer = ctypes.create_string_buffer(size)
        received = ctypes.c_size_t()
        kernel.ReadProcessMemory(process, info.base + offset, buffer, size, ctypes.byref(received))
        count = min(received.value, size)
        snapshot[offset:offset + count] = buffer.raw[:count]
        readable += count
    data = bytes(snapshot)
    lowered = data.lower()
    markers = ("onnxruntime", "directml", "tensorrt", "cudnn", "ncnn", "tensorflow", "pytorch",
               "beatrice", "llvc", "rmvpe", "hifigan", "vocoder", "contentvec", "hubert", "gemm",
               "conv1d", "conv2d", "speaker_embedding", "causal", "int8", "fp16", "sse2", "avx2")
    matches = {name: len(re.findall(rb"(?<![a-z])" + name.encode() + rb"(?![a-z])", lowered))
               for name in markers}
    matches = {name: count for name, count in matches.items() if count}
    classes = sorted(set(raw.decode("ascii") for raw in re.findall(rb"\.\?AV[A-Za-z0-9_?$@]{3,160}@@", data)
                         if re.search(rb"model|engine|neural|vocoder|encoder|decoder|conv|tensor", raw, re.I)))[:60]
    result = {"name": path.name, "status": "loaded", "image_bytes": info.size, "readable_bytes": readable,
              "memory_sha256": hashlib.sha256(data).hexdigest(), "backend_markers": matches, "classes": classes}
    # A named, zero-argument public factory; no virtual methods or login calls.
    if path.name == "dubbing-sdk-windows.dll":
        kernel.GetProcAddress.argtypes = [w.HMODULE, ctypes.c_char_p]
        kernel.GetProcAddress.restype = w.LPVOID
        address = kernel.GetProcAddress(module, b"?V_@IVoiceTranscribe@DUBBING@@SAPEAV12@XZ")
        result["factory_export_present"] = bool(address)
        if address:
            factory = ctypes.CFUNCTYPE(ctypes.c_void_p)(address)
            instance = factory()
            result["factory_returned_instance"] = bool(instance)
            if instance:
                pointer = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_void_p))
                received = ctypes.c_size_t()
                if kernel.ReadProcessMemory(process, instance, pointer, len(pointer), ctypes.byref(received)):
                    table = struct.unpack("<Q", pointer.raw)[0]
                    result["factory_vtable_in_sdk"] = info.base <= table < info.base + info.size
                    if initialization is not None:
                        # Signature recovered from DubbingAI.exe RVA 0x1c6d48:
                        # self, binary-files JSON, speaker-files JSON, input Hz, output Hz.
                        data_dir = path.parent / "vc_model"
                        files = sorted((p for p in data_dir.iterdir() if p.is_file()), key=lambda p: p.name)
                        large = [p for p in files if p.stat().st_size > 8192]
                        small = [p for p in files if p.stat().st_size <= 8192]
                        if len(large) != 3 or not small:
                            raise ValueError("Unexpected reference model inventory")
                        binaries, speakers = [], []
                        if initialization != "empty":
                            order = [int(value) for value in initialization]
                            binaries = [{"id": index + 1, "bid": index + 1, "localPath": str(large[value])}
                                        for index, value in enumerate(order)]
                            speakers = [{"id": 1, "binFileIds": [1, 2, 3], "localPath": str(small[0])}]
                        function_pointer = ctypes.create_string_buffer(8)
                        if not kernel.ReadProcessMemory(process, table + 8, function_pointer, 8, ctypes.byref(received)):
                            raise ctypes.WinError(ctypes.get_last_error())
                        initialize = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p,
                                                       ctypes.c_char_p, ctypes.c_int, ctypes.c_int)(
                                                           struct.unpack("<Q", function_pointer.raw)[0])
                        started = time.perf_counter()
                        result["initialize_code"] = initialize(instance, json.dumps(binaries).encode(),
                                                                json.dumps(speakers).encode(), 48000, 48000)
                        result["initialize_ms"] = round((time.perf_counter() - started) * 1000)
                        result["initialization_case"] = initialization
    directory.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path)
    parser.add_argument("--output", type=Path, default=Path("sdk-probe.json"))
    parser.add_argument("--child", type=Path)
    parser.add_argument("--initialization")
    parser.add_argument("--probe-initialization", action="store_true")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("The loader probe requires Windows")
    if args.child:
        print(json.dumps(inspect_loaded(args.child.resolve(), args.initialization)), flush=True)
        return
    environment = {key: value for key, value in os.environ.items()
                   if not any(word in key.upper() for word in ("TOKEN", "SECRET", "PASSWORD"))}
    results = []
    cases = [("dubbing-base.dll", None), ("dubbing-sdk-windows.dll", None)]
    if args.probe_initialization:
        cases += [("dubbing-sdk-windows.dll", "empty")]
        cases += [("dubbing-sdk-windows.dll", "".join(map(str, order))) for order in itertools.permutations(range(3))]
    for name, initialization in cases:
        try:
            command = [sys.executable, str(Path(__file__).resolve()), "--child", str((args.app / name).resolve())]
            if initialization is not None:
                command.extend(["--initialization", initialization])
            child = subprocess.run(command, env=environment,
                                   capture_output=True, text=True, timeout=45)
            if child.returncode:
                result = {"name": name, "status": "child_failed", "exit_code": child.returncode,
                          "stderr": child.stderr[-1200:]}
            else:
                result = json.loads(child.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            result = {"name": name, "status": "timeout", "timeout_seconds": 45}
        if initialization is not None:
            result["initialization_case"] = initialization
        results.append(result)
        print("::warning title=Native SDK probe::" + json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"mode": "public-sdk-api-probe", "results": results}, indent=2),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
