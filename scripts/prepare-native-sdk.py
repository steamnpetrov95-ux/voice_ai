"""Prepare a personal native pack from the exact installer supplied by its user."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil


COMMON_FILES = (
    ("dd9f3ba78e223d341fab75a7767b10a4", "f925caca58166966758d97a5ee39e7a3cab231dbdd2fcd52aa2d011e84859440"),
    ("d136725af877d2b05083e4dd2867af71", "53a7aaffb008e795db1ecc9b2c1f4e6213af8a70a61e83dfa1d51ed1fc088e1d"),
    ("6f51031213cc38a90992b69e7ec433c4", "64f094ed7c482dc5a8207c8ba5447a3c85f018e19da956b6c825b58e555a434d"),
)


def prepare(source, destination, catalog=None):
    source, destination = source.resolve(), destination.resolve()
    if source.is_relative_to(destination):
        raise ValueError("The output must not contain the source directory")
    if destination.exists():
        if any(destination.iterdir()) and not (destination / "manifest.json").is_file():
            raise ValueError("Refusing to replace an unrelated directory")
        shutil.rmtree(destination)
    (destination / "sdk").mkdir(parents=True)
    (destination / "vc_model").mkdir()
    checksums, binaries, voices = [], [], []

    def copy(path, relative):
        target = destination / relative
        shutil.copyfile(path, target)
        checksums.append({"file": relative, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})

    for path in sorted(source.glob("*.dll")):
        copy(path, "sdk/" + path.name)
    licenses = destination / "sdk" / "licenses"
    licenses.mkdir()
    for path in source.iterdir():
        if path.is_file() and ("LICENSE" in path.name.upper() or "NOTICE" in path.name.upper()):
            shutil.copyfile(path, licenses / path.name)
    if (source / "csorigin.config").is_file():
        copy(source / "csorigin.config", "sdk/csorigin.config")
    if not (destination / "sdk/dubbing-sdk-windows.dll").is_file():
        raise ValueError("Native SDK is missing")
    common = {name for name, _ in COMMON_FILES}
    for number, (name, expected) in enumerate(COMMON_FILES, 1):
        path = source / "vc_model" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Unexpected common model: " + name)
        copy(path, "vc_model/" + name)
        binaries.append({"id": number, "bid": number, "file": "vc_model/" + name})
    names = json.loads(catalog.read_text(encoding="utf-8")) if catalog and catalog.is_file() else {}
    for path in sorted((source / "vc_model").iterdir()):
        if not path.is_file() or path.name in common:
            continue
        if len(path.name) != 32 or not 0 < path.stat().st_size <= 8192:
            raise ValueError("Unexpected voice profile: " + path.name)
        base64.b64decode(path.read_bytes(), validate=True)
        number = len(voices) + 1
        copy(path, "vc_model/" + path.name)
        voices.append({"id": number, "key": path.name, "file": "vc_model/" + path.name,
                       "name": names.get(path.name, f"AI-голос {number:04d}")})
    if len(voices) != 1241:
        raise ValueError("Unexpected native voice count")
    manifest = {"schema": 1, "sdk_version": "Dubbing AI 1.8.2", "source_installer_sha256":
                "af7ea3aa9c743473d0cb927e9d388da192af64681870b3c2aedc11a7e987f231",
                "binaries": binaries, "voices": voices, "checksums": checksums}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "SOURCE.txt").write_text("Native Dubbing AI SDK and profiles supplied by the user. "
                                          "Personal compatibility pack; the engine is not an independent reimplementation.\n",
                                          encoding="utf-8")
    print(f"Prepared {len(voices)} personal native profiles in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist/native-sdk"))
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output, args.catalog)
