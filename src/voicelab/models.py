"""Atomic local voice storage and verified model downloads. Never runs Python checkpoints."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import quote, unquote, urlparse
import urllib.request
import uuid

from .neural import NeuralVoice, _session, voice_metadata
from .settings import data_dir


@dataclass(frozen=True)
class ModelAsset:
    url: str
    sha256: str
    size: int
    filename: str


COMMON = {
    256: ModelAsset(
        "https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/resolve/3ff2e37e161ebcb3c4711e7ec870472342260842/vec-256-layer-9.onnx",
        "61d0d0598803f74d5a4bcba65ca367571889accc0e86196f0982e8e455d56393", 293348935, "vec-256.onnx"),
    768: ModelAsset(
        "https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/resolve/3ff2e37e161ebcb3c4711e7ec870472342260842/vec-768-layer-12.onnx",
        "b3886e7dff1495cda514f94f4680a7b1261e05d6929f5c764cdb17934b413c2a", 377655729, "vec-768.onnx"),
    "pitch": ModelAsset(
        "https://huggingface.co/wok000/weights_gpl/resolve/d1b578cfcdf944dd3910786f7604754c79234b36/rmvpe/rmvpe_20231006.onnx",
        "84f0586308e36157f75b77c8591bf636d6719c0c4ba95f8faf3df479e7566219", 362003174, "rmvpe.onnx"),
}


class DownloadCancelled(Exception):
    pass


def _hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(asset, destination, progress=lambda message, percent: None, cancel=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == asset.size and _hash(destination) == asset.sha256:
        return
    temporary = destination.with_name(destination.name + ".part-" + uuid.uuid4().hex)
    try:
        digest, received = hashlib.sha256(), 0
        request = urllib.request.Request(asset.url, headers={"User-Agent": "npetrov-VoiceLab/0.2"})
        with urllib.request.urlopen(request, timeout=45) as response, temporary.open("xb") as output:
            while True:
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled()
                block = response.read(512 * 1024)
                if not block:
                    break
                received += len(block)
                if received > asset.size:
                    raise ValueError("Размер скачанного файла не соответствует модели.")
                output.write(block)
                digest.update(block)
                progress(f"{asset.filename}: {received // 1048576} / {asset.size // 1048576} МБ", int(received * 100 / asset.size))
        if received != asset.size or digest.hexdigest() != asset.sha256:
            raise ValueError("Проверка SHA256 модели не прошла. Повторите скачивание.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(4 * 1024 * 1024 + 1)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("Ответ каталога слишком большой.")
    return json.loads(data)


def resolve_huggingface(url):
    parsed = urlparse(url)
    parts = unquote(parsed.path).strip("/").split("/")
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co" or len(parts) < 5 or parts[2] != "resolve":
        raise ValueError("Нужна HTTPS-ссылка Hugging Face вида /owner/repo/resolve/main/model.onnx.")
    owner, repository, _, revision, *segments = parts
    filename = "/".join(segments)
    if not filename.lower().endswith(".onnx") or ".." in parts:
        raise ValueError("Нужна модель ONNX.")
    info = read_json(f"https://huggingface.co/api/models/{quote(owner)}/{quote(repository)}/revision/{quote(revision, safe='')}?blobs=true")
    sibling = next((item for item in info.get("siblings", []) if item.get("rfilename") == filename), None)
    if not sibling or not sibling.get("lfs") or not re.fullmatch(r"[0-9a-f]{64}", sibling["lfs"].get("sha256", "")):
        raise ValueError("В источнике нет SHA256 для выбранной модели.")
    size = sibling["lfs"]["size"]
    if not 0 < size <= 1500 * 1024 * 1024:
        raise ValueError("Эта модель слишком велика для данного режима.")
    commit = info["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Некорректная версия модели в источнике.")
    pinned = f"https://huggingface.co/{quote(owner)}/{quote(repository)}/resolve/{commit}/{quote(filename, safe='/')}"
    return ModelAsset(pinned, sibling["lfs"]["sha256"], size, Path(filename).name)


def voice_directory():
    return data_dir() / "models"


def _builtin_directory():
    return (Path(sys.executable).resolve().parent / "models" if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parents[2] / "dist" / "neural-models")


def ensure_common(asset, root, progress, cancel):
    destination = root / "common" / asset.filename
    builtin = _builtin_directory() / "common" / asset.filename
    if (not destination.exists() and builtin.is_file() and builtin.stat().st_size == asset.size
            and _hash(builtin) == asset.sha256):
        if cancel is not None and cancel.is_set():
            raise DownloadCancelled()
        progress("Подготовка общей модели из установленного приложения…", 0)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part-" + uuid.uuid4().hex)
        try:
            try:
                os.link(builtin, temporary)
            except OSError:
                shutil.copyfile(builtin, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return
    download(asset, destination, progress, cancel)


def _contained(root, relative):
    if not isinstance(relative, str):
        raise ValueError("Некорректный путь модели.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Файл модели отсутствует или находится вне библиотеки.")
    return path


def installed_voices():
    builtin = _builtin_directory()
    voices = {}
    priorities = {}
    for path in sorted(builtin.glob("voices/*/voice.json")) + sorted(voice_directory().glob("voices/*/voice.json")):
        root = path.parents[2]
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(value, dict) or not isinstance(value.get("name"), str)
                    or not re.fullmatch(r"[0-9a-f]{24}", str(value.get("id", "")))
                    or value.get("sample_rate") not in (32000, 40000, 48000)
                    or value.get("channels") not in (256, 768)):
                continue
            voices[value["id"]] = NeuralVoice(
                id=value["id"], name=value["name"], model=_contained(root, value["model"]),
                encoder=_contained(root, value["encoder"]), pitch=_contained(root, value["pitch"]),
                sample_rate=value["sample_rate"], channels=value["channels"],
                credit=str(value.get("credit", "")), terms_url=str(value.get("terms_url", "")),
                decoder=_contained(root, value["decoder"]) if value.get("decoder") else None,
            )
            priorities[value["id"]] = value.get("priority", 100) if isinstance(value.get("priority", 100), int) else 100
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(voices.values(), key=lambda voice: (priorities[voice.id], voice.name))


def install_voice(source, name, progress=lambda message, percent: None, cancel=None, credit="", terms_url=""):
    root = voice_directory()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=root) as staging:
        staged = Path(staging) / "voice.onnx"
        if str(source).startswith("https://"):
            asset = resolve_huggingface(str(source))
            download(asset, staged, progress, cancel)
            identity = asset.sha256
        else:
            source = Path(source)
            if source.suffix.lower() != ".onnx":
                raise ValueError("Поддерживаются модели .onnx. Файл .pth нужно сначала экспортировать в RVC ONNX.")
            if source.stat().st_size > 1500 * 1024 * 1024:
                raise ValueError("Модель слишком велика.")
            shutil.copyfile(source, staged)
            identity = _hash(staged)
        if cancel is not None and cancel.is_set():
            raise DownloadCancelled()
        progress("Проверка голосовой модели…", 0)
        session = _session(staged, "cpu")
        rate, channels = voice_metadata(session)
        del session
        for key in (channels, "pitch"):
            ensure_common(COMMON[key], root, progress, cancel)
        if cancel is not None and cancel.is_set():
            raise DownloadCancelled()
        identifier = identity[:24]
        destination = root / "voices" / identifier
        destination.mkdir(parents=True, exist_ok=True)
        staged.replace(destination / "voice.onnx")
        metadata = {"id": identifier, "name": str(name).strip()[:120] or "AI-голос",
                    "model": f"voices/{identifier}/voice.onnx",
                    "encoder": "common/" + COMMON[channels].filename,
                    "pitch": "common/" + COMMON["pitch"].filename,
                    "sample_rate": rate, "channels": channels, "sha256": identity,
                    "credit": str(credit)[:300], "terms_url": str(terms_url)[:2000]}
        temporary = destination / ("voice-" + uuid.uuid4().hex + ".json")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination / "voice.json")
    return next(voice for voice in installed_voices() if voice.id == identifier)
