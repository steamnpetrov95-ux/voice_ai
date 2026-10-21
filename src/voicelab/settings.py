"""Non-secret preferences stored outside the installation directory."""

import json
import os
from pathlib import Path


def data_dir() -> Path:
    override = os.environ.get("VOICELAB_DATA_DIR")
    if override:
        return Path(override)
    base = os.environ.get("APPDATA") if os.name == "nt" else os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "VoiceLab"


def load_preferences() -> dict:
    try:
        value = json.loads((data_dir() / "settings.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_preferences(value: dict):
    directory = data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "settings.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
