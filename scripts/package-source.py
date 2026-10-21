"""Create a portable source archive without caches, recordings or local settings."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parent.parent
target = root / "dist" / "VoiceLab-Windows-source.zip"
target.parent.mkdir(exist_ok=True)
files = [root / name for name in (
    "README.md", ".gitignore", "pyproject.toml", "requirements.lock", "uv.lock",
    "Start-VoiceLab.bat", "Build-Windows.bat",
)]
for folder in ("src", "tests", "scripts", "installer", "docs", ".github"):
    files.extend(path for path in (root / folder).rglob("*")
                 if path.is_file() and "__pycache__" not in path.parts)
with ZipFile(target, "w", ZIP_DEFLATED) as archive:
    for path in sorted(files):
        content = path.read_bytes()
        if path.suffix == ".bat":
            content = content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        archive.writestr("VoiceLab/" + path.relative_to(root).as_posix(), content)
print(target)
