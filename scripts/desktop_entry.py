"""PyInstaller entry point; the source entry point is python -m voicelab."""

from voicelab.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
