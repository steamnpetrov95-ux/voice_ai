import argparse
import json

from . import APP_NAME
from .presets import PRESET_BY_ID


def main():
    parser = argparse.ArgumentParser(description=f"{APP_NAME} — локальные эффекты голоса")
    parser.add_argument("--render", nargs=2, metavar=("INPUT", "OUTPUT"), help="Обработать аудиофайл в WAV")
    parser.add_argument("--preset", choices=PRESET_BY_ID, default="clean")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--list-voices", action="store_true", help="Показать установленные нейросетевые голоса")
    parser.add_argument("--install-voice", metavar="MODEL", help="Импортировать RVC ONNX: файл или ссылка Hugging Face")
    parser.add_argument("--voice-name", default="AI-голос")
    parser.add_argument("--ai-voice", metavar="ID", help="Использовать AI-модель при --render")
    parser.add_argument("--ai-provider", choices=("auto", "cpu", "dml"), default="auto")
    args = parser.parse_args()
    if args.render:
        from .effects import render_file
        try:
            voice = None
            if args.ai_voice:
                from .models import installed_voices
                voice = next((item for item in installed_voices() if item.id == args.ai_voice), None)
                if voice is None:
                    raise ValueError("AI-голос не найден. Используйте --list-voices.")
            result = render_file(*args.render, PRESET_BY_ID[args.preset].settings, voice, args.ai_provider)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"Ошибка: {exc}\n")
        print(json.dumps(result))
    elif args.install_voice:
        from .models import install_voice
        try:
            voice = install_voice(args.install_voice, args.voice_name)
        except Exception as exc:
            parser.exit(1, f"Ошибка установки AI-голоса: {exc}\n")
        print(json.dumps({"id": voice.id, "name": voice.name}, ensure_ascii=False))
    elif args.list_voices:
        from .models import installed_voices
        print(json.dumps([{"id": voice.id, "name": voice.name} for voice in installed_voices()], ensure_ascii=False))
    elif args.list_devices:
        from .audio import list_devices
        inputs, outputs = list_devices()
        for name, devices in (("Микрофоны", inputs), ("Выходы", outputs)):
            print(name + ":")
            for device in devices:
                print(f"  {device.index}: {device.label}")
    else:
        from .app import run
        return run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
