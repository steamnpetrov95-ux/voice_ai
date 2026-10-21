import argparse
import json

from .presets import PRESET_BY_ID


def main():
    parser = argparse.ArgumentParser(description="VoiceLab — локальные эффекты голоса")
    parser.add_argument("--render", nargs=2, metavar=("INPUT", "OUTPUT"), help="Обработать аудиофайл в WAV")
    parser.add_argument("--preset", choices=PRESET_BY_ID, default="clean")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if args.render:
        from .effects import render_file
        try:
            result = render_file(*args.render, PRESET_BY_ID[args.preset].settings)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"Ошибка: {exc}\n")
        print(json.dumps(result))
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
