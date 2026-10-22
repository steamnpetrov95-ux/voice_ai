"""Try anonymous, read-only catalogue URLs; save only display names and file keys."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    keys = {p.name for p in args.profiles.iterdir() if p.is_file() and p.stat().st_size <= 8192}
    names = {}

    def walk(value, inherited=""):
        if isinstance(value, list):
            for item in value:
                walk(item, inherited)
        elif isinstance(value, dict):
            name = value.get("name", inherited)
            if not isinstance(name, str) or not 0 < len(name) <= 120 or any(c in name for c in "<>\r\n"):
                name = inherited
            if name:
                for field in ("localFileName", "localPath", "url", "fileName", "modelUrl"):
                    source = value.get(field)
                    if not isinstance(source, str):
                        continue
                    candidates = re.findall(r"[0-9a-f]{32}", source.lower())
                    candidates.append(hashlib.md5(source.encode()).hexdigest())
                    for key in candidates:
                        if key in keys:
                            names[key] = name
            for item in value.values():
                if isinstance(item, (list, dict)):
                    walk(item, name)

    for host in ("a1.dubbingai.io", "a2.dubbingai.io", "www.dubbingai.io"):
        for path in ("/api/rvc/getAllSpeakers", "/rvc-client/assets/config.json"):
            url = "https://" + host + path
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "npetrov compatibility research"})
                with urllib.request.urlopen(request, timeout=12) as response:
                    raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    continue
                walk(json.loads(raw))
                print(f"Catalogue source {host}{path}: {len(names)} matching names", flush=True)
            except (urllib.error.URLError, ValueError, TimeoutError):
                print(f"Catalogue source {host}{path}: unavailable anonymously", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"::notice title=Native catalogue::Recovered {len(names)} display names without account data.")


if __name__ == "__main__":
    main()
