"""Inspect anonymous public bootstrap metadata; export no account information."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request


HOSTS = ("a1.dubbingai.io", "a2.dubbingai.io", "a3.dubbingai.io",
         "a1.dubbing-ai.com", "a2.dubbing-ai.com", "a3.dubbing-ai.com")


def inspect(url):
    result = {"url": url}
    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            result.update(status=response.status, bytes=len(raw), content_type=response.headers.get("Content-Type"))
        if len(raw) > 8 * 1024 * 1024:
            return result
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            result["format"] = "non-JSON"
            return result
        if isinstance(data, dict) and data.get("data") is None:
            result["api_code"] = data.get("code")
            result["api_message"] = str(data.get("msg", ""))[:200]
        result["shape"] = list(data)[:100] if isinstance(data, dict) else {"type": type(data).__name__, "length": len(data)}
        urls, records = set(), []

        def walk(value, inherited=""):
            if isinstance(value, dict):
                name = value.get("name") or value.get("displayName") or value.get("speakerName") or inherited
                if isinstance(name, str) and name:
                    fields = ("id", "name", "displayName", "localFileName", "fileName", "url", "modelUrl", "speakerId", "groupModelId", "binFileIds", "fileKey", "localPath")
                    record = {key: value[key] for key in fields if key in value and isinstance(value[key], (str, int, list))}
                    record["name"] = name
                    records.append(record)
                for key, item in value.items():
                    if re.search(r"token|password|secret|authorization|cookie", key, re.I):
                        continue
                    walk(item, name if isinstance(name, str) else inherited)
            elif isinstance(value, list):
                for item in value:
                    walk(item, inherited)
            elif isinstance(value, str) and value.startswith("https://"):
                parts = urllib.parse.urlsplit(value)
                urls.add(urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")))

        walk(data)
        result.update(public_urls=sorted(urls)[:150], voice_records=records)
    except urllib.error.HTTPError as exc:
        result["status"] = exc.code
    except (urllib.error.URLError, TimeoutError) as exc:
        result["error"] = type(exc).__name__
    return result


if __name__ == "__main__":
    targets = ["https://d" + number + ".dubbing-ai.com/rvc-client/assets/config.json" for number in ("1", "2", "6")]
    targets += ["https://" + host + "/api/rvc/getAllSpeakers" for host in ("console.dubbing-ai.com", "www.dubbing-ai.com")]
    targets += ["https://console.dubbing-ai.com/api/rvc/getAliasSpeakers", "https://console.dubbing-ai.com/api/rvc/getSquareSpeakers"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(inspect, targets))
    Path("public-native-catalog.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in results:
        summary = {key: value for key, value in item.items() if key != "voice_records"}
        summary["voice_records"] = len(item.get("voice_records", []))
        print("::notice title=Public native catalogue::" + json.dumps(summary), flush=True)
