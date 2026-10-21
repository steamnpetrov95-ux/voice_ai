"""Inspect upstream RVC model metadata on the native CI network, without audio or credentials."""

import json
import urllib.request
from urllib.parse import quote


def read_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def notice(title, value):
    message = json.dumps(value, ensure_ascii=True).replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")
    print(f"::notice title={title}::{message}", flush=True)


catalogue = read_json("https://huggingface.co/wok000/vcclient_model/raw/main/samples_0004_o.json")
notice("RVC catalogue", catalogue)
for repository in ("NaruseMioShirakana/MoeSS-SUBModel", "wok000/weights_gpl"):
    info = read_json(f"https://huggingface.co/api/models/{repository}?blobs=true")
    notice("RVC common assets", {
        "id": info["id"], "sha": info["sha"], "cardData": info.get("cardData"),
        "siblings": [s for s in info["siblings"] if any(word in s["rfilename"] for word in ("vec-", "rmvpe", "README"))],
    })
for query in ("Тони", "Старк", "Локи", "Гаврилин", "Дэдпул", "Бэтмен", "Танос", "Deadpool", "Batman", "Loki", "Thanos", "rvc russian"):
    matches = read_json(f"https://huggingface.co/api/models?search={quote(query)}&limit=15")
    notice("Voice search " + query, [{"id": item["id"], "tags": item.get("tags", [])} for item in matches])
