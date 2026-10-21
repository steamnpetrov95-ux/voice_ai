"""Inspect upstream RVC model metadata on the native CI network, without audio or credentials."""

import json
import os
import urllib.request
from urllib.parse import quote


def read_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def notice(title, value):
    message = json.dumps(value, ensure_ascii=True).replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")
    print(f"::notice title={title}::{message}", flush=True)


sources = json.loads(os.environ.get("MODEL_SOURCES", "[]") or "[]")
queries = json.loads(os.environ.get("MODEL_SEARCHES", "[]") or "[]")
if queries:
    for query in queries[:8]:
        found = read_json(f"https://huggingface.co/api/models?search={quote(query)}&limit=60")
        notice("Search " + query, [item["id"] for item in found if "dialogpt" not in item["id"].lower()
                                   and not set(item.get("tags", [])) & {"text-to-image", "text-generation"}])
    raise SystemExit(0)
if sources:
    for repository in sources[:8]:
        info = read_json(f"https://huggingface.co/api/models/{repository}?blobs=true")
        relevant = [item for item in info["siblings"] if any(word in item["rfilename"].casefold() for word in
                    ("deadpool", "дэдпул", "бэтмен", "бетмен", "batman", "loki", "локи", "танос", "thanos",
                     "tony", "тони", "старк", "желез", "iron", "гаврил", "мясник", "зайцев", "thor"))]
        notice("Voice source " + repository, {
            "id": info["id"], "sha": info["sha"], "tags": info.get("tags", []),
            "cardData": info.get("cardData"),
            "siblings": [{"name": item["rfilename"], "size": item.get("size"), "lfs": item.get("lfs")}
                         for item in (relevant or info["siblings"][:12])[:18]],
        })
    raise SystemExit(0)

catalogue = read_json("https://huggingface.co/wok000/vcclient_model/raw/main/samples_0004_o.json")
notice("RVC catalogue", catalogue)
for repository in ("NaruseMioShirakana/MoeSS-SUBModel", "wok000/weights_gpl"):
    info = read_json(f"https://huggingface.co/api/models/{repository}?blobs=true")
    notice("RVC common assets", {
        "id": info["id"], "sha": info["sha"], "cardData": info.get("cardData"),
        "siblings": [s for s in info["siblings"] if any(word in s["rfilename"] for word in ("vec-", "rmvpe", "README"))],
    })
searches = {}
for query in ("Deadpool", "Batman", "Loki", "Thanos", "IronMan", "Gavrilin", "TonyStark", "rvc russian", "rvc ru"):
    matches = read_json(f"https://huggingface.co/api/models?search={quote(query)}&limit=30")
    searches[query] = [item["id"] for item in matches]
for query, matches in searches.items():
    notice("Voice search " + query, matches)
