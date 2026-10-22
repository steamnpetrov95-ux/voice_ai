"""Surface native-process failures in CI without relying on downloadable logs."""

import json
from pathlib import Path
import subprocess
import sys


result = subprocess.run(
    [sys.executable, "-X", "faulthandler", str(Path(__file__).with_name("smoke-native-sdk.py")), *sys.argv[1:]],
    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240,
)
print(result.stdout, flush=True)
if result.stderr:
    print(result.stderr, file=sys.stderr, flush=True)
if result.returncode:
    diagnostic = {"exit_code": result.returncode, "stdout": result.stdout[-5000:], "stderr": result.stderr[-5000:]}
    print("::error title=Native subprocess failure::" + json.dumps(diagnostic), flush=True)
raise SystemExit(0 if result.returncode == 0 else 1)
