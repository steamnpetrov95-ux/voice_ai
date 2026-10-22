import pytest


@pytest.fixture(autouse=True)
def isolate_native_library(monkeypatch, tmp_path):
    # A developer's personal model pack must not change the test library.
    monkeypatch.setenv("VOICELAB_NATIVE_DIR", str(tmp_path / "no-native-pack"))
