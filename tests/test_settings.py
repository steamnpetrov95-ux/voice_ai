from voicelab.settings import data_dir, load_preferences, save_preferences


def test_preferences_roundtrip_and_corruption(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICELAB_DATA_DIR", str(tmp_path / "prefs"))
    assert load_preferences() == {}
    save_preferences({"preset": "sakura", "input": "Микрофон"})
    assert load_preferences()["input"] == "Микрофон"
    (data_dir() / "settings.json").write_text("invalid", encoding="utf-8")
    assert load_preferences() == {}
