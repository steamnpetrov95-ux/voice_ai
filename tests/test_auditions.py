import base64
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

spec = importlib.util.spec_from_file_location("dnd_auditions", Path(__file__).resolve().parents[1] / "scripts/make-dnd-auditions.py")
auditions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditions)


def test_diversity_selects_examples_from_different_acoustic_groups():
    groups = np.array([[0, 0], [0.1, 0], [10, 0], [10.1, 0], [0, 10], [0, 10.1]])
    selected = auditions.choose_diverse(groups, 3)
    assert {i // 2 for i in selected} == {0, 1, 2}
    assert selected == auditions.choose_diverse(groups, 3)
    assert len(set(auditions.choose_diverse(np.zeros((5, 3)), 3))) == 3


def test_render_uses_unmodified_native_output_and_flushes_tail():
    source = np.linspace(-0.2, 0.2, 17003, dtype=np.float32)
    calls = []

    class Converter:
        def __init__(self, voice, rate):
            assert voice == "test" and rate == 48000

        def process(self, block, pitch):
            assert len(block) == 3840 and pitch == 0
            return -block

        def close(self):
            calls.append("closed")

    output = auditions.render_voice(source, "test", Converter)
    np.testing.assert_array_equal(output[:len(source)], -source)
    assert len(output) == len(source) + 30720
    assert not output[len(source):].any()
    assert calls == ["closed"]


def test_reference_fingerprint_rejects_silence_and_tracks_pitch_difference():
    t = np.arange(48000) / 48000
    low = (np.sin(2 * np.pi * 110 * t) * 0.1).astype(np.float32)
    high = (np.sin(2 * np.pi * 330 * t) * 0.1).astype(np.float32)
    assert auditions.fingerprint(high)[-2] > auditions.fingerprint(low)[-2]
    with pytest.raises(ValueError, match="speech"):
        auditions.fingerprint(np.zeros(48000, np.float32))


def test_page_contains_playable_inline_audio_and_safe_catalogue_text(tmp_path):
    source = np.sin(2 * np.pi * 180 * np.arange(48000) / 48000).astype(np.float32) * 0.1
    url = auditions.wav_url(source)
    audio, rate = sf.read(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert rate == 16000 and len(audio) == 16000
    assert 0.01 < np.max(np.abs(audio)) <= 0.88
    name = '</script><script>alert("x")</script>'
    page = tmp_path / "auditions.html"
    auditions.write_page(page, source, [{"id": "native-id", "name": name, "audio": url}], "Русская реплика")
    html = page.read_text(encoding="utf-8")
    assert name not in html
    embedded = html.split('<script type="application/json" id="data">', 1)[1].split('</script>', 1)[0]
    data = json.loads(embedded)
    assert data["voices"][0]["name"] == name
    assert data["original"].startswith("data:audio/wav;base64,")
    assert "__AUDITION_DATA__" not in html
