import numpy as np
import pytest
import soundfile as sf

from voicelab.effects import VoiceProcessor, render_file
from voicelab.pitch import QuickPitchShifter, StreamingPitchShifter
from voicelab.presets import CARTOON_PRESETS, PRESETS, VoiceSettings

RATE = 48000


def tone(frequency=220, seconds=0.5):
    return (0.25 * np.sin(2 * np.pi * frequency * np.arange(int(RATE * seconds)) / RATE)).astype(np.float32)


def stream(processor, samples):
    return np.concatenate([processor.process(samples[i:i + 1024]) for i in range(0, len(samples), 1024)])


@pytest.mark.parametrize("preset", PRESETS, ids=lambda preset: preset.id)
def test_each_preset_processes_streaming_audio(preset):
    rng = np.random.default_rng(13)
    samples = tone() + rng.normal(0, 0.03, int(RATE * 0.5)).astype(np.float32)
    result = stream(VoiceProcessor(RATE, preset.settings), samples)
    assert result.shape == samples.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) <= 1
    assert np.max(np.abs(result)) > 0.01


def test_bypass_preserves_samples_even_with_effects_selected():
    samples = tone()
    result = stream(VoiceProcessor(RATE, VoiceSettings(pitch=-8, robot=1, room=0.8, enabled=False)), samples)
    np.testing.assert_array_equal(result, samples)


def test_clean_preset_has_no_effects_or_added_tail(tmp_path):
    from voicelab.presets import PRESET_BY_ID
    samples = tone()
    processor = VoiceProcessor(RATE, PRESET_BY_ID["clean"].settings)
    assert processor.pitch_shifter is None and not len(processor.board)
    np.testing.assert_array_equal(stream(processor, samples), samples)
    source, target = tmp_path / "source.wav", tmp_path / "clean.wav"
    sf.write(source, samples, RATE)
    render_file(source, target, PRESET_BY_ID["clean"].settings)
    assert sf.info(target).frames == sf.info(source).frames


def test_pitch_changes_fundamental_frequency():
    samples = tone(220, 2)
    result = stream(VoiceProcessor(RATE, VoiceSettings(pitch=12, clarity=0)), samples)
    stable = result[RATE:]
    spectrum = np.abs(np.fft.rfft(stable * np.hanning(len(stable))))
    frequency = np.fft.rfftfreq(len(stable), 1 / RATE)[np.argmax(spectrum)]
    assert frequency == pytest.approx(440, abs=5)


def test_robot_produces_modulation_sidebands_and_continuity():
    samples = tone(220, 1)
    settings = VoiceSettings(robot=1, clarity=0)
    output = stream(VoiceProcessor(RATE, settings), samples)
    whole = VoiceProcessor(RATE, settings).process(samples)
    np.testing.assert_allclose(output, whole, atol=1e-6)
    spectrum = np.abs(np.fft.rfft(output))
    peaks = np.argsort(spectrum)[-2:]
    assert set(peaks) == {145, 295}


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("pitch", [-10, 9])
def test_quick_pitch_is_causal_continuous_and_bounded(rate, pitch):
    samples = np.random.default_rng(42).normal(0, 0.1, rate).astype(np.float32)
    processor = QuickPitchShifter(pitch, rate)
    blocks = [processor.process(samples[i:i + 229]) for i in range(0, len(samples), 229)]
    whole = QuickPitchShifter(pitch, rate).process(samples)
    np.testing.assert_allclose(np.concatenate(blocks), whole, atol=1e-6)
    assert len(processor.history) <= rate * 0.026
    # An impulse must emerge inside the stated delay bound, not a full AI block.
    impulse = np.zeros(rate // 10, np.float32)
    impulse[0] = 1
    result = QuickPitchShifter(pitch, rate).process(impulse)
    nonzero = np.flatnonzero(np.abs(result) > 1e-6)
    assert len(nonzero) and nonzero[-1] < rate * 0.026


@pytest.mark.parametrize("pitch", [-12, 12])
def test_quick_pitch_changes_frequency_without_a_vocoder(pitch):
    result = stream(VoiceProcessor(RATE, VoiceSettings(pitch=pitch, clarity=0, quick_pitch=True)), tone(500, 2))
    stable = result[RATE:]
    spectrum = np.abs(np.fft.rfft(stable * np.hanning(len(stable))))
    frequency = np.fft.rfftfreq(len(stable), 1 / RATE)[np.argmax(spectrum)]
    # Short moving delays deliberately introduce some granular sidebands.
    assert frequency == pytest.approx(500 * 2 ** (pitch / 12), abs=25)


def test_cartoon_styles_have_short_pitch_and_no_long_effect_tails():
    assert len(CARTOON_PRESETS) == 12
    for preset in CARTOON_PRESETS:
        processor = VoiceProcessor(RATE, preset.settings)
        assert processor.pitch_shifter is None or isinstance(processor.pitch_shifter, QuickPitchShifter)
        assert not (preset.settings.room or preset.settings.echo or preset.settings.chorus)


def test_radio_attenuates_low_frequencies():
    low = tone(80, 1)
    settings = VoiceSettings(radio=True, clarity=0)
    result = stream(VoiceProcessor(RATE, settings), low)
    assert np.std(result[RATE // 2:]) < np.std(low) * 0.3


def test_russian_speech_profile_increases_consonant_presence():
    samples = tone(2600, 1)
    plain = VoiceProcessor(RATE, VoiceSettings(clarity=0)).process(samples)
    enhanced = VoiceProcessor(RATE, VoiceSettings(clarity=6)).process(samples * 0.1)
    # Low input keeps the compressor inactive so presence EQ can be measured.
    assert np.std(enhanced[RATE // 2:]) > np.std(plain[RATE // 2:]) * 0.15


def test_render_file_saves_valid_mono_wav_and_effect_tail(tmp_path):
    source, target = tmp_path / "source.wav", tmp_path / "target.wav"
    samples = tone()
    sf.write(source, np.column_stack([samples, samples]), RATE)
    info = render_file(source, target, VoiceSettings(pitch=-4))
    audio, rate = sf.read(target)
    assert rate == RATE
    assert audio.ndim == 1
    assert len(audio) == info["frames"] == len(samples) + RATE
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.01


def test_render_never_overwrites_existing_file(tmp_path):
    source, target = tmp_path / "source.wav", tmp_path / "existing.wav"
    sf.write(source, tone(), RATE)
    target.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        render_file(source, target, VoiceSettings())
    assert target.read_bytes() == b"keep me"
    with pytest.raises(ValueError):
        render_file(source, source, VoiceSettings())


def test_render_bypass_retains_duration(tmp_path):
    source, target = tmp_path / "source.wav", tmp_path / "dry.wav"
    sf.write(source, tone(), RATE)
    render_file(source, target, VoiceSettings(enabled=False))
    assert sf.info(target).frames == sf.info(source).frames


@pytest.mark.parametrize("pitch", [-12, -7, -1, 1, 7, 12])
def test_pitch_buffers_stay_bounded_with_irregular_live_blocks(pitch):
    processor = StreamingPitchShifter(pitch)
    rng = np.random.default_rng(42)
    for _ in range(100):
        count = int(rng.integers(64, 2049))
        result = processor.process(rng.normal(0, 0.1, count).astype(np.float32))
        assert len(result) == count
        assert np.isfinite(result).all()
        assert len(processor._input) < processor.WINDOW
        assert len(processor._ready) < processor.WINDOW + 512
        assert len(processor._resample) < 4
