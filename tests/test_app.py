import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from shiboken6 import delete
from PySide6.QtWidgets import QApplication

from voicelab import app as ui
from voicelab.audio import Device
from voicelab.presets import PRESETS, PRESET_BY_ID, VoiceSettings


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(ui.STYLE)
    return app


@pytest.fixture
def window(qt_app, monkeypatch, tmp_path):
    monkeypatch.setenv("VOICELAB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ui, "list_devices", lambda: ([], []))
    window = ui.MainWindow()
    window.ai.mode.setCurrentIndex(1)
    window.show()
    qt_app.processEvents()
    yield window
    window.close()
    delete(window)
    qt_app.processEvents()


def test_no_hardware_disables_start_but_keeps_file_rendering(window):
    assert "npetrov VoiceLab" in window.windowTitle()
    assert not window.monitor_check.isChecked()
    assert not window.monitor_check.isEnabled()
    assert not window.start_button.isEnabled()
    assert window.render_button.isEnabled()
    assert len(window.preset_buttons) == len(PRESETS)
    assert window.routing_combo.currentData() == "virtual"
    assert window.install_cable_button.isVisible()


def test_virtual_route_selects_cable_and_excludes_feedback_input(window, monkeypatch):
    microphone = Device(1, "USB Microphone", "WASAPI", 1)
    cable_capture = Device(2, "CABLE Output (VB-Audio Virtual Cable)", "WASAPI", 2)
    headphones = Device(3, "Headphones", "WASAPI", 2)
    cable_playback = Device(4, "CABLE Input (VB-Audio Virtual Cable)", "WASAPI", 2)
    monkeypatch.setattr(ui, "list_devices", lambda: ([microphone, cable_capture], [headphones, cable_playback]))
    window.routing_combo.setCurrentIndex(0)
    window.refresh_devices()
    assert window.input_combo.count() == 1
    assert window.input_combo.currentData() == 1
    assert window.output_combo.count() == 1
    assert window.output_combo.currentData() == 4
    assert window.start_button.isEnabled()
    assert window.install_cable_button.isHidden()
    assert window.monitor_combo.count() == 1
    assert window.monitor_combo.currentData() == 3
    assert window.monitor_check.isEnabled()
    window.routing_combo.setCurrentIndex(1)
    assert window.output_combo.count() == 1
    assert window.output_combo.currentData() == 3
    assert window.monitor_check.isEnabled()


def test_checkbox_toggles_live_monitor_while_main_route_keeps_running(window, monkeypatch):
    monkeypatch.setattr(ui, "list_devices", lambda: (
        [Device(1, "USB Microphone", "WASAPI", 1)],
        [Device(3, "Headphones", "WASAPI", 2), Device(4, "CABLE Input", "WASAPI", 2)],
    ))
    window.refresh_devices()
    window.routing_combo.setCurrentIndex(0)
    calls = []

    class Monitor:
        def close(self):
            calls.append("close")

    def monitor(device):
        calls.append(device)
        window.engine._monitor = Monitor()

    monkeypatch.setattr(window.engine, "start_monitoring", monitor)
    window.engine.running = True
    window.set_running(True)
    window.monitor_check.setChecked(True)
    assert calls == [3]
    assert not window.monitor_combo.isEnabled()
    assert not window.output_combo.isEnabled()
    window.monitor_check.setChecked(False)
    assert calls == [3, "close"]
    assert window.engine.running
    assert window.monitor_combo.isEnabled()
    assert not window.output_combo.isEnabled()


def test_monitor_failure_unchecks_box_and_keeps_main_route(window, monkeypatch):
    window.monitor_combo.addItem("Headphones", 3)
    window.engine.running = True
    window.update_monitor_controls()

    def fail(device):
        raise RuntimeError("busy")

    messages = []
    monkeypatch.setattr(window.engine, "start_monitoring", fail)
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *args: messages.append(args[-1]))
    window.monitor_check.setChecked(True)
    assert not window.monitor_check.isChecked()
    assert window.engine.running
    assert window.monitor_combo.isEnabled()
    assert "busy" in messages[0]


def test_virtual_mode_never_falls_back_to_speakers(window, monkeypatch):
    monkeypatch.setattr(ui, "list_devices", lambda: (
        [Device(1, "USB Microphone", "WASAPI", 1)],
        [Device(2, "Speakers", "WASAPI", 2)],
    ))
    window.routing_combo.setCurrentIndex(0)
    window.refresh_devices()
    assert window.output_combo.count() == 0
    assert not window.start_button.isEnabled()
    assert window.install_cable_button.isVisible()
    assert "VB-CABLE" in window.device_note.text()


def test_standalone_mode_starts_without_virtual_output_or_driver(window, monkeypatch):
    monkeypatch.setattr(ui, "list_devices", lambda: (
        [Device(1, "USB Microphone", "WASAPI", 1)],
        [Device(3, "Headphones", "WASAPI", 2)],
    ))
    window.routing_combo.setCurrentIndex(1)
    window.refresh_devices()
    assert window.start_button.isEnabled()
    assert window.output_combo.isHidden()
    assert window.install_cable_button.isHidden()
    calls = []

    def start(microphone, output):
        calls.append((microphone, output))
        window.engine.running = True

    monkeypatch.setattr(window.engine, "start", start)
    window.toggle_audio()
    assert calls == [(1, None)]
    assert window.engine.running
    assert window.monitor_check.isEnabled()


def test_installed_app_starts_bundled_driver_setup(window, monkeypatch, tmp_path):
    helper = tmp_path / "Install-VirtualMicrophone.ps1"
    helper.write_text("# driver setup", encoding="utf-8")
    monkeypatch.setattr(ui.sys, "platform", "win32")
    monkeypatch.setattr(ui.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ui.sys, "executable", str(tmp_path / "VoiceLab.exe"))
    launches = []
    monkeypatch.setattr(ui.QProcess, "startDetached", lambda program, args: (launches.append((program, args)) is None, 123))
    window.install_virtual_microphone()
    assert len(launches) == 1
    assert launches[0][0] == "powershell.exe"
    assert launches[0][1][-1] == str(helper)


def test_search_and_category_filter(window, qt_app):
    window.search.setText("сакура")
    qt_app.processEvents()
    visible = [key for key, button in window.preset_buttons.items() if not button.isHidden()]
    assert visible == ["sakura"]
    window.search.clear()
    window.category_combo.setCurrentText("Кибер")
    visible = [key for key, button in window.preset_buttons.items() if not button.isHidden()]
    assert len(visible) == 6
    window.search.setText("не существующий голос")
    assert window.empty_label.isVisible()


def test_controls_update_audio_settings_and_russian_profile(window):
    window.select_preset(PRESET_BY_ID["sakura"])
    assert window.engine.settings.pitch == 6
    window.pitch_slider.setValue(-3)
    window.timbre_slider.setValue(50)
    window.clarity_slider.setValue(5)
    assert window.engine.settings.pitch == -3
    assert window.engine.settings.timbre == 0.5
    assert window.engine.settings.clarity == 5
    window.russian_speech.setChecked(False)
    assert window.engine.settings.clarity == 0
    assert not window.clarity_slider.isEnabled()
    window.bypass.setChecked(True)
    assert not window.engine.settings.enabled


def test_cartoon_shortcut_leaves_ai_and_selects_fast_style(window):
    window.ai.mode.setCurrentIndex(0)
    window.cartoon_button.click()
    assert not window.ai.active
    assert window.category_combo.currentText() == "Мультяшные"
    assert window.engine.settings.quick_pitch
    assert window.selected.id == "toon_squirrel"
    assert sum(not button.isHidden() for button in window.preset_buttons.values()) == 12
    window.select_preset(PRESET_BY_ID["toon_giant"])
    assert window.engine.settings.pitch == -7
    assert window.engine.settings.quick_pitch
    # Re-entering AI must clear the cartoon DSP recipe.
    window.ai.mode.setCurrentIndex(0)
    assert window.selected.id == "clean"
    assert not window.engine.settings.quick_pitch


def test_cartoon_shortcut_cannot_interrupt_live_ai(window):
    window.ai.mode.setCurrentIndex(0)
    window.engine.running = True
    window.set_running(True)
    assert not window.cartoon_button.isEnabled()
    window.open_cartoons()
    assert window.ai.active


def test_clean_ai_removes_all_post_effects_without_bypassing_conversion(window):
    window.select_preset(PRESET_BY_ID["monster"])
    window.pitch_slider.setValue(-9)
    window.room_slider.setValue(60)
    window.gain_slider.setValue(10)
    window.ai.mode.setCurrentIndex(0)
    window.select_preset(PRESET_BY_ID["toon_alien"])
    window.room_slider.setValue(70)
    assert window.engine.settings == VoiceSettings(clarity=0)
    assert window.engine.settings.enabled
    assert not window.pitch_slider.isEnabled()
    assert not window.room_slider.isEnabled()
    assert not window.russian_speech.isEnabled()
    window.clean_ai_check.setChecked(False)
    assert window.engine.settings.robot == 0.6
    assert window.pitch_slider.isEnabled()
    window.bypass.setChecked(True)
    window.clean_ai_button.click()
    assert window.engine.settings == VoiceSettings(clarity=0)
    assert window.ai.active and not window.bypass.isChecked()
    assert window.selected.id == "clean"


def test_clean_ai_preserves_selected_voice_and_can_reset_dsp_while_running(window, monkeypatch, tmp_path):
    from voicelab import neural_ui
    from voicelab.native import NativeVoice
    voice = NativeVoice("native-" + "2" * 32, "Мой голос", tmp_path, 2)
    monkeypatch.setattr(neural_ui, "installed_voices", lambda: [voice])
    window.ai.reload(voice.id)
    window.ai.mode.setCurrentIndex(0)
    window.engine.neural_voice = voice
    window.engine.running = True
    window.set_running(True)
    window.clean_ai_check.setChecked(False)
    window.room_slider.setValue(60)
    assert window.engine.settings.room == 0.6
    window.clean_ai_button.click()
    assert window.engine.running
    assert window.ai.voice == voice and window.engine.neural_voice == voice
    assert window.engine.settings == VoiceSettings(clarity=0)


def test_clean_ai_choice_is_saved_without_changing_effect_mode(window):
    from voicelab.settings import load_preferences
    assert not window.ai.active
    assert window.clean_ai_check.isHidden()
    window.select_preset(PRESET_BY_ID["sakura"])
    assert window.engine.settings.pitch == 6
    window.clean_ai_check.setChecked(False)
    window.close()
    assert load_preferences()["clean_ai"] is False


def test_ai_mode_does_not_present_effects_as_neural_voices(window):
    window.ai.mode.setCurrentIndex(0)
    assert window.ai.active and not window.ai.ready
    assert not window.start_button.isEnabled()
    assert all(button.isHidden() for button in window.preset_buttons.values())
    assert not window.ai_buttons
    assert window.empty_label.isVisible()
    assert "AI" in window.voice_count.text()


def test_installed_ai_voices_are_selectable_and_persisted(window, monkeypatch, tmp_path):
    from voicelab import neural_ui
    from voicelab.neural import NeuralVoice
    voices = [NeuralVoice(str(i), name, tmp_path / "voice.onnx", tmp_path / "vec.onnx", tmp_path / "pitch.onnx", 40000, 768)
              for i, name in enumerate(("Голос один", "Голос два"))]
    monkeypatch.setattr(neural_ui, "installed_voices", lambda: voices)
    window.ai.reload()
    window.ai.mode.setCurrentIndex(0)
    assert len(window.ai_buttons) == 2
    assert window.ai.voice.name == "Голос один"
    window.ai_buttons["1"].click()
    assert window.ai.voice.name == "Голос два"
    assert window.ai.settings()["ai_voice"] == "1"
    window.search.setText("два")
    assert window.ai_buttons["0"].isHidden() and not window.ai_buttons["1"].isHidden()


def test_ai_profile_defaults_to_light_and_respects_saved_choice(qt_app):
    from voicelab.neural_ui import NeuralPanel
    for preferences, expected in (({}, "light"), ({"ai_profile": "full"}, "full")):
        panel = NeuralPanel(preferences)
        assert panel.profile.currentData() == panel.settings()["ai_profile"] == expected
        panel.set_running(True)
        assert not panel.profile.isEnabled()
        panel.set_running(False)
        assert panel.profile.isEnabled()
        delete(panel)


def test_large_ai_catalog_keeps_only_one_page_of_buttons(window, monkeypatch, tmp_path):
    from voicelab import neural_ui
    from voicelab.native import NativeVoice
    voices = [NativeVoice(f"native-{i:032x}", f"AI-голос {i:04d}", tmp_path, i) for i in range(1, 1242)]
    monkeypatch.setattr(neural_ui, "installed_voices", lambda: voices)
    window.ai.reload()
    window.ai.mode.setCurrentIndex(0)
    assert len(window.ai_buttons) == 40
    assert window.next_voices.isVisible()
    window.next_voices.click()
    assert len(window.ai_buttons) == 40
    assert voices[40].id in window.ai_buttons and voices[0].id not in window.ai_buttons
    window.search.setText("1241")
    assert len(window.ai_buttons) == 1
    assert voices[-1].id in window.ai_buttons
    window.ai_buttons[voices[-1].id].click()
    assert window.ai.voice.speaker_id == 1241
    assert window.ai.provider.isHidden() and window.ai.profile.isHidden()
    assert window.ai.rename_button.isVisible()


def test_short_buffers_are_available_only_for_native_voices(window, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from voicelab import neural_ui
    from voicelab.native import NativeVoice
    native = NativeVoice("native-" + "1" * 32, "Native", tmp_path, 1)
    rvc = SimpleNamespace(id="rvc", name="RVC", credit="")
    monkeypatch.setattr(neural_ui, "installed_voices", lambda: [native, rvc])
    window.ai.reload(native.id)
    window.ai.mode.setCurrentIndex(0)
    window.ai.buffer.setCurrentIndex(window.ai.buffer.findData(80))
    assert window.ai.buffer.currentData() == 80
    window.ai.voice_combo.setCurrentIndex(1)
    assert window.ai.buffer.currentData() >= 160
    for milliseconds in (40, 80):
        assert not window.ai.buffer.model().item(window.ai.buffer.findData(milliseconds)).isEnabled()
