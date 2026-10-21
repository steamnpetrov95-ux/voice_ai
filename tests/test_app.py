import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from shiboken6 import delete
from PySide6.QtWidgets import QApplication

from voicelab import app as ui
from voicelab.audio import Device
from voicelab.presets import PRESET_BY_ID


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
    window.show()
    qt_app.processEvents()
    yield window
    window.close()
    delete(window)
    qt_app.processEvents()


def test_no_hardware_disables_start_but_keeps_file_rendering(window):
    assert not window.start_button.isEnabled()
    assert window.render_button.isEnabled()
    assert len(window.preset_buttons) == 48
    assert window.routing_combo.currentData() == "virtual"
    assert window.install_cable_button.isVisible()


def test_virtual_route_selects_cable_and_excludes_feedback_input(window, monkeypatch):
    microphone = Device(1, "USB Microphone", "WASAPI", 1)
    cable_capture = Device(2, "CABLE Output (VB-Audio Virtual Cable)", "WASAPI", 2)
    headphones = Device(3, "Headphones", "WASAPI", 2)
    cable_playback = Device(4, "CABLE Input (VB-Audio Virtual Cable)", "WASAPI", 2)
    monkeypatch.setattr(ui, "list_devices", lambda: ([microphone, cable_capture], [headphones, cable_playback]))
    window.refresh_devices()
    assert window.input_combo.count() == 1
    assert window.input_combo.currentData() == 1
    assert window.output_combo.count() == 1
    assert window.output_combo.currentData() == 4
    assert window.start_button.isEnabled()
    assert window.install_cable_button.isHidden()
    window.routing_combo.setCurrentIndex(1)
    assert window.output_combo.count() == 1
    assert window.output_combo.currentData() == 3


def test_virtual_mode_never_falls_back_to_speakers(window, monkeypatch):
    monkeypatch.setattr(ui, "list_devices", lambda: (
        [Device(1, "USB Microphone", "WASAPI", 1)],
        [Device(2, "Speakers", "WASAPI", 2)],
    ))
    window.refresh_devices()
    assert window.output_combo.count() == 0
    assert not window.start_button.isEnabled()
    assert window.install_cable_button.isVisible()
    assert "VB-CABLE" in window.device_note.text()


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
