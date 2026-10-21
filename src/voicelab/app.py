import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QProcess, QSize, QThread, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QCloseEvent, QDesktopServices, QFont, QIcon, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QToolButton, QVBoxLayout, QWidget,
)

from .audio import AudioEngine, list_devices
from .avatars import avatar_icon
from .effects import render_file
from .presets import CATEGORIES, PRESETS, PRESET_BY_ID
from .routing import is_virtual_input, is_virtual_output
from .settings import load_preferences, save_preferences


STYLE = """
QWidget { background: #0c1018; color: #edf0f7; font-family: 'Segoe UI', 'DejaVu Sans'; font-size: 13px; }
QLabel { background: transparent; }
QLabel#brand { font-size: 24px; font-weight: 800; }
QLabel#eyebrow { color: #93e6cb; font-size: 11px; font-weight: 700; }
QLabel#hero { font-size: 28px; font-weight: 800; }
QLabel#section { font-size: 18px; font-weight: 700; }
QLabel#muted { color: #8f9aae; }
QLabel#small { color: #8f9aae; font-size: 11px; }
QLabel#status { color: #94a3ba; background: #18212e; border-radius: 14px; padding: 7px 14px; }
QFrame#panel { background: #131a26; border: 1px solid #263041; border-radius: 18px; }
QFrame#panel QLabel, QFrame#panel QCheckBox { background: transparent; }
QFrame#monitor { background: #13221f; border: 1px solid #28483c; border-radius: 18px; }
QPushButton { background: #202a3a; border: 1px solid #303d51; padding: 10px 16px; border-radius: 10px; font-weight: 600; }
QPushButton:hover { background: #2b374b; border-color: #62728a; }
QPushButton:disabled { color: #637089; background: #19202c; border-color: #263041; }
QPushButton#primary { background: #9aefcf; color: #10251d; border: none; font-size: 15px; padding: 16px; }
QPushButton#primary:hover { background: #b7f7e0; }
QPushButton#primary:disabled { background: #254c40; color: #6b9d8c; }
QToolButton#preset { background: #141c29; border: 1px solid #293447; border-radius: 14px; padding: 10px; font-size: 13px; }
QToolButton#preset:hover { background: #202e3c; border-color: #61788a; }
QToolButton#preset:checked { background: #1b342e; border: 2px solid #8ae6bf; color: #b1ffdc; }
QLineEdit { background: #141c29; border: 1px solid #303d51; border-radius: 8px; padding: 10px; }
QLineEdit:focus { border-color: #8ae6bf; }
QComboBox { background: #0e1520; border: 1px solid #303d51; border-radius: 8px; padding: 10px; min-height: 20px; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background: #1a2535; selection-background-color: #345044; }
QComboBox:disabled { color: #64718a; }
QSlider::groove:horizontal { height: 5px; background: #303d51; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #93e6cb; border-radius: 2px; }
QSlider::handle:horizontal { width: 15px; margin: -5px 0; background: #c5ffe8; border-radius: 7px; }
QCheckBox { spacing: 9px; }
QCheckBox::indicator { width: 17px; height: 17px; border-radius: 5px; border: 1px solid #506079; background: #0e1520; }
QCheckBox::indicator:checked { background: #93e6cb; border-color: #93e6cb; image: none; }
QProgressBar { background: #283344; border: none; height: 5px; border-radius: 2px; }
QProgressBar::chunk { background: #93e6cb; border-radius: 2px; }
QScrollArea { border: none; }
QScrollBar:vertical { background: #101722; width: 8px; margin: 0; border-radius: 4px; }
QScrollBar::handle:vertical { background: #3b4a60; min-height: 35px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QToolTip { background: #273447; color: white; border: none; padding: 6px; }
"""


def label(text, name=None, wrap=False):
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    return widget


def panel(name="panel"):
    widget = QFrame()
    widget.setObjectName(name)
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(22, 22, 22, 22)
    layout.setSpacing(14)
    return widget, layout


class Waveform(QWidget):
    def __init__(self):
        super().__init__()
        self.samples = np.zeros(128, dtype=np.float32)
        self.setMinimumHeight(36)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = self.height() / 2
        painter.setPen(QColor("#355146"))
        painter.drawLine(0, int(center), self.width(), int(center))
        painter.setPen(QColor("#9aefcf"))
        path = QPainterPath()
        for i, value in enumerate(self.samples):
            point = QPointF(i * self.width() / (len(self.samples) - 1), center - float(value) * center * 0.9)
            if i == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        painter.drawPath(path)


class RenderWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, source, destination, settings):
        super().__init__()
        self.source, self.destination, self.settings = source, destination, settings

    @Slot()
    def run(self):
        try:
            render_file(self.source, self.destination, self.settings)
            self.finished.emit(self.destination)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoiceLab — локальное изменение голоса")
        self.resize(1280, 940)
        self.setMinimumSize(1000, 760)
        self.engine = AudioEngine()
        self.preferences = load_preferences()
        self.selected = PRESET_BY_ID.get(self.preferences.get("preset"), PRESETS[0])
        self._render_thread = None
        self._last_error = ""
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(32, 26, 32, 20)
        outer.setSpacing(22)

        header = QHBoxLayout()
        header.addWidget(label("∿  VoiceLab", "brand"))
        header.addSpacing(16)
        header.addWidget(label("LOCAL VOICE STUDIO", "eyebrow"))
        header.addStretch()
        self.status = label("●  Микрофон выключен", "status")
        header.addWidget(self.status)
        help_button = QPushButton("Подключить Discord ↗")
        help_button.clicked.connect(self.show_help)
        header.addWidget(help_button)
        self.render_button = QPushButton("Обработать файл…")
        self.render_button.clicked.connect(self.render_audio)
        header.addWidget(self.render_button)
        outer.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        columns = QHBoxLayout(body)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(24)
        left = QVBoxLayout()
        left.setSpacing(12)
        left.addWidget(label("Твой голос. Твои правила.", "hero"))
        left.addWidget(label("Эффекты в реальном времени для общения, игр и стримов.", "muted", True))

        monitor, monitor_layout = panel("monitor")
        monitor_layout.setContentsMargins(18, 14, 18, 14)
        monitor_layout.setSpacing(8)
        monitor_header = QHBoxLayout()
        monitor_header.addWidget(label("ЖИВОЙ СИГНАЛ", "eyebrow"))
        monitor_header.addStretch()
        self.rate_label = label("Ожидание микрофона", "small")
        monitor_header.addWidget(self.rate_label)
        monitor_layout.addLayout(monitor_header)
        self.waveform = Waveform()
        monitor_layout.addWidget(self.waveform)
        self.signal_hint = label("Наденьте наушники перед включением прослушивания.", "small", True)
        monitor_layout.addWidget(self.signal_hint)
        left.addWidget(monitor)

        presets_header = QHBoxLayout()
        presets_header.addWidget(label("Коллекция голосов", "section"))
        presets_header.addStretch()
        self.voice_count = label(f"{len(PRESETS)} ГОЛОСОВ", "small")
        presets_header.addWidget(self.voice_count)
        left.addLayout(presets_header)
        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Найти голос…")
        self.search.textChanged.connect(self.filter_presets)
        self.category_combo = QComboBox()
        self.category_combo.addItems(CATEGORIES)
        self.category_combo.currentTextChanged.connect(self.filter_presets)
        filters.addWidget(self.search, 1)
        filters.addWidget(self.category_combo)
        left.addLayout(filters)
        catalog_scroll = QScrollArea()
        catalog_scroll.setWidgetResizable(True)
        catalog_scroll.setMinimumHeight(260)
        catalog_scroll.setMaximumHeight(480)
        catalog = QWidget()
        self.preset_grid = QGridLayout(catalog)
        self.preset_grid.setContentsMargins(0, 0, 8, 0)
        self.preset_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.preset_grid.setSpacing(10)
        self.preset_buttons = {}
        for index, preset in enumerate(PRESETS):
            button = QToolButton()
            button.setText(preset.name)
            button.setIcon(avatar_icon(preset.id, preset.category))
            button.setIconSize(QSize(88, 88))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setObjectName("preset")
            button.setCheckable(True)
            button.setMinimumHeight(140)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setToolTip(preset.description)
            button.clicked.connect(lambda checked=False, item=preset: self.select_preset(item))
            self.preset_buttons[preset.id] = button
            self.preset_grid.addWidget(button, index // 4, index % 4)
        catalog_scroll.setWidget(catalog)
        left.addWidget(catalog_scroll, 1)
        self.empty_label = label("Голоса не найдены. Попробуйте другой запрос.", "muted")
        self.empty_label.hide()
        left.addWidget(self.empty_label)
        self.description = label("", "muted", True)
        self.description.setMinimumHeight(24)
        left.addWidget(self.description)

        right = QVBoxLayout()
        right.setSpacing(18)
        devices_panel, devices_layout = panel()
        devices_layout.addWidget(label("Аудиоустройства", "section"))
        devices_layout.addWidget(label("КУДА ПЕРЕДАВАТЬ ГОЛОС", "small"))
        self.routing_combo = QComboBox()
        self.routing_combo.addItem("Виртуальный микрофон", "virtual")
        self.routing_combo.addItem("Прослушивание в наушниках", "monitor")
        self.routing_combo.setCurrentIndex(1 if self.preferences.get("routing") == "monitor" else 0)
        self.routing_combo.currentIndexChanged.connect(self.refresh_devices)
        devices_layout.addWidget(self.routing_combo)
        devices_layout.addWidget(label("МИКРОФОН / ВХОД", "small"))
        self.input_combo = QComboBox()
        self.input_combo.setMinimumWidth(250)
        self.input_combo.setMaximumWidth(360)
        devices_layout.addWidget(self.input_combo)
        devices_layout.addWidget(label("ОБРАБОТАННЫЙ ЗВУК / ВЫХОД", "small"))
        self.output_combo = QComboBox()
        self.output_combo.setMaximumWidth(360)
        devices_layout.addWidget(self.output_combo)
        self.route_hint = label("", "small", True)
        devices_layout.addWidget(self.route_hint)
        self.install_cable_button = QPushButton("Установить виртуальный микрофон ↗")
        self.install_cable_button.setToolTip("Скачать официальный VB-CABLE и запустить установку. Windows запросит права администратора.")
        self.install_cable_button.clicked.connect(self.install_virtual_microphone)
        devices_layout.addWidget(self.install_cable_button)
        self.refresh_button = QPushButton("↻  Обновить устройства")
        self.refresh_button.clicked.connect(self.refresh_devices)
        devices_layout.addWidget(self.refresh_button)
        self.device_note = label("", "small", True)
        devices_layout.addWidget(self.device_note)
        right.addWidget(devices_panel)

        effect_panel, effect_layout = panel()
        effect_layout.addWidget(label("Настрой свой звук", "section"))
        self.pitch_slider, self.pitch_value = self.make_slider(effect_layout, "Высота голоса", -12, 12, "st")
        self.timbre_slider, self.timbre_value = self.make_slider(effect_layout, "Тембр: яркий ↔ тёплый", -100, 100, "%")
        self.clarity_slider, self.clarity_value = self.make_slider(effect_layout, "Разборчивость речи", 0, 6, "dB")
        self.russian_speech = QCheckBox("Профиль «Русская речь»")
        self.russian_speech.setChecked(True)
        self.russian_speech.setToolTip("Эквализация согласных и мягкая компрессия. Не меняет язык и произношение; работает и с другими языками.")
        self.russian_speech.toggled.connect(self.update_settings)
        effect_layout.addWidget(self.russian_speech)
        self.room_slider, self.room_value = self.make_slider(effect_layout, "Пространство", 0, 80, "%")
        self.gain_slider, self.gain_value = self.make_slider(effect_layout, "Усиление", -18, 12, "dB")
        self.bypass = QCheckBox("Отключить эффекты")
        self.bypass.toggled.connect(self.update_settings)
        effect_layout.addWidget(self.bypass)
        self.start_button = QPushButton("Включить микрофон")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.toggle_audio)
        levels = QHBoxLayout()
        levels.addWidget(label("IN", "small"))
        self.input_meter = QProgressBar()
        self.input_meter.setRange(0, 100)
        self.input_meter.setTextVisible(False)
        levels.addWidget(self.input_meter)
        levels.addWidget(label("OUT", "small"))
        self.output_meter = QProgressBar()
        self.output_meter.setRange(0, 100)
        self.output_meter.setTextVisible(False)
        levels.addWidget(self.output_meter)
        right.addWidget(effect_panel)

        self.record_button = QPushButton("●  Записать голос в WAV")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_label = label("Запись сохраняется только на вашем компьютере.", "small", True)
        right.addStretch()
        columns.addLayout(left, 7)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_widget)
        right_scroll.setMinimumWidth(330)
        columns.addWidget(right_scroll, 3)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        transport, transport_layout = panel()
        transport_layout.setContentsMargins(18, 12, 18, 12)
        transport_row = QHBoxLayout()
        transport_row.setSpacing(20)
        transport_row.addWidget(self.start_button, 1)
        transport_row.addLayout(levels, 1)
        transport_row.addWidget(self.record_button, 1)
        transport_layout.addLayout(transport_row)
        transport_layout.addWidget(self.record_label)
        outer.addWidget(transport)
        footer = QHBoxLayout()
        footer.addWidget(label("●  Обработка на устройстве · Без аккаунта и облака", "small"))
        footer.addStretch()
        footer.addWidget(label("VoiceLab  /  0.1.0", "small"))
        outer.addLayout(footer)
        self.setCentralWidget(root)
        self.select_preset(self.selected)
        tuning = self.preferences.get("tuning")
        if isinstance(tuning, dict):
            for key, slider in (("pitch", self.pitch_slider), ("timbre", self.timbre_slider),
                                ("clarity", self.clarity_slider), ("room", self.room_slider),
                                ("gain", self.gain_slider)):
                value = tuning.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    slider.setValue(round(value))
            self.russian_speech.setChecked(bool(tuning.get("russian_speech", True)))
        self.refresh_devices()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(50)

    def filter_presets(self, *args):
        query = self.search.text().strip().casefold()
        category = self.category_combo.currentText()
        visible = 0
        for preset in PRESETS:
            button = self.preset_buttons[preset.id]
            self.preset_grid.removeWidget(button)
            matches = (category == CATEGORIES[0] or preset.category == category) and (
                query in f"{preset.name} {preset.id} {preset.category}".casefold()
            )
            button.setVisible(matches)
            if matches:
                self.preset_grid.addWidget(button, visible // 4, visible % 4)
                visible += 1
        self.voice_count.setText(f"{visible} / {len(PRESETS)} ГОЛОСОВ")
        self.empty_label.setVisible(visible == 0)

    def make_slider(self, layout, title, low, high, suffix):
        row = QHBoxLayout()
        row.addWidget(label(title, "muted"))
        row.addStretch()
        value = label("0 " + suffix)
        row.addWidget(value)
        layout.addLayout(row)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.valueChanged.connect(self.update_settings)
        layout.addWidget(slider)
        return slider, value

    def select_preset(self, preset):
        self.selected = preset
        for key, button in self.preset_buttons.items():
            button.setChecked(key == preset.id)
        for slider, value in ((self.pitch_slider, preset.settings.pitch),
                              (self.timbre_slider, preset.settings.timbre * 100),
                              (self.clarity_slider, preset.settings.clarity),
                              (self.room_slider, preset.settings.room * 100),
                              (self.gain_slider, preset.settings.gain)):
            slider.blockSignals(True)
            slider.setValue(round(value))
            slider.blockSignals(False)
        self.description.setText(preset.description)
        self.update_settings()

    def update_settings(self, *args):
        settings = replace(self.selected.settings, pitch=self.pitch_slider.value(),
                           timbre=self.timbre_slider.value() / 100,
                           clarity=self.clarity_slider.value() if self.russian_speech.isChecked() else 0,
                           room=self.room_slider.value() / 100, gain=self.gain_slider.value(),
                           enabled=not self.bypass.isChecked())
        self.engine.configure(settings)
        self.pitch_value.setText(f"{settings.pitch:+.0f} st")
        self.timbre_value.setText(f"{settings.timbre * 100:+.0f} %")
        self.clarity_value.setText(f"{settings.clarity:.0f} dB")
        self.clarity_slider.setEnabled(self.russian_speech.isChecked())
        self.room_value.setText(f"{settings.room * 100:.0f} %")
        self.gain_value.setText(f"{settings.gain:+.0f} dB")

    def refresh_devices(self):
        previous_input = self.input_combo.currentText() or self.preferences.get("input")
        previous_output = self.output_combo.currentText() or self.preferences.get("output")
        self.input_combo.clear()
        self.output_combo.clear()
        try:
            inputs, outputs = list_devices()
            # Never feed the virtual capture endpoint back into its own playback.
            inputs = [device for device in inputs if not is_virtual_input(device.name) and not is_virtual_output(device.name)]
            virtual_mode = self.routing_combo.currentData() == "virtual"
            outputs = [device for device in outputs if is_virtual_output(device.name) == virtual_mode]
            for combo, devices, previous in ((self.input_combo, inputs, previous_input),
                                              (self.output_combo, outputs, previous_output)):
                for device in devices:
                    combo.addItem(device.label, device.index)
                selected = combo.findText(previous or "")
                if selected >= 0:
                    combo.setCurrentIndex(selected)
            ready = bool(inputs and outputs)
            self.start_button.setEnabled(ready)
            self.install_cable_button.setVisible(virtual_mode and not outputs)
            if virtual_mode:
                self.route_hint.setText("Микрофон → эффекты → VB-CABLE. В Discord и игре выберите вход CABLE Output.")
                if not outputs:
                    self.device_note.setText("Виртуальный микрофон не найден. Установите VB-CABLE, перезагрузите Windows и обновите устройства.")
                elif not inputs:
                    self.device_note.setText("Виртуальный выход найден. Подключите физический микрофон и обновите устройства.")
                else:
                    self.device_note.setText("Готово: обработанный голос пойдёт в виртуальное устройство.")
            else:
                self.route_hint.setText("Обработанный голос идёт в наушники. Для Discord переключитесь на виртуальный микрофон.")
                self.device_note.setText("" if ready else "Нет доступного микрофона или обычного аудиовыхода.")
        except Exception as exc:
            self.start_button.setEnabled(False)
            self.device_note.setText(f"Не удалось получить устройства: {exc}")

    def toggle_audio(self):
        if self.engine.running:
            self.engine.stop()
            self.set_running(False)
            return
        if self.input_combo.currentData() is None or self.output_combo.currentData() is None:
            return
        if self.routing_combo.currentData() == "monitor":
            result = QMessageBox.question(
                self, "Прослушивание голоса",
                "Выбран обычный аудиовыход. Используйте наушники, чтобы звук не попадал обратно в микрофон.\n\nВключить прослушивание?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if result != QMessageBox.StandardButton.Yes:
                return
        try:
            self.engine.start(self.input_combo.currentData(), self.output_combo.currentData())
            self._last_error = ""
            self.set_running(True)
        except Exception as exc:
            self.set_running(False)
            QMessageBox.critical(self, "Не удалось включить микрофон", str(exc))

    def set_running(self, running):
        self.routing_combo.setEnabled(not running)
        self.input_combo.setEnabled(not running)
        self.output_combo.setEnabled(not running)
        self.refresh_button.setEnabled(not running)
        self.start_button.setText("Остановить микрофон" if running else "Включить микрофон")
        self.record_button.setEnabled(running)
        route_name = "Виртуальный микрофон включён" if self.routing_combo.currentData() == "virtual" else "Прослушивание включено"
        self.status.setText(f"●  {route_name}" if running else "●  Микрофон выключен")
        self.rate_label.setText(f"{self.engine.sample_rate / 1000:g} kHz · MONO" if running else "Ожидание микрофона")
        if not running:
            self.record_button.setText("●  Записать голос в WAV")

    def toggle_recording(self):
        if self.engine.recording_path:
            path = self.engine.recording_path
            self.engine.stop_recording()
            self.record_button.setText("●  Записать голос в WAV")
            self.record_label.setText(f"Сохранено: {path}")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Новая запись", str(Path.home() / "VoiceLab-recording.wav"), "WAV (*.wav)")
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        try:
            self.engine.start_recording(path)
            self.record_button.setText("■  Остановить запись")
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось начать запись", str(exc))

    def render_audio(self):
        source, _ = QFileDialog.getOpenFileName(self, "Открыть аудио", "", "Аудиофайлы (*.wav *.flac *.ogg *.mp3);;Все файлы (*)")
        if not source:
            return
        destination, _ = QFileDialog.getSaveFileName(self, "Сохранить обработанный WAV", str(Path(source).with_name(Path(source).stem + "-voicelab.wav")), "WAV (*.wav)")
        if not destination:
            return
        if not destination.lower().endswith(".wav"):
            destination += ".wav"
        self.render_button.setEnabled(False)
        self.render_button.setText("Обработка…")
        thread = QThread(self)
        worker = RenderWorker(source, destination, self.engine.settings)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.render_finished)
        worker.failed.connect(self.render_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self.render_cleanup)
        self._render_thread, self._render_worker = thread, worker
        thread.start()

    @Slot(str)
    def render_finished(self, path):
        QMessageBox.information(self, "Готово", f"Обработанный файл сохранён:\n{path}")

    @Slot(str)
    def render_failed(self, error):
        QMessageBox.warning(self, "Не удалось обработать файл", error)

    @Slot()
    def render_cleanup(self):
        self._render_thread.deleteLater()
        self._render_thread = self._render_worker = None
        self.render_button.setEnabled(True)
        self.render_button.setText("Обработать файл…")

    def tick(self):
        self.engine.check_health()
        self.input_meter.setValue(round(min(1, self.engine.input_level) * 100))
        self.output_meter.setValue(round(min(1, self.engine.output_level) * 100))
        samples = self.engine.waveform if self.engine.running else np.zeros(128)
        self.waveform.samples = samples
        self.waveform.update()
        if self.engine.running:
            self.signal_hint.setText(f"Потери аудиобуфера: {self.engine.dropouts}. При росте счётчика закройте тяжёлые приложения.")
        else:
            self.signal_hint.setText("Наденьте наушники перед включением прослушивания.")
        if self.engine.recording_path:
            seconds = int(self.engine.recorded_frames / self.engine.sample_rate)
            self.record_label.setText(f"● Запись  {seconds // 60:02d}:{seconds % 60:02d}")
        elif self.engine.running:
            self.record_button.setText("●  Записать голос в WAV")
        error = self.engine.error
        if error and error != self._last_error:
            self._last_error = error
            if self.engine.failed:
                self.engine.stop()
                self.set_running(False)
            QMessageBox.warning(self, "Аудиодвижок", error)

    def show_help(self):
        QMessageBox.information(self, "Discord и игры", (
            "1. Если виртуального микрофона нет, нажмите «Установить виртуальный микрофон» в VoiceLab. "
            "Приложение скачает официальный VB-CABLE и запустит установку. "
            "В окне VB-CABLE нажмите Install Driver. Windows запросит права администратора; может понадобиться перезагрузка.\n\n"
            "2. В VoiceLab выберите физический микрофон, а выход — CABLE Input (VB-Audio Virtual Cable).\n\n"
            "3. В Discord → Голос и видео → Устройство ввода выберите CABLE Output. "
            "В игре также выберите CABLE Output.\n\n"
            "4. Включите микрофон в VoiceLab. Выход Discord оставьте на наушниках. "
            "Если эффекты подавляются, попробуйте выключить шумоподавление и автоматическую регулировку усиления в Discord.\n\n"
            "Без виртуального кабеля можно слушать результат в наушниках и записывать WAV. "
            "Это DSP-эффекты, а не нейросетевое клонирование голоса."
        ))

    def install_virtual_microphone(self):
        if sys.platform == "win32":
            if getattr(sys, "frozen", False):
                helper = Path(sys.executable).parent / "Install-VirtualMicrophone.ps1"
            else:
                helper = Path(__file__).resolve().parents[2] / "installer" / "Install-VirtualMicrophone.ps1"
            if helper.is_file():
                started, _ = QProcess.startDetached("powershell.exe", [
                    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(helper),
                ])
                if not started:
                    QMessageBox.warning(self, "Виртуальный микрофон", "Не удалось запустить установку VB-CABLE.")
                return
        QDesktopServices.openUrl(QUrl("https://vb-audio.com/Cable/"))

    def closeEvent(self, event: QCloseEvent):
        if self._render_thread is not None:
            QMessageBox.information(self, "Обработка файла", "Дождитесь завершения обработки перед выходом.")
            event.ignore()
            return
        self.timer.stop()
        self.engine.stop()
        try:
            save_preferences({"preset": self.selected.id, "input": self.input_combo.currentText(),
                              "output": self.output_combo.currentText(),
                              "routing": self.routing_combo.currentData(),
                              "tuning": {"pitch": self.pitch_slider.value(), "timbre": self.timbre_slider.value(),
                                         "clarity": self.clarity_slider.value(), "room": self.room_slider.value(),
                                         "gain": self.gain_slider.value(), "russian_speech": self.russian_speech.isChecked()}})
        except OSError:
            pass
        event.accept()


def run():
    app = QApplication(sys.argv)
    app.setApplicationName("VoiceLab")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Segoe UI", 10))
    icon = QPixmap(64, 64)
    icon.fill(QColor("#13221f"))
    painter = QPainter(icon)
    painter.setPen(QColor("#9aefcf"))
    painter.setFont(QFont("Segoe UI", 42))
    painter.drawText(icon.rect(), Qt.AlignmentFlag.AlignCenter, "∿")
    painter.end()
    app.setWindowIcon(QIcon(icon))
    window = MainWindow()
    window.show()
    return app.exec()
