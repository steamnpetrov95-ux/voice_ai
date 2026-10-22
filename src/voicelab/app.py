import os
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
from . import APP_NAME, __version__
from .avatars import avatar_icon
from .effects import render_file
from .neural_ui import NeuralPanel
from .presets import CARTOON_PRESETS, CATEGORIES, PRESETS, PRESET_BY_ID, VoiceSettings
from .routing import is_virtual_input, is_virtual_output
from .settings import load_preferences, save_preferences
from .voice_roles import actor_roles, roles_caption, roles_description


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


def virtual_microphone_helper():
    if sys.platform != "win32":
        return None
    if getattr(sys, "frozen", False):
        helper = Path(sys.executable).parent / "Install-VirtualMicrophone.ps1"
    elif (os.environ.get("VOICELAB_VBCABLE_INTEGRATION_AUTHORIZED") == "1"
          or os.environ.get("VOICELAB_PERSONAL_DRIVER_SETUP") == "1"):
        helper = Path(__file__).resolve().parents[2] / "installer" / "Install-VirtualMicrophone.ps1"
    else:
        return None
    return helper if helper.is_file() else None


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

    def __init__(self, source, destination, settings, voice=None, provider="auto", profile="light"):
        super().__init__()
        self.source, self.destination, self.settings = source, destination, settings
        self.voice, self.provider = voice, provider
        self.profile = profile

    @Slot()
    def run(self):
        try:
            render_file(self.source, self.destination, self.settings, self.voice, self.provider, self.profile)
            self.finished.emit(self.destination)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — локальное изменение голоса")
        self.resize(1280, 940)
        self.setMinimumSize(1000, 760)
        self.engine = AudioEngine()
        self.preferences = load_preferences()
        self.ai = NeuralPanel(self.preferences)
        self.ai_buttons = {}
        self._voice_page = 0
        self._voice_query = None
        self._previous_ai_active = self.ai.active
        self.selected = PRESET_BY_ID.get(self.preferences.get("preset"), PRESETS[0])
        if self.ai.active and self.preferences.get("voice_mode") != "ai":
            self.selected = PRESETS[0]
        self._render_thread = None
        self._last_error = ""
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(32, 26, 32, 20)
        outer.setSpacing(22)

        header = QHBoxLayout()
        header.addWidget(label(f"∿  {APP_NAME}", "brand"))
        header.addSpacing(16)
        header.addWidget(label("LOCAL VOICE STUDIO", "eyebrow"))
        header.addStretch()
        self.status = label("●  Микрофон выключен", "status")
        header.addWidget(self.status)
        help_button = QPushButton("Как пользоваться ↗")
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
        left.addWidget(label("Нейросетевые голоса и эффекты для общения, игр и стримов.", "muted", True))

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

        voice_mode_row = QHBoxLayout()
        voice_mode_row.addWidget(self.ai.mode, 1)
        voice_mode_row.addWidget(self.ai.add_button)
        left.addLayout(voice_mode_row)
        self.clean_ai_button = QPushButton("Чистый AI · без дополнительных эффектов")
        self.clean_ai_button.setToolTip("Выбранная нейросетевая модель без сдвига высоты, эха, эквализации и модуляции. "
                                        "Перед переходом из эффектов остановите микрофон.")
        self.clean_ai_button.clicked.connect(self.open_clean_ai)
        left.addWidget(self.clean_ai_button)
        self.cartoon_button = QPushButton("Экспериментальные мульт-эффекты")
        self.cartoon_button.setToolTip("Упрощённые DSP-эффекты. Возможны двоение, дребезг и металлическое звучание. "
                                       "Это не нейросетевые голоса персонажей. Перед переходом из AI остановите микрофон.")
        self.cartoon_button.clicked.connect(self.open_cartoons)
        left.addWidget(self.cartoon_button)
        presets_header = QHBoxLayout()
        self.catalog_title = label("Коллекция голосов", "section")
        presets_header.addWidget(self.catalog_title)
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
        page_row = QHBoxLayout()
        self.previous_voices = QPushButton("← Назад")
        self.previous_voices.clicked.connect(lambda: self.change_voice_page(-1))
        self.next_voices = QPushButton("Далее →")
        self.next_voices.clicked.connect(lambda: self.change_voice_page(1))
        self.voice_page_label = label("", "small")
        page_row.addWidget(self.previous_voices)
        page_row.addWidget(self.voice_page_label, 1)
        page_row.addWidget(self.next_voices)
        left.addLayout(page_row)
        self.empty_label = label("Голоса не найдены. Попробуйте другой запрос.", "muted")
        self.empty_label.hide()
        left.addWidget(self.empty_label)
        self.description = label("", "muted", True)
        self.description.setMinimumHeight(24)
        left.addWidget(self.description)

        right = QVBoxLayout()
        right.setSpacing(18)
        ai_panel, ai_layout = panel()
        ai_layout.addWidget(label("Замена голоса", "section"))
        ai_layout.addWidget(self.ai)
        devices_panel, devices_layout = panel()
        devices_layout.addWidget(label("Аудиоустройства", "section"))
        devices_layout.addWidget(label("КУДА ПЕРЕДАВАТЬ ГОЛОС", "small"))
        self.routing_combo = QComboBox()
        self.routing_combo.addItem("Виртуальный микрофон", "virtual")
        self.routing_combo.addItem("Только наушники — без драйвера", "monitor")
        self.routing_combo.setCurrentIndex(1 if self.preferences.get("routing") == "monitor" else 0)
        self.routing_combo.currentIndexChanged.connect(self.refresh_devices)
        devices_layout.addWidget(self.routing_combo)
        devices_layout.addWidget(label("МИКРОФОН / ВХОД", "small"))
        self.input_combo = QComboBox()
        self.input_combo.setMinimumWidth(250)
        self.input_combo.setMaximumWidth(360)
        devices_layout.addWidget(self.input_combo)
        self.output_caption = label("ОБРАБОТАННЫЙ ЗВУК / ВЫХОД", "small")
        devices_layout.addWidget(self.output_caption)
        self.output_combo = QComboBox()
        self.output_combo.setMaximumWidth(360)
        devices_layout.addWidget(self.output_combo)
        self.monitor_check = QCheckBox("Слышать себя в реальном времени")
        self.monitor_check.setToolTip("Включить или выключить обработанный голос в выбранных наушниках.")
        self.monitor_check.toggled.connect(self.toggle_monitoring)
        devices_layout.addWidget(self.monitor_check)
        devices_layout.addWidget(label("НАУШНИКИ ДЛЯ ПРОСЛУШИВАНИЯ", "small"))
        self.monitor_combo = QComboBox()
        self.monitor_combo.setMaximumWidth(360)
        self.monitor_combo.setAccessibleName("Наушники для прослушивания")
        devices_layout.addWidget(self.monitor_combo)
        devices_layout.addWidget(label("Для прослушивания используйте наушники, чтобы избежать эха.", "small", True))
        self.route_hint = label("", "small", True)
        devices_layout.addWidget(self.route_hint)
        self.install_cable_button = QPushButton("Установить виртуальный микрофон ↗")
        self.install_cable_button.setToolTip(
            "Скачать официальный VB-CABLE и запустить установку. Windows запросит права администратора."
            if virtual_microphone_helper() else "Открыть официальный сайт VB-CABLE для отдельной установки драйвера."
        )
        self.install_cable_button.clicked.connect(self.install_virtual_microphone)
        devices_layout.addWidget(self.install_cable_button)
        self.refresh_button = QPushButton("↻  Обновить устройства")
        self.refresh_button.clicked.connect(self.refresh_devices)
        devices_layout.addWidget(self.refresh_button)
        self.device_note = label("", "small", True)
        devices_layout.addWidget(self.device_note)
        right.addWidget(devices_panel)
        right.addWidget(ai_panel)

        effect_panel, effect_layout = panel()
        effect_layout.addWidget(label("Настрой свой звук", "section"))
        self.clean_ai_check = QCheckBox("Чистый AI — без дополнительных эффектов")
        self.clean_ai_check.setChecked(self.preferences.get("clean_ai", True) is not False)
        self.clean_ai_check.setToolTip("Оставить только нейросетевую замену. Не удаляет шум, уже присутствующий в микрофоне или модели.")
        self.clean_ai_check.toggled.connect(self.update_settings)
        effect_layout.addWidget(self.clean_ai_check)
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
        self.bypass = QCheckBox("Исходный микрофон — отключить обработку")
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
        transport_footer = QHBoxLayout()
        transport_footer.addWidget(self.record_label, 1)
        transport_footer.addWidget(self.monitor_check)
        transport_layout.addLayout(transport_footer)
        outer.addWidget(transport)
        footer = QHBoxLayout()
        footer.addWidget(label("●  Обработка на устройстве · Без аккаунта и облака", "small"))
        footer.addStretch()
        footer.addWidget(label(f"{APP_NAME}  /  {__version__}", "small"))
        outer.addLayout(footer)
        self.setCentralWidget(root)
        self.select_preset(self.selected)
        tuning = self.preferences.get("tuning")
        if isinstance(tuning, dict) and (not self.ai.active or self.preferences.get("voice_mode") == "ai"):
            for key, slider in (("pitch", self.pitch_slider), ("timbre", self.timbre_slider),
                                ("clarity", self.clarity_slider), ("room", self.room_slider),
                                ("gain", self.gain_slider)):
                value = tuning.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    slider.setValue(round(value))
            self.russian_speech.setChecked(bool(tuning.get("russian_speech", True)))
        self.refresh_devices()
        self.ai.changed.connect(self.ai_changed)
        self.ai_changed()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(50)

    def filter_presets(self, *args):
        query = self.search.text().strip().casefold().replace("ё", "е")
        if query != self._voice_query:
            self._voice_page = 0
            self._voice_query = query
        category = self.category_combo.currentText()
        visible = 0
        self.category_combo.setEnabled(not self.ai.active)
        self.category_combo.setVisible(not self.ai.active)
        for button in self.ai_buttons.values():
            self.preset_grid.removeWidget(button)
            button.hide()
        if self.ai.active:
            self.catalog_title.setText("Нейросетевые голоса")
            for button in self.preset_buttons.values():
                self.preset_grid.removeWidget(button)
                button.hide()
            matches = [voice for voice in self.ai.voices
                       if query in f"{voice.name} {roles_description(voice.name)}".casefold().replace("ё", "е")]
            page_count = max(1, (len(matches) + 39) // 40)
            self._voice_page = min(self._voice_page, page_count - 1)
            page = matches[self._voice_page * 40:(self._voice_page + 1) * 40]
            if len(self.ai.voices) > 40:
                keys = {voice.id for voice in page}
                for key, button in list(self.ai_buttons.items()):
                    if key not in keys:
                        button.deleteLater()
                        del self.ai_buttons[key]
            self.previous_voices.setVisible(page_count > 1)
            self.next_voices.setVisible(page_count > 1)
            self.voice_page_label.setVisible(page_count > 1)
            self.previous_voices.setEnabled(self._voice_page > 0)
            self.next_voices.setEnabled(self._voice_page + 1 < page_count)
            self.voice_page_label.setText(f"Страница {self._voice_page + 1} / {page_count}")
            for voice in page:
                button = self.ai_buttons.get(voice.id)
                if button is None:
                    button = QToolButton()
                    button.setObjectName("preset")
                    button.setCheckable(True)
                    button.setIcon(avatar_icon(voice.id, "Игровые"))
                    button.setIconSize(QSize(88, 88))
                    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
                    button.setMinimumHeight(140)
                    button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                    button.clicked.connect(lambda checked=False, key=voice.id: self.ai.voice_combo.setCurrentIndex(self.ai.voice_combo.findData(key)))
                    self.ai_buttons[voice.id] = button
                roles = actor_roles(voice.name)
                button.setText("\n".join([voice.name.split(" — ")[0], *(role for role, _ in roles)]))
                button.setMinimumHeight(180 if roles else 140)
                button.setToolTip(f"Нейросетевая модель: {voice.name}\n{roles_description(voice.name)}\n{voice.credit}".strip())
                button.setChecked(self.ai.voice is not None and voice.id == self.ai.voice.id)
                live_switch = (self.engine.running and getattr(self.engine.neural_voice, "backend", "rvc") == "native"
                               and getattr(voice, "backend", "rvc") == "native")
                button.setEnabled((not self.engine.running or live_switch) and self._render_thread is None)
                button.setVisible(True)
                self.preset_grid.addWidget(button, visible // 4, visible % 4)
                visible += 1
            self.voice_count.setText(f"{len(matches)} AI-ГОЛОСОВ")
            self.empty_label.setText("Добавьте модель AI-голоса кнопкой справа." if not self.ai.voices else "Голоса не найдены.")
            self.empty_label.setVisible(visible == 0)
            return
        for widget in (self.previous_voices, self.next_voices, self.voice_page_label):
            widget.hide()
        self.catalog_title.setText("Эффекты голоса")
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
        self.voice_count.setText(f"{visible} / {len(PRESETS)} ЭФФЕКТОВ")
        self.empty_label.setText("Эффекты не найдены. Попробуйте другой запрос.")
        self.empty_label.setVisible(visible == 0)

    def change_voice_page(self, delta):
        self._voice_page = max(0, self._voice_page + delta)
        self.filter_presets()

    def ai_changed(self):
        if (self.engine.running and self.ai.voice is not None
                and getattr(self.engine.neural_voice, "backend", "rvc") == "native"
                and getattr(self.ai.voice, "backend", "rvc") == "native"
                and self.engine.neural_voice.id != self.ai.voice.id):
            self.engine.switch_native_voice(self.ai.voice)
        if self.ai.active and not self._previous_ai_active:
            self.select_preset(PRESETS[0])
        self._previous_ai_active = self.ai.active
        self.update_settings()
        self.filter_presets()
        if self.ai.active:
            if self.ai.voice:
                caption = roles_caption(self.ai.voice.name)
                self.description.setText(f"AI: {self.ai.voice.name}. "
                                         + (roles_description(self.ai.voice.name) + " " if caption else "")
                                         + "Голос преобразуется локальной нейросетью.")
            else:
                self.description.setText("Для естественной замены нужна обученная модель голоса.")
        else:
            self.description.setText(self.selected.description)
        if not self.engine.running:
            self.start_button.setEnabled(self.ai.ready and self.input_combo.currentData() is not None and self.output_combo.currentData() is not None)

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

    def open_cartoons(self):
        if self._render_thread is not None or (self.engine.running and self.ai.active):
            return
        self.ai.mode.setCurrentIndex(self.ai.mode.findData("effects"))
        self.search.clear()
        self.category_combo.setCurrentText("Мультяшные")
        self.select_preset(CARTOON_PRESETS[0])

    def open_clean_ai(self):
        if self._render_thread is not None or (self.engine.running and not self.ai.active):
            return
        self.ai.mode.setCurrentIndex(self.ai.mode.findData("ai"))
        self.search.clear()
        self.bypass.setChecked(False)
        self.clean_ai_check.setChecked(True)
        self.select_preset(PRESET_BY_ID["clean"])
        self.ai_changed()

    def update_settings(self, *args):
        settings = replace(self.selected.settings, pitch=self.pitch_slider.value(),
                           timbre=self.timbre_slider.value() / 100,
                           clarity=self.clarity_slider.value() if self.russian_speech.isChecked() else 0,
                           room=self.room_slider.value() / 100, gain=self.gain_slider.value(),
                           enabled=not self.bypass.isChecked())
        clean_ai = self.ai.active and self.clean_ai_check.isChecked()
        if clean_ai:
            # Keep the neural conversion enabled; remove only our post-effects.
            settings = VoiceSettings(clarity=0, enabled=not self.bypass.isChecked())
        self.engine.configure(settings)
        self.clean_ai_check.setVisible(self.ai.active)
        for widget in (self.pitch_slider, self.timbre_slider, self.room_slider,
                       self.gain_slider, self.russian_speech):
            widget.setEnabled(not clean_ai)
        self.pitch_value.setText(f"{settings.pitch:+.0f} st")
        self.timbre_value.setText(f"{settings.timbre * 100:+.0f} %")
        self.clarity_value.setText(f"{settings.clarity:.0f} dB")
        self.clarity_slider.setEnabled(not clean_ai and self.russian_speech.isChecked())
        self.room_value.setText(f"{settings.room * 100:.0f} %")
        self.gain_value.setText(f"{settings.gain:+.0f} dB")

    def refresh_devices(self):
        previous_input = self.input_combo.currentText() or self.preferences.get("input")
        previous_output = self.output_combo.currentText() or self.preferences.get("output")
        previous_monitor = self.monitor_combo.currentText() or self.preferences.get("monitor_output")
        self.input_combo.clear()
        self.output_combo.clear()
        self.monitor_combo.clear()
        try:
            inputs, outputs = list_devices()
            # Never feed the virtual capture endpoint back into its own playback.
            inputs = [device for device in inputs if not is_virtual_input(device.name) and not is_virtual_output(device.name)]
            virtual_mode = self.routing_combo.currentData() == "virtual"
            self.output_caption.setVisible(virtual_mode)
            self.output_combo.setVisible(virtual_mode)
            monitor_outputs = [device for device in outputs
                               if not is_virtual_output(device.name) and not is_virtual_input(device.name)]
            outputs = [device for device in outputs if is_virtual_output(device.name) == virtual_mode]
            for combo, devices, previous in ((self.input_combo, inputs, previous_input),
                                              (self.output_combo, outputs, previous_output),
                                              (self.monitor_combo, monitor_outputs, previous_monitor)):
                for device in devices:
                    combo.addItem(device.label, device.index)
                selected = combo.findText(previous or "")
                if selected >= 0:
                    combo.setCurrentIndex(selected)
            ready = bool(inputs and outputs)
            self.start_button.setEnabled(ready and self.ai.ready)
            self.install_cable_button.setVisible(virtual_mode and not outputs)
            if virtual_mode:
                self.route_hint.setText("Микрофон → эффекты → VB-CABLE. В нужном приложении выберите вход CABLE Output.")
                if not outputs:
                    self.device_note.setText("Виртуальный микрофон не найден. Установите VB-CABLE, перезагрузите Windows и обновите устройства.")
                elif not inputs:
                    self.device_note.setText("Виртуальный выход найден. Подключите физический микрофон и обновите устройства.")
                else:
                    self.device_note.setText("Готово: обработанный голос пойдёт в виртуальное устройство.")
            else:
                self.route_hint.setText("Микрофон → эффекты → наушники. Включите микрофон и галочку «Слышать себя». VB-CABLE не нужен.")
                self.device_note.setText("" if ready else "Нет доступного микрофона или обычного аудиовыхода.")
            self.update_monitor_controls()
        except Exception as exc:
            self.start_button.setEnabled(False)
            self.monitor_check.setChecked(False)
            self.monitor_check.setEnabled(False)
            self.monitor_combo.setEnabled(False)
            self.device_note.setText(f"Не удалось получить устройства: {exc}")

    def update_monitor_controls(self):
        available = self.monitor_combo.count() > 0
        if not available:
            self.monitor_check.setChecked(False)
        self.monitor_check.setEnabled(available)
        self.monitor_combo.setEnabled(available and not self.engine.monitoring)

    def toggle_monitoring(self, checked):
        if self.engine.running:
            try:
                if checked:
                    self.engine.start_monitoring(self.monitor_combo.currentData())
                else:
                    self.engine.stop_monitoring()
            except Exception as exc:
                self.monitor_check.setChecked(False)
                QMessageBox.warning(self, "Прослушивание голоса",
                                    f"Не удалось включить прослушивание: {exc}\n\n"
                                    "Основная обработка микрофона продолжает работать.")
        self.update_monitor_controls()

    def toggle_audio(self):
        if self.engine.running:
            self.engine.stop()
            self.set_running(False)
            return
        if self.input_combo.currentData() is None or self.output_combo.currentData() is None:
            return
        if not self.ai.ready:
            QMessageBox.information(self, "AI-голос", "Сначала добавьте и выберите нейросетевую модель голоса.")
            return
        try:
            self.engine.configure_neural(self.ai.voice if self.ai.active else None,
                                         self.ai.provider.currentData(), self.ai.buffer.currentData(),
                                         self.ai.profile.currentData())
            output = self.output_combo.currentData() if self.routing_combo.currentData() == "virtual" else None
            self.engine.start(self.input_combo.currentData(), output)
            self._last_error = ""
            self.set_running(True)
            if self.monitor_check.isChecked():
                self.toggle_monitoring(True)
        except Exception as exc:
            self.set_running(False)
            QMessageBox.critical(self, "Не удалось включить микрофон", str(exc))

    def set_running(self, running):
        self.clean_ai_button.setEnabled(self._render_thread is None and (not running or self.ai.active))
        self.cartoon_button.setEnabled(self._render_thread is None and (not running or not self.ai.active))
        live_switch = running and getattr(self.engine.neural_voice, "backend", "rvc") == "native"
        self.ai.set_running(running, allow_voice_switch=live_switch)
        native_keys = {voice.id for voice in self.ai.voices if getattr(voice, "backend", "rvc") == "native"}
        for key, button in self.ai_buttons.items():
            button.setEnabled(not running or (live_switch and key in native_keys))
        self.update_monitor_controls()
        self.routing_combo.setEnabled(not running)
        self.input_combo.setEnabled(not running)
        self.output_combo.setEnabled(not running)
        self.refresh_button.setEnabled(not running)
        self.start_button.setText("Остановить микрофон" if running else "Включить микрофон")
        self.record_button.setEnabled(running)
        route_name = "Виртуальный микрофон включён" if self.routing_combo.currentData() == "virtual" else "Обработка микрофона включена"
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
        if not self.ai.ready:
            QMessageBox.information(self, "AI-голос", "Сначала добавьте и выберите модель голоса либо включите режим эффектов.")
            return
        if self.engine.running and self.ai.active:
            QMessageBox.information(self, "Обработка файла", "Остановите микрофон перед обработкой файла AI, чтобы освободить видеопамять.")
            return
        if self.engine.busy and not self.engine.running:
            QMessageBox.information(self, "Обработка файла", "Дождитесь завершения предыдущей обработки AI.")
            return
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
        if self.ai.active:
            self.ai.set_running(True)
            self.start_button.setEnabled(False)
        thread = QThread(self)
        worker = RenderWorker(source, destination, self.engine.settings,
                              self.ai.voice if self.ai.active else None, self.ai.provider.currentData(),
                              self.ai.profile.currentData())
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
        if not self.engine.running:
            self.ai.set_running(False)
            self.ai_changed()

    def tick(self):
        self.engine.check_health()
        if self.engine.running:
            self.record_button.setEnabled(not self.engine.loading)
            if self.engine.loading:
                self.status.setText("●  Загрузка AI-модели…")
            elif self.engine.neural_voice:
                self.status.setText(f"●  AI: {self.engine.neural_voice.name.split(' — ')[0]}")
                duration = self.engine.block_size * 1000 / self.engine.sample_rate
                profile = ("нативный AI" if getattr(self.engine.neural_voice, "backend", "rvc") == "native"
                           else "лёгкий AI" if self.engine.neural_profile == "light" else "полный AI")
                self.rate_label.setText(f"{self.engine.provider_label} · обработка {self.engine.processing_ms:.0f} мс · блок {duration:.0f} мс")
                names = {"encoder": "Анализ речи", "pitch": "Высота голоса", "voice": "Модель голоса", "decoder": "Синтез", "native": "Нативный движок"}
                self.rate_label.setToolTip(profile + ". Время обработки не равно полной задержке: добавляются "
                                          "накопление звука, аудиодрайверы и виртуальный кабель.\n"
                                          + "\n".join(f"{names.get(key, key)}: {value:.0f} мс" for key, value in self.engine.stage_ms.items()))
            elif self.selected.settings.quick_pitch:
                self.rate_label.setText(f"Мульт-стиль · обработка {self.engine.processing_ms:.1f} мс")
                self.rate_label.setToolTip("Быстрый DSP без нейросети. Сдвиг высоты использует историю 1–25 мс; "
                                          "полная задержка также зависит от устройств и виртуального кабеля.")
        if self.engine.running and self.monitor_check.isChecked() and not self.engine.monitoring:
            self.monitor_check.setChecked(False)
        if self.engine.monitor_error:
            message = self.engine.monitor_error
            self.engine.monitor_error = ""
            QMessageBox.warning(self, "Прослушивание голоса", message)
        self.input_meter.setValue(round(min(1, self.engine.input_level) * 100))
        self.output_meter.setValue(round(min(1, self.engine.output_level) * 100))
        samples = self.engine.waveform if self.engine.running else np.zeros(128)
        self.waveform.samples = samples
        self.waveform.update()
        if self.engine.running:
            if self.engine.loading:
                self.signal_hint.setText("Загрузка нейросетей. Звук начнёт передаваться после подготовки модели.")
            elif self.engine.neural_voice and self.engine.processing_ms > self.engine.block_size * 1000 / self.engine.sample_rate:
                self.signal_hint.setText("Мало запаса по времени. Если звук прерывается, остановите микрофон и увеличьте буфер AI."
                                         if self.engine.native else
                                         "AI не успевает за речью. Остановите микрофон: выберите лёгкий AI, видеокарту и буфер 480/640 мс.")
            else:
                self.signal_hint.setText(f"Потери аудиобуфера: {self.engine.dropouts}. "
                                         + ("Если счётчик растёт, увеличьте буфер AI." if self.engine.native else
                                            "При росте счётчика закройте тяжёлые приложения."))
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
        QMessageBox.information(self, "Как пользоваться", (
            "Нажмите «Чистый AI · без дополнительных эффектов» и выберите карточку голоса. "
            "Дождитесь загрузки модели после включения микрофона. "
            "Если звук прерывается, остановите микрофон и увеличьте буфер AI в настройках справа.\n\n"
            f"1. В {APP_NAME} выберите режим «Виртуальный микрофон» и ваш физический микрофон.\n\n"
            "2. Если виртуального выхода нет, нажмите «Установить виртуальный микрофон». "
            "В официальном установщике VB-CABLE нажмите Install Driver и перезагрузите Windows, если потребуется.\n\n"
            "3. Выход приложения — CABLE Input. В списке микрофонов Windows и в нужных программах "
            "выбирайте CABLE Output (VB-Audio Virtual Cable). Прямое подключение к Discord не требуется.\n\n"
            "4. Выберите голос и нажмите «Включить микрофон». Для прослушивания выберите наушники "
            "и включите галочку «Слышать себя в реальном времени». Её можно переключать во время работы. "
            "Используйте наушники, чтобы избежать эха.\n\n"
            "Для прослушивания без драйвера есть режим «Только наушники — без драйвера». "
            "Обработанный голос также можно записать в WAV или применить эффекты к готовому файлу.\n\n"
            "В AI-режиме звучание задаётся нейросетевой моделью. Чистый AI отключает дополнительные DSP-эффекты; "
            "он не удаляет шум из микрофона или артефакты модели. Экспериментальные мульт-эффекты "
            "могут давать двоение и металлическую окраску и не являются голосами конкретных персонажей."
        ))

    def install_virtual_microphone(self):
        helper = virtual_microphone_helper()
        if helper:
            started, _ = QProcess.startDetached("powershell.exe", [
                "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(helper),
            ])
            if not started:
                QMessageBox.warning(self, "Виртуальный микрофон", "Не удалось запустить установку VB-CABLE.")
            return
        QDesktopServices.openUrl(QUrl("https://vb-audio.com/Cable/"))

    def closeEvent(self, event: QCloseEvent):
        if self.ai.dialog is not None and self.ai.dialog.thread is not None:
            self.ai.dialog.reject()
            event.ignore()
            return
        if self._render_thread is not None:
            QMessageBox.information(self, "Обработка файла", "Дождитесь завершения обработки перед выходом.")
            event.ignore()
            return
        self.timer.stop()
        self.engine.stop()
        try:
            save_preferences({**self.ai.settings(), "clean_ai": self.clean_ai_check.isChecked(),
                              "preset": self.selected.id, "input": self.input_combo.currentText(),
                              "output": self.output_combo.currentText(),
                              "routing": self.routing_combo.currentData(),
                              "monitor_output": self.monitor_combo.currentText(),
                              "tuning": {"pitch": self.pitch_slider.value(), "timbre": self.timbre_slider.value(),
                                         "clarity": self.clarity_slider.value(), "room": self.room_slider.value(),
                                         "gain": self.gain_slider.value(), "russian_speech": self.russian_speech.isChecked()}})
        except OSError:
            pass
        event.accept()


def run():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
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
