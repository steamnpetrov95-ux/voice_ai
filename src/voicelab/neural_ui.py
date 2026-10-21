"""AI voice selection and cancellable installation, outside the audio/GUI threads."""

import threading

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from .models import DownloadCancelled, install_voice, installed_voices


class InstallWorker(QObject):
    progress = Signal(str, int)
    done = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, source, name, cancel):
        super().__init__()
        self.source, self.name, self.cancel = source, name, cancel

    @Slot()
    def run(self):
        try:
            voice = install_voice(self.source, self.name, self.progress.emit, self.cancel)
            self.done.emit(voice)
        except DownloadCancelled:
            pass
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class InstallDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Добавить нейросетевой голос")
        self.setMinimumWidth(620)
        self.thread = None
        self.worker = None
        self.result_voice = None
        self.cancel = threading.Event()
        layout = QVBoxLayout(self)
        text = QLabel("Выберите готовую модель RVC ONNX или вставьте ссылку на неё с Hugging Face. "
                      "Общие модели речи скачаются автоматически один раз (около 700 МБ). "
                      "После установки голос работает без интернета.")
        text.setWordWrap(True)
        layout.addWidget(text)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Название голоса в вашей библиотеке")
        layout.addWidget(self.name_edit)
        row = QHBoxLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Файл .onnx или https://huggingface.co/…/resolve/…/voice.onnx")
        row.addWidget(self.source_edit, 1)
        self.browse = QPushButton("Выбрать файл…")
        self.browse.clicked.connect(self.browse_file)
        row.addWidget(self.browse)
        layout.addLayout(row)
        self.status = QLabel("Узнаваемый голос персонажа требует модели именно его озвучки.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Установить голос")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        self.buttons.accepted.connect(self.install)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def browse_file(self):
        source, _ = QFileDialog.getOpenFileName(self, "Голосовая модель", "", "RVC ONNX (*.onnx)")
        if source:
            self.source_edit.setText(source)

    def install(self):
        if self.thread is not None:
            return
        if not self.name_edit.text().strip() or not self.source_edit.text().strip():
            self.status.setText("Введите название и выберите модель голоса.")
            return
        self.cancel.clear()
        self.result_voice = None
        self.name_edit.setEnabled(False)
        self.source_edit.setEnabled(False)
        self.browse.setEnabled(False)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.status.setText("Проверка источника и установка…")
        self.thread = QThread(self)
        self.worker = InstallWorker(self.source_edit.text().strip(), self.name_edit.text().strip(), self.cancel)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    @Slot(str, int)
    def on_progress(self, text, percent):
        self.status.setText(text)
        self.progress.setValue(percent)

    @Slot(object)
    def on_done(self, voice):
        self.result_voice = voice

    @Slot(str)
    def on_error(self, message):
        self.status.setText(f"Голос не установлен: {message}")

    @Slot()
    def on_finished(self):
        thread, self.thread = self.thread, None
        self.worker = None
        thread.deleteLater()
        self.name_edit.setEnabled(True)
        self.source_edit.setEnabled(True)
        self.browse.setEnabled(True)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        if self.cancel.is_set():
            super().reject()
        elif self.result_voice is not None:
            self.accept()

    def reject(self):
        if self.thread is not None:
            self.cancel.set()
            self.status.setText("Отмена установки… Дождитесь завершения текущего чтения файла.")
            return
        super().reject()

    def closeEvent(self, event):
        if self.thread is not None:
            self.reject()
            event.ignore()
        else:
            event.accept()


class NeuralPanel(QWidget):
    changed = Signal()

    def __init__(self, preferences, parent=None):
        super().__init__(parent)
        self.dialog = None
        self.voices = installed_voices()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.mode = QComboBox()
        self.mode.addItem("AI — замена голоса нейросетью", "ai")
        self.mode.addItem("Эффекты — высота, тембр, робот", "effects")
        self.mode.setCurrentIndex(1 if preferences.get("voice_mode") == "effects" else 0)
        layout.addWidget(self.mode)
        row = QHBoxLayout()
        self.voice_combo = QComboBox()
        self.voice_combo.setAccessibleName("Нейросетевой голос")
        row.addWidget(self.voice_combo, 1)
        self.add_button = QPushButton("Добавить AI-голос…")
        self.add_button.clicked.connect(self.add_voice)
        row.addWidget(self.add_button)
        layout.addLayout(row)
        self.provider = QComboBox()
        self.provider.addItem("Ускорение: автоматически (видеокарта / CPU)", "auto")
        self.provider.addItem("Видеокарта (DirectML)", "dml")
        self.provider.addItem("Процессор (CPU, медленнее)", "cpu")
        chosen = self.provider.findData(preferences.get("ai_provider"))
        if chosen >= 0:
            self.provider.setCurrentIndex(chosen)
        layout.addWidget(self.provider)
        self.buffer = QComboBox()
        for milliseconds in (160, 320, 480, 640):
            self.buffer.addItem(f"Буфер AI: {milliseconds} мс", milliseconds)
        selected_buffer = self.buffer.findData(preferences.get("ai_buffer_ms", 320))
        self.buffer.setCurrentIndex(selected_buffer if selected_buffer >= 0 else 1)
        self.buffer.setToolTip("Меньший буфер уменьшает задержку. При прерываниях звука выберите больший.")
        layout.addWidget(self.buffer)
        for combo in (self.voice_combo, self.provider, self.buffer):
            combo.setMinimumContentsLength(12)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.reload(preferences.get("ai_voice"))
        for combo in (self.mode, self.voice_combo, self.provider, self.buffer):
            combo.currentIndexChanged.connect(self.update_state)
        self.update_state()

    @property
    def active(self):
        return self.mode.currentData() == "ai"

    @property
    def voice(self):
        identifier = self.voice_combo.currentData()
        return next((voice for voice in self.voices if voice.id == identifier), None)

    @property
    def ready(self):
        return not self.active or self.voice is not None

    def reload(self, identifier=None):
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        self.voices = installed_voices()
        for voice in self.voices:
            self.voice_combo.addItem(voice.name, voice.id)
        index = self.voice_combo.findData(identifier)
        if index >= 0:
            self.voice_combo.setCurrentIndex(index)
        self.voice_combo.blockSignals(False)

    def update_state(self, *args):
        self.voice_combo.setVisible(self.active)
        self.provider.setVisible(self.active)
        self.buffer.setVisible(self.active)
        self.add_button.setVisible(self.active)
        if self.active:
            self.note.setText("Модель: " + self.voice.name + ". Интонации и язык следуют вашей речи."
                              if self.voice else "AI-голоса пока не установлены. Добавьте обученную модель нужного персонажа.")
        else:
            self.note.setText("48 наборов эффектов. Для естественной замены личности голоса выберите режим AI.")
        self.changed.emit()

    def set_running(self, running):
        for widget in (self.mode, self.voice_combo, self.provider, self.buffer, self.add_button):
            widget.setEnabled(not running)

    def add_voice(self):
        dialog = InstallDialog(self)
        self.dialog = dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.reload(dialog.result_voice.id)
            self.mode.setCurrentIndex(0)
            self.update_state()
        self.dialog = None
        dialog.deleteLater()

    def settings(self):
        return {"voice_mode": self.mode.currentData(), "ai_voice": self.voice_combo.currentData(),
                "ai_provider": self.provider.currentData(), "ai_buffer_ms": self.buffer.currentData()}
