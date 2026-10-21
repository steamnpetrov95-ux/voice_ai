# Проверка в облачной среде

Целевой продукт — Windows-приложение. Текущая машина — Linux без физических аудиоустройств. Дополнительно подготовлены Windows Python 3.12.10 и Wine 10 для сборки и проверки Windows EXE. Фактический захват микрофона, установка системного драйвера VB-CABLE и Discord требуют настоящей Windows.

В этой машине зависимости установлены в `/workspace/voice_ai/.venv`. PortAudio и libjack извлечены из пакетов Debian trixie в `/workspace/.local/audio-libs`. Пакеты получены через подписанный индекс официального репозитория Debian; проверка TLS и подписей не отключалась. Symlink `libportaudio.so` нужен, чтобы `ctypes.util.find_library` нашёл библиотеку вне системного каталога.

```bash
cd /workspace/voice_ai
export LD_LIBRARY_PATH=/workspace/.local/audio-libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export XDG_CACHE_HOME=/workspace/.cache
export VOICELAB_DATA_DIR=/workspace/voice_ai/.local/settings
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

Для новой Linux-машины предпочтительна установка PortAudio штатным пакетным менеджером. Резервный способ без записи в `/usr` автоматизирован в `scripts/setup-cloud.sh`. Он предназначен только для Debian-подобной облачной машины, не для Windows-пользователя.

Список устройств в этой среде пуст; кнопка включения микрофона корректно отключена. Обработка аудиофайлов доступна. Результат тестов в этой машине не подтверждает работу физического микрофона или драйвера на Windows.

Проверено на этапе добавления установщика:

- 86 тестов проходят в Linux; те же тесты выполняются Windows Python через Wine. Устройства в тестах подменяются, DSP и Qt настоящие.
- Собран настоящий Windows EXE через PyInstaller. Собранный EXE обрабатывает WAV: 48 000 входных кадров превращаются в 96 000 кадров с учётом секундного хвоста, сигнал ненулевой.
- Собран один `dist/installer/VoiceLab-Setup.exe` через Inno Setup 7.1.0 под Wine. Версии Qt 6.8.3 и NumPy 2.2.6 закреплены для совместимости с Windows 10.
- Выполнена установка приложения в тестовую папку, обработка WAV установленным EXE, открытие и штатное закрытие настоящего Windows-окна. Проверено наличие помощника установки виртуального микрофона. Удаление приложения через собранный деинсталлятор проверено.
- Тестовая установка выполняется с `/TASKS=desktopicon`: драйвер на Linux/Wine не устанавливается.
- Загрузка VB-CABLE из облачной машины блокируется сетевой политикой (`403 Forbidden` для `vb-audio.com` и `download.vb-audio.com`). Проверка официального архива, действительности подписи издателя и установка драйвера в этой машине не подтверждены. Windows CI выполняет `Install-VirtualMicrophone.ps1 -VerifyOnly` до публикации релиза; при ошибке проверки релиз не публикуется.

Нужные дополнения в сетевой черновик среды: `api.github.com`, `uploads.github.com`, `vb-audio.com`, `download.vb-audio.com`. Сохранение черновика не применяет их к работающей машине. API GitHub также заблокирован текущей политикой, поэтому состояние GitHub Actions нельзя подтверждать через API из этой машины.

Дополнительные инструменты сборки находятся вне checkout: `/workspace/.local/bin/voicelab-wine`, `/workspace/.cache/windows-build/python/tools/python.exe`, `/workspace/.local/inno7/ISCC.exe`. Пакеты Wine получены из подписанных индексов Debian. Windows Python и Universal CRT из официального Windows SDK NuGet проверены по SHA512 из каталога NuGet; компилятор Inno Setup получен из официального GitHub-релиза. Они не входят в исходный архив проекта.

Для Wine GUI нужен Xvfb на дисплее `:91`; процесс не следует считать сохранённым в новой задаче. Если инструменты сохранены и дисплей свободен, Xvfb можно запустить так:

```bash
/workspace/.local/wine/usr/bin/Xvfb :91 -screen 0 1280x960x24 -nolisten tcp -ac > /tmp/voicelab-xvfb.log 2>&1 &
```

Штатный способ воспроизводимой сборки — Windows CI или `Build-Windows.bat` на Windows с Inno Setup 6/7. Дополнительная сборка через Wine не заменяет проверку реальных Windows-аудиоустройств.
