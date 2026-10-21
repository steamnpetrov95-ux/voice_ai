#!/usr/bin/env bash
set -euo pipefail
cd /workspace/voice_ai
export UV_CACHE_DIR=/workspace/.cache/uv
uv sync --frozen --extra dev

audio_root=/workspace/.local/audio-libs
audio_lib="$audio_root/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$audio_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if ! .venv/bin/python -c 'import sounddevice' >/dev/null 2>&1; then
    # Use official signed Debian indexes without changing system package sources.
    # apt verifies the downloaded packages against these authenticated indexes.
    . /etc/os-release
    if [[ "$ID" != debian || "$(uname -m)" != x86_64 ]]; then
        echo 'Install the platform PortAudio library with your package manager.' >&2
        exit 1
    fi
    apt_root=/workspace/.cache/voicelab-apt
    mkdir -p "$apt_root/lists/partial" "$apt_root/packages" "$audio_root"
    printf 'deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] https://deb.debian.org/debian %s main\n' "$VERSION_CODENAME" > "$apt_root/sources.list"
    apt_options=(-o "Dir::Etc::sourcelist=$apt_root/sources.list" -o Dir::Etc::sourceparts=- -o "Dir::State::Lists=$apt_root/lists" -o Debug::NoLocking=1)
    apt-get "${apt_options[@]}" -o APT::Update::Error-Mode=any update
    (
        cd "$apt_root/packages"
        apt-get "${apt_options[@]}" download libportaudio2 libjack-jackd2-0
        for package in ./*.deb; do
            dpkg-deb -x "$package" "$audio_root"
        done
    )
    ln -sfn libportaudio.so.2 "$audio_lib/libportaudio.so"
fi
.venv/bin/python -c 'import sounddevice; print("PortAudio:", sounddevice.get_portaudio_version()[1])'
export XDG_CACHE_HOME=/workspace/.cache
export VOICELAB_DATA_DIR=/workspace/voice_ai/.local/settings
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
