import pytest

from voicelab.routing import is_virtual_input, is_virtual_output


@pytest.mark.parametrize("name", [
    "CABLE Input (VB-Audio Virtual Cable)",
    "CABLE-A Input (VB-Audio Cable A)",
    "CABLE-B Input (VB-Audio Cable B)",
    "CABLE In 16ch (VB-Audio Virtual Cable)",
])
def test_recognizes_virtual_playback(name):
    assert is_virtual_output(name)
    assert not is_virtual_input(name)


@pytest.mark.parametrize("name", ["CABLE Output (VB-Audio Virtual Cable)", "CABLE-A Output", "CABLE Out 16ch"])
def test_capture_endpoint_cannot_be_used_as_playback(name):
    assert is_virtual_input(name)
    assert not is_virtual_output(name)


@pytest.mark.parametrize("name", ["USB Microphone", "Speakers", "Headphones", "Cable microphone"])
def test_physical_devices_are_not_virtual_cable(name):
    assert not is_virtual_output(name)
    assert not is_virtual_input(name)
