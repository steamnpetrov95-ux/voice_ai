"""VB-CABLE playback endpoints feed its matching virtual capture endpoints."""

import re

_CABLE_PLAYBACK = re.compile(r"\bcable(?:-[a-d])?\s+(?:input\b|in\s+16ch\b)", re.IGNORECASE)
_CABLE_CAPTURE = re.compile(r"\bcable(?:-[a-d])?\s+(?:output\b|out\s+16ch\b)", re.IGNORECASE)


def is_virtual_output(name: str) -> bool:
    # Never select CABLE Output here: that is the capture side, not playback.
    return bool(_CABLE_PLAYBACK.search(name))


def is_virtual_input(name: str) -> bool:
    return bool(_CABLE_CAPTURE.search(name))
