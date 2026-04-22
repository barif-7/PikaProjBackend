"""Minimal compatibility shim for XTTS imports on Python 3.9.

The `TTS` package imports Bangla phonemizer modules eagerly, which in turn import
the third-party `bangla` package. The currently resolved `bangla` release uses
Python 3.10-only type syntax and crashes during import on the existing backend
runtime. XTTS English inference does not rely on the full package, so we provide
just the helper accessed by `TTS.tts.utils.text.bangla.phonemizer`.
"""

_EN_TO_BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def convert_english_digit_to_bangla_digit(text: str) -> str:
    return text.translate(_EN_TO_BN_DIGITS)
