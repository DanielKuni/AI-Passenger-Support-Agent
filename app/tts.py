"""
Optional local Hebrew text-to-speech with Piper.

Used only when both are present:
  - the piper-tts package (pip install -r requirements-voice.txt), and
  - the voice files voices/he_IL-saspeech-medium.onnx and .onnx.json
    (python -m piper.download_voices --download-dir voices he_IL-saspeech-medium)

Facts that matter for anyone deploying this (see README, section "Voice interface"):
  - piper-tts 1.8 is licensed GPL-3.0-or-later.
  - The he_IL-saspeech-medium voice was trained on the SASPEECH corpus (openslr.org/134), whose licence is a
    custom NON-COMMERCIAL licence with the Israeli Public Broadcasting Corporation as copyright owner.
    This demo is non-commercial. Do not ship this voice in a commercial product.
  - piper-tts restores Hebrew vowel points with a bundled Nakdimon model before converting to phonemes, so
    plain unvoweled Hebrew can be synthesised.
"""
from __future__ import annotations

import importlib.util
import io
import threading
import wave

from . import config

VOICE_NAME = "he_IL-saspeech-medium"
VOICE_DIR = config.ROOT / "voices"
VOICE_PATH = VOICE_DIR / f"{VOICE_NAME}.onnx"
MAX_CHARS = 600

_voice = None
_lock = threading.Lock()


def available() -> bool:
    return VOICE_PATH.exists() and VOICE_PATH.with_suffix(".onnx.json").exists() \
        and importlib.util.find_spec("piper") is not None


def info() -> dict:
    return {
        "engine": "piper" if available() else None,
        "voice": VOICE_NAME if available() else None,
        "licence_note": "piper-tts: GPL-3.0-or-later. Voice dataset SASPEECH: custom non-commercial licence (IPBC). Demo use only.",
    }


def _load():
    global _voice
    if _voice is None:
        from piper import PiperVoice
        _voice = PiperVoice.load(str(VOICE_PATH))
    return _voice


def synthesize_wav(text: str) -> bytes:
    """Synchronous synthesis; call through asyncio.to_thread from the server."""
    text = " ".join(text.split())[:MAX_CHARS]
    with _lock:
        voice = _load()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            voice.synthesize_wav(text, w)
    return buf.getvalue()
