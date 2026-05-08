from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def get_int_env(name: str, default: int) -> int:
    raw = get_env(name, "")
    if not raw:
        return default
    return int(raw)


def get_float_env(name: str, default: float) -> float:
    raw = get_env(name, "")
    if not raw:
        return default
    return float(raw)


EVENT_SERVICE_TOKEN = get_env("EKKO_AUDIO_EVENT_SERVICE_TOKEN", "")
EVENT_MODEL_PATH = get_env("EKKO_AUDIO_EVENT_MODEL_PATH", get_env("EKKO_AUDIO_EVENT_MODEL_HANDLE", ""))
EVENT_TARGET_SAMPLE_RATE = get_int_env("EKKO_AUDIO_EVENT_TARGET_SAMPLE_RATE", 16000)
EVENT_SPEECH_THRESHOLD = get_float_env("EKKO_AUDIO_EVENT_SPEECH_THRESHOLD", 0.35)
EVENT_DROP_MARGIN = get_float_env("EKKO_AUDIO_EVENT_DROP_MARGIN", 0.08)
