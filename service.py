from __future__ import annotations

import base64
import csv
import io
import logging
from pathlib import Path
from functools import lru_cache

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

from config import (
    EVENT_DROP_MARGIN,
    EVENT_MODEL_PATH,
    EVENT_SPEECH_THRESHOLD,
    EVENT_TARGET_SAMPLE_RATE,
)


logger = logging.getLogger("audio_event_service")

_SPEECH_KEYWORDS = (
    "speech",
    "conversation",
    "narration",
    "monologue",
    "babbling",
    "whisper",
    "shout",
    "yell",
    "screaming",
)
_BREATHING_KEYWORDS = (
    "breathing",
    "sniff",
    "wheeze",
    "gasp",
    "pant",
    "snort",
    "cough",
    "throat clearing",
    "sneeze",
)
_NOISE_KEYWORDS = (
    "noise",
    "hiss",
    "hum",
    "buzz",
    "wind",
    "static",
    "white noise",
    "pink noise",
    "rustle",
    "crackle",
    "distortion",
)


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / max(peak, 1e-6)
    return audio


def _decode_audio(audio_base64: str, audio_format: str) -> tuple[np.ndarray, int]:
    raw = base64.b64decode(audio_base64)
    fmt = (audio_format or "wav").lower()

    if fmt == "wav":
        with sf.SoundFile(io.BytesIO(raw)) as wav_file:
            audio = wav_file.read(dtype="float32")
            sample_rate = int(wav_file.samplerate)
        return _normalize_audio(audio), sample_rate

    raise ValueError(f"Unsupported audio_format: {audio_format}")


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    if audio.size == 0:
        return audio
    return resampy.resample(audio, source_rate, target_rate).astype(np.float32)


def _read_labels_from_model(model) -> list[str]:
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with tf.io.gfile.GFile(class_map_path) as csv_file:
        rows = list(csv.DictReader(csv_file))
    labels = [row["display_name"] for row in rows]
    if not labels:
        raise RuntimeError("YAMNet class map is empty")
    return labels


def _score_bucket(label_scores: list[tuple[str, float]], keywords: tuple[str, ...]) -> float:
    total = 0.0
    for label, score in label_scores:
        lower = label.lower()
        if any(keyword in lower for keyword in keywords):
            total += score
    return float(total)


@lru_cache(maxsize=1)
def _load_runtime() -> tuple[object, list[str]]:
    if not EVENT_MODEL_PATH:
        raise RuntimeError("EKKO_AUDIO_EVENT_MODEL_PATH is not configured")
    model_path = Path(EVENT_MODEL_PATH).expanduser()
    if not model_path.exists():
        raise RuntimeError(f"YAMNet model path does not exist: {model_path}")
    logger.info("loading yamnet model path=%s", model_path)
    model = hub.load(str(model_path))
    labels = _read_labels_from_model(model)
    logger.info("yamnet model loaded labels=%s", len(labels))
    return model, labels


def warmup_model() -> tuple[bool, str | None]:
    try:
        _load_runtime()
    except Exception as exc:
        logger.exception("yamnet warmup failed")
        return False, repr(exc)
    return True, None


def get_runtime_status() -> dict[str, object]:
    return {
        "model_handle": EVENT_MODEL_PATH,
        "target_sample_rate": EVENT_TARGET_SAMPLE_RATE,
        "speech_threshold": EVENT_SPEECH_THRESHOLD,
        "drop_margin": EVENT_DROP_MARGIN,
    }


class AudioEventService:
    def classify(self, *, audio_base64: str, audio_format: str = "wav", top_k: int = 8) -> dict:
        model, labels = _load_runtime()
        audio, sample_rate = _decode_audio(audio_base64, audio_format)
        audio = _resample_audio(audio, sample_rate, EVENT_TARGET_SAMPLE_RATE)

        if audio.size == 0:
            raise ValueError("Audio payload is empty after decode")

        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, _embeddings, _spectrogram = model(waveform)
        mean_scores = np.asarray(scores.numpy(), dtype=np.float32).mean(axis=0)

        ranked_indices = np.argsort(mean_scores)[::-1]
        top_indices = ranked_indices[:top_k]
        top_labels = [
            {"label": labels[int(index)], "score": float(mean_scores[int(index)])}
            for index in top_indices
        ]

        all_label_scores = [(labels[int(index)], float(mean_scores[int(index)])) for index in ranked_indices]
        speech_score = _score_bucket(all_label_scores, _SPEECH_KEYWORDS)
        breathing_score = _score_bucket(all_label_scores, _BREATHING_KEYWORDS)
        noise_score = _score_bucket(all_label_scores, _NOISE_KEYWORDS)

        dominant_label = top_labels[0]["label"] if top_labels else "unknown"
        is_speech = speech_score >= EVENT_SPEECH_THRESHOLD and speech_score >= breathing_score + EVENT_DROP_MARGIN
        should_drop = not is_speech and (breathing_score > 0.0 or noise_score > 0.0)

        logger.info(
            "classify success dominant=%s speech=%.4f breathing=%.4f noise=%.4f top_labels=%s",
            dominant_label,
            speech_score,
            breathing_score,
            noise_score,
            [item["label"] for item in top_labels[:3]],
        )
        return {
            "is_speech": bool(is_speech),
            "should_drop": bool(should_drop),
            "speech_score": float(speech_score),
            "breathing_score": float(breathing_score),
            "noise_score": float(noise_score),
            "dominant_label": dominant_label,
            "top_labels": top_labels,
        }
