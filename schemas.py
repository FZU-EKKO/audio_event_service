from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AudioEventClassifyRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1)
    audio_format: str = Field(default="wav")
    sample_rate: int = Field(default=16000, gt=0)
    channels: int = Field(default=1, gt=0)
    sample_width: int = Field(default=2, gt=0)
    top_k: int = Field(default=8, ge=1, le=20)


class AudioEventLabelScore(BaseModel):
    label: str
    score: float


class AudioEventClassifyResponse(BaseModel):
    is_speech: bool
    should_drop: bool
    speech_score: float
    breathing_score: float
    noise_score: float
    dominant_label: str
    top_labels: list[AudioEventLabelScore] = Field(default_factory=list)


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    ready: bool
    model_handle: str
    target_sample_rate: int
    speech_threshold: float
    drop_margin: float
    model_loaded: bool = False
    model_load_error: str | None = None
    last_classify_error: str | None = None
