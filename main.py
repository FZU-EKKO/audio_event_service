from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from config import EVENT_SERVICE_TOKEN
from schemas import AudioEventClassifyRequest, AudioEventClassifyResponse, HealthResponse
from service import AudioEventService, get_runtime_status, warmup_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("audio_event_service")

app = FastAPI(title="audio_event_service")
app.state.model_loaded = False
app.state.model_load_error = None
app.state.last_classify_error = None


def verify_token(authorization: str | None = Header(default=None)) -> None:
    if not EVENT_SERVICE_TOKEN:
        return
    expected = f"Bearer {EVENT_SERVICE_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.on_event("startup")
async def startup_warmup() -> None:
    ready, error = warmup_model()
    app.state.model_loaded = ready
    app.state.model_load_error = error
    if ready:
        logger.info("startup warmup success")
    else:
        logger.error("startup warmup failed detail=%s", error)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    logger.exception("unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": repr(exc)},
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    runtime = get_runtime_status()
    ready = bool(app.state.model_loaded and not app.state.model_load_error)
    return HealthResponse(
        status="ok" if ready else "degraded",
        ready=ready,
        model_handle=runtime["model_handle"],
        target_sample_rate=runtime["target_sample_rate"],
        speech_threshold=runtime["speech_threshold"],
        drop_margin=runtime["drop_margin"],
        model_loaded=bool(app.state.model_loaded),
        model_load_error=app.state.model_load_error,
        last_classify_error=app.state.last_classify_error,
    )


@app.post(
    "/audio-events/classify",
    response_model=AudioEventClassifyResponse,
    dependencies=[Depends(verify_token)],
)
def classify(req: AudioEventClassifyRequest) -> AudioEventClassifyResponse:
    logger.info(
        "http classify request format=%s audio_base64_chars=%s top_k=%s",
        req.audio_format,
        len(req.audio_base64 or ""),
        req.top_k,
    )
    app.state.last_classify_error = None

    try:
        result = AudioEventService().classify(
            audio_base64=req.audio_base64,
            audio_format=req.audio_format,
            top_k=req.top_k,
        )
    except ValueError as exc:
        app.state.last_classify_error = str(exc)
        logger.warning("http classify bad_request detail=%s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        app.state.last_classify_error = repr(exc)
        logger.exception("http classify failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=repr(exc)) from exc

    logger.info(
        "http classify success dominant=%s is_speech=%s should_drop=%s",
        result.get("dominant_label"),
        result.get("is_speech"),
        result.get("should_drop"),
    )
    return AudioEventClassifyResponse(**result)
