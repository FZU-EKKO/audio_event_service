audio_event_service

Standalone audio event classification service for `ekko`.

Purpose:

- classify a sentence-level audio clip after `ekko` VAD segmentation
- estimate whether the clip is mainly `speech`, `breathing`, or `noise`
- decide whether the clip should be dropped before sending it to `ekko_asr_service`

Interfaces:

- `GET /health`
- `POST /audio-events/classify`

Run:

```bash
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 19003 --reload
```

Download local YAMNet model:

```bash
python download_yamnet.py
```

Environment:

```env
EKKO_AUDIO_EVENT_SERVICE_TOKEN=
EKKO_AUDIO_EVENT_MODEL_PATH=./models/yamnet
EKKO_AUDIO_EVENT_TARGET_SAMPLE_RATE=16000
EKKO_AUDIO_EVENT_SPEECH_THRESHOLD=0.35
EKKO_AUDIO_EVENT_DROP_MARGIN=0.08
```

Notes:

- this service is intentionally isolated from `ekko` and `ekko_asr_service`
- `ekko` should call this service after sentence segmentation and before ASR
- this service now expects a local YAMNet model directory
- `python download_yamnet.py` will download the official model into `audio_event_service/models/yamnet`
- you can download the model yourself and point `EKKO_AUDIO_EVENT_MODEL_PATH` to that local directory
- `EKKO_AUDIO_EVENT_MODEL_HANDLE` is still accepted as a backward-compatible alias, but it should also point to a local path
