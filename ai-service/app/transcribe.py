"""Speech-to-text for the consultation summarizer (bonus task: voice transcription -> structured summary).

faster-whisper (CTranslate2 Whisper, int8 on CPU): no PyTorch, runs the same on a laptop, in Docker and on
Kubernetes. Audio is processed locally and the temporary file is deleted right after transcription; nothing is
sent to a third party. The model is downloaded once on first use (WHISPER_MODEL, default "small").
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

MAX_AUDIO_BYTES = 15 * 1024 * 1024
AUDIO_TYPES = ("audio/", "video/webm", "application/octet-stream")


class TranscriberUnavailable(RuntimeError):
    """The model could not be loaded (not downloaded yet and no network, disk full, ...)."""


class Transcriber:
    def __init__(self, model_size: str = "small", compute_type: str = "int8"):
        self.model_size = model_size
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()  # one model instance, loaded once, used one request at a time

    @staticmethod
    def available() -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def preload(self) -> None:
        """Load (and on first start download) the model in the background so the first request does not wait."""
        try:
            with self._lock:
                self._load()
        except Exception as exc:  # noqa: BLE001 - surfaced as 503 when a request needs it
            log.warning("Whisper preload failed: %s", exc)

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            started = time.perf_counter()
            try:  # downloads to HF_HOME (/data/hf in the container image) on first use
                self._model = WhisperModel(self.model_size, device="cpu", compute_type=self.compute_type)
            except Exception as exc:  # noqa: BLE001
                raise TranscriberUnavailable(str(exc)) from exc
            log.info("Whisper %s loaded in %.1fs", self.model_size, time.perf_counter() - started)
        return self._model

    def transcribe(self, audio: bytes, filename: str = "audio") -> dict[str, Any]:
        suffix = os.path.splitext(filename)[1] or ".bin"
        started = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:  # deleted on close
            tmp.write(audio)
            tmp.flush()
            with self._lock:
                model = self._load()
                segments, info = model.transcribe(tmp.name, beam_size=1, vad_filter=True)
                segs = [{"start": round(s.start, 1), "end": round(s.end, 1), "text": s.text.strip()} for s in segments]
        return {
            "transcript": "\n".join(s["text"] for s in segs),
            "segments": segs,
            "language": info.language,
            "language_probability": round(info.language_probability, 2),
            "audio_seconds": round(info.duration, 1),
            "transcribe_ms": int((time.perf_counter() - started) * 1000),
            "model": f"faster-whisper-{self.model_size}",
        }
