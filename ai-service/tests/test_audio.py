"""Audio summarizer endpoint (offline): the Whisper model is replaced by a stub, the real model runs only with
RUN_WHISPER=1 (it downloads ~480 MB on first use)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SAMPLE = Path(__file__).resolve().parents[2] / "mock-data" / "audio" / "consultation-sample.mp3"


class StubTranscriber:
    model_size = "stub"

    def __init__(self, text: str = "Doctor: what brings you in?\nPatient: sore throat for three days."):
        self.text = text
        self.calls = 0

    def transcribe(self, audio: bytes, filename: str = "audio") -> dict:
        self.calls += 1
        return {"transcript": self.text, "segments": [], "language": "en", "language_probability": 1.0,
                "audio_seconds": 3.0, "transcribe_ms": 1, "model": "stub"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "ckpt.sqlite3"))
    monkeypatch.setenv("MEMORY_DB", str(tmp_path / "mem.sqlite3"))
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("WHISPER_PRELOAD", "false")  # never load / download the real model in unit tests
    from app.config import get_settings

    get_settings.cache_clear()
    from app import main

    with TestClient(main.app) as c:
        main.app.state.transcriber = StubTranscriber()
        yield c
    get_settings.cache_clear()


def test_audio_is_transcribed_then_summarized(client):
    r = client.post("/v1/summarize/audio", files={"file": ("visit.mp3", b"ID3fake-audio", "audio/mpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transcript"].startswith("Doctor:") and body["language"] == "en"
    assert body["summary"]["chief_complaint"]  # structured summary from the transcript


def test_non_audio_upload_is_rejected(client):
    r = client.post("/v1/summarize/audio", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415 and r.json()["detail"]["code"] == "UNSUPPORTED_MEDIA"


def test_empty_and_silent_recordings_are_rejected(client):
    assert client.post("/v1/summarize/audio", files={"file": ("a.mp3", b"", "audio/mpeg")}).status_code == 422
    client.app.state.transcriber = StubTranscriber(text="   ")
    r = client.post("/v1/summarize/audio", files={"file": ("a.mp3", b"ID3", "audio/mpeg")})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "NO_SPEECH"


def test_without_speech_to_text_installed_the_endpoint_says_so(client):
    client.app.state.transcriber = None
    r = client.post("/v1/summarize/audio", files={"file": ("a.mp3", b"ID3", "audio/mpeg")})
    assert r.status_code == 501 and r.json()["detail"]["code"] == "TRANSCRIPTION_UNAVAILABLE"


def test_model_that_cannot_load_is_a_503_not_a_bad_file(client):
    from app.transcribe import TranscriberUnavailable

    class Broken(StubTranscriber):
        def transcribe(self, audio, filename="audio"):
            raise TranscriberUnavailable("no network to download the model")

    client.app.state.transcriber = Broken()
    r = client.post("/v1/summarize/audio", files={"file": ("a.mp3", b"ID3", "audio/mpeg")})
    assert r.status_code == 503 and "network" not in r.text  # internal details are not leaked to the client


def test_oversized_upload_is_refused_before_reading(client):
    r = client.post("/v1/summarize/audio", files={"file": ("a.mp3", b"0" * (16 * 1024 * 1024), "audio/mpeg")})
    assert r.status_code == 413 and client.app.state.transcriber.calls == 0


@pytest.mark.skipif(os.environ.get("RUN_WHISPER") != "1", reason="real Whisper model: set RUN_WHISPER=1")
def test_real_whisper_transcribes_the_sample_recording():
    from app.transcribe import Transcriber

    r = Transcriber("small").transcribe(SAMPLE.read_bytes(), SAMPLE.name)
    text = r["transcript"].lower()
    assert r["language"] == "en" and "penicillin" in text and "paracetamol" in text and "500" in text
