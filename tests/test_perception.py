"""PRD FR-11, Appendix A.2: parsing, Gemma-to-fallback behaviour, format checks. Mocked HTTP."""

import asyncio
import io
import json
import math
import shutil
import struct
import subprocess
import wave

import httpx
import pytest
from PIL import Image

from hub import perception
from hub.perception import PerceptionError
from hub.settings import Settings


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, format="PNG")
    return buf.getvalue()


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(
            b"".join(struct.pack("<h", int(8000 * math.sin(i / 10))) for i in range(8000))
        )
    return buf.getvalue()


def _use(monkeypatch, **kw):
    s = Settings(_env_file=None, **kw)
    monkeypatch.setattr(perception, "get_settings", lambda: s)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ollama(content: str, seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})

    return handler


# --- parsing ---


def test_parse_sender_and_message():
    raw = "SENDER: VK-SBIUPD\nMESSAGE:\nYour account is blocked.\nVisit http://x.example/a"
    assert perception.parse_image_output(raw) == (
        "VK-SBIUPD",
        "Your account is blocked.\nVisit http://x.example/a",
    )


@pytest.mark.parametrize("unknown", ["unknown", "Unknown", "UNKNOWN", ""])
def test_unknown_sender_is_none(unknown):
    sender, text = perception.parse_image_output(f"SENDER: {unknown}\nMESSAGE:\nhello")
    assert sender is None and text == "hello"


def test_parse_variants():
    assert perception.parse_image_output("```\nSENDER: Bank\nMESSAGE:\nhi\n```") == ("Bank", "hi")
    assert perception.parse_image_output("SENDER: +91 70000 00000\nMESSAGE: one line") == (
        "+91 70000 00000",
        "one line",
    )
    assert perception.parse_image_output("just text, no markers") == (None, "just text, no markers")
    # the word MESSAGE inside the body is kept
    assert perception.parse_image_output("SENDER: A\nMESSAGE:\nMESSAGE: x")[1] == "MESSAGE: x"


def test_sniff_image():
    assert perception.sniff_image(_png()) == "png"
    assert perception.sniff_image(b"\xff\xd8\xff\xe0abc") == "jpeg"
    assert perception.sniff_image(b"RIFF\0\0\0\0WEBPVP8 ") == "webp"
    assert perception.sniff_image(b"%PDF-1.4") is None


# --- image ---


def test_gemma_image_request_and_parse(monkeypatch):
    _use(monkeypatch, ocr_fallback="gemma")
    seen: list[dict] = []
    client = _client(_ollama("SENDER: VK-SBIUPD\nMESSAGE:\nKYC pending", seen))
    sender, text = asyncio.run(perception.transcribe_image(_png(), "image/png", client=client))
    assert (sender, text) == ("VK-SBIUPD", "KYC pending")
    (body,) = seen
    assert body["think"] is False and body["stream"] is False
    msg = body["messages"][0]
    assert msg["content"].startswith("Transcribe the message in this screenshot")
    assert len(msg["images"]) == 1


def test_gemma_failure_falls_back_to_tesseract(monkeypatch):
    _use(monkeypatch, ocr_fallback="gemma")
    monkeypatch.setattr(perception, "_tesseract", lambda image: "ocr text")

    def boom(request):
        return httpx.Response(500)

    sender, text = asyncio.run(perception.transcribe_image(_png(), None, client=_client(boom)))
    assert (sender, text) == (None, "ocr text")


def test_gemma_empty_transcript_falls_back(monkeypatch):
    _use(monkeypatch, ocr_fallback="gemma")
    monkeypatch.setattr(perception, "_tesseract", lambda image: "ocr text")
    client = _client(_ollama("SENDER: unknown\nMESSAGE:\n"))
    assert asyncio.run(perception.transcribe_image(_png(), None, client=client)) == (
        None,
        "ocr text",
    )


def test_tesseract_setting_never_calls_gemma(monkeypatch):
    _use(monkeypatch, ocr_fallback="tesseract")
    monkeypatch.setattr(perception, "_tesseract", lambda image: "ocr text")

    def fail(request):
        raise AssertionError("gemma must not be called")

    assert asyncio.run(perception.transcribe_image(_png(), None, client=_client(fail))) == (
        None,
        "ocr text",
    )


def test_both_fail_raises_without_text(monkeypatch):
    _use(monkeypatch, ocr_fallback="gemma")
    monkeypatch.setattr(perception, "_tesseract", lambda image: "")
    with pytest.raises(PerceptionError):
        asyncio.run(
            perception.transcribe_image(_png(), None, client=_client(lambda r: httpx.Response(500)))
        )


def test_tesseract_runs_on_a_real_image(monkeypatch):
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract not installed")
    _use(monkeypatch, ocr_fallback="tesseract")
    with pytest.raises(PerceptionError):  # blank image: no text
        asyncio.run(perception.transcribe_image(_png(), "image/png"))


@pytest.mark.parametrize("data,mime", [(b"%PDF-1.4 ...", "application/pdf"), (b"", None)])
def test_unsupported_image_type(monkeypatch, data, mime):
    _use(monkeypatch, ocr_fallback="gemma")
    with pytest.raises(PerceptionError, match="unsupported image type"):
        asyncio.run(perception.transcribe_image(data, mime))


def test_non_image_mime_rejected(monkeypatch):
    _use(monkeypatch, ocr_fallback="gemma")
    with pytest.raises(PerceptionError):
        asyncio.run(perception.transcribe_image(_png(), "application/pdf"))


# --- audio ---

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _gemma_audio(content: str, seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return handler


@needs_ffmpeg
def test_gemma_audio_request_shape(monkeypatch):
    _use(monkeypatch, asr_fallback="gemma", gemma_audio_url="http://127.0.0.1:8082/v1")
    seen: list = []
    text = asyncio.run(
        perception.transcribe_audio(
            _wav(), "audio/wav", client=_client(_gemma_audio(" hello ", seen))
        )
    )
    assert text == "hello"
    ((url, body),) = seen
    assert url == "http://127.0.0.1:8082/v1/chat/completions"
    parts = body["messages"][0]["content"]
    assert parts[0]["text"].startswith("Transcribe this voice note")
    assert parts[1]["type"] == "input_audio" and parts[1]["input_audio"]["format"] == "wav"


@needs_ffmpeg
def test_m4a_input_is_converted(monkeypatch, tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(_wav())
    dst = tmp_path / "a.m4a"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), str(dst)])
    if r.returncode != 0:
        pytest.skip("ffmpeg cannot encode m4a here")
    _use(monkeypatch, asr_fallback="gemma", gemma_audio_url="http://x/v1")
    seen: list = []
    out = asyncio.run(
        perception.transcribe_audio(
            dst.read_bytes(), "audio/mp4", client=_client(_gemma_audio("ok", seen))
        )
    )
    assert out == "ok"
    assert seen[0][1]["messages"][0]["content"][1]["input_audio"]["format"] == "wav"


@needs_ffmpeg
def test_gemma_audio_failure_uses_whisper(monkeypatch):
    _use(monkeypatch, asr_fallback="gemma", gemma_audio_url="http://x/v1")
    monkeypatch.setattr(perception, "_whisper", lambda wav: "whisper text")
    out = asyncio.run(
        perception.transcribe_audio(
            _wav(), "audio/wav", client=_client(lambda r: httpx.Response(500))
        )
    )
    assert out == "whisper text"


@needs_ffmpeg
def test_whisper_when_gemma_not_configured(monkeypatch):
    _use(monkeypatch, asr_fallback="gemma", gemma_audio_url=None)
    monkeypatch.setattr(perception, "_whisper", lambda wav: "whisper text")
    assert asyncio.run(perception.transcribe_audio(_wav(), "audio/wav")) == "whisper text"
    _use(monkeypatch, asr_fallback="mlx-whisper", gemma_audio_url="http://x/v1")
    assert asyncio.run(perception.transcribe_audio(_wav(), "audio/wav")) == "whisper text"


@needs_ffmpeg
def test_whisper_missing_raises_perception_error(monkeypatch):
    _use(monkeypatch, asr_fallback="mlx-whisper")
    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", None)  # force ImportError
    with pytest.raises(PerceptionError, match="mlx-whisper"):
        asyncio.run(perception.transcribe_audio(_wav(), "audio/wav"))


@needs_ffmpeg
def test_undecodable_audio(monkeypatch):
    _use(monkeypatch, asr_fallback="mlx-whisper")
    with pytest.raises(PerceptionError, match="decode"):
        asyncio.run(perception.transcribe_audio(b"not audio at all" * 10, "audio/webm"))


def test_empty_audio():
    with pytest.raises(PerceptionError):
        asyncio.run(perception.transcribe_audio(b"", None))
