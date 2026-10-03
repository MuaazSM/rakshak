"""Perceive node: image/audio transcription via Gemma with Tesseract / mlx-whisper fallbacks
(PRD FR-11, Appendix A.2, §17 OCR/ASR decisions).

Images: Gemma via Ollama (`"think": false`) when OCR_FALLBACK=gemma, else Tesseract `eng+hin`;
a Gemma failure falls back to Tesseract. Audio: the Gemma-audio llama-server (OpenAI-style
`input_audio`) when ASR_FALLBACK=gemma and GEMMA_AUDIO_URL is set, else mlx-whisper (optional
"mlx" extra, imported lazily). Android sends webm/opus and iOS m4a; both are converted to
16 kHz mono WAV with ffmpeg first. Errors never carry transcript text and nothing here logs it.
"""

import asyncio
import base64
import io
import logging
import re
import shutil
import tempfile
from pathlib import Path

import httpx

from hub import prompts
from hub.settings import get_settings

log = logging.getLogger(__name__)

WHISPER_MODEL = "mlx-community/whisper-small-mlx"
OLLAMA_KEEP_ALIVE = -1  # keep Gemma resident so it doesn't reload between checks
GEMMA_IMAGE_TIMEOUT_S = 120.0
GEMMA_AUDIO_TIMEOUT_S = 300.0
FFMPEG_TIMEOUT_S = 60.0

_NO_SENDER = {"", "unknown", "none", "n/a", "<sender or unknown>"}
_AUDIO_SUFFIX = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


class PerceptionError(Exception):
    """Image/audio could not be turned into text. Messages never contain transcript text."""


# --- image ---


def sniff_image(data: bytes) -> str | None:
    """Image type from magic bytes: 'png' | 'jpeg' | 'webp' | 'gif', else None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def parse_image_output(raw: str) -> tuple[str | None, str]:
    """Parse Appendix A.2 output ("SENDER: …\\nMESSAGE:\\n…") into (sender, text)."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\s*|\s*```$", "", s).strip()
    sender_m = re.search(r"^\s*SENDER:[ \t]*(.*)$", s, re.M)
    msg_m = re.search(r"^\s*MESSAGE:[ \t]*\n?(.*)\Z", s, re.M | re.S)
    if msg_m:
        text = msg_m.group(1).strip()
    elif sender_m:  # no MESSAGE marker: everything after the SENDER line
        text = s[sender_m.end() :].strip()
    else:
        text = s
    sender = sender_m.group(1).strip().strip("*`\"'") if sender_m else ""
    return (None if sender.lower() in _NO_SENDER else sender), text


def _to_png(data: bytes) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(out, format="PNG")
    return out.getvalue()


async def _gemma_image(
    client: httpx.AsyncClient, image: bytes, kind: str
) -> tuple[str | None, str]:
    s = get_settings()
    if kind not in ("png", "jpeg"):
        image = await asyncio.to_thread(_to_png, image)
    body = {
        "model": s.gemma_model,
        "messages": [
            {
                "role": "user",
                "content": prompts.PERCEIVE_IMAGE,
                "images": [base64.b64encode(image).decode()],
            }
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 7},
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    r = await client.post(f"{s.ollama_host}/api/chat", json=body, timeout=GEMMA_IMAGE_TIMEOUT_S)
    r.raise_for_status()
    sender, text = parse_image_output(r.json().get("message", {}).get("content") or "")
    if not text:
        raise PerceptionError("gemma returned an empty transcript")
    return sender, text


def _tesseract(image: bytes) -> str:
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(image))
    try:
        text = pytesseract.image_to_string(img, lang="eng+hin")
    except pytesseract.TesseractError:  # hin traineddata not installed: English only
        text = pytesseract.image_to_string(img, lang="eng")
    return text.strip()


async def transcribe_image(
    image: bytes, mime: str | None, *, client: httpx.AsyncClient | None = None
) -> tuple[str | None, str]:
    """Screenshot to (sender, text). Sender is None when unknown."""
    kind = sniff_image(image)
    if kind is None or (mime and not mime.lower().startswith("image/")):
        raise PerceptionError("unsupported image type")
    if get_settings().ocr_fallback == "gemma":
        own = client is None
        http = client or httpx.AsyncClient()
        try:
            return await _gemma_image(http, image, kind)
        except Exception as e:  # any Gemma failure: fall back to Tesseract (FR-11)
            log.warning("gemma image failed (%s), using tesseract", type(e).__name__)
        finally:
            if own:
                await http.aclose()
    try:
        text = await asyncio.to_thread(_tesseract, image)
    except Exception as e:
        raise PerceptionError(f"tesseract failed ({type(e).__name__})") from None
    if not text:
        raise PerceptionError("no text found in the image")
    return None, text


# --- audio ---


async def to_wav(audio: bytes, mime: str | None, workdir: Path) -> Path:
    """Convert webm/opus, m4a, ogg … to 16 kHz mono WAV with ffmpeg (files, not pipes, so
    m4a with a trailing moov atom works)."""
    if shutil.which("ffmpeg") is None:
        raise PerceptionError("ffmpeg is not installed")
    base = (mime or "").split(";")[0].strip().lower()
    src = workdir / f"in{_AUDIO_SUFFIX.get(base, '.bin')}"
    dst = workdir / "out.wav"
    src.write_bytes(audio)
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )  # fmt: skip
    try:
        code = await asyncio.wait_for(proc.wait(), FFMPEG_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        raise PerceptionError("audio conversion timed out") from None
    if code != 0 or not dst.exists():
        raise PerceptionError("could not decode the audio")
    return dst


async def _gemma_audio(client: httpx.AsyncClient, url: str, wav: Path) -> str:
    body = {
        "model": "gemma-audio",
        "temperature": 0,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts.PERCEIVE_AUDIO},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(wav.read_bytes()).decode(),
                            "format": "wav",
                        },
                    },
                ],
            }
        ],
    }
    r = await client.post(
        f"{url.rstrip('/')}/chat/completions", json=body, timeout=GEMMA_AUDIO_TIMEOUT_S
    )
    r.raise_for_status()
    text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    if not text:
        raise PerceptionError("gemma returned an empty transcript")
    return text


def _whisper(wav: Path) -> str:
    try:
        import mlx_whisper
    except ImportError:
        raise PerceptionError("mlx-whisper is not installed (uv sync --extra mlx)") from None
    out = mlx_whisper.transcribe(str(wav), path_or_hf_repo=WHISPER_MODEL)
    return (out.get("text") or "").strip()


async def transcribe_audio(
    audio: bytes, mime: str | None, *, client: httpx.AsyncClient | None = None
) -> str:
    """Voice note to transcript."""
    if not audio:
        raise PerceptionError("empty audio")
    s = get_settings()
    use_gemma = s.asr_fallback == "gemma" and bool(s.gemma_audio_url)
    with tempfile.TemporaryDirectory(prefix="rakshak-audio-") as tmp:
        wav = await to_wav(audio, mime, Path(tmp))
        if use_gemma:
            own = client is None
            http = client or httpx.AsyncClient()
            try:
                return await _gemma_audio(http, s.gemma_audio_url or "", wav)
            except Exception as e:  # fall back to mlx-whisper
                log.warning("gemma audio failed (%s), trying mlx-whisper", type(e).__name__)
            finally:
                if own:
                    await http.aclose()
        try:
            text = await asyncio.to_thread(_whisper, wav)
        except PerceptionError:
            raise
        except Exception as e:
            raise PerceptionError(f"mlx-whisper failed ({type(e).__name__})") from None
    if not text:
        raise PerceptionError("no speech found in the audio")
    return text
