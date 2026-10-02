"""FastAPI app: every PRD §5.3 endpoint plus the built PWA (PRD FR-1..FR-9, §5.3).

Binds to 127.0.0.1 (see settings); external access only through `tailscale serve`.
Handlers never log message text, only event ids and metadata (PRD §9.3, §13).
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from hub import db, fusion, tracing
from hub.graph import EmptyInput, UnknownParent, run_check
from hub.schemas import Channel, CheckRequest, Feedback, Lang, Verdict
from hub.settings import get_parents, get_settings, load_parents

log = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_AUDIO_BYTES = 2 * 1024 * 1024  # PRD FR-3: 60 s of opus/m4a is well under this
MAX_AUDIO_SECONDS = 60.0
DIST_DIR = Path(__file__).resolve().parent.parent / "pwa" / "dist"
_parents_lock = threading.Lock()


class ParentPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: Lang


def _error(status: int, code: str, detail: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": code, "detail": detail})


def _require_parent(parent_id: str | None) -> str:
    if not parent_id or parent_id not in get_parents():
        raise _error(400, "unknown_parent", "parent_id is missing or not configured")
    return parent_id


async def _read_upload(f: UploadFile | None, limit: int, what: str) -> bytes | None:
    """Bytes of an upload (None when absent/empty); 413 beyond `limit`."""
    if f is None:
        return None
    data = await f.read(limit + 1)
    if len(data) > limit:
        raise _error(413, f"{what}_too_large", f"{what} larger than {limit // (1024 * 1024)} MB")
    return data or None


async def _audio_seconds(data: bytes) -> float | None:
    """Best-effort duration via ffprobe (None if unavailable or unreadable)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    fd, name = tempfile.mkstemp(suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        proc = await asyncio.create_subprocess_exec(
            ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )  # fmt: skip
        out, _ = await asyncio.wait_for(proc.communicate(), 5)
        return float(out.decode().strip())
    except (OSError, ValueError, TimeoutError):
        return None
    finally:
        Path(name).unlink(missing_ok=True)


def _lang(value: str | None) -> str | None:
    """Optional `lang` form field: blank -> None; anything but en/hi -> 422."""
    value = (value or "").strip().lower()
    if not value:
        return None
    if value not in ("en", "hi"):
        raise _error(422, "bad_lang", "lang must be en or hi")
    return value


async def _check(**kw) -> Verdict:
    try:
        return await run_check(**kw)
    except UnknownParent:
        raise _error(400, "unknown_parent", "parent_id is not configured") from None
    except EmptyInput:
        raise _error(422, "empty_input", "send text or an image") from None


def _p95(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


async def _ping(client: httpx.AsyncClient, url: str | None) -> bool | None:
    if not url:
        return None
    try:
        return (await client.get(url)).status_code < 500
    except httpx.HTTPError:
        return False


def _persist_language(parent_id: str, language: str) -> None:
    """Atomically rewrite parents_file with the new language, then reload the cache."""
    path = Path(get_settings().parents_file)
    with _parents_lock:
        data = json.loads(path.read_text("utf-8"))
        for entry in data:
            if entry.get("id") == parent_id:
                entry["language"] = language
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")
        try:
            load_parents(tmp)  # validate before replacing
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        get_parents.cache_clear()


def create_app(dist_dir: Path | None = None) -> FastAPI:
    dist = DIST_DIR if dist_dir is None else dist_dir

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        tracing.init_tracing(get_settings())
        db.init_db()
        yield

    app = FastAPI(title="Rakshak hub", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException):
        body = (
            exc.detail if isinstance(exc.detail, dict) else {"error": "http", "detail": exc.detail}
        )
        return JSONResponse(body, status_code=exc.status_code)

    # --- capture endpoints -------------------------------------------------

    @app.post("/share")
    async def share(
        parent: Annotated[str | None, Query()] = None,
        parent_id: Annotated[str | None, Form()] = None,
        title: Annotated[str | None, Form()] = None,
        text: Annotated[str | None, Form()] = None,
        url: Annotated[str | None, Form()] = None,
        channel: Annotated[Channel, Form()] = "sms",
        image: Annotated[UploadFile | None, File()] = None,
    ):
        """Android Share Target: combine title/text/url, check, 303 to the verdict page."""
        pid = _require_parent(parent_id or parent)
        parts = [p.strip() for p in (title, text, url) if p and p.strip()]
        combined = "\n".join(dict.fromkeys(parts))  # drop exact duplicates (url inside text)
        img = await _read_upload(image, MAX_IMAGE_BYTES, "image")
        v = await _check(
            parent_id=pid,
            text=combined or None,
            channel=channel,
            image=img,
            image_mime=image.content_type if img and image else None,
        )
        return RedirectResponse(f"/v/{v.event_id}", status_code=303)

    @app.post("/api/share", response_model=Verdict)
    async def api_share(
        parent_id: Annotated[str | None, Form()] = None,
        lang: Annotated[str | None, Form()] = None,
        text: Annotated[str | None, Form()] = None,
        image: Annotated[UploadFile | None, File()] = None,
    ):
        """iOS Shortcut: multipart in, Verdict JSON out."""
        pid = _require_parent(parent_id)
        img = await _read_upload(image, MAX_IMAGE_BYTES, "image")
        return await _check(
            parent_id=pid,
            text=text,
            channel="sms",
            image=img,
            image_mime=image.content_type if img and image else None,
            lang=_lang(lang),
        )

    @app.post("/api/check", response_model=Verdict)
    async def api_check(req: CheckRequest):
        _require_parent(req.parent_id)
        return await _check(
            parent_id=req.parent_id,
            text=req.text,
            channel=req.channel,
            sender=req.sender,
            lang=req.lang,
        )

    @app.post("/voice", response_model=Verdict)
    async def voice(
        audio: Annotated[UploadFile, File()],
        parent_id: Annotated[str | None, Form()] = None,
        lang: Annotated[str | None, Form()] = None,
    ):
        pid = _require_parent(parent_id)
        data = await _read_upload(audio, MAX_AUDIO_BYTES, "audio")
        if data is None:
            raise _error(422, "empty_input", "no audio received")
        seconds = await _audio_seconds(data)
        if seconds is not None and seconds > MAX_AUDIO_SECONDS:
            raise _error(
                413, "audio_too_long", f"voice notes are limited to {int(MAX_AUDIO_SECONDS)} s"
            )
        return await _check(
            parent_id=pid,
            channel="call_description",
            audio=data,
            audio_mime=audio.content_type,
            lang=_lang(lang),
        )

    # --- profile, verdicts, feedback --------------------------------------

    @app.patch("/api/parents/{parent_id}", status_code=204)
    async def patch_parent(parent_id: str, body: ParentPatch):
        _require_parent(parent_id)
        await asyncio.to_thread(_persist_language, parent_id, body.language)
        return Response(status_code=204)

    @app.get("/api/verdict/{event_id}", response_model=Verdict)
    async def get_verdict(event_id: str):
        v = await asyncio.to_thread(db.get_verdict, event_id)
        if v is None:
            raise _error(404, "not_found", "unknown event")
        return v

    @app.post("/api/feedback/{event_id}", status_code=204)
    async def post_feedback(event_id: str, fb: Feedback):
        if not await asyncio.to_thread(db.event_exists, event_id):
            raise _error(404, "not_found", "unknown event")
        await asyncio.to_thread(db.save_feedback, event_id, fb.correct, fb.true_verdict, fb.note)
        return Response(status_code=204)

    @app.get("/health")
    async def health():
        s = get_settings()
        th = fusion.get_thresholds()
        base = s.detector_url.rstrip("/").removesuffix("/v1")
        audio = s.gemma_audio_url.rstrip("/").removesuffix("/v1") if s.gemma_audio_url else None
        async with httpx.AsyncClient(timeout=2.0) as c:
            det, oll, aud = await asyncio.gather(
                _ping(c, f"{base}/health"),
                _ping(c, f"{s.ollama_host.rstrip('/')}/api/tags"),
                _ping(c, f"{audio}/health" if audio else None),
            )
        timings = await asyncio.to_thread(db.recent_timings, 50)
        totals = [t["total"] for t in timings if "total" in t]
        return {
            "status": "ok" if det and oll and aud is not False else "degraded",
            "models": {"detector": det, "gemma": oll, "gemma_audio": aud},
            "model_versions": {"detector": s.detector_version, "gemma": s.gemma_model},
            "thresholds": {"t_high": th.t_high, "t_low": th.t_low, "calibrated": th.calibrated},
            "latency_p95_ms": _p95(totals),
            "latency_samples": len(totals),
        }

    try:  # the status page agent adds hub/status.py exposing `router`
        from hub.status import router as status_router

        app.include_router(status_router)
    except ImportError:
        pass

    # --- built PWA (registered last so API routes take precedence) --------

    if (dist / "index.html").is_file():
        if (dist / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path == "api" or path.startswith("api/"):
                raise _error(404, "not_found", "no such endpoint")
            root = dist.resolve()
            target = (root / path).resolve()
            if path and target.is_file() and target.is_relative_to(root):
                return FileResponse(target)
            return FileResponse(root / "index.html")

    return app


app = create_app()
