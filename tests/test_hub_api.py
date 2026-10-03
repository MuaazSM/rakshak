"""HTTP API (PRD §5.3) through FastAPI's TestClient with fake pipeline services."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from hub import alerts, db, detector, explainer, perception, settings
from hub import app as hub_app
from tests.test_hub_env import hub_env, make_result  # noqa: F401

SCAM_TEXT = "Dear customer your account will be blocked today. Share your OTP now."
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


@pytest.fixture
def client(hub_env, monkeypatch):  # noqa: F811
    state = {"detect": [], "result": make_result("SCAM", 0.97), "down": False, "alerts": []}

    async def fake_detect(channel, sender, signals, text, *, client=None):
        state["detect"].append((channel, sender, text))
        if state["down"]:
            raise detector.DetectorUnavailable("ConnectError")
        return state["result"]

    async def fake_explain(verdict, parent, lang, input_text, *, client=None):
        return f"EXPLAIN {verdict['verdict']} {lang}"

    async def fake_image(image, mime, *, client=None):
        state["image"] = (image, mime)
        return "AX-HDFCBK", SCAM_TEXT

    async def fake_audio(audio, mime, *, client=None):
        state["audio"] = (audio, mime)
        return "someone said my account will be blocked"

    async def fake_notify(parent, category, when, *, client=None):
        state["alerts"].append(parent.id)
        return True

    monkeypatch.setattr(detector, "detect", fake_detect)
    monkeypatch.setattr(explainer, "explain", fake_explain)
    monkeypatch.setattr(perception, "transcribe_image", fake_image)
    monkeypatch.setattr(perception, "transcribe_audio", fake_audio)
    monkeypatch.setattr(alerts, "notify_scam", fake_notify)
    with TestClient(hub_app.create_app(dist_dir=hub_env / "no-dist")) as c:
        c.state = state
        yield c


def _post_check(client, **over):
    body = {"parent_id": "mom", "text": SCAM_TEXT, "channel": "sms", **over}
    return client.post("/api/check", json=body)


# --- /api/check -----------------------------------------------------------------


def test_api_check_scam(client):
    r = _post_check(client, sender="+91 98765 43210")
    assert r.status_code == 200
    v = r.json()
    assert v["verdict"] == "SCAM" and v["parent_id"] == "mom" and v["language"] == "en"
    assert v["red_flags"][0]["source"] == "model" and v["explanation"] == "EXPLAIN SCAM en"
    assert set(v) == {
        "event_id", "verdict", "category", "p_scam", "red_flags", "explanation",
        "language", "parent_id", "timings_ms", "model_versions",
    }  # fmt: skip
    assert client.state["alerts"] == ["mom"]


def test_api_check_lang_override_and_validation(client):
    assert _post_check(client, lang="hi").json()["language"] == "hi"
    assert _post_check(client, lang="mr").status_code == 422
    assert _post_check(client, text="").status_code == 422
    assert _post_check(client, channel="fax").status_code == 422


def test_unknown_parent_is_400_everywhere(client):
    assert _post_check(client, parent_id="uncle").status_code == 400
    assert _post_check(client, parent_id="uncle").json()["error"] == "unknown_parent"
    assert client.post("/api/share", data={"parent_id": "uncle", "text": "x"}).status_code == 400
    assert client.post("/api/share", data={"text": "x"}).status_code == 400
    assert client.post("/share", data={"parent_id": "uncle", "text": "x"}).status_code == 400
    r = client.post("/voice", data={"parent_id": "uncle"}, files={"audio": ("a.m4a", b"x")})
    assert r.status_code == 400
    assert client.patch("/api/parents/uncle", json={"language": "en"}).status_code == 400
    assert client.state["detect"] == []


def test_detector_down_returns_unknown(client):
    client.state["down"] = True
    v = _post_check(client, text="Hi, are you coming for dinner tonight?").json()
    assert v["verdict"] == "UNKNOWN" and v["category"] is None and v["p_scam"] is None
    assert client.state["alerts"] == []


# --- /api/share and /share ------------------------------------------------------------


def test_api_share_text(client):
    r = client.post("/api/share", data={"parent_id": "dad", "text": SCAM_TEXT, "lang": "en"})
    assert r.status_code == 200 and r.json()["verdict"] == "SCAM"
    assert r.json()["parent_id"] == "dad"
    assert client.state["detect"][0][2] == SCAM_TEXT


def test_api_share_image(client):
    r = client.post(
        "/api/share", data={"parent_id": "dad"}, files={"image": ("s.png", PNG, "image/png")}
    )
    assert r.status_code == 200 and r.json()["verdict"] == "SCAM"
    assert client.state["image"] == (PNG, "image/png")
    assert client.state["detect"][0][0] == "screenshot"
    assert client.state["detect"][0][1] == "AX-HDFCBK"


def test_api_share_empty_is_422(client):
    assert client.post("/api/share", data={"parent_id": "dad"}).status_code == 422
    assert client.post("/api/share", data={"parent_id": "dad", "text": "  "}).status_code == 422


def test_api_share_blank_lang_ignored_bad_lang_rejected(client):
    ok = client.post("/api/share", data={"parent_id": "dad", "text": "hi", "lang": ""})
    assert ok.status_code == 200 and ok.json()["language"] == "en"
    bad = client.post("/api/share", data={"parent_id": "dad", "text": "hi", "lang": "fr"})
    assert bad.status_code == 422


def test_share_redirects_303_to_verdict_page(client):
    r = client.post(
        "/share?parent=mom",
        data={"title": "Bank", "text": SCAM_TEXT, "url": "http://x.example/a"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"].startswith("/v/evt_")
    event_id = r.headers["location"].removeprefix("/v/")
    assert client.get(f"/api/verdict/{event_id}").json()["verdict"] == "SCAM"
    assert client.state["detect"][0][2] == f"Bank\n{SCAM_TEXT}\nhttp://x.example/a"


def test_share_dedupes_url_already_in_text_and_prefers_form_parent(client):
    client.post(
        "/share?parent=dad",
        data={"parent_id": "mom", "text": "see http://a.example", "url": "see http://a.example"},
        follow_redirects=False,
    )
    assert client.state["detect"][0][2] == "see http://a.example"
    assert db.list_events_since("2000-01-01")[0]["parent_id"] == "mom"


def test_share_with_image(client):
    r = client.post(
        "/share?parent=mom",
        files={"image": ("shot.png", PNG, "image/png")},
        follow_redirects=False,
    )
    assert r.status_code == 303 and client.state["image"][1] == "image/png"


def test_share_without_content_is_422(client):
    assert client.post("/share?parent=mom", data={"title": ""}).status_code == 422


# --- /voice ---------------------------------------------------------------------------


def test_voice_m4a(client):
    r = client.post(
        "/voice",
        data={"parent_id": "dad"},
        files={"audio": ("note.m4a", b"fake-m4a", "audio/mp4")},
    )
    assert r.status_code == 200 and r.json()["verdict"] == "SCAM"
    assert client.state["audio"] == (b"fake-m4a", "audio/mp4")
    assert client.state["detect"][0][0] == "call_description"


def test_voice_webm_with_lang(client):
    r = client.post(
        "/voice",
        data={"parent_id": "mom", "lang": "hi"},
        files={"audio": ("rec.webm", b"fake-webm", "audio/webm;codecs=opus")},
    )
    assert r.status_code == 200 and r.json()["language"] == "hi"


def test_voice_too_large_is_413(client):
    big = b"0" * (hub_app.MAX_AUDIO_BYTES + 1)
    r = client.post("/voice", data={"parent_id": "mom"}, files={"audio": ("a.webm", big)})
    assert r.status_code == 413 and r.json()["error"] == "audio_too_large"
    assert "audio" not in client.state


def test_voice_too_long_is_413(client, monkeypatch):
    async def fake_seconds(data):
        return 75.0

    monkeypatch.setattr(hub_app, "_audio_seconds", fake_seconds)
    r = client.post("/voice", data={"parent_id": "mom"}, files={"audio": ("a.webm", b"x")})
    assert r.status_code == 413 and r.json()["error"] == "audio_too_long"


def test_voice_missing_or_empty_audio(client):
    assert client.post("/voice", data={"parent_id": "mom"}).status_code == 422
    r = client.post("/voice", data={"parent_id": "mom"}, files={"audio": ("a.webm", b"")})
    assert r.status_code == 422


# --- parents, verdicts, feedback ----------------------------------------------------


def test_patch_parent_language_persists_and_applies(client, hub_env):  # noqa: F811
    r = client.patch("/api/parents/mom", json={"language": "hi"})
    assert r.status_code == 204 and r.content == b""
    saved = json.loads((hub_env / "parents.json").read_text())
    assert [(p["id"], p["language"]) for p in saved] == [("mom", "hi"), ("dad", "en")]
    assert saved[0]["name"] == "Mom" and not list(hub_env.glob("parents.json.tmp"))
    assert settings.get_parents()["mom"].language == "hi"
    assert _post_check(client).json()["language"] == "hi"  # no per-request lang needed
    assert _post_check(client, parent_id="dad").json()["language"] == "en"


def test_patch_parent_rejects_bad_language(client):
    assert client.patch("/api/parents/mom", json={"language": "mr"}).status_code == 422
    assert client.patch("/api/parents/mom", json={}).status_code == 422
    assert client.patch("/api/parents/mom", json={"language": "en", "x": 1}).status_code == 422


def test_get_verdict(client):
    v = _post_check(client).json()
    r = client.get(f"/api/verdict/{v['event_id']}")
    assert r.status_code == 200 and r.json() == v
    missing = client.get("/api/verdict/evt_nope")
    assert missing.status_code == 404 and missing.json()["error"] == "not_found"


def test_feedback(client):
    event_id = _post_check(client).json()["event_id"]
    r = client.post(
        f"/api/feedback/{event_id}", json={"correct": False, "true_verdict": "SAFE", "note": "bank"}
    )
    assert r.status_code == 204
    assert db.pending_feedback(10) == []
    assert client.post(f"/api/feedback/{event_id}", json={"correct": True}).status_code == 204
    assert client.post("/api/feedback/evt_nope", json={"correct": True}).status_code == 404
    assert client.post(f"/api/feedback/{event_id}", json={}).status_code == 422
    bad = client.post(
        f"/api/feedback/{event_id}", json={"correct": True, "true_verdict": "UNKNOWN"}
    )
    assert bad.status_code == 422


# --- /health --------------------------------------------------------------------------


def test_health_reports_models_thresholds_and_p95(client, monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.port == 8081:
            return httpx.Response(200, json={"status": "ok"})
        raise httpx.ConnectError("down")

    real = httpx.AsyncClient
    monkeypatch.setattr(
        hub_app.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    for _ in range(3):
        _post_check(client)
    h = client.get("/health").json()
    assert h["status"] == "degraded"
    assert h["models"] == {"detector": True, "gemma": False, "gemma_audio": False}
    assert h["model_versions"] == {"detector": "rakshak-detector-v1-q4km", "gemma": "gemma4:e2b"}
    assert h["thresholds"] == {"t_high": 0.8, "t_low": 0.35, "calibrated": False}
    assert h["latency_samples"] == 3 and isinstance(h["latency_p95_ms"], int)
    assert any(u.endswith(":8081/health") for u in seen)


def test_health_all_up(client, monkeypatch):
    real = httpx.AsyncClient
    monkeypatch.setattr(
        hub_app.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(lambda r: httpx.Response(200)), **kw),
    )
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["latency_p95_ms"] is None


# --- PWA serving ----------------------------------------------------------------------


def test_spa_fallback_and_static_assets(hub_env, client):  # noqa: F811
    dist = hub_env / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>PWA</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (dist / "manifest.webmanifest").write_text("{}")
    with TestClient(hub_app.create_app(dist_dir=dist)) as c:
        for path in ("/", "/check", "/talk", "/settings", "/v/evt_abc"):
            r = c.get(path)
            assert r.status_code == 200 and "PWA" in r.text, path
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert c.get("/manifest.webmanifest").text == "{}"
        (hub_env / "secret.txt").write_text("SECRET")
        for evil in ("/%2e%2e/secret.txt", "/..%2fsecret.txt", "/assets/..%2f..%2fsecret.txt"):
            assert "SECRET" not in c.get(evil).text, evil
        assert c.get("/api/nope").status_code == 404
        assert c.get("/api/verdict/evt_nope").status_code == 404  # API wins over the fallback
        assert c.get("/health").json()["model_versions"]
        r = c.post("/api/check", json={"parent_id": "mom", "text": "hi", "channel": "sms"})
        assert r.status_code == 200


def test_no_dist_means_no_spa(client):
    assert client.get("/").status_code == 404
