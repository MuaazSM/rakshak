"""Eval adapters and split parsing (PRD §12.3, Appendix A.4, §7.4). Synthetic data, no network."""

import asyncio
import json

import httpx
import pytest

from eval import systems
from eval.systems import (
    GemmaZeroShot,
    QwenBaseFewShot,
    RulesOnly,
    gold_of,
    item_from_row,
    parse_reply,
    parse_user_message,
    run_system,
)
from hub.detector import chat_example, user_message
from hub.schemas import DetectorOutput

SCAM_TEXT = "Your SBI account will be blocked today. Install http://sbi-help.example/app.apk now"
OTP_TEXT = "Your OTP is <OTP>. Do not share it with anyone. -SBI"


def row(id_, text, out, sender="VK-SBIUPD", signals=(), channel="sms", **meta):
    msgs = chat_example(channel, sender, list(signals), text, DetectorOutput.model_validate(out))
    return {"messages": msgs, "meta": {"id": id_, "source": "synthetic", **meta}}


SCAM_OUT = {
    "verdict": "SCAM",
    "category": "malicious_apk",
    "red_flags": [{"quote": "will be blocked today", "reason": "urgency_deadline"}],
}
SAFE_OUT = {"verdict": "SAFE", "category": "genuine_otp", "red_flags": []}


def test_parse_user_message_round_trips():
    for sender in ("VK-SBIUPD", "+919812345678", None):
        msg = user_message("whatsapp", sender, ["threat_lexicon", "apk_link"], "line1\nline2")
        channel, s, sig, text = parse_user_message(msg)
        assert user_message(channel, s, sig, text) == msg
        assert text == "line1\nline2"


def test_parse_user_message_rejects_other_format():
    with pytest.raises(ValueError):
        parse_user_message("hello")


def test_item_and_gold_from_row():
    it = item_from_row(row("a1", OTP_TEXT, SAFE_OUT, hard_negative=False, language="en"))
    assert (it.id, it.channel, it.sender, it.text) == ("a1", "sms", "VK-SBIUPD", OTP_TEXT)
    g = gold_of(it)
    assert g.verdict == "SAFE" and g.hard_negative  # genuine_otp counts as hard negative
    assert [m["role"] for m in it.prompt_messages] == ["system", "user"]


def test_parse_reply_strict_valid():
    p = parse_reply(json.dumps(SCAM_OUT))
    assert p.verdict == "SCAM" and p.category == "malicious_apk"
    assert p.json_valid and p.json_parsed
    assert p.quotes == ["will be blocked today"]


def test_parse_reply_lenient_for_base_models():
    raw = '```json\n{"verdict": "scam", "category": "Phishing", "red_flags": [{"quote": "x", "reason": "a long sentence"}]}\n```'
    p = parse_reply(raw)
    assert p.verdict == "SCAM" and p.category is None and p.quotes == ["x"]
    assert p.json_parsed and not p.json_valid


def test_parse_reply_garbage_is_unknown():
    p = parse_reply("I think this is a scam")
    assert p.verdict == "UNKNOWN" and p.json_valid is False and p.json_parsed is False
    assert parse_reply('{"verdict": "MAYBE"}').verdict == "UNKNOWN"


def test_rules_only_mapping():
    scam = item_from_row(row("s", SCAM_TEXT, SCAM_OUT))
    otp = item_from_row(row("o", OTP_TEXT, SAFE_OUT, sender="AD-SBIOTP"))
    preds = asyncio.run(run_system(RulesOnly(), [scam, otp]))
    assert preds[0].verdict == "SCAM" and preds[0].quotes  # apk_link is a hard signal
    assert all(q in SCAM_TEXT for q in preds[0].quotes)
    assert preds[1].verdict == "UNKNOWN"  # rules-only never says SAFE (§7.4)
    assert preds[0].json_valid is None and preds[0].latency_ms is not None


def _client(handler, seen):
    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(wrapped))


def test_gemma_zeroshot_request_and_parse():
    seen = []
    client = _client(
        lambda r: httpx.Response(200, json={"message": {"content": json.dumps(SCAM_OUT)}}), seen
    )
    it = item_from_row(row("s", SCAM_TEXT, SCAM_OUT))
    [pred] = asyncio.run(run_system(GemmaZeroShot(client=client), [it]))
    body = seen[0]
    assert body["think"] is False and body["format"] == "json" and body["stream"] is False
    assert body["messages"] == it.prompt_messages  # A.1 + §8.2 user, no examples
    assert pred.verdict == "SCAM" and pred.json_valid


def test_server_error_becomes_unknown_with_error_type():
    client = _client(lambda r: httpx.Response(500, json={}), [])
    it = item_from_row(row("s", SCAM_TEXT, SCAM_OUT))
    [pred] = asyncio.run(run_system(GemmaZeroShot(client=client), [it]))
    assert pred.verdict == "UNKNOWN" and pred.error == "HTTPStatusError"


def _write_fewshot(path, n=6):
    lines = []
    for i in range(n):
        msgs = row(f"f{i}", OTP_TEXT, SAFE_OUT)["messages"]
        lines.append(json.dumps({"id": f"f{i}", "kind": "x", "messages": msgs[1:]}))
    path.write_text("\n".join(lines) + "\n", "utf-8")


def test_qwen_fewshot_sends_system_six_pairs_then_user(tmp_path):
    shots = tmp_path / "fewshot.jsonl"
    _write_fewshot(shots)
    seen = []
    client = _client(
        lambda r: httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(SAFE_OUT)}}]}
        ),
        seen,
    )
    it = item_from_row(row("o", OTP_TEXT, SAFE_OUT))
    [pred] = asyncio.run(run_system(QwenBaseFewShot(client=client, fewshot_path=shots), [it]))
    msgs = seen[0]["messages"]
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 6 + ["user"]
    assert msgs[-1] == it.prompt_messages[1]
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[0]["temperature"] == 0
    assert pred.verdict == "SAFE"


def test_fewshot_file_must_have_six(tmp_path):
    shots = tmp_path / "fewshot.jsonl"
    _write_fewshot(shots, n=5)
    with pytest.raises(ValueError):
        systems.load_fewshot(shots)


def test_run_system_keeps_order():
    items = [item_from_row(row(f"i{k}", OTP_TEXT, SAFE_OUT)) for k in range(5)]

    class Echo(systems.System):
        async def predict(self, item):
            await asyncio.sleep(0.001 * (5 - int(item.id[1:])))
            return systems.Pred(verdict="SAFE", category=item.id)

    preds = asyncio.run(run_system(Echo(), items))
    assert [p.category for p in preds] == [it.id for it in items]


def test_full_system_maps_verdict_and_never_alerts(monkeypatch, tmp_path):
    from hub import alerts, graph, settings
    from hub.schemas import RedFlag, Verdict

    s = settings.get_settings()
    monkeypatch.setattr(s, "ntfy_topic", "rakshak-test-topic")  # as if set in .env
    orig_topic, orig_db = s.ntfy_topic, s.db_path
    monkeypatch.setattr(settings, "get_parents", lambda: {"mom": object()})
    calls = []

    async def fake_run_check(**kw):
        calls.append(kw)
        assert settings.get_settings().ntfy_topic is None  # alerts off during eval
        assert settings.get_settings().db_path != orig_db  # never the real DB
        assert await alerts.notify_scam(None, "x", None) is False
        return Verdict(
            event_id="evt_1",
            verdict="SCAM",
            category="malicious_apk",
            p_scam=0.9,
            red_flags=[RedFlag(quote="app.apk", reason="apk_link", source="rule")],
            explanation="",
            language="en",
            parent_id="mom",
        )

    monkeypatch.setattr(graph, "run_check", fake_run_check)
    it = item_from_row(row("s", SCAM_TEXT, SCAM_OUT))
    sys_ = systems.FullSystem()
    [pred] = asyncio.run(run_system(sys_, [it]))
    asyncio.run(sys_.aclose())
    assert calls[0] == {
        "parent_id": "mom",
        "text": SCAM_TEXT,
        "channel": "sms",
        "sender": "VK-SBIUPD",
    }
    assert (pred.verdict, pred.category, pred.p_scam, pred.quotes) == (
        "SCAM",
        "malicious_apk",
        0.9,
        ["app.apk"],
    )
    assert (s.ntfy_topic, s.db_path) == (orig_topic, orig_db)  # restored
    assert alerts.notify_scam.__name__ == "notify_scam"


def test_full_system_requires_configured_parent(monkeypatch):
    from hub import settings

    monkeypatch.setattr(settings, "get_parents", lambda: {})
    with pytest.raises(NotImplementedError):
        systems.FullSystem()


def test_served_detector_is_tuned():
    def client(ids):
        return httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"data": [{"id": i} for i in ids]})
            )
        )

    assert not systems.served_detector_is_tuned(client(["models/base/qwen3.5-4b-q4km.gguf"]))
    assert systems.served_detector_is_tuned(client(["models/rakshak-detector-v1-q4km.gguf"]))
    down = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    assert not systems.served_detector_is_tuned(down)


def _shots(path, groups):
    lines = []
    for k, g in enumerate(groups):
        msgs = row(f"f{k}", OTP_TEXT, SAFE_OUT)["messages"]
        lines.append(
            json.dumps({"id": f"f{k}", "seed_group": g, "kind": "x", "messages": msgs[1:]})
        )
    path.write_text("\n".join(lines) + "\n", "utf-8")


def test_fewshot_guard_rejects_overlap_by_id_or_group(tmp_path):
    shots = tmp_path / "fewshot.jsonl"
    _shots(shots, [f"g{k}" for k in range(6)])
    clean = [item_from_row(row("d1", OTP_TEXT, SAFE_OUT, seed_group="other"))]
    systems.check_fewshot_disjoint(shots, clean)
    same_id = [item_from_row(row("f2", OTP_TEXT, SAFE_OUT, seed_group="other"))]
    same_group = [item_from_row(row("d2", OTP_TEXT, SAFE_OUT, seed_group="g4"))]
    for items in (same_id, same_group):
        with pytest.raises(ValueError, match="overlap"):
            systems.check_fewshot_disjoint(shots, items)


def test_fewshot_guard_requires_seed_groups(tmp_path):
    shots = tmp_path / "fewshot.jsonl"
    _write_fewshot(shots)  # old format, no seed_group
    with pytest.raises(ValueError, match="seed_group"):
        systems.check_fewshot_disjoint(shots, [])
    sys_ = QwenBaseFewShot(client=httpx.AsyncClient(), fewshot_path=shots)
    with pytest.raises(ValueError):
        sys_.check_items([])


def test_strip_sender_renders_unknown_and_recomputes_signals():
    text = "Your SBI KYC is pending, account blocked today. Call now"
    it = item_from_row(
        row("s", text, SCAM_OUT, sender="+919812345678", signals=["unregistered_sender"])
    )
    out = systems.strip_sender(it)
    user = out.prompt_messages[1]["content"]
    assert user.splitlines()[1] == "SENDER: unknown (unknown)"
    assert user == user_message("sms", None, out.rule_signals, text)
    assert "unregistered_sender" not in out.rule_signals
    assert out.sender is None and out.text == it.text and out.gold == it.gold
    assert out.messages[2] == it.messages[2] and it.sender == "+919812345678"  # original intact
