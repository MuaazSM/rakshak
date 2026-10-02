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


def test_full_system_not_wired_yet():
    with pytest.raises(NotImplementedError):
        systems.FullSystem()


def test_run_system_keeps_order():
    items = [item_from_row(row(f"i{k}", OTP_TEXT, SAFE_OUT)) for k in range(5)]

    class Echo(systems.System):
        async def predict(self, item):
            await asyncio.sleep(0.001 * (5 - int(item.id[1:])))
            return systems.Pred(verdict="SAFE", category=item.id)

    preds = asyncio.run(run_system(Echo(), items))
    assert [p.category for p in preds] == [it.id for it in items]
