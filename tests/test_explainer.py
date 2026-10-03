"""PRD FR-16, §7.5, Appendix A.3: post-check, one regeneration, template fallback."""

import asyncio
import json

import httpx
import pytest

from hub import explainer
from hub.settings import Parent, Settings

MOM = Parent(id="mom", name="Asha", age=60, language="en", device="android")
INPUT = (
    "Dear customer your SBI account will be blocked today. Update at http://sbi-kyc.example/login"
)
SCAM = {
    "verdict": "SCAM",
    "category": "kyc_account_block",
    "red_flags": [
        {"quote": "will be blocked today", "reason": "urgency_deadline", "source": "model"}
    ],
}
GOOD = "This is a scam. The message threatens to block your account. Do not click, call Muaaz."


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    s = Settings(_env_file=None, son_name="Muaaz")
    monkeypatch.setattr(explainer, "get_settings", lambda: s)


class Fake:
    """Mock Ollama: returns the queued replies in order and records request bodies."""

    def __init__(self, *replies: str | int):
        self.replies = list(replies)
        self.bodies: list[dict] = []

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            self.bodies.append(json.loads(request.content))
            reply = self.replies.pop(0)
            if isinstance(reply, int):
                return httpx.Response(reply)
            return httpx.Response(200, json={"message": {"role": "assistant", "content": reply}})

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def run(verdict, fake, lang="en", text=INPUT):
    return asyncio.run(explainer.explain(verdict, MOM, lang, text, client=fake.client()))


def test_good_output_passes_and_request_shape():
    fake = Fake(GOOD)
    assert run(SCAM, fake) == GOOD
    (body,) = fake.bodies
    assert body["think"] is False and body["stream"] is False
    system, user = (m["content"] for m in body["messages"])
    assert "Asha" in system and "60" in system and "Muaaz" in system
    assert user.startswith("VERDICT_JSON = ")
    sent = json.loads(user.removeprefix("VERDICT_JSON = "))
    assert sent["verdict"] == "SCAM" and sent["red_flags"][0] == {
        "quote": "will be blocked today",
        "reason": "urgency_deadline",
    }


def test_missing_verdict_word_regenerates_then_ok():
    fake = Fake("Your account may be blocked. Call Muaaz.", GOOD)
    assert run(SCAM, fake) == GOOD
    assert len(fake.bodies) == 2


@pytest.mark.parametrize(
    "bad",
    [
        "This is a scam. Open http://evil.example now. Call Muaaz.",  # foreign URL
        "This is a scam. Call 9876543210 now.",  # foreign number
        "This is a scam. It is bad. Do not click. Call Muaaz.",  # 4 sentences
        "This is a scam " + "word " * 60,  # too long
        "",  # empty
    ],
)
def test_bad_output_regenerates_then_template(bad):
    fake = Fake(bad, bad)
    out = run(SCAM, fake)
    assert len(fake.bodies) == 2
    assert out == explainer.template("SCAM", "kyc_account_block", "en", "Muaaz")
    assert out.startswith("This is a scam. Real banks never ask you to update KYC")


def test_number_in_input_is_allowed():
    fake = Fake("This is a scam. Do not call 1930 numbers from it. Call Muaaz.")
    text = INPUT + " call 1930"
    assert run(SCAM, fake, text=text).startswith("This is a scam.")


def test_suspicious_goes_to_gemma():
    fake = Fake("Be careful, this message is rushing you. Check with Muaaz first.")
    out = run({"verdict": "SUSPICIOUS", "category": "other_scam", "red_flags": []}, fake)
    assert out.startswith("Be careful") and len(fake.bodies) == 1


@pytest.mark.parametrize(
    "verdict",
    [
        {"verdict": "SAFE", "category": "delivery_update", "red_flags": []},
        {"verdict": "SAFE", "category": "genuine_otp", "red_flags": []},
        {"verdict": "UNKNOWN", "category": None, "red_flags": []},
    ],
)
def test_safe_and_unknown_never_call_gemma(verdict):
    fake = Fake()  # any call would pop from an empty list and fail
    out = run(verdict, fake)
    assert fake.bodies == []
    assert out == explainer.template(verdict["verdict"], verdict["category"], "en", "Muaaz")


def test_template_copy_matches_design():
    t = explainer.template
    assert t("SAFE", "genuine_otp", "en", "Muaaz") == (
        "This looks like a normal message. This is a real OTP from your bank. "
        "Never share an OTP with anyone."
    )
    assert t("UNKNOWN", None, "en", "Muaaz") == (
        "Couldn't check right now. The home computer is offline. "
        "Don't open any links, and call Muaaz."
    )
    assert t("SUSPICIOUS", None, "en", "Muaaz").startswith("Be careful.")
    assert "Call Muaaz" in t("SCAM", "digital_arrest", "en", "Muaaz")


def test_gemma_down_falls_back_without_raising():
    fake = Fake(500)
    out = run(SCAM, fake)
    assert out == explainer.template("SCAM", "kyc_account_block", "en", "Muaaz")
    assert len(fake.bodies) == 1  # no pointless retry against a dead server


def test_connect_error_falls_back():
    def boom(request):
        raise httpx.ConnectError("down")

    c = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    out = asyncio.run(explainer.explain(SCAM, MOM, "en", INPUT, client=c))
    assert out.startswith("This is a scam.")


def test_templates_cover_every_verdict_and_category_in_both_languages():
    from typing import get_args

    from hub.schemas import ScamCategory

    for lang in ("en", "hi"):
        for v in ("SCAM", "SUSPICIOUS", "SAFE", "UNKNOWN"):
            out = explainer.template(v, None, lang, "Muaaz")
            assert out and "{" not in out
            words = explainer._verdict_words()[lang][v]
            assert any(w in out.lower() for w in words), (lang, v)
        for cat in get_args(ScamCategory):
            assert (
                explainer.template("SCAM", cat, lang, "Muaaz")
                != explainer.template("SCAM", None, lang, "Muaaz")
                or cat == "other_scam"
            )


def test_hindi_postcheck_and_digits():
    problems = explainer.post_check("यह स्कैम है। लिंक न खोलें। Muaaz को फ़ोन करें।", "SCAM", "hi", "x")
    assert problems == []
    assert "foreign_number" in explainer.post_check("यह स्कैम है। ९८७६ पर फ़ोन करें।", "SCAM", "hi", "x")
    assert explainer.post_check("यह स्कैम है। १९३० पर फ़ोन न करें।", "SCAM", "hi", "call 1930") == []


@pytest.mark.parametrize(
    "bad",
    [
        "This is not a scam. Call Muaaz.",
        "This isn't a scam, it looks fine.",
        "This is no scam. Do not worry.",
        "This looks like a normal message. It is a scam though. Call Muaaz.",  # other verdict
    ],
)
def test_negated_or_contradicting_verdict_rejected_then_template(bad):
    fake = Fake(bad, bad)
    assert run(SCAM, fake) == explainer.template("SCAM", "kyc_account_block", "en", "Muaaz")
    assert len(fake.bodies) == 2


def test_negated_first_try_then_good_regeneration():
    fake = Fake("This is not a scam. Call Muaaz.", GOOD)
    assert run(SCAM, fake) == GOOD


@pytest.mark.parametrize(
    "ok",
    [
        "This is a scam. Do not click, call Muaaz.",
        "This is a scam, not a drill. Call Muaaz.",
        "Be careful. Do not reply, call Muaaz.",
    ],
)
def test_legit_negations_elsewhere_pass(ok):
    verdict = "SUSPICIOUS" if ok.startswith("Be careful") else "SCAM"
    assert explainer.post_check(ok, verdict, "en", "x") == []


def test_suspicious_must_not_say_scam_or_safe():
    pc = explainer.post_check
    assert "other_verdict_word" in pc("Be careful, this could be a scam.", "SUSPICIOUS", "en", "x")
    assert "negated_verdict_word" in pc(
        "This is not suspicious. Be careful.", "SUSPICIOUS", "en", "x"
    )


def test_hindi_negation_and_contradiction():
    pc = explainer.post_check
    assert "negated_verdict_word" in pc("यह स्कैम नहीं है।", "SCAM", "hi", "x")
    assert "other_verdict_word" in pc("यह सामान्य मैसेज है। यह स्कैम है।", "SCAM", "hi", "x")
    assert pc("यह स्कैम है। लिंक न खोलें।", "SCAM", "hi", "x") == []
    assert pc("सावधान रहें, लिंक न खोलें।", "SUSPICIOUS", "hi", "x") == []


def test_keep_alive_is_resident():
    fake = Fake(GOOD)
    run(SCAM, fake)
    assert fake.bodies[0]["keep_alive"] == -1 == explainer.OLLAMA_KEEP_ALIVE
