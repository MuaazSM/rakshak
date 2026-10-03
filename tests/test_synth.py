"""Synthetic generator: seed/quote validation, deterministic axes, resumable run (mocked Ollama)."""

import asyncio
import json

import httpx
import pytest

from training import synth

SEED_SCAM = {
    "id": "seed-001",
    "text": "Your account will be blocked today. Update KYC: http://hdfc-kycupdate.co.in/login",
    "sender": "+91 99876 54321",
    "channel": "sms",
    "verdict": "SCAM",
    "category": "kyc_account_block",
    "red_flags": [
        {"quote": "will be blocked today", "reason": "urgency_deadline"},
        {"quote": "http://hdfc-kycupdate.co.in/login", "reason": "lookalike_link"},
    ],
    "language": "en",
    "meta": {"source": "synthetic_seed"},
}
SEED_SAFE = {
    "id": "seed-002",
    "text": "<OTP> is your OTP for Rs 2,499 at AMAZON. Do not share this OTP with anyone.",
    "sender": "AX-HDFCBK",
    "channel": "sms",
    "verdict": "SAFE",
    "category": "genuine_otp",
    "red_flags": [],
    "language": "en",
    "meta": {"source": "synthetic_seed"},
}


def gemma_reply(text, flags, sender="+91 98100 22334"):
    return json.dumps({"text": text, "sender": sender, "red_flags": flags})


def test_validate_item_quotes_and_vocab():
    assert synth.validate_item(SEED_SCAM) == []
    bad_quote = {**SEED_SCAM, "red_flags": [{"quote": "not in text", "reason": "urgency_deadline"}]}
    assert "quote is not an exact substring" in synth.validate_item(bad_quote)
    bad_reason = {**SEED_SCAM, "red_flags": [{"quote": "will be blocked", "reason": "rude"}]}
    assert "bad reason" in synth.validate_item(bad_reason)
    assert "needs at least one red flag" in synth.validate_item({**SEED_SCAM, "red_flags": []})
    assert synth.validate_item(SEED_SAFE) == []
    assert "SAFE must have no red flags" in synth.validate_item(
        {**SEED_SAFE, "red_flags": SEED_SCAM["red_flags"]}
    )
    assert "category does not fit verdict" in synth.validate_item(
        {**SEED_SAFE, "category": "personal2"}
    )


def test_load_seeds_limit(tmp_path):
    p = tmp_path / "seeds.jsonl"
    p.write_text(json.dumps(SEED_SCAM) + "\n" + json.dumps(SEED_SAFE) + "\n")
    assert len(synth.load_seeds(p)) == 2
    with pytest.raises(synth.SynthError):
        synth.load_seeds(p, limit=1)


def test_accept_response_validates_quotes_against_normalized_text():
    allowed = {"urgency_deadline", "lookalike_link"}
    text = "Dear  customer,​ your KYC is expiring. Verify now: http://kyc-sbi.in/x"
    flags = [{"quote": "Verify  now", "reason": "urgency_deadline"}]  # normalized form matches
    got = synth.accept_response(json.loads(gemma_reply(text, flags)), SEED_SCAM, allowed)
    assert got and got["red_flags"][0]["quote"] == "Verify now"
    assert "​" not in got["text"]
    # a hallucinated quote, an out-of-vocab reason or a reason outside the seed's set: dropped
    for flags in (
        [{"quote": "Pay now", "reason": "urgency_deadline"}],
        [{"quote": "Verify now", "reason": "asks_payment"}],
        [{"quote": "Verify now", "reason": "made_up"}],
        [],
    ):
        assert (
            synth.accept_response(json.loads(gemma_reply(text, flags)), SEED_SCAM, allowed) is None
        )


def test_accept_response_safe_rules():
    ok = {
        "text": "Rs 500 debited from A/c XX<ACCT>. Not you? Call 1800 2662.",
        "sender": "AX-SBIUPI",
    }
    got = synth.accept_response(ok, SEED_SAFE, set())
    assert got and got["red_flags"] == []
    # unregistered sender, apk link and leftover placeholders are rejected
    assert synth.accept_response({**ok, "sender": "+91 98765 43210"}, SEED_SAFE, set()) is None
    apk = {"text": "Install https://x-bank.in/app.apk now", "sender": "AX-SBIUPI"}
    assert synth.accept_response(apk, SEED_SAFE, set()) is None
    assert (
        synth.accept_response({**ok, "text": "Hello [link here] ok then"}, SEED_SAFE, set()) is None
    )


def test_make_job_is_deterministic_and_covers_axes():
    a, b = synth.make_job(SEED_SCAM, 1, 2), synth.make_job(SEED_SCAM, 1, 2)
    assert a == b and a.key == "seed-001/s1/v2"
    jobs = [synth.make_job(SEED_SCAM, 0, v) for v in range(1, 6)]
    assert len({j.obfuscation for j in jobs}) == 5  # distinct obfuscations across 5 variants
    assert synth.make_job(SEED_SCAM, 0, 0).obfuscation == "none"
    assert {synth.make_job(SEED_SCAM, s, 0).language for s in range(3)} == set(synth.LANGUAGES)
    call = {**SEED_SCAM, "channel": "call_description"}
    assert synth.make_job(call, 0, 1).channel == "call_description"
    assert synth.make_job(SEED_SAFE, 0, 3).obfuscation == "none"


def fake_complete(calls):
    async def complete(prompt: str) -> str:
        calls.append(prompt)
        if "GENUINE" in prompt:
            return json.dumps(
                {
                    "text": "<OTP> is your OTP for Rs 77. Do not share this OTP with anyone.",
                    "sender": "AX-ICICIB",
                }
            )
        marker = "SEED (kyc_account_block): "
        src = prompt.split(marker, 1)[1]
        text = f"{src.split('.')[0]}. Variant {len(calls)}"
        return gemma_reply(text, [{"quote": "will be blocked", "reason": "urgency_deadline"}])

    return complete


def test_generate_two_levels_and_resume(tmp_path):
    out = tmp_path / "b.jsonl"
    calls: list[str] = []
    stats = asyncio.run(
        synth.generate(
            [SEED_SCAM, SEED_SAFE],
            fake_complete(calls),
            out,
            scenarios=2,
            variants=2,
            log=lambda _: None,
        )
    )
    items = synth.read_items(out)
    assert stats["generated"] == len(items) == 2 * 3 * 2  # (scenarios x (1+variants)) per seed
    assert not stats.get("dropped")
    assert all(
        i["meta"]["source"] == "synthetic" and i["meta"]["generator"] == "gemma4:e2b" for i in items
    )
    assert {i["meta"]["seed_group"] for i in items} == {"seed-001", "seed-002"}
    safe = [i for i in items if i["verdict"] == "SAFE"]
    assert safe and all(i["red_flags"] == [] and i["meta"]["hard_negative"] for i in safe)
    scam = [i for i in items if i["verdict"] == "SCAM"]
    assert any(i["meta"]["obfuscated"] for i in scam)
    assert not any(i["meta"]["obfuscated"] for i in scam if i["meta"]["variant"] == 0)
    # variants are rewritten from the scenario text, not from the seed
    v_prompts = [p for p in calls if "Variant" in p.split("SEED", 1)[-1]]
    assert v_prompts
    # resume: a second run generates nothing new
    n_calls = len(calls)
    stats2 = asyncio.run(
        synth.generate(
            [SEED_SCAM, SEED_SAFE],
            fake_complete(calls),
            out,
            scenarios=2,
            variants=2,
            log=lambda _: None,
        )
    )
    assert len(calls) == n_calls and stats2.get("generated", 0) == 0
    assert len(synth.read_items(out)) == len(items)


def test_generate_retries_invalid_json_then_drops(tmp_path):
    n = {"c": 0}

    async def flaky(prompt: str) -> str:
        n["c"] += 1
        return (
            "not json"
            if n["c"] <= 2
            else gemma_reply(
                "Your account will be blocked now",
                [{"quote": "will be blocked", "reason": "urgency_deadline"}],
            )
        )

    job = synth.make_job(SEED_SCAM, 0, 0)
    item = asyncio.run(synth.run_job(job, None, flaky))
    assert item and n["c"] == 3

    async def always_bad(prompt: str) -> str:
        return "{"

    assert asyncio.run(synth.run_job(job, None, always_bad)) is None


def test_make_complete_sends_thinking_off_json_mode():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content), url=str(request.url))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await synth.make_complete("http://ollama.test", "gemma4:e2b", client)("hi")

    assert asyncio.run(go()) == "{}"
    assert seen["think"] is False and seen["format"] == "json" and seen["stream"] is False
    assert seen["options"]["temperature"] == 0.9 and seen["url"].endswith("/api/chat")


def test_review_sample_is_ten_percent_and_deterministic():
    items = [{"id": str(i)} for i in range(100)]
    s1, s2 = synth.review_sample(items), synth.review_sample(items)
    assert len(s1) == 10 and s1 == s2


def test_suspect_reasons():
    safe = {
        "text": "Update now at http://evil-bank.in/login",
        "sender": "AX-HDFCBK",
        "verdict": "SAFE",
        "language": "en",
    }
    assert "safe_nonofficial_link" in synth.suspect_reasons(safe)
    ok = {
        "text": "Rs 77 debited from A/c XX<ACCT>.",
        "sender": "AX-HDFCBK",
        "verdict": "SAFE",
        "language": "en",
    }
    assert synth.suspect_reasons(ok) == []
    assert "lang_hi_without_devanagari" in synth.suspect_reasons({**ok, "language": "hi"})


def test_relabel_language_follows_script():
    assert synth.relabel_language("hi", "आपका खाता बंद हो जाएगा") == "hi"
    assert synth.relabel_language("hi", "Aapka account band ho jayega") == "hinglish"
    assert synth.relabel_language("hinglish", "आपका खाता बंद हो जाएगा, तुरंत बताएं") == "hi"
    assert synth.relabel_language("en", "Your account will be blocked") == "en"
    assert synth.relabel_language("en", "Your खाता will be blocked") == "hinglish"


def test_suspicious_must_stay_ambiguous():
    susp = {**SEED_SCAM, "id": "seed-009", "verdict": "SUSPICIOUS", "category": "other_scam"}
    allowed = {"urgency_deadline"}
    flags = [{"quote": "important", "reason": "urgency_deadline"}]
    calm = {
        "text": "Hi, is this your number? I have something important.",
        "sender": "+91 98100 22334",
        "red_flags": flags,
    }
    assert synth.accept_response(calm, susp, allowed)
    linked = {**calm, "text": "Hi, something important: http://x-promo.in/win"}
    assert synth.accept_response(linked, susp, allowed) is None


def test_reason_fits_removes_flags_that_cannot_carry_their_reason():
    assert synth.reason_fits("http://bit.ly/x", "lookalike_link")
    assert not synth.reason_fits("acount", "lookalike_link")
    assert synth.reason_fits("Challan.apk", "apk_link") and not synth.reason_fits(
        "bit.ly/x", "apk_link"
    )
    assert synth.reason_fits("share your OTP", "asks_otp_or_pin")
    assert not synth.reason_fits("Hello sir", "asks_otp_or_pin")
    allowed = {"lookalike_link", "urgency_deadline"}
    obj = {
        "red_flags": [
            {"quote": "acount", "reason": "lookalike_link"},
            {"quote": "blocked", "reason": "urgency_deadline"},
        ]
    }
    assert synth.clean_flags(obj, "acount is blocked", allowed) == [
        {"quote": "blocked", "reason": "urgency_deadline"}
    ]
    assert (
        synth.clean_flags({"red_flags": obj["red_flags"][:1]}, "acount is blocked", allowed) is None
    )


def test_finalize_item_repairs_sender_and_category():
    def make(cat, text, sender, verdict="SAFE"):
        return {"verdict": verdict, "category": cat, "text": text, "sender": sender,
                "meta": {"key": "seed-9/s0", "hard_negative": False}}  # fmt: skip

    bank = "Rs 480 debited from A/c XX<ACCT>. Not you? Call 1800 2662."
    fixed = synth.finalize_item(make("delivery_update", bank, "AX-987654"))
    assert (
        fixed["category"] == "transaction_alert"
        and fixed["meta"]["category_requested"] == "delivery_update"
    )
    assert fixed["sender"] != "AX-987654" and fixed["meta"]["hard_negative"]
    assert synth.finalize_item(make("delivery_update", bank, "AX-987654"), strict=True) is None
    assert synth.finalize_item(make("legit_promo", "Hello how are you", "AX-987654")) is None
    # idempotent and deterministic
    again = synth.finalize_item(dict(fixed))
    assert again["sender"] == fixed["sender"] and again["category"] == fixed["category"]
    scam = synth.finalize_item(make("kyc_account_block", "Pay now", "+91 7890 123456", "SCAM"))
    assert scam["sender"] != "+91 7890 123456"


def test_call_responses_and_sender_redraw():
    spec = next(sp for sp in synth.CALL_SPECS if sp[0] == "c1")
    ok = {"text": "Caller said the bank confirmed my card is blocked.", "sender": "+91 98765 43210"}
    got = synth.accept_call_response(ok, spec)
    assert got and got["red_flags"] == []
    asks = {"text": "Caller said to share the OTP to confirm the block.", "sender": "1800 123 4567"}
    assert synth.accept_call_response(asks, spec) is None
    assert synth.accept_call_response({**ok, "text": "Open http://x-bank.in now"}, spec) is None
    susp = next(sp for sp in synth.CALL_SPECS if sp[1] == "SUSPICIOUS")
    flags = [{"quote": "survey for the bank", "reason": "impersonates_authority"}]
    sus_ok = {
        "text": "Caller did a survey for the bank and will call later.",
        "sender": "x",
        "red_flags": flags,
    }
    assert synth.accept_call_response(sus_ok, susp)["red_flags"] == flags
    assert synth.accept_call_response({**sus_ok, "red_flags": []}, susp) is None
    item = {
        "category": "personal",
        "verdict": "SAFE",
        "sender": "+91 98765 43210",
        "meta": {"key": "calls-c4/1"},
    }
    assert synth.redraw_call_sender(
        dict(item),
    ) == synth.redraw_call_sender(dict(item))


def test_generate_calls_writes_call_description_items(tmp_path):
    async def complete(prompt: str) -> str:
        if "red_flags" in prompt:
            return json.dumps(
                {
                    "text": "Caller did a survey for the bank, will call again.",
                    "sender": "x",
                    "red_flags": [
                        {"quote": "survey for the bank", "reason": "impersonates_authority"}
                    ],
                }
            )
        return json.dumps(
            {"text": "Caller said the maintenance cut is tomorrow.", "sender": "+91 98765 43210"}
        )

    out = tmp_path / "b2.jsonl"
    stats = asyncio.run(synth.generate_calls(complete, out, per_spec=1, log=lambda _: None))
    items = synth.read_items(out)
    assert stats["generated"] == len(items) == len(synth.CALL_SPECS)
    assert {i["channel"] for i in items} == {"call_description"}
    assert {i["verdict"] for i in items} == {"SAFE", "SUSPICIOUS"}
    assert all(i["meta"]["seed_group"].startswith("calls-") for i in items)
    assert asyncio.run(synth.generate_calls(complete, out, per_spec=1, log=lambda _: None))[
        "resumed"
    ] == len(items)


def test_link_batch_accepts_only_official_links():
    cat, link = "transaction_alert", "https://www.hdfcbank.com/support"
    ok = {
        "text": f"Rs 480 debited from your A/c. Not you? Call 1800 2662. More: {link}",
        "sender": "AX-HDFCBK",
    }
    assert synth.accept_link_response(ok, cat, link)
    fake = {**ok, "text": ok["text"] + " or http://hdfc-secure-login.in/x"}
    assert synth.accept_link_response(fake, cat, link) is None
    no_link = {**ok, "text": "Rs 480 debited from your A/c. Not you? Call 1800 2662."}
    assert synth.accept_link_response(no_link, cat, link) is None
    other = {**ok, "text": "Rs 480 debited from your A/c. More: https://www.icicibank.com/support"}
    assert synth.accept_link_response(other, cat, link) is None  # must contain the given link
    otp = {**ok, "text": f"Please share your OTP now at {link}"}
    assert synth.accept_link_response(otp, cat, link) is None


def test_generate_links_groups_and_resume(tmp_path):
    async def complete(prompt: str) -> str:
        link = prompt.split("written exactly like this: ")[1].split("\n")[0]
        cat = prompt.split("GENUINE message of type ")[1].split(" ")[0]
        body = {
            "genuine_otp": "<OTP> is your OTP. Do not share it with anyone.",
            "transaction_alert": "Rs 99 debited from your A/c. Not you? Call 1800 2662.",
            "delivery_update": "Your order is out for delivery today with the courier.",
            "legit_promo": "Big sale offer this weekend with 20% off, T&C apply.",
            "govt_genuine": "Your electricity bill is generated, due date 15-10-2026.",
        }[cat]
        return json.dumps({"text": f"{body} Details: {link}", "sender": "AX-HDFCBK"})

    out = tmp_path / "b3.jsonl"
    stats = asyncio.run(synth.generate_links(complete, out, per_group=2, log=lambda _: None))
    items = synth.read_items(out)
    assert (
        stats["generated"]
        == len(items)
        == 5 * synth.LINK_GROUPS_PER_CATEGORY * 2 - stats.get("dropped", 0)
    )
    assert all(
        i["verdict"] == "SAFE" and i["meta"]["seed_group"].startswith("links-") for i in items
    )
    assert all(normalize_has_url(i["text"]) for i in items)
    assert asyncio.run(synth.generate_links(complete, out, per_group=2, log=lambda _: None))[
        "resumed"
    ] == len(items)


def normalize_has_url(text: str) -> bool:
    from hub.normalize import extract_urls

    return bool(extract_urls(text))
