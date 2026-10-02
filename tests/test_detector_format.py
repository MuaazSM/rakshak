"""§8.2 detector I/O format and FR-10 text normalization shared by training and inference."""

import json

import pytest

from hub import prompts
from hub.detector import chat_example, target_json, user_message
from hub.normalize import normalize_text, sender_status
from hub.schemas import DetectorOutput


def test_normalize_nfkc_zero_width_and_whitespace():
    raw = "  Your  a/c​ will be\tblocked  \n\n\n ＫＹＣ   now ­ "
    assert normalize_text(raw) == "Your a/c will be blocked\nKYC now"


def test_normalize_keeps_devanagari():
    assert normalize_text("आपका  OTP   है।") == "आपका OTP है।"


@pytest.mark.parametrize(
    "sender, status",
    [
        ("VK-SBIUPD", "registered"),
        ("AX-HDFCBK-S", "registered"),
        ("+91 98765 43210", "unregistered"),
        ("+91 9876<PHONE>", "unregistered"),
        ("<PHONE>", "unknown"),
        ("<NAME>", "unknown"),
        ("SBI Bank", "unknown"),
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_sender_status(sender, status):
    assert sender_status(sender) == status


def test_user_message_matches_section_8_2():
    msg = user_message("sms", "VK-SBIUPD", ["shortener_link", "apk_link", "apk_link"], "Hello")
    assert msg == (
        "CHANNEL: sms\nSENDER: VK-SBIUPD (registered)\n"
        "RULE_SIGNALS: [apk_link, shortener_link]\nMESSAGE:\nHello"
    )
    assert user_message("whatsapp", None, [], "x").splitlines()[1:3] == [
        "SENDER: unknown (unknown)",
        "RULE_SIGNALS: []",
    ]


def test_target_json_compact_and_key_order():
    out = DetectorOutput(
        verdict="SCAM",
        category="kyc_account_block",
        red_flags=[{"quote": "blocked today", "reason": "urgency_deadline"}],
    )
    s = target_json(out)
    assert s == (
        '{"verdict":"SCAM","category":"kyc_account_block",'
        '"red_flags":[{"quote":"blocked today","reason":"urgency_deadline"}]}'
    )
    assert DetectorOutput.model_validate(json.loads(s)) == out


def test_chat_example_roles_and_system_prompt():
    out = DetectorOutput(verdict="SAFE", category="genuine_otp", red_flags=[])
    msgs = chat_example("sms", "AD-SBIOTP", [], "OTP <OTP>", out)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == prompts.DETECTOR_SYSTEM
