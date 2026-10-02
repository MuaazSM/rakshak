"""Redaction (PRD §10.3). Synthetic examples only: no real messages, names or numbers."""

import json
from collections import Counter

import pytest

from training.redact import Names, load_names, main, redact_record, redact_text

NAMES = Names(
    names=["Sunita", "सुनीता", "Ramesh Kumar"],
    addrs=["12 Example Nagar"],
    phones=["9000000000"],
)


def red(text: str) -> tuple[str, Counter]:
    counts: Counter = Counter()
    return redact_text(text, NAMES, counts), counts


# --- masks ---


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Call +91 98765 43210 now", "Call +91 9876<PHONE> now"),
        ("Call 9876543210 now", "Call 9876<PHONE> now"),
        ("Call +91-98765-43210", "Call +91-9876<PHONE>"),
        ("WhatsApp +92 301 2345678", "WhatsApp +92 301 2<PHONE>"),
    ],
)
def test_other_phones_keep_country_code_and_first_four_digits(text, expected):
    out, counts = red(text)
    assert out == expected
    assert counts["phone_partial"] == 1


@pytest.mark.parametrize(
    "number", ["9000000000", "+91 90000 00000", "+91-9000000000", "09000000000"]
)
def test_listed_personal_phone_fully_masked(number):
    out, counts = red(f"Mom here, call me on {number} ok")
    assert out == "Mom here, call me on <PHONE> ok"
    assert counts["phone_personal"] == 1


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Your OTP is 482913. Do not share.", "Your OTP is <OTP>. Do not share."),
        ("482913 is your OTP for login", "<OTP> is your OTP for login"),
        ("आपका OTP 482913 है।", "आपका OTP <OTP> है।"),
        ("ओटीपी 5566 किसी को न बताएं", "ओटीपी <OTP> किसी को न बताएं"),
        ("Your PIN is 1234", "Your PIN is <OTP>"),
        ("verification code: 77881", "verification code: <OTP>"),
    ],
)
def test_otp_and_pin_masked(text, expected):
    assert red(text)[0] == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("a/c XX4321 debited", "a/c <ACCT> debited"),
        ("A/c no. 123456789012 frozen", "A/c no. <ACCT> frozen"),
        ("card ending 4321 used", "card ending <ACCT> used"),
        ("Card 4111 1111 1111 1111 blocked", "Card <ACCT> blocked"),
        ("Aadhaar 1234 5678 9012 linked", "Aadhaar <ACCT> linked"),
        ("UPI Ref 612345678901 done", "UPI Ref <ACCT> done"),
        ("आपका खाता 55667788 बंद", "आपका खाता <ACCT> बंद"),
    ],
)
def test_account_and_card_fragments_masked(text, expected):
    assert red(text)[0] == expected


def test_names_and_addresses_masked_whole_word_case_insensitive():
    out, counts = red("Dear SUNITA, Ramesh Kumar at 12 Example Nagar. सुनीता जी")
    assert out == "Dear <NAME>, <NAME> at <ADDR>. <NAME> जी"
    assert counts["name"] == 3 and counts["addr"] == 1


def test_name_inside_a_longer_word_is_not_masked():
    assert red("Sunitaben and Sunitas")[0] == "Sunitaben and Sunitas"


# --- kept ---


@pytest.mark.parametrize(
    "text",
    [
        "Rs 25000 debited",
        "Pay ₹4999 now",
        "Send 5000 rupees processing fee",
        "Prize of Rs.250000000 waiting",
        "Visit http://sbi-kyc-verify.example/login/9876543210 now",
        "Download from bit.ly/3Xy9876 today",
        "Pay to refund.help@ybl immediately",
        "Customer care 1800-425-3800",
        "Meet at 9:30 on 03-10-26",
        "Valid for 10 min",
    ],
)
def test_amounts_links_upi_tollfree_and_times_kept(text):
    out, counts = red(text)
    assert out == text
    assert sum(counts.values()) == 0


def test_otp_keyword_in_another_sentence_does_not_mask_a_year():
    out = red("Your verification code is 7788. Meet in 2026 plan")[0]
    assert out == "Your verification code is <OTP>. Meet in 2026 plan"


def test_amount_near_pin_keyword_kept():
    assert red("Enter PIN to receive Rs 5000")[0] == "Enter PIN to receive Rs 5000"


def test_idempotent():
    once = red("Sunita, OTP 482913, call +91 98765 43210, a/c XX4321")[0]
    assert red(once)[0] == once


# --- records, sender, names file, CLI ---


def test_record_sender_header_kept_and_phone_sender_masked():
    rec = {
        "id": "x1",
        "text": "hi",
        "sender": "VK-SBIUPD",
        "channel": "sms",
        "source": "family_real",
    }
    assert redact_record(rec, NAMES)[0]["sender"] == "VK-SBIUPD"
    rec["sender"] = "+91 98765 43210"
    assert redact_record(rec, NAMES)[0]["sender"] == "+91 9876<PHONE>"
    rec["sender"] = "Sunita"
    assert redact_record(rec, NAMES)[0]["sender"] == "<NAME>"


def test_record_missing_field_rejected():
    with pytest.raises(ValueError, match="text"):
        redact_record({"id": "x", "channel": "sms", "source": "own_inbox"}, NAMES)


def test_load_names_file_format(tmp_path):
    p = tmp_path / "names.txt"
    p.write_text("# comment\nSunita\naddr: 12 Example Nagar\nphone: +91 90000 00000\n\n", "utf-8")
    n = load_names(p)
    assert n.names == ["Sunita"] and n.addrs == ["12 Example Nagar"] and n.phones == ["9000000000"]


def test_cli_redacts_fake_raw_file_and_reports_counts_only(tmp_path, capsys):
    raw, out = tmp_path / "raw", tmp_path / "redacted"
    raw.mkdir()
    names = tmp_path / "names.txt"
    names.write_text("Sunita\nphone: 9000000000\n", "utf-8")
    secret = "Sunita your OTP 482913 call 9000000000 a/c XX4321 or +91 98765 43210"
    records = [
        {
            "id": "f1",
            "text": secret,
            "sender": "VK-SBIUPD",
            "channel": "sms",
            "source": "family_real",
        },
        {
            "id": "f2",
            "text": "Rs 500 debited. Not you? Call 1800-425-3800",
            "channel": "sms",
            "source": "own_inbox",
        },
    ]
    (raw / "mom.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", "utf-8")

    assert main(["--raw", str(raw), "--out", str(out), "--names", str(names)]) == 0
    got = [json.loads(line) for line in (out / "mom.jsonl").read_text("utf-8").splitlines()]
    assert got[0]["text"] == "<NAME> your OTP <OTP> call <PHONE> a/c <ACCT> or +91 9876<PHONE>"
    assert got[0]["sender"] == "VK-SBIUPD"
    assert got[1]["text"] == records[1]["text"]
    assert {k: got[0][k] for k in ("id", "channel", "source")} == {
        "id": "f1",
        "channel": "sms",
        "source": "family_real",
    }

    report = capsys.readouterr().out
    for fragment in ("Sunita", "482913", "9000000000", "4321", "43210", "debited"):
        assert fragment not in report  # counts only, no text

    # A second run must not clobber files that may hold manual review edits.
    assert main(["--raw", str(raw), "--out", str(out), "--names", str(names)]) == 0
    assert "skip mom.jsonl" in capsys.readouterr().out


def test_digits_left_flags_unmasked_numbers_only():
    from training.redact import digits_left

    assert digits_left({"text": "call +91 9876<PHONE>, pay Rs 5000, see bit.ly/12345"}) == 0
    assert digits_left({"text": "send 20000 on GPay"}) == 1
