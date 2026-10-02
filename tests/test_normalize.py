"""Normalize node (PRD FR-10). Synthetic messages only."""

from hub.normalize import normalize


def test_extracts_urls_phones_upi_as_substrings():
    text = (
        "Update at https://sbi-kyc.in/login, or sbi-help.xyz/form. "
        "Call +91 98765 43210 or 9123456789. Pay refund.help@ybl"
    )
    n = normalize(text)
    assert n.urls == ["https://sbi-kyc.in/login", "sbi-help.xyz/form"]
    assert n.phones == ["+91 98765 43210", "9123456789"]
    assert n.upi_handles == ["refund.help@ybl"]
    assert all(x in n.text for x in n.urls + n.phones + n.upi_handles)


def test_not_urls_amounts_abbreviations_or_upi_domains():
    n = normalize("Rs.500 debited from a/c. i.e. today. Pay to refund.help@ybl at 9.30")
    assert n.urls == []


def test_idn_and_redacted_phone():
    n = normalize("Visit sbі.co.in. Call +92 301 2<PHONE> or 9876<PHONE>")
    assert n.urls == ["sbі.co.in"]
    assert n.phones == ["+92 301 2<PHONE>", "9876<PHONE>"]


def test_sender_classification():
    assert normalize("x", "VK-SBIUPD").sender_status == "registered"
    assert normalize("x", "+91 98765 43210").sender_status == "unregistered"
    assert normalize("x", "Sunita Didi").sender_status == "unknown"
    assert normalize("x").sender_status == "unknown"


def test_perception_header_supplies_sender():
    n = normalize("SENDER: AX-HDFCBK\nMESSAGE:\nYour a/c is debited")
    assert n.sender == "AX-HDFCBK" and n.sender_status == "registered"
    assert n.text == "Your a/c is debited"
    n = normalize("SENDER: unknown\nMESSAGE:\nHello", "VK-X")
    assert n.sender == "VK-X" and n.text == "Hello"


def test_zero_width_inside_keywords_is_removed():
    assert normalize("O​T‌P share karo").text == "OTP share karo"
