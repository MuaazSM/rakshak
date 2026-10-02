"""Rules engine (PRD §7.3, FR-12). Synthetic messages only.

CASES[signal]["pos"] must fire `signal`; CASES[signal]["neg"] must not. Each case is
(text, sender). Every signal has ≥ 3 positives and ≥ 2 negatives, across English,
Devanagari Hindi and romanized Hinglish where the signal is language-dependent.
"""

import pytest

from hub.normalize import normalize
from hub.rules import HARD, SOFT, evaluate, load_rule_data, rule_signals

CASES: dict[str, dict[str, list[tuple[str, str | None]]]] = {
    "apk_link": {
        "pos": [
            ("Download the app from http://bijli-bill-update.xyz/app.apk", None),
            ("Install SBI_KYC.apk to update your details", None),
            ("Apna bijli bill dekhne ke liye app download karo: https://mseb-bill.top/view", None),
            ("यह ऐप डाउनलोड करें: http://parcel-track.live/app", None),
        ],
        "neg": [
            (
                "Download the YONO app from https://play.google.com/store/apps/details?id=com.sbi.lotusintouch",
                None,
            ),
            ("Your statement is ready on https://www.hdfcbank.com/statements", None),
            ("Please download your e-ticket from the IRCTC app", None),
        ],
    },
    "lookalike_domain": {
        "pos": [
            ("Update KYC at https://sbi-yono-kyc.in/login", None),
            ("Login now: hdfcbnak.com/verify", None),
            ("Visit http://xn--sbi-pqa.com to claim reward", None),
            ("Complete KYC at sbі.co.in today", None),  # Cyrillic і
            ("Pay bill at http://mahadiscom-bill.xyz", None),
        ],
        "neg": [
            ("Login to https://onlinesbi.sbi to view your statement", None),
            ("File returns at https://www.incometax.gov.in/iec/foportal", None),
            ("Track at https://www.indiapost.gov.in", None),
            ("Your order from https://www.amazon.in has shipped", None),
        ],
    },
    "upi_pin_to_receive": {
        "pos": [
            ("Enter your UPI PIN to receive Rs 5000 cashback", None),
            ("Approve the collect request to get your refund of Rs 2,000", None),
            ("रिफंड प्राप्त करने के लिए अपना यूपीआई पिन डालें", None),
            ("Paise receive karne ke liye apna UPI PIN daalo", None),
        ],
        "neg": [
            ("You never need to enter your UPI PIN to receive money", "AX-NPCI"),
            ("Rs 500 received in your account from Ramesh", "AD-HDFCBK"),
            ("Enter UPI PIN to pay Rs 200 to the merchant", None),
            ("पैसे पाने के लिए यूपीआई पिन कभी न डालें", None),
        ],
    },
    "unregistered_sender": {
        "pos": [
            ("Dear customer your SBI account KYC is pending", "+91 98765 43210"),
            ("This is Mumbai Police. Call back urgently", "9876543210"),
            ("आपका बैंक खाता बंद हो जाएगा", "+91 9123456789"),
        ],
        "neg": [
            ("Dear customer your SBI account KYC is pending", "VK-SBIUPD"),
            ("Mummy kal aana, khana saath khayenge", "+91 98765 43210"),
            ("Your bank statement is ready", None),
        ],
    },
    "shortener_link": {
        "pos": [
            ("Pay your bill at bit.ly/3XyZab", None),
            ("Check details: https://tinyurl.com/kyc-now", None),
            ("Parcel held, see cutt.ly/abc123", None),
        ],
        "neg": [
            ("Pay your bill at https://www.mahadiscom.in", None),
            ("Bit of a delay, will reach by 7", None),
            ("Read the advisory at https://cybercrime.gov.in", None),
        ],
    },
    "threat_lexicon": {
        "pos": [
            ("A case has been registered against you. You will face digital arrest", None),
            ("आपको गिरफ्तार कर लिया जाएगा", None),
            ("Aapki bijli kaat di jayegi aaj raat", None),
            ("Your account will be blocked today", None),
        ],
        "neg": [
            ("Your order has been delivered. Thank you for shopping", None),
            ("Happy birthday beta, khoob khush raho", None),
            ("आपका बिल जमा हो गया है, धन्यवाद", None),
        ],
    },
    "urgency_plus_payment": {
        "pos": [
            ("Pay Rs 5000 processing fee immediately to release your parcel", None),
            ("Turant 2000 rupaye GPay karo warna connection kat jayega", None),
            ("तुरंत भुगतान करें नहीं तो कनेक्शन काट दिया जाएगा", None),
        ],
        "neg": [
            ("Your payment of Rs 500 was successful", "AD-HDFCBK"),
            ("Please come home immediately, dinner is ready", None),
            ("Your electricity bill of Rs 840 is generated", "VM-MSEDCL"),
        ],
    },
    "foreign_code_authority": {
        "pos": [
            ("This is CBI officer. Call +92 301 2345678 regarding your case", None),
            ("Police verification pending, contact on WhatsApp", "+44 7700 900123"),
            ("मुंबई पुलिस से बात करें: +880 1712 345678", None),
        ],
        "neg": [
            ("This is Mumbai Police helpline, call +91 22 2262 1855", None),
            ("Hi beta, reached Dubai safely, call me on +971 50 123 4567", None),
            ("Police station address updated", "+91 98765 43210"),
        ],
    },
    "asks_otp_or_pin": {
        "pos": [
            ("Please share the OTP you just received to verify your account", None),
            ("OTP aaya hoga, bhej do na", None),
            ("आपके फोन पर आया ओटीपी बताएं", None),
            ("Don't worry, just tell me your ATM PIN", None),
        ],
        "neg": [
            ("Your OTP is <OTP>. Do not share it with anyone. -SBI", "AD-SBIOTP"),
            ("OTP kisi ke saath share mat karo", None),
            ("आपका OTP <OTP> है। किसी के साथ साझा न करें।", "AD-SBIOTP"),
            ("Never share your OTP, PIN or CVV with anyone", "AX-HDFCBK"),
            ("Share this link with your friends to win", None),
        ],
    },
}


def _fires(text: str, sender: str | None) -> set[str]:
    r = evaluate(normalize(text, sender))
    return set(r.hard) | set(r.soft)


def _params(kind: str):
    return [
        pytest.param(sig, text, sender, id=f"{sig}-{kind}{i}")
        for sig, c in CASES.items()
        for i, (text, sender) in enumerate(c[kind])
    ]


def test_every_signal_has_enough_cases():
    assert set(CASES) == set(HARD) | set(SOFT)
    for sig, c in CASES.items():
        assert len(c["pos"]) >= 3 and len(c["neg"]) >= 2, sig


@pytest.mark.parametrize("signal, text, sender", _params("pos"))
def test_signal_fires(signal, text, sender):
    assert signal in _fires(text, sender)


@pytest.mark.parametrize("signal, text, sender", _params("neg"))
def test_signal_does_not_fire(signal, text, sender):
    assert signal not in _fires(text, sender)


@pytest.mark.parametrize("signal, text, sender", _params("pos"))
def test_evidence_is_exact_substring(signal, text, sender):
    n = normalize(text, sender)
    r = evaluate(n)
    assert r.evidence[signal]
    assert all(e in n.text for e in r.evidence[signal])


def test_hard_and_soft_split_and_state_shape():
    r = evaluate(normalize("Install SBI_KYC.apk now, digital arrest pending", None))
    assert r.hard == ["apk_link"] and r.soft == ["threat_lexicon"]
    assert r.as_state() == {"hard": ["apk_link"], "soft": ["threat_lexicon"]}


def test_genuine_otp_sms_is_clean():
    r = evaluate(
        normalize("482913 is your OTP for txn of Rs 1,250. Do not share. -HDFC Bank", "AD-HDFCBK")
    )
    assert r.hard == [] and r.soft == []


def test_rule_signals_flat_list_for_dataset_builders():
    sigs = rule_signals(
        "Update KYC at https://sbi-yono-kyc.in/login immediately or account blocked",
        "+91 98765 43210",
        "sms",
    )
    assert sigs[0] == "lookalike_domain"
    assert {"unregistered_sender", "threat_lexicon"} <= set(sigs)


def test_rule_data_loaded_and_cached():
    d = load_rule_data()
    assert "onlinesbi.sbi" in d.official and "gov.in" in d.restricted
    assert "bit.ly" in d.shorteners and "sbi" in d.brands
    assert load_rule_data() is d
