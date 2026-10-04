"""Terminal labeler (PRD §10.1), driven with scripted answers. Synthetic items only."""

import json

from training.label import Labeler, load_labels, run

SCAM = {
    "id": "s1",
    "text": "Your SBI a/c <ACCT> will be blocked today. Update KYC at http://sbi-kyc.example",
    "channel": "sms",
    "source": "family_real",
}
ADVISORY = {"id": "a1", "text": "Lottery! Pay Rs 5000 fee", "channel": "sms", "source": "advisory"}


def scripted(answers):
    it = iter(answers)
    out = []
    return Labeler(ask=lambda _prompt: next(it), say=out.append), out


def write_items(tmp_path, *items):
    d = tmp_path / "redacted"
    d.mkdir()
    (d / "mom.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n", "utf-8")
    return d, d / "_labels.jsonl"


def test_labels_scam_with_exact_quote_validation(tmp_path):
    d, labels = write_items(tmp_path, SCAM)
    labeler, out = scripted(
        [
            "1",  # SCAM
            "kyc",  # kyc_account_block (unique prefix)
            "blocked tomorrow",  # not a substring → rejected
            "will be blocked today",
            "urgency",  # urgency_deadline
            "",  # done with flags
            "en",
            "mom",
            "n",
            "y",  # test
        ]
    )
    assert run(d, labels, labeler) == 1
    lab = load_labels(labels)["s1"]
    assert lab["verdict"] == "SCAM" and lab["category"] == "kyc_account_block"
    assert lab["red_flags"] == [{"quote": "will be blocked today", "reason": "urgency_deadline"}]
    assert lab["language"] == "en" and lab["source_phone"] == "mom"
    assert lab["obfuscated"] is False and lab["is_test"] is True
    assert lab["seed_group"] == "s1"
    assert any("not an exact substring" in line for line in out)


def test_advisory_item_cannot_be_test_and_safe_has_no_flags(tmp_path):
    d, labels = write_items(tmp_path, ADVISORY)
    labeler, _ = scripted(["SAFE", "legit_promo", "hinglish", "", "y"])
    run(d, labels, labeler)
    lab = load_labels(labels)["a1"]
    assert lab["red_flags"] == [] and lab["is_test"] is False and lab["source_phone"] is None


def test_saves_after_each_item_and_resumes(tmp_path):
    d, labels = write_items(tmp_path, SCAM, ADVISORY)
    labeler, _ = scripted(["SAFE", "personal", "en", "own", "n", "n", "q"])
    assert run(d, labels, labeler) == 1  # quit during the second item
    assert set(load_labels(labels)) == {"s1"}
    labeler, _ = scripted(["s"])  # resumes at the unlabeled item and skips it
    assert run(d, labels, labeler) == 0
    assert set(load_labels(labels)) == {"s1"}


def test_public_report_is_labelable_but_never_test(tmp_path):
    # Advisory (public_report) items are a separate eval slice, never the frozen test set (§10.2).
    report = {
        "id": "p1",
        "text": "Your parcel is held at customs. Pay Rs 499 at http://customs-fee.example",
        "channel": "sms",
        "source": "public_report",
        "source_url": "https://example.org/advisory/1",
    }
    d, labels = write_items(tmp_path, report)
    labeler, _ = scripted(["SCAM", "courier", "", "en", "", "n"])  # no TEST prompt for advisory
    run(d, labels, labeler)
    lab = load_labels(labels)["p1"]
    assert lab["source"] == "public_report" and lab["source_phone"] is None
    assert lab["is_test"] is False
