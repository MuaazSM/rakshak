"""Data contracts (PRD §8.1 Verdict, §8.2 Detector I/O, §8.3 CheckState, §5.3 request bodies).

Field names, order and enums must match the PRD exactly. Vocabularies: PRD §10.1.
"""

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

VerdictLabel = Literal["SCAM", "SUSPICIOUS", "SAFE", "UNKNOWN"]
DetectorVerdict = Literal["SCAM", "SUSPICIOUS", "SAFE"]
Channel = Literal["sms", "whatsapp", "call_description", "screenshot"]
Lang = Literal["hi", "en"]

ScamCategory = Literal[
    "digital_arrest",
    "kyc_account_block",
    "malicious_apk",
    "electricity_disconnect",
    "courier_parcel",
    "family_emergency",
    "task_job_offer",
    "investment_tips",
    "lottery_prize",
    "upi_collect_refund",
    "otp_phishing",
    "other_scam",
]
SafeCategory = Literal[
    "genuine_otp",
    "transaction_alert",
    "delivery_update",
    "legit_promo",
    "govt_genuine",
    "personal",
]
Category = Literal[ScamCategory, SafeCategory]

Reason = Literal[
    "lookalike_link",
    "apk_link",
    "threat_or_arrest",
    "urgency_deadline",
    "asks_payment",
    "asks_otp_or_pin",
    "upi_collect",
    "impersonates_authority",
    "unregistered_sender",
    "too_good_to_be_true",
    "secrecy_request",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- §8.1 Verdict (API response) ---


class RedFlag(_Strict):
    quote: str
    # Model flags use §10.1 reasons; rule flags carry §7.3 signal names (fusion adds rules.soft).
    reason: str
    source: Literal["model", "rule"]


class Verdict(_Strict):
    event_id: str
    verdict: VerdictLabel
    category: Category | None = None  # None when UNKNOWN (detector unavailable)
    p_scam: float | None = Field(default=None, ge=0.0, le=1.0)  # None in rules-only mode
    red_flags: list[RedFlag] = Field(default_factory=list)
    explanation: str
    language: Lang
    parent_id: str
    timings_ms: dict[str, int] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)


# --- §8.2 Detector output (JSON only, keys in this order) ---


class DetectorRedFlag(_Strict):
    quote: str
    reason: Reason


class DetectorOutput(_Strict):
    verdict: DetectorVerdict
    category: Category
    red_flags: list[DetectorRedFlag]


# --- §5.3 request bodies ---


class CheckRequest(_Strict):
    """POST /api/check."""

    parent_id: str
    text: str = Field(min_length=1)
    channel: Channel
    sender: str | None = None
    lang: Lang | None = None  # FR-8: default is the parent's profile language


class Feedback(_Strict):
    """POST /api/feedback/{event_id}."""

    correct: bool
    true_verdict: DetectorVerdict | None = None
    note: str | None = None


# --- §8.3 CheckState (LangGraph) ---


class CheckState(TypedDict, total=False):
    event_id: str
    parent_id: str
    channel: Literal["sms", "whatsapp", "call_description", "screenshot"]
    raw_text: str | None
    image: bytes | None
    audio: bytes | None
    text: str
    urls: list[str]
    phones: list[str]
    upi_handles: list[str]
    sender: str | None
    rule_signals: dict  # {"hard": [...], "soft": [...]}
    detector: dict | None  # parsed JSON
    p_scam: float | None
    verdict: str
    red_flags: list[dict]
    grounding_rate: float | None
    explanation: str
    language: str
    timings_ms: dict
    errors: list[str]
