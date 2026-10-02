"""LangGraph pipeline over CheckState (PRD §5.2, §8.3):

    normalize -> perceive (only for image/audio) -> rules -> detect -> fuse -> explain -> notify_store

Every node records its latency in `state["timings_ms"]` and runs in a tracing span with
metadata-only attributes (PRD FR-19, §9.3). Failures are recorded as exception *type names* in
`state["errors"]`; a failed perception or an unavailable detector never crashes a check and
never yields SAFE (PRD §13). Media bytes are dropped from the state after perception and are
only written to disk when KEEP_MEDIA=true.
"""

import asyncio
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from hub import alerts, db, detector, explainer, fusion, perception, tracing
from hub.normalize import normalize
from hub.rules import RuleSignals, evaluate
from hub.schemas import CheckState, Verdict
from hub.settings import Parent, get_parents, get_settings

log = logging.getLogger(__name__)

NODES = ("normalize", "perceive", "rules", "detect", "fuse", "explain", "notify_store")


class UnknownParent(ValueError):
    """`parent_id` is not in config/parents.json (PRD FR-7 -> HTTP 400)."""


class EmptyInput(ValueError):
    """Nothing to check: no text, image or audio (-> HTTP 400/422)."""


class GraphState(CheckState, total=False):
    """CheckState plus internal fields (not part of the API)."""

    image_mime: str | None
    audio_mime: str | None
    lang_override: str | None
    category: str | None
    perception_failed: bool
    normalized: Any  # hub.normalize.NormalizedInput
    rules: Any  # hub.rules.RuleSignals
    detector_result: Any  # hub.detector.DetectorResult | None
    result: Any  # hub.schemas.Verdict


def new_event_id() -> str:
    """Time-ordered, unique id: `evt_<ms since epoch, hex><random>`."""
    return f"evt_{int(time.time() * 1000):011x}{secrets.token_hex(4)}"


def _node(name: str, fallback: Callable[[GraphState], dict] | None = None):
    """Wrap an async node: span, latency, and error capture (types only, no text).

    If the node raises, `fallback(state)` supplies a safe update so later nodes still run
    (a crashed node can only make the verdict more cautious, never SAFE)."""

    def deco(fn: Callable[[GraphState], Awaitable[dict]]):
        async def run(state: GraphState) -> dict:
            timings = dict(state.get("timings_ms") or {})
            errors = list(state.get("errors") or [])
            t0 = time.perf_counter()
            with tracing.node_span(name) as span:
                try:
                    out = await fn(state)
                except Exception as e:
                    errors.append(f"{name}:{type(e).__name__}")
                    tracing.set_attrs(span, error_type=type(e).__name__)
                    out = fallback(state) if fallback else {}
                ms = int((time.perf_counter() - t0) * 1000)
                tracing.set_attrs(span, latency_ms=ms, **out.pop("_attrs", {}))
            timings[name] = ms
            out["timings_ms"] = timings
            out["errors"] = errors + out.get("errors", [])
            return out

        run.__name__ = name
        return run

    return deco


def _drop_media(state: GraphState, event_id: str) -> None:
    """KEEP_MEDIA=true: write the raw media next to the DB; otherwise nothing touches disk."""
    s = get_settings()
    if not s.keep_media:
        return
    folder = Path(s.db_path).parent / "media"
    folder.mkdir(parents=True, exist_ok=True)
    for key, ext in (("image", "img"), ("audio", "aud")):
        if state.get(key):
            (folder / f"{event_id}.{ext}").write_bytes(state[key])


@_node("normalize", lambda s: {"text": s.get("raw_text") or "", "normalized": normalize("")})
async def normalize_node(state: GraphState) -> dict:
    n = normalize(state.get("raw_text") or "", state.get("sender"))
    return {
        "normalized": n,
        "text": n.text,
        "urls": n.urls,
        "phones": n.phones,
        "upi_handles": n.upi_handles,
        "sender": n.sender,
    }


@_node("perceive")
async def perceive_node(state: GraphState) -> dict:
    _drop_media(state, state["event_id"])
    sender, transcript, channel = state.get("sender"), "", state["channel"]
    try:
        if state.get("image"):
            found, transcript = await perception.transcribe_image(
                state["image"], state.get("image_mime")
            )
            sender = sender or found
            channel = "screenshot"
        if state.get("audio"):
            spoken = await perception.transcribe_audio(state["audio"], state.get("audio_mime"))
            transcript = f"{transcript}\n{spoken}".strip()
            channel = "call_description"
        if not (transcript or "").strip():
            raise perception.PerceptionError("empty transcription")
    except Exception as e:
        log.warning("perception failed: %s", type(e).__name__)
        return {
            "image": None,
            "audio": None,
            "perception_failed": True,
            "errors": [f"perceive:{type(e).__name__}"],
        }
    combined = "\n".join(p for p in (state.get("raw_text"), transcript) if p)
    n = normalize(combined, sender)
    return {
        "image": None,  # media bytes dropped from state (FR-18)
        "audio": None,
        "channel": channel,
        "normalized": n,
        "raw_text": None,
        "text": n.text,
        "urls": n.urls,
        "phones": n.phones,
        "upi_handles": n.upi_handles,
        "sender": n.sender,
    }


@_node("rules", lambda s: {"rules": RuleSignals(), "rule_signals": {"hard": [], "soft": []}})
async def rules_node(state: GraphState) -> dict:
    if state.get("perception_failed"):
        return {"rules": RuleSignals(), "rule_signals": {"hard": [], "soft": []}}
    r = evaluate(state["normalized"])
    return {
        "rules": r,
        "rule_signals": r.as_state(),
        "_attrs": {"rule_signals": r.hard + r.soft},
    }


@_node(
    "detect",
    lambda s: {"detector": None, "detector_result": None, "p_scam": None, "grounding_rate": None},
)
async def detect_node(state: GraphState) -> dict:
    none = {"detector": None, "detector_result": None, "p_scam": None, "grounding_rate": None}
    if state.get("perception_failed"):
        return none
    sig = state["rule_signals"]
    try:
        res = await detector.detect(
            state["channel"], state.get("sender"), sig["hard"] + sig["soft"], state["text"]
        )
    except detector.DetectorUnavailable:
        return {**none, "errors": ["detect:DetectorUnavailable"]}
    if res is None:
        return {**none, "errors": ["detect:InvalidJson"]}
    return {
        "detector": res.output.model_dump(),
        "detector_result": res,
        "p_scam": res.p_scam,
        "grounding_rate": res.grounding_rate,
        "red_flags": res.red_flags,
        "_attrs": {
            "retries": res.retries,
            "model_version": get_settings().detector_version,
            "p_scam": res.p_scam,
        },
    }


@_node("fuse", lambda s: {"verdict": "UNKNOWN", "category": None, "red_flags": [], "p_scam": None})
async def fuse_node(state: GraphState) -> dict:
    if state.get("perception_failed"):
        return {"verdict": "UNKNOWN", "category": None, "red_flags": [], "p_scam": None}
    f = fusion.fuse(state["rules"], state.get("detector_result"), state["text"])
    return {
        "verdict": f.verdict,
        "category": f.category,
        "red_flags": f.red_flags,
        "p_scam": f.p_scam,
        "_attrs": {"verdict": f.verdict, "category": f.category or ""},
    }


def _language(state: GraphState, parent: Parent) -> str:
    return state.get("lang_override") or parent.language  # FR-8


def _explain_fallback(state: GraphState) -> dict:
    parent = get_parents()[state["parent_id"]]
    lang = state.get("lang_override") or parent.language
    text = explainer.template(
        state["verdict"], state.get("category"), lang, get_settings().son_name
    )
    return {"explanation": text, "language": lang}


@_node("explain", _explain_fallback)
async def explain_node(state: GraphState) -> dict:
    parent = get_parents()[state["parent_id"]]
    lang = _language(state, parent)
    text = await explainer.explain(
        {
            "verdict": state["verdict"],
            "category": state.get("category"),
            "red_flags": state.get("red_flags", []),
        },
        parent,
        lang,
        state.get("text", ""),
    )
    return {
        "explanation": text,
        "language": lang,
        "_attrs": {"model_version": get_settings().gemma_model},
    }


@_node("notify_store")
async def notify_store_node(state: GraphState) -> dict:
    s = get_settings()
    parent = get_parents()[state["parent_id"]]
    done = state.get("timings_ms") or {}
    timings = {n: done.get(n, 0) for n in NODES[:-1]}  # PRD §8.1 order; skipped nodes are 0
    timings["total"] = sum(v for k, v in timings.items() if k != "total")
    verdict = Verdict(
        event_id=state["event_id"],
        verdict=state["verdict"],
        category=state.get("category"),
        p_scam=state.get("p_scam"),
        red_flags=state.get("red_flags", []),
        explanation=state.get("explanation", ""),
        language=state.get("language") or parent.language,
        parent_id=state["parent_id"],
        timings_ms=timings,
        model_versions={"detector": s.detector_version, "gemma": s.gemma_model},
    )
    out: dict = {"result": verdict}
    try:
        await asyncio.to_thread(
            db.save_event,
            state["event_id"],
            state["parent_id"],
            state["channel"],
            state.get("text"),
            state.get("sender"),
        )
        await asyncio.to_thread(db.save_verdict, verdict.model_dump())
    except Exception as e:
        out["errors"] = [f"notify_store:db:{type(e).__name__}"]
    if verdict.verdict == "SCAM":
        try:
            await alerts.notify_scam(parent, verdict.category, datetime.now())
        except Exception as e:
            out["errors"] = [*out.get("errors", []), f"notify_store:ntfy:{type(e).__name__}"]
    return out


def _needs_perception(state: GraphState) -> str:
    return "perceive" if state.get("image") or state.get("audio") else "rules"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("normalize", normalize_node)
    g.add_node("perceive", perceive_node)
    g.add_node("rules", rules_node)
    g.add_node("detect", detect_node)
    g.add_node("fuse", fuse_node)
    g.add_node("explain", explain_node)
    g.add_node("notify_store", notify_store_node)
    g.add_edge(START, "normalize")
    g.add_conditional_edges("normalize", _needs_perception, ["perceive", "rules"])
    g.add_edge("perceive", "rules")
    g.add_edge("rules", "detect")
    g.add_edge("detect", "fuse")
    g.add_edge("fuse", "explain")
    g.add_edge("explain", "notify_store")
    g.add_edge("notify_store", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


async def run_check(
    *,
    parent_id: str,
    text: str | None = None,
    channel: str,
    sender: str | None = None,
    image: bytes | None = None,
    image_mime: str | None = None,
    audio: bytes | None = None,
    audio_mime: str | None = None,
    lang: str | None = None,
) -> Verdict:
    """Run one check through the graph. Used by the API and by eval's full_system adapter.

    Raises UnknownParent / EmptyInput for bad requests; everything else degrades to a
    verdict (UNKNOWN when the detector or perception is unavailable)."""
    if parent_id not in get_parents():
        raise UnknownParent(parent_id)
    if not ((text or "").strip() or image or audio):
        raise EmptyInput("no text, image or audio")
    state: GraphState = {
        "event_id": new_event_id(),
        "parent_id": parent_id,
        "channel": "call_description" if audio else ("screenshot" if image else channel),
        "raw_text": text,
        "sender": sender,
        "image": image,
        "image_mime": image_mime,
        "audio": audio,
        "audio_mime": audio_mime,
        "lang_override": lang,
        "timings_ms": {},
        "errors": [],
    }
    with tracing.check_transaction(parent_id, state["channel"]):
        final = await get_graph().ainvoke(state)
    return final["result"]
