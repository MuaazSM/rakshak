"""System adapters for the ablation table (PRD §12.3, Appendix A.4, §7.4 rules-only mode).

Every adapter turns one split item into an `eval.metrics.Pred`:

    rules_only         hub.rules + §7.4 rules-only mapping (hard → SCAM, ≥1 soft → SUSPICIOUS,
                       else UNKNOWN "couldn't check"; never SAFE)
    gemma_zeroshot     Ollama /api/chat, A.1 system + §8.2 user, format=json, think=false
    qwen_base_fewshot  llama-server (DETECTOR_URL, base Qwen3.5-4B GGUF) with A.1 + 6 few-shot
                       turns from eval/fewshot.jsonl, thinking off
    tuned_tinker       Tinker sampler for a sampler checkpoint (greedy, renderer
                       qwen3_5_disable_thinking); also used per epoch by train_tinker
    tuned_detector     llama-server via hub.detector.detect (retry + logprob p_scam)
    full_system        hub.graph.run_check in-process (parent "mom"; ntfy off, temp DB)

All model inputs are the split file's own §8.2 user message, so eval sees exactly what training
saw. Nothing here logs message text; errors are reported by exception type only.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import get_args

import httpx

from eval.metrics import LABELS, Gold, Pred, is_hard_negative
from hub.schemas import Category, DetectorOutput
from hub.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]
FEWSHOT_PATH = ROOT / "eval" / "fewshot.jsonl"
TINKER_RENDERER = "qwen3_5_disable_thinking"
TIMEOUT_S = 180.0
MAX_TOKENS = 512
SEED = 7

_USER_RE = re.compile(
    r"\ACHANNEL: (?P<channel>[^\n]*)\n"
    r"SENDER: (?P<sender>[^\n]*) \((?P<status>registered|unregistered|unknown)\)\n"
    r"RULE_SIGNALS: \[(?P<signals>[^\n]*)\]\n"
    r"MESSAGE:\n(?P<text>.*)\Z",
    re.DOTALL,
)
_CATEGORIES = frozenset(get_args(Category))


# --- split items ---


@dataclass
class Item:
    id: str
    messages: list[dict]  # [system A.1, user §8.2, assistant target]
    channel: str
    sender: str | None
    rule_signals: list[str]
    text: str
    gold: DetectorOutput
    meta: dict = field(default_factory=dict)

    @property
    def prompt_messages(self) -> list[dict]:
        """System + user, without the gold answer."""
        return [m for m in self.messages if m["role"] != "assistant"]


def parse_user_message(content: str) -> tuple[str, str | None, list[str], str]:
    """Invert `hub.detector.user_message`: (channel, sender, rule_signals, text)."""
    m = _USER_RE.match(content)
    if not m:
        raise ValueError("user message is not in §8.2 format")
    sender = m["sender"].strip()
    signals = [s.strip() for s in m["signals"].split(",") if s.strip()]
    return m["channel"], (None if sender == "unknown" else sender), signals, m["text"]


def item_from_row(row: dict) -> Item:
    msgs = row["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    target = next(m["content"] for m in msgs if m["role"] == "assistant")
    channel, sender, signals, text = parse_user_message(user)
    meta = row.get("meta", {})
    return Item(
        id=str(meta.get("id", "")),
        messages=msgs,
        channel=channel,
        sender=sender,
        rule_signals=signals,
        text=text,
        gold=DetectorOutput.model_validate_json(target),
        meta=meta,
    )


def load_items(path: Path, limit: int | None = None) -> list[Item]:
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    items = [item_from_row(r) for r in rows]
    return items[:limit] if limit else items


def strip_sender(item: Item) -> Item:
    """The item as if it arrived without a sender (runtime text shares): SENDER renders as
    `unknown (unknown)` and RULE_SIGNALS are recomputed with sender None (PRD §8.2, §7.3)."""
    from hub.detector import user_message
    from hub.rules import rule_signals

    signals = rule_signals(item.text, None, item.channel)
    user = user_message(item.channel, None, signals, item.text)
    messages = [{**m, "content": user} if m["role"] == "user" else m for m in item.messages]
    return replace(item, messages=messages, sender=None, rule_signals=signals)


def gold_of(item: Item) -> Gold:
    g = item.gold
    return Gold(
        verdict=g.verdict,
        category=g.category,
        quotes=tuple(f.quote for f in g.red_flags),
        hard_negative=is_hard_negative(item.meta, g.verdict, g.category),
        text=item.text,
    )


# --- reply parsing (lenient for baselines; validity is reported separately) ---


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        s = s.removeprefix("json").strip()
    return s


def _first_object(s: str) -> dict | None:
    try:
        data = json.loads(s)
    except ValueError:
        start, end = s.find("{"), s.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(s[start : end + 1])
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


def parse_reply(raw: str) -> Pred:
    """Score a model reply. `json_valid` = the reply as-is is strict §8.2 JSON; the verdict,
    category and quotes are still read leniently (fences, extra text, off-vocabulary values)
    so a base model's free-text category doesn't hide a correct verdict."""
    try:
        DetectorOutput.model_validate_json(raw.strip())
        valid = True
    except ValueError:
        valid = False
    data = _first_object(_strip_fences(raw))
    if data is None:
        return Pred(verdict="UNKNOWN", json_valid=False, json_parsed=False)
    verdict = str(data.get("verdict", "")).strip().upper()
    category = data.get("category")
    flags = data.get("red_flags") if isinstance(data.get("red_flags"), list) else []
    quotes = [str(f["quote"]) for f in flags if isinstance(f, dict) and f.get("quote")]
    return Pred(
        verdict=verdict if verdict in LABELS else "UNKNOWN",
        category=category if category in _CATEGORIES else None,
        quotes=quotes,
        json_valid=valid,
        json_parsed=True,
    )


# --- adapters ---


class System:
    name = "system"
    concurrency = 2

    async def predict(self, item: Item) -> Pred:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None

    def check_items(self, items: list["Item"]) -> None:
        """Raise ValueError if this system must not be evaluated on these items."""
        return None

    async def timed(self, item: Item) -> Pred:
        t0 = time.perf_counter()
        try:
            pred = await self.predict(item)
        except (httpx.HTTPError, TimeoutError, ConnectionError) as e:
            pred = Pred(verdict="UNKNOWN", error=type(e).__name__)
        pred.latency_ms = (time.perf_counter() - t0) * 1000
        return pred


class RulesOnly(System):
    """§7.4 rules-only mode. Category is not predicted (None)."""

    name = "rules_only"
    concurrency = 8

    async def predict(self, item: Item) -> Pred:
        from hub.normalize import normalize
        from hub.rules import evaluate

        r = evaluate(normalize(item.text, item.sender))
        if r.hard:
            verdict, names = "SCAM", r.hard
        elif r.soft:
            verdict, names = "SUSPICIOUS", r.soft
        else:
            verdict, names = "UNKNOWN", []
        quotes = [q for n in names for q in r.evidence.get(n, [])]
        return Pred(verdict=verdict, quotes=quotes)


class GemmaZeroShot(System):
    """Appendix A.4: A.1 system prompt, same user format, no examples (Ollama)."""

    name = "gemma_zeroshot"

    def __init__(self, client: httpx.AsyncClient | None = None):
        s = get_settings()
        self.url = f"{s.ollama_host.rstrip('/')}/api/chat"
        self.model = s.gemma_model
        self.client = client or httpx.AsyncClient(timeout=TIMEOUT_S)

    async def predict(self, item: Item) -> Pred:
        body = {
            "model": self.model,
            "messages": item.prompt_messages,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0, "seed": SEED, "num_predict": MAX_TOKENS},
            "keep_alive": "30m",
        }
        r = await self.client.post(self.url, json=body)
        r.raise_for_status()
        return parse_reply((r.json().get("message") or {}).get("content") or "")

    async def aclose(self) -> None:
        await self.client.aclose()


async def llama_chat(
    client: httpx.AsyncClient, messages: list[dict], max_tokens: int = MAX_TOKENS
) -> str:
    """One OpenAI-compatible chat completion on llama-server, thinking off (PRD §7.1)."""
    body = {
        "model": "detector",
        "messages": messages,
        "temperature": 0,
        "seed": SEED,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{get_settings().detector_url.rstrip('/')}/chat/completions"
    r = await client.post(url, json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


def load_fewshot(path: Path = FEWSHOT_PATH) -> list[dict]:
    """Few-shot user/assistant turns (A.4), in file order."""
    turns: list[dict] = []
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            turns += [m for m in json.loads(line)["messages"] if m["role"] != "system"]
    if len(turns) != 12:
        raise ValueError(f"{path}: expected 6 examples (12 turns), got {len(turns)} turns")
    return turns


def check_fewshot_disjoint(path: Path, items: list[Item]) -> None:
    """Refuse when a few-shot example (by id or seed group) is in the evaluated split, or when
    the few-shot file predates seed-group recording (rebuild with --build-fewshot)."""
    shots = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]
    if any(not s.get("seed_group") for s in shots):
        raise ValueError(f"{path}: few-shot lines lack seed_group; rebuild from train")
    ids = {it.id for it in items}
    groups = {it.meta.get("seed_group") for it in items}
    leaked = sorted(s["id"] for s in shots if s["id"] in ids or s["seed_group"] in groups)
    if leaked:
        raise ValueError(f"few-shot examples overlap the evaluated split: {leaked}")


class QwenBaseFewShot(System):
    """Appendix A.4: A.1 + 6 fixed train examples, base Qwen3.5-4B on llama-server."""

    name = "qwen_base_fewshot"

    def __init__(self, client: httpx.AsyncClient | None = None, fewshot_path: Path = FEWSHOT_PATH):
        self.fewshot_path = fewshot_path
        self.shots = load_fewshot(fewshot_path)
        self.client = client or httpx.AsyncClient(timeout=TIMEOUT_S)

    def check_items(self, items: list[Item]) -> None:
        check_fewshot_disjoint(self.fewshot_path, items)

    async def predict(self, item: Item) -> Pred:
        system, user = item.prompt_messages
        return parse_reply(await llama_chat(self.client, [system, *self.shots, user]))

    async def aclose(self) -> None:
        await self.client.aclose()


class TunedDetector(System):
    """The served tuned GGUF through the hub's own client (retry + logprob p_scam)."""

    name = "tuned_detector"

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=TIMEOUT_S)

    async def predict(self, item: Item) -> Pred:
        from hub.detector import DetectorUnavailable, detect

        try:
            res = await detect(
                item.channel, item.sender, item.rule_signals, item.text, client=self.client
            )
        except DetectorUnavailable as e:
            return Pred(verdict="UNKNOWN", error=f"DetectorUnavailable:{e}")
        if res is None:  # invalid JSON twice → rules-only in the hub
            return Pred(verdict="UNKNOWN", json_valid=False, json_parsed=False)
        out = res.output
        return Pred(
            verdict=out.verdict,
            category=out.category,
            quotes=[f.quote for f in out.red_flags],
            json_valid=res.retries == 0,
            json_parsed=True,
            p_scam=res.p_scam,
        )

    async def aclose(self) -> None:
        await self.client.aclose()


class TunedTinker(System):
    """Greedy sampling from a Tinker sampler checkpoint (or base model), thinking off."""

    name = "tuned_tinker"
    concurrency = 8

    def __init__(
        self,
        model_path: str | None = None,
        *,
        sampling_client=None,
        base_model: str | None = None,
    ):
        import tinker
        from tinker_cookbook import renderers

        if sampling_client is None:
            s = get_settings()
            key = s.tinker_api_key.get_secret_value() if s.tinker_api_key else None
            service = tinker.ServiceClient(api_key=key)
            sampling_client = service.create_sampling_client(
                model_path=model_path, base_model=None if model_path else base_model
            )
        self.sampler = sampling_client
        self.renderer = renderers.get_renderer(TINKER_RENDERER, sampling_client.get_tokenizer())
        self.params = tinker.SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            seed=SEED,
            stop=self.renderer.get_stop_sequences(),
        )
        self.prompt_tokens = 0
        self.sampled_tokens = 0

    async def predict(self, item: Item) -> Pred:
        from tinker_cookbook import renderers

        prompt = self.renderer.build_generation_prompt(item.prompt_messages)
        resp = await self.sampler.sample_async(
            prompt=prompt, num_samples=1, sampling_params=self.params
        )
        toks = resp.sequences[0].tokens
        self.prompt_tokens += prompt.length
        self.sampled_tokens += len(toks)
        msg, _ = self.renderer.parse_response(toks)
        return parse_reply(renderers.get_text_content(msg))

    async def timed(self, item: Item) -> Pred:
        t0 = time.perf_counter()
        try:
            pred = await self.predict(item)
        except Exception as e:  # Tinker raises its own exception types
            pred = Pred(verdict="UNKNOWN", error=type(e).__name__)
        pred.latency_ms = (time.perf_counter() - t0) * 1000
        return pred


class FullSystem(System):
    """The whole hub graph in-process (hub.graph.run_check) as parent "mom".

    Eval must never alert anyone or touch the real DB: while this adapter is open, the cached
    settings get `ntfy_topic=None` and a throwaway `db_path`, and `hub.alerts.notify_scam` is
    replaced by a no-op. `aclose` restores all three. Red flags are the post-filter flags the
    parent would see (model + rule), so grounding is ~1 by construction.
    """

    name = "full_system"
    parent_id = "mom"

    def __init__(self):
        import tempfile

        from hub import alerts, graph
        from hub.settings import get_parents

        if self.parent_id not in get_parents():
            raise NotImplementedError(f"full_system: parent {self.parent_id!r} not configured")
        self._run_check = graph.run_check
        self._alerts = alerts
        self._orig_notify = alerts.notify_scam
        self._settings = get_settings()
        self._orig = {k: getattr(self._settings, k) for k in ("ntfy_topic", "db_path")}
        self._tmp = tempfile.TemporaryDirectory(prefix="rakshak-eval-")

        async def _no_alert(*_a, **_k) -> bool:
            return False

        alerts.notify_scam = _no_alert
        self._settings.ntfy_topic = None
        self._settings.db_path = Path(self._tmp.name) / "eval.db"
        from hub import db

        db.init_db()

    async def predict(self, item: Item) -> Pred:
        v = await self._run_check(
            parent_id=self.parent_id, text=item.text, channel=item.channel, sender=item.sender
        )
        return Pred(
            verdict=v.verdict,
            category=v.category,
            quotes=[f.quote for f in v.red_flags],
            p_scam=v.p_scam,
        )

    async def aclose(self) -> None:
        self._alerts.notify_scam = self._orig_notify
        for k, val in self._orig.items():
            setattr(self._settings, k, val)
        self._tmp.cleanup()


def served_models(client: httpx.Client | None = None) -> list[str]:
    """Model ids llama-server at DETECTOR_URL reports ([] if unreachable)."""
    url = f"{get_settings().detector_url.rstrip('/')}/models"
    try:
        r = (client or httpx).get(url, timeout=5)
        r.raise_for_status()
        return [str(m.get("id", "")) for m in r.json().get("data", [])]
    except (httpx.HTTPError, ValueError, AttributeError):
        return []


def served_detector_is_tuned(client: httpx.Client | None = None) -> bool:
    """True if llama-server at DETECTOR_URL serves a rakshak-detector GGUF (not the base)."""
    return any("rakshak-detector" in i for i in served_models(client))


SYSTEMS = {
    "rules_only": RulesOnly,
    "gemma_zeroshot": GemmaZeroShot,
    "qwen_base_fewshot": QwenBaseFewShot,
    "tuned_tinker": TunedTinker,
    "tuned_detector": TunedDetector,
    "full_system": FullSystem,
}


async def run_system(system: System, items: list[Item]) -> list[Pred]:
    """Predict every item with bounded concurrency; output order = input order."""
    sem = asyncio.Semaphore(system.concurrency)

    async def one(item: Item) -> Pred:
        async with sem:
            return await system.timed(item)

    return list(await asyncio.gather(*(one(it) for it in items)))
