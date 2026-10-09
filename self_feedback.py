"""
self_feedback.py — Self-feedback 3-turn pipeline for BioASQ yes/no questions

Turn 1 (Generate) : Initial yes/no answer with step-by-step reasoning
Turn 2 (Critique) : Self-review of own reasoning
Turn 3 (Finalize) : Final yes/no answer incorporating the critique

Features (fully aligned with main.py):
  - async / aiohttp with Semaphore concurrency control
  - Exponential backoff retry (identical to main.py)
  - latency_ms, finish_reason, native_finish_reason, thinking_text per turn
  - Claude extended thinking support (thinkingLevel)
  - Full wandb integration: config, prompts, questions table, per-doc metrics, overall metrics
  - Self-feedback-specific wandb extras: initial_answer, changed flag, critique text
  - PredRecord + FBRecord dataclasses
  - Identical compute_metrics / WandbLogger as main.py
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None  # type: ignore

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

YesNo = Literal["yes", "no"]
GenerationStatus = Literal["ok", "api_error", "timeout", "parse_error"]

# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = ("""You are a critical biomedical reviewer. Answer Yes/No questions only when the provided snippets contain direct and unambiguous evidence. If snippets conflict with each other, are only tangentially related to the question, or do not clearly support a yes answer, answer no. Your response must consist of exactly and only the single word "yes" or "no". Do not include any reasoning, explanation, punctuation, or extra characters.
""")

GENERATE_TEMPLATE = """### INSTRUCTIONS:
1. Read the question and snippets carefully.
2. Write down your step-by-step reasoning on how the snippets answer the question.
3. Do not use outside knowledge.
4. End your response with a new line containing exactly and only the word: yes or no

### QUESTION:
{q_body}

### SNIPPETS:
{snippets_block}

### REASONING AND INITIAL ANSWER:
"""

CRITIQUE_TEMPLATE = """Review your reasoning above. Check for:
1. Did you misread any snippet?
2. Are there snippets that CONTRADICT your answer?
3. Is the evidence clearly about the EXACT question asked, or only related?
4. Did any irrelevant snippet bias your reasoning?

Write a brief critique of your own reasoning. Identify any weaknesses."""

FINALIZE_TEMPLATE = """Based on your reasoning and self-critique, provide your FINAL answer.
Respond with a new line containing exactly and only the word: yes or no

### FINAL REASONING AND ANSWER:
"""

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExpConfig:
    # ── Identity ──────────────────────────────────────────────────────────
    wandb_run_name: Optional[str] = "SF-4"

    # ── Model (per-turn; set model_critique / model_finalize to override) ──
    model: str = "anthropic/claude-sonnet-4.6"          # Turn 1 default (also fallback)
    model_critique: Optional[str] = "anthropic/claude-sonnet-4.6"            # Turn 2 — None = same as model
    model_finalize: Optional[str] = "anthropic/claude-sonnet-4.6"            # Turn 3 — None = same as model

    # ── Sampling ──────────────────────────────────────────────────────────
    temperature: float = 0.0
    topP: Optional[float] = None
    topK: Optional[int] = None
    presencePenalty: Optional[float] = None
    frequencyPenalty: Optional[float] = None
    # Extended thinking: e.g. "low" | "medium" | "high" | None
    thinkingLevel: Optional[str] = None
    maxOutputTokens: int = 8192

    # ── Data paths ────────────────────────────────────────────────────────
    doc_ids: List[str] = field(default_factory=lambda: ["12b_01", "12b_02", "12b_03", "12b_04"])
    test_dir: Path = Path("data/testData")
    test_results_dir: Path = Path("data/testResults")
    gold_path: Path = Path("data/predictions/training13b.json")

    # ── API ───────────────────────────────────────────────────────────────
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    request_timeout_s: float = 600.0   # 3 turns — needs longer timeout
    concurrency: int = 10

    # ── Retry / backoff ───────────────────────────────────────────────────
    max_retries: int = 2
    backoff_base_s: float = 0.8
    backoff_cap_s: float = 20.0

    # ── Misc ──────────────────────────────────────────────────────────────
    max_chars_per_snippet: Optional[int] = None
    missing_as: YesNo = "no"

    # ── Prompt overrides (optional file paths) ────────────────────────────
    system_prompt_path: Optional[Path] = None
    generate_template_path: Optional[Path] = None
    critique_template_path: Optional[Path] = None
    finalize_template_path: Optional[Path] = None

    # ── W&B ───────────────────────────────────────────────────────────────
    use_wandb: bool = True
    wandb_project: str = "yesno"
    wandb_entity: Optional[str] = "bioasq"
    wandb_group: Optional[str] = "CSHS"
    wandb_mode: Optional[str] = "online"
    wandb_table_max_chars: int = 100_000


# ---------------------------------------------------------------------------
# Prompt spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptSpec:
    system: str = DEFAULT_SYSTEM_PROMPT
    generate_template: str = GENERATE_TEMPLATE
    critique_template: str = CRITIQUE_TEMPLATE
    finalize_template: str = FINALIZE_TEMPLATE


def load_prompt_spec(cfg: ExpConfig) -> PromptSpec:
    system = DEFAULT_SYSTEM_PROMPT
    gen_t = GENERATE_TEMPLATE
    crit_t = CRITIQUE_TEMPLATE
    fin_t = FINALIZE_TEMPLATE

    if cfg.system_prompt_path and cfg.system_prompt_path.exists():
        system = cfg.system_prompt_path.read_text(encoding="utf-8").strip()
    if cfg.generate_template_path and cfg.generate_template_path.exists():
        gen_t = cfg.generate_template_path.read_text(encoding="utf-8")
    if cfg.critique_template_path and cfg.critique_template_path.exists():
        crit_t = cfg.critique_template_path.read_text(encoding="utf-8")
    if cfg.finalize_template_path and cfg.finalize_template_path.exists():
        fin_t = cfg.finalize_template_path.read_text(encoding="utf-8")

    return PromptSpec(
        system=system,
        generate_template=gen_t,
        critique_template=crit_t,
        finalize_template=fin_t,
    )


# ---------------------------------------------------------------------------
# Dataclasses: metrics + records
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    n: int
    missing_pred: int
    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float
    precision_yes: float
    recall_yes: float
    f1_yes: float
    precision_no: float
    recall_no: float
    f1_no: float
    maF1: float


@dataclass
class TurnRecord:
    """Metadata for a single API call within the 3-turn pipeline."""
    content: str
    thinking_text: str
    finish_reason: str
    native_finish_reason: str
    latency_ms: int
    retry_count: int
    status: GenerationStatus
    error_text: str


@dataclass
class FBRecord:
    """Complete record for one yes/no question processed by self-feedback."""
    qid: str
    qtype: str
    body: str
    snippet_count: int

    # Predictions
    initial_answer: Optional[YesNo]
    final_answer: YesNo
    changed: bool                    # did critique flip the answer?

    # Per-turn data
    turn1: TurnRecord                # Generate
    turn2: TurnRecord                # Critique
    turn3: TurnRecord                # Finalize

    # Final turn summary (mirrors PredRecord for wandb table compat.)
    generation_status: GenerationStatus
    latency_ms: int                  # total wall-clock ms across all 3 turns
    retry_count: int                 # total retries across all 3 turns

    # Config snapshot
    model: str                   # Turn 1 model
    model_critique: Optional[str]  # Turn 2 model (None = same as model)
    model_finalize: Optional[str]  # Turn 3 model (None = same as model)
    temperature: float
    topP: Optional[float]
    topK: Optional[int]
    maxOutputTokens: int
    presencePenalty: Optional[float]
    frequencyPenalty: Optional[float]
    thinkingLevel: Optional[str]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_qtype(qtype: str) -> str:
    qt = (qtype or "").strip().lower()
    if qt in {"yes/no", "yesno", "yes-no", "yn"}:
        return "yesno"
    return qt


def coerce_yesno(val: Any) -> Optional[YesNo]:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().lower().strip(" .,:;!\"'")
        if s == "yes":
            return "yes"
        if s == "no":
            return "no"
        return None
    if isinstance(val, list) and val:
        first = val[0]
        if isinstance(first, list) and first:
            first = first[0]
        return coerce_yesno(first)
    return None


def _coerce_yesno_from_text(text: str) -> Optional[YesNo]:
    if not text:
        return None
    lines = [line.strip().lower() for line in text.strip().splitlines() if line.strip()]
    if lines:
        last = lines[-1].strip(" .,:;!\"'")
        if last == "yes":
            return "yes"
        if last == "no":
            return "no"
    matches = re.findall(r"\b(yes|no)\b", text.lower())
    return matches[-1] if matches else None  # type: ignore[return-value]


def load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}") from e


def load_questions(path: Path) -> List[dict]:
    obj = load_json_file(path)
    if not isinstance(obj, dict):
        return []
    qs = obj.get("questions")
    if not isinstance(qs, list):
        return []
    return [q for q in qs if isinstance(q, dict)]


def safe_json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def derive_doc_id_from_phase_filename(path: Path) -> Optional[str]:
    m = re.search(r"(?:phaseB_|PhaseB_)?(\d{1,2}b_\d{2})", path.stem)
    return m.group(1) if m else None


def derive_doc_id(path: Path) -> str:
    return derive_doc_id_from_phase_filename(path) or path.stem


def resolve_test_paths(test_dir: Path, doc_ids: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for did in doc_ids:
        p = test_dir / f"phaseB_{did}.json"
        if p.exists():
            paths.append(p)
        else:
            print(f"[WARN] Missing test file: {p}", file=sys.stderr)
    return paths


def normalize_ideal_answer_for_submission(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""
    return str(value).strip()


def collect_snippet_texts(q: dict) -> List[str]:
    texts: List[str] = []
    for s in q.get("snippets", []):
        t = s.get("text") if isinstance(s, dict) else None
        if t is not None:
            texts.append(t)
    return texts


def build_snippets_block(snippets: List[str], max_chars: Optional[int]) -> str:
    parts: List[str] = []
    for i, txt in enumerate(snippets, 1):
        if max_chars and len(txt) > max_chars:
            txt = txt[:max_chars].rstrip() + "…"
        parts.append(f"[{i}]: {txt}")
    return "\n".join(parts) if parts else "(No snippets provided.)"


def ids_in_test_file(test_path: Path) -> List[str]:
    ids: List[str] = []
    for q in load_questions(test_path):
        if normalize_qtype(str(q.get("type", ""))) != "yesno":
            continue
        qid = str(q.get("id", "")).strip()
        if qid:
            ids.append(qid)
    return ids


def map_yesno_labels(questions: Sequence[dict]) -> Dict[str, YesNo]:
    out: Dict[str, YesNo] = {}
    for q in questions:
        if normalize_qtype(str(q.get("type", ""))) != "yesno":
            continue
        qid = str(q.get("id", "")).strip()
        if not qid:
            continue
        label = coerce_yesno(q.get("exact_answer"))
        if label is not None:
            out[qid] = label
    return out


def map_gold_labels(gold_path: Path) -> Dict[str, YesNo]:
    return map_yesno_labels(load_questions(gold_path))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2.0 * p * r, p + r)


def compute_metrics(
    gold: Dict[str, YesNo],
    pred: Dict[str, YesNo],
    *,
    missing_as: YesNo = "no",
) -> Metrics:
    tp = fp = fn = tn = missing = 0
    for qid, g in gold.items():
        p = pred.get(qid)
        if p is None:
            missing += 1
            p = missing_as
        if   g == "yes" and p == "yes": tp += 1
        elif g == "no"  and p == "yes": fp += 1
        elif g == "yes" and p == "no":  fn += 1
        else:                           tn += 1

    n = len(gold)
    acc   = _safe_div(tp + tn, n)
    p_yes = _safe_div(tp, tp + fp)
    r_yes = _safe_div(tp, tp + fn)
    f1y   = _f1(p_yes, r_yes)
    p_no  = _safe_div(tn, tn + fn)
    r_no  = _safe_div(tn, tn + fp)
    f1n   = _f1(p_no, r_no)

    return Metrics(
        n=n, missing_pred=missing,
        tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=acc,
        precision_yes=p_yes, recall_yes=r_yes, f1_yes=f1y,
        precision_no=p_no,  recall_no=r_no,  f1_no=f1n,
        maF1=0.5 * (f1y + f1n),
    )


def metrics_to_dict(m: Metrics) -> Dict[str, Any]:
    data = asdict(m)
    return {k: float(f"{v:.6f}") if isinstance(v, float) else v for k, v in data.items()}


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def print_metrics(title: str, m: Metrics) -> None:
    print(f"\n=== {title} ===")
    print(f"N={m.n} | missing_pred={m.missing_pred}")
    print(f"Confusion (YES positive): TP={m.tp} FP={m.fp} FN={m.fn} TN={m.tn}")
    print(f"accuracy={_fmt(m.accuracy)}  maF1={_fmt(m.maF1)}")
    print(f"YES: P={_fmt(m.precision_yes)} R={_fmt(m.recall_yes)} F1={_fmt(m.f1_yes)}")
    print(f"NO : P={_fmt(m.precision_no)}  R={_fmt(m.recall_no)}  F1={_fmt(m.f1_no)}")


def run_evaluation(
    *,
    cfg: ExpConfig,
    test_paths: List[Path],
    records: List[FBRecord],
) -> Tuple[Dict[str, Metrics], Metrics]:
    gold = map_gold_labels(cfg.gold_path)
    if not gold:
        raise RuntimeError(f"No gold yes/no labels found in: {cfg.gold_path}")

    pred_map: Dict[str, YesNo] = {r.qid: r.final_answer for r in records if r.qid}

    per_doc: Dict[str, Metrics] = {}
    union_ids: Set[str] = set()
    for tp in test_paths:
        did = derive_doc_id(tp)
        ids_set = {qid for qid in ids_in_test_file(tp) if qid in gold}
        union_ids |= ids_set
        per_doc[did] = compute_metrics(
            {qid: gold[qid] for qid in ids_set},
            {qid: pred_map[qid] for qid in ids_set if qid in pred_map},
            missing_as=cfg.missing_as,
        )

    overall = compute_metrics(
        {qid: gold[qid] for qid in union_ids},
        {qid: pred_map[qid] for qid in union_ids if qid in pred_map},
        missing_as=cfg.missing_as,
    )
    return per_doc, overall


# ---------------------------------------------------------------------------
# HTTP client (identical to main.py OpenRouterClient)
# ---------------------------------------------------------------------------

class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        http_referer: Optional[str] = None,
        app_title: Optional[str] = None,
        timeout_s: float = 90.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.http_referer = http_referer
        self.app_title = app_title
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "OpenRouterClient":
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, *_) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _headers(self) -> Dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.http_referer:
            h["HTTP-Referer"] = self.http_referer
        if self.app_title:
            h["X-Title"] = self.app_title
        return h

    async def chat_completions(self, *, payload: dict) -> dict:
        if self._session is None:
            raise RuntimeError("Client not started. Use 'async with OpenRouterClient(...)'.")
        url = f"{self.base_url}/chat/completions"
        async with self._session.post(url, headers=self._headers(), json=payload) as resp:
            raw_text = await resp.text()
            if resp.status >= 400:
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=raw_text[:5000],
                    headers=resp.headers,
                )
            try:
                return json.loads(raw_text)
            except Exception as e:
                raise RuntimeError(f"Non-JSON response: {raw_text[:2000]}") from e


# ---------------------------------------------------------------------------
# Payload builder — supports extended thinking (identical to main.py)
# ---------------------------------------------------------------------------

def build_payload(cfg: ExpConfig, messages: List[dict], *, model_override: Optional[str] = None) -> dict:
    payload: Dict[str, Any] = {
        "model": model_override or cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": int(cfg.maxOutputTokens),
        "tool_choice": "none",
    }
    if cfg.topP is not None:
        payload["top_p"] = cfg.topP
    if cfg.topK is not None:
        payload["top_k"] = cfg.topK
    if cfg.presencePenalty is not None:
        payload["presence_penalty"] = cfg.presencePenalty
    if cfg.frequencyPenalty is not None:
        payload["frequency_penalty"] = cfg.frequencyPenalty
    if cfg.thinkingLevel is not None:
        payload["thinkingLevel"] = cfg.thinkingLevel
    return payload


# ---------------------------------------------------------------------------
# Response parsing — identical to main.py extract_content_and_thinking
# ---------------------------------------------------------------------------

def extract_content_and_thinking(resp_json: dict) -> Tuple[str, str, str, str]:
    choices = resp_json.get("choices") or []
    if not choices:
        return "", "", "", ""
    c0 = choices[0] or {}
    finish_reason = str(c0.get("finish_reason") or "")
    native_finish_reason = str(c0.get("native_finish_reason") or "")
    msg = c0.get("message") if isinstance(c0.get("message"), dict) else {}
    msg = msg or {}
    content = str(msg.get("content") or "")
    thinking = str(msg.get("reasoning") or "")
    if not thinking:
        reasoning_details = msg.get("reasoning_details")
        if isinstance(reasoning_details, list):
            parts: List[str] = []
            for item in reasoning_details:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            thinking = "\n\n".join(parts)
    return content, thinking, finish_reason, native_finish_reason


# ---------------------------------------------------------------------------
# Retry logic — identical to main.py
# ---------------------------------------------------------------------------

async def retry_chat(
    client: OpenRouterClient,
    *,
    payload: dict,
    max_retries: int,
    backoff_base_s: float,
    backoff_cap_s: float,
) -> Tuple[dict, int, int, GenerationStatus, str]:
    start_all = time.perf_counter()
    last_err = ""
    last_status: GenerationStatus = "api_error"

    for attempt in range(max_retries + 1):
        try:
            t0 = time.perf_counter()
            resp_json = await client.chat_completions(payload=payload)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return resp_json, attempt, latency_ms, "ok", ""
        except asyncio.TimeoutError:
            last_status = "timeout"
            last_err = "TIMEOUT"
        except aiohttp.ClientResponseError as e:
            last_status = "api_error"
            last_err = f"HTTP_ERROR status={e.status} msg={str(e)[:2000]}"
        except aiohttp.ClientError as e:
            last_status = "api_error"
            last_err = f"CLIENT_ERROR {repr(e)[:2000]}"
        except Exception as e:
            last_status = "api_error"
            last_err = f"UNEXPECTED_ERROR {repr(e)[:2000]}"

        if attempt >= max_retries:
            break
        sleep_s = min(backoff_cap_s, backoff_base_s * (2 ** attempt)) * (0.8 + 0.4 * random.random())
        await asyncio.sleep(sleep_s)

    latency_ms = int((time.perf_counter() - start_all) * 1000)
    return {}, max_retries, latency_ms, last_status, last_err


# ---------------------------------------------------------------------------
# Self-feedback pipeline per question
# ---------------------------------------------------------------------------

def _empty_turn(status: GenerationStatus = "api_error", err: str = "") -> TurnRecord:
    return TurnRecord(
        content="", thinking_text="", finish_reason="", native_finish_reason="",
        latency_ms=0, retry_count=0, status=status, error_text=err,
    )


async def process_question(
    client: OpenRouterClient,
    cfg: ExpConfig,
    prompt_spec: PromptSpec,
    q: dict,
    sem: asyncio.Semaphore,
) -> FBRecord:
    qid = str(q.get("id", "")).strip()
    qtype = normalize_qtype(str(q.get("type", "")))
    body = str(q.get("body", "")).strip()
    snippets = collect_snippet_texts(q)
    block = build_snippets_block(snippets, cfg.max_chars_per_snippet)

    total_latency = 0
    total_retries = 0
    overall_status: GenerationStatus = "ok"

    # Resolve per-turn models (fall back to cfg.model when not set)
    model_t1 = cfg.model
    model_t2 = cfg.model_critique or cfg.model
    model_t3 = cfg.model_finalize or cfg.model

    # Conversation state — grows across turns
    conv: List[dict] = [{"role": "system", "content": prompt_spec.system}]

    async def call_turn(user_msg: str, *, model_override: Optional[str] = None) -> TurnRecord:
        nonlocal total_latency, total_retries, overall_status
        conv.append({"role": "user", "content": user_msg})
        payload = build_payload(cfg, conv, model_override=model_override)
        async with sem:
            resp_json, retry_count, latency_ms, status, err_text = await retry_chat(
                client,
                payload=payload,
                max_retries=cfg.max_retries,
                backoff_base_s=cfg.backoff_base_s,
                backoff_cap_s=cfg.backoff_cap_s,
            )
        total_latency += latency_ms
        total_retries += retry_count
        if status != "ok":
            overall_status = status

        content, thinking, finish_reason, native_fr = extract_content_and_thinking(resp_json)
        conv.append({"role": "assistant", "content": content})

        return TurnRecord(
            content=content,
            thinking_text=thinking,
            finish_reason=finish_reason,
            native_finish_reason=native_fr,
            latency_ms=latency_ms,
            retry_count=retry_count,
            status=status,
            error_text=err_text,
        )

    # Turn 1 — Generate
    t1 = await call_turn(
        prompt_spec.generate_template.format(q_body=body, snippets_block=block).strip(),
        model_override=model_t1,
    )
    initial = _coerce_yesno_from_text(t1.content)
    if t1.status == "ok" and initial is None:
        t1 = TurnRecord(**{**asdict(t1), "status": "parse_error"})  # type: ignore[arg-type]
        overall_status = "parse_error"

    # Turn 2 — Critique
    t2 = await call_turn(prompt_spec.critique_template, model_override=model_t2)

    # Turn 3 — Finalize
    t3 = await call_turn(prompt_spec.finalize_template, model_override=model_t3)
    final = _coerce_yesno_from_text(t3.content)
    if t3.status == "ok" and final is None:
        t3 = TurnRecord(**{**asdict(t3), "status": "parse_error"})  # type: ignore[arg-type]
        overall_status = "parse_error"
    final = final or cfg.missing_as

    return FBRecord(
        qid=qid,
        qtype=qtype,
        body=body,
        snippet_count=len(snippets),
        initial_answer=initial,
        final_answer=final,
        changed=(initial != final),
        turn1=t1,
        turn2=t2,
        turn3=t3,
        generation_status=overall_status,
        latency_ms=total_latency,
        retry_count=total_retries,
        model=cfg.model,
        model_critique=cfg.model_critique,
        model_finalize=cfg.model_finalize,
        temperature=float(cfg.temperature),
        topP=cfg.topP,
        topK=cfg.topK,
        maxOutputTokens=int(cfg.maxOutputTokens),
        presencePenalty=cfg.presencePenalty,
        frequencyPenalty=cfg.frequencyPenalty,
        thinkingLevel=cfg.thinkingLevel,
    )


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

async def process_file(
    path: Path,
    cfg: ExpConfig,
    prompt_spec: PromptSpec,
    client: OpenRouterClient,
) -> Tuple[Path, List[FBRecord]]:
    questions = load_questions(path)
    yesno_qs = [q for q in questions if normalize_qtype(str(q.get("type", ""))) == "yesno"]

    sem = asyncio.Semaphore(max(1, cfg.concurrency))
    tasks = [
        asyncio.create_task(process_question(client, cfg, prompt_spec, q, sem))
        for q in yesno_qs
    ]

    records: List[FBRecord] = []
    pbar = tqdm(total=len(tasks), desc=f"self-feedback: {path.name}")
    for coro in asyncio.as_completed(tasks):
        records.append(await coro)
        pbar.update(1)
    pbar.close()

    by_id: Dict[str, FBRecord] = {r.qid: r for r in records}

    out_questions: List[dict] = []
    for q in questions:
        q_out = dict(q)
        qid = str(q_out.get("id", "")).strip()
        q_out["ideal_answer"] = normalize_ideal_answer_for_submission(q_out.get("ideal_answer"))
        if normalize_qtype(str(q_out.get("type", ""))) == "yesno":
            rec = by_id.get(qid)
            q_out["exact_answer"] = rec.final_answer if rec else cfg.missing_as
        out_questions.append(q_out)

    doc_id = derive_doc_id(path)
    out_path = cfg.test_results_dir / f"result_{doc_id}_self_feedback.json"
    safe_json_dump(out_path, {"questions": out_questions})

    # Save full detail log
    detail_path = cfg.test_results_dir / f"self_feedback_details_{doc_id}.json"
    safe_json_dump(detail_path, [
        {
            "qid": r.qid,
            "initial_answer": r.initial_answer,
            "final_answer": r.final_answer,
            "changed": r.changed,
            "generation_status": r.generation_status,
            "latency_ms": r.latency_ms,
            "retry_count": r.retry_count,
            "turn1_content": r.turn1.content,
            "turn1_thinking": r.turn1.thinking_text,
            "turn1_status": r.turn1.status,
            "turn2_content": r.turn2.content,
            "turn2_thinking": r.turn2.thinking_text,
            "turn2_status": r.turn2.status,
            "turn3_content": r.turn3.content,
            "turn3_thinking": r.turn3.thinking_text,
            "turn3_status": r.turn3.status,
        }
        for r in records
    ])

    return out_path, records


# ---------------------------------------------------------------------------
# W&B Logger — extended from main.py WandbLogger
# ---------------------------------------------------------------------------

class WandbLogger:
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self._run = None

    def start(self, run_config: Dict[str, Any]) -> None:
        if not self.cfg.use_wandb:
            return
        if wandb is None:
            raise RuntimeError("wandb is not installed.")

        init_kwargs: Dict[str, Any] = {
            "project": self.cfg.wandb_project,
            "config": run_config,
        }
        if self.cfg.wandb_entity:
            init_kwargs["entity"] = self.cfg.wandb_entity
        if self.cfg.wandb_run_name:
            init_kwargs["name"] = self.cfg.wandb_run_name
        if self.cfg.wandb_group:
            init_kwargs["group"] = self.cfg.wandb_group
        if self.cfg.wandb_mode:
            init_kwargs["mode"] = self.cfg.wandb_mode

        self._run = wandb.init(**init_kwargs)

    def finish(self) -> None:
        if self._run and wandb is not None:
            wandb.finish()
            self._run = None

    def log_prompts(self, spec: PromptSpec) -> None:
        if not self._run or wandb is None:
            return
        wandb.config.update(
            {
                "prompt/system": spec.system,
                "prompt/generate_template": spec.generate_template,
                "prompt/critique_template": spec.critique_template,
                "prompt/finalize_template": spec.finalize_template,
            },
            allow_val_change=True,
        )

    def _trunc(self, value: Any) -> str:
        text = "" if value is None else str(value)
        mc = self.cfg.wandb_table_max_chars
        return text if len(text) <= mc else text[:max(0, mc - 1)] + "…"

    def log_questions_table(
        self,
        rows: Sequence[Dict[str, Any]],
        *,
        table_name: str = "questions",
    ) -> None:
        """
        Columns mirror main.py plus self-feedback-specific fields:
        initial_answer, changed, critique_text, t1_thinking, t2_thinking, t3_thinking.
        """
        if not self._run or wandb is None:
            return

        columns = [
            "doc_id", "id", "body", "snippet_count",
            "gold_answer", "system_answer", "correct",
            "initial_answer", "changed",
            "generation_status",
            "system_prompt",
            "generate_template", "critique_template", "finalize_template",
            "turn1_output", "turn1_thinking",
            "turn2_critique", "turn2_thinking",
            "turn3_output", "turn3_thinking",
            "finish_reason",
            "temperature", "model", "thinkingLevel",
            "latency_ms", "retry_count", "error_text",
        ]

        table = wandb.Table(columns=columns)
        for r in rows:
            table.add_data(
                r.get("doc_id", ""),
                r.get("id", ""),
                self._trunc(r.get("body", "")),
                int(r.get("snippet_count", 0) or 0),
                r.get("gold_answer", None),
                r.get("system_answer", None),
                bool(r.get("correct", False)),
                r.get("initial_answer", None),
                bool(r.get("changed", False)),
                r.get("generation_status", ""),
                self._trunc(r.get("system_prompt", "")),
                self._trunc(r.get("generate_template", "")),
                self._trunc(r.get("critique_template", "")),
                self._trunc(r.get("finalize_template", "")),
                self._trunc(r.get("turn1_output", "")),
                self._trunc(r.get("turn1_thinking", "")),
                self._trunc(r.get("turn2_critique", "")),
                self._trunc(r.get("turn2_thinking", "")),
                self._trunc(r.get("turn3_output", "")),
                self._trunc(r.get("turn3_thinking", "")),
                r.get("finish_reason", ""),
                r.get("temperature", None),
                r.get("model", ""),
                r.get("thinkingLevel", None),
                int(r.get("latency_ms", 0) or 0),
                r.get("retry_count", None),
                self._trunc(r.get("error_text", "")),
            )
        wandb.log({table_name: table})

    def log_doc_metrics_table(
        self,
        rows: Sequence[Dict[str, Any]],
        *,
        table_name: str = "per_doc_metrics",
    ) -> None:
        if not self._run or wandb is None:
            return
        columns = [
            "doc_id", "n", "missing_pred", "accuracy", "maF1",
            "precision_yes", "recall_yes", "f1_yes",
            "precision_no", "recall_no", "f1_no",
        ]
        table = wandb.Table(columns=columns)
        for r in rows:
            table.add_data(
                r.get("doc_id", ""),
                int(r.get("n", 0) or 0),
                int(r.get("missing_pred", 0) or 0),
                float(r.get("accuracy", 0.0) or 0.0),
                float(r.get("maF1", 0.0) or 0.0),
                float(r.get("precision_yes", 0.0) or 0.0),
                float(r.get("recall_yes", 0.0) or 0.0),
                float(r.get("f1_yes", 0.0) or 0.0),
                float(r.get("precision_no", 0.0) or 0.0),
                float(r.get("recall_no", 0.0) or 0.0),
                float(r.get("f1_no", 0.0) or 0.0),
            )
        wandb.log({table_name: table})

    def log_self_feedback_table(
        self,
        records: List[FBRecord],
        gold: Dict[str, YesNo],
        *,
        table_name: str = "self_feedback_changes",
    ) -> None:
        """Extra table: one row per question that changed its answer."""
        if not self._run or wandb is None:
            return
        columns = [
            "qid", "gold_answer", "initial_answer", "final_answer",
            "became_correct", "became_wrong",
            "turn1_output", "turn2_critique", "turn3_output",
        ]
        table = wandb.Table(columns=columns)
        for r in records:
            if not r.changed:
                continue
            g = gold.get(r.qid)
            table.add_data(
                r.qid,
                g,
                r.initial_answer,
                r.final_answer,
                bool(g is not None and r.final_answer == g and r.initial_answer != g),
                bool(g is not None and r.final_answer != g and r.initial_answer == g),
                self._trunc(r.turn1.content),
                self._trunc(r.turn2.content),
                self._trunc(r.turn3.content),
            )
        wandb.log({table_name: table})

    def log_overall_metrics(self, metrics: Dict[str, Any]) -> None:
        if not self._run or wandb is None:
            return
        wandb.log({f"overall/{k}": v for k, v in metrics.items()})

    def log_self_feedback_stats(
        self, records: List[FBRecord], gold: Dict[str, YesNo]
    ) -> None:
        """Log high-level self-feedback statistics."""
        if not self._run or wandb is None:
            return
        n = len(records)
        changed = sum(r.changed for r in records)
        changed_better = sum(
            r.changed
            and gold.get(r.qid) == r.final_answer
            and gold.get(r.qid) != r.initial_answer
            for r in records if r.qid in gold
        )
        changed_worse = sum(
            r.changed
            and gold.get(r.qid) != r.final_answer
            and gold.get(r.qid) == r.initial_answer
            for r in records if r.qid in gold
        )
        avg_latency = sum(r.latency_ms for r in records) / n if n else 0.0
        wandb.log({
            "self_feedback/total_questions": n,
            "self_feedback/changed_count": changed,
            "self_feedback/changed_pct": changed / n if n else 0.0,
            "self_feedback/changed_better": changed_better,
            "self_feedback/changed_worse": changed_worse,
            "self_feedback/avg_latency_ms": avg_latency,
        })


# ---------------------------------------------------------------------------
# W&B row builders
# ---------------------------------------------------------------------------

def build_questions_rows(
    *,
    cfg: ExpConfig,
    test_paths: List[Path],
    records: List[FBRecord],
    prompt_spec: PromptSpec,
) -> List[Dict[str, Any]]:
    gold_map = map_gold_labels(cfg.gold_path) if cfg.gold_path.exists() else {}
    rec_by_id: Dict[str, FBRecord] = {r.qid: r for r in records if r.qid}
    rows: List[Dict[str, Any]] = []

    for tp in test_paths:
        did = derive_doc_id(tp)
        for q in load_questions(tp):
            if normalize_qtype(str(q.get("type", ""))) != "yesno":
                continue
            qid = str(q.get("id", "")).strip()
            if not qid:
                continue
            gold_answer = gold_map.get(qid)
            rec = rec_by_id.get(qid)
            system_answer = rec.final_answer if rec else None

            rows.append({
                "doc_id": did,
                "id": qid,
                "body": str(q.get("body", "")).strip(),
                "snippet_count": rec.snippet_count if rec else len(collect_snippet_texts(q)),
                "gold_answer": gold_answer,
                "system_answer": system_answer,
                "correct": (gold_answer is not None) and (system_answer == gold_answer),
                "initial_answer": rec.initial_answer if rec else None,
                "changed": rec.changed if rec else False,
                "generation_status": rec.generation_status if rec else "api_error",
                "system_prompt": prompt_spec.system,
                "generate_template": prompt_spec.generate_template,
                "critique_template": prompt_spec.critique_template,
                "finalize_template": prompt_spec.finalize_template,
                "turn1_output": rec.turn1.content if rec else "",
                "turn1_thinking": rec.turn1.thinking_text if rec else "",
                "turn2_critique": rec.turn2.content if rec else "",
                "turn2_thinking": rec.turn2.thinking_text if rec else "",
                "turn3_output": rec.turn3.content if rec else "",
                "turn3_thinking": rec.turn3.thinking_text if rec else "",
                "finish_reason": rec.turn3.finish_reason if rec else "",
                "temperature": rec.temperature if rec else cfg.temperature,
                "model": rec.model if rec else cfg.model,
                "model_critique": rec.model_critique if rec else cfg.model_critique,
                "model_finalize": rec.model_finalize if rec else cfg.model_finalize,
                "thinkingLevel": rec.thinkingLevel if rec else cfg.thinkingLevel,
                "latency_ms": rec.latency_ms if rec else 0,
                "retry_count": rec.retry_count if rec else None,
                "error_text": rec.turn3.error_text if rec else "",
            })
    return rows


def build_doc_metrics_rows(per_doc: Dict[str, Metrics]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for doc_id, m in sorted(per_doc.items()):
        rows.append({
            "doc_id": doc_id,
            "n": m.n,
            "missing_pred": m.missing_pred,
            "accuracy": m.accuracy,
            "maF1": m.maF1,
            "precision_yes": m.precision_yes,
            "recall_yes": m.recall_yes,
            "f1_yes": m.f1_yes,
            "precision_no": m.precision_no,
            "recall_no": m.recall_no,
            "f1_no": m.f1_no,
        })
    return rows


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def main_async(cfg: ExpConfig) -> None:
    load_dotenv()
    api_key = os.getenv(cfg.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing {cfg.api_key_env} in .env")

    resolved_base_url = os.getenv("OPENROUTER_BASE_URL", cfg.base_url).strip() or cfg.base_url
    resolved_model = os.getenv("OPENROUTER_MODEL", cfg.model).strip() or cfg.model
    http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip() or None
    app_title = os.getenv("OPENROUTER_APP_TITLE", "").strip() or None

    test_paths = resolve_test_paths(cfg.test_dir, cfg.doc_ids)
    if not test_paths:
        raise SystemExit("No test files found.")

    prompt_spec = load_prompt_spec(cfg)
    cfg.test_results_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[FBRecord] = []
    result_paths: List[Path] = []

    async with OpenRouterClient(
        api_key=api_key,
        base_url=resolved_base_url,
        http_referer=http_referer,
        app_title=app_title,
        timeout_s=cfg.request_timeout_s,
    ) as client:
        for tp in test_paths:
            out_path, recs = await process_file(tp, cfg, prompt_spec, client)
            result_paths.append(out_path)
            all_records.extend(recs)
            print(f"[OK] {out_path}")

    # ── Evaluation ────────────────────────────────────────────────────────
    gold: Dict[str, YesNo] = {}
    per_doc_metrics: Dict[str, Metrics] = {}
    overall_metrics: Optional[Metrics] = None

    if cfg.gold_path.exists():
        per_doc_metrics, overall_metrics = run_evaluation(
            cfg=cfg, test_paths=test_paths, records=all_records
        )
        gold = map_gold_labels(cfg.gold_path)

        for did in sorted(per_doc_metrics.keys()):
            print_metrics(f"DOC {did}", per_doc_metrics[did])
        print_metrics("OVERALL", overall_metrics)

        # Self-feedback change statistics
        changed = sum(r.changed for r in all_records)
        changed_better = sum(
            r.changed and gold.get(r.qid) == r.final_answer and gold.get(r.qid) != r.initial_answer
            for r in all_records if r.qid in gold
        )
        changed_worse = sum(
            r.changed and gold.get(r.qid) != r.final_answer and gold.get(r.qid) == r.initial_answer
            for r in all_records if r.qid in gold
        )
        print(f"\n=== Self-feedback change stats ===")
        print(f"Changed after critique: {changed}/{len(all_records)}")
        print(f"  → became correct : {changed_better}")
        print(f"  → became wrong   : {changed_worse}")

    # ── W&B ───────────────────────────────────────────────────────────────
    if cfg.use_wandb:
        wb = WandbLogger(cfg)

        run_config = {
            "experiment": "self_feedback",
            "model": resolved_model,
            "model_critique": cfg.model_critique,
            "model_finalize": cfg.model_finalize,
            "temperature": cfg.temperature,
            "topP": cfg.topP,
            "topK": cfg.topK,
            "presencePenalty": cfg.presencePenalty,
            "frequencyPenalty": cfg.frequencyPenalty,
            "thinkingLevel": cfg.thinkingLevel,
            "maxOutputTokens": cfg.maxOutputTokens,
            "concurrency": cfg.concurrency,
            "request_timeout_s": cfg.request_timeout_s,
            "max_retries": cfg.max_retries,
            "backoff_base_s": cfg.backoff_base_s,
            "backoff_cap_s": cfg.backoff_cap_s,
            "max_chars_per_snippet": cfg.max_chars_per_snippet,
            "missing_as": cfg.missing_as,
            "test_dir": str(cfg.test_dir),
            "test_results_dir": str(cfg.test_results_dir),
            "gold_path": str(cfg.gold_path),
            "docs": cfg.doc_ids,
        }

        wb.start(run_config)
        wb.log_prompts(prompt_spec)
        wb.log_questions_table(
            build_questions_rows(
                cfg=cfg,
                test_paths=test_paths,
                records=all_records,
                prompt_spec=prompt_spec,
            ),
            table_name="questions",
        )

        if per_doc_metrics:
            wb.log_doc_metrics_table(
                build_doc_metrics_rows(per_doc_metrics),
                table_name="per_doc_metrics",
            )

        if overall_metrics is not None:
            overall_dict = metrics_to_dict(overall_metrics)
            keep_keys = [
                "accuracy", "maF1",
                "precision_yes", "recall_yes", "f1_yes",
                "precision_no", "recall_no", "f1_no",
            ]
            wb.log_overall_metrics({k: overall_dict[k] for k in keep_keys if k in overall_dict})

        wb.log_self_feedback_stats(all_records, gold)
        wb.log_self_feedback_table(all_records, gold, table_name="self_feedback_changes")
        wb.finish()

    print("\n[self_feedback.py] Wrote result files:")
    for rp in result_paths:
        print(" -", rp)


def main() -> None:
    cfg = ExpConfig()   # ← modify ExpConfig fields here
    print(
        f"Model={cfg.model} | thinkingLevel={cfg.thinkingLevel} | "
        f"docs={cfg.doc_ids} | concurrency={cfg.concurrency} | wandb={cfg.use_wandb}"
    )
    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()