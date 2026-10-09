"""
single_snippet.py — Experiment 3c: Snippet-by-Snippet Majority Vote

每道 yes/no 題目：
  1. 把每條 snippet 單獨送給 LLM，得到一個 yes/no
  2. 對所有 snippet 的答案做 majority vote → 最終答案

並發設計：
  - 所有題目的所有 snippet call 同時進入 asyncio task pool
  - 全域一個 Semaphore 控制最大同時 API 呼叫數
  - 進度條以「題目」為單位更新

W&B 上傳：
  - questions table（每題摘要 + 每個 snippet 的投票）
  - per_doc_metrics table
  - overall/ scalar metrics
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None  # type: ignore

# ─── types ────────────────────────────────────────────────────────────────────

YesNo = Literal["yes", "no"]
GenerationStatus = Literal["ok", "api_error", "timeout", "parse_error"]

# ─── prompts ──────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """You are a critical biomedical reviewer. Answer Yes/No questions only when the provided snippets contain direct and unambiguous evidence. If snippets conflict with each other, are only tangentially related to the question, or do not clearly support a yes answer, answer no. Your response must consist of exactly and only the single word "yes" or "no". Do not include any reasoning, explanation, punctuation, or extra characters.
"""

# 注意：這裡只有一條 snippet，所以標題改成 SNIPPET（單數）
SINGLE_SNIPPET_USER_TEMPLATE = """### INSTRUCTIONS:
1. Read the question and each numbered snippet carefully.
2. For each relevant snippet, note its number [N] and what evidence it provides.
3. Based only on the snippets, reason toward your final answer.
4. Do not use outside knowledge.
5. Your response must consist of exactly and only the single word "yes" or "no". Do not include any reasoning, explanation, punctuation, or extra characters.

### QUESTION:
{q_body}

### SNIPPETS:
{snippet_text}

### SNIPPET ANALYSIS AND ANSWER:

"""

# ─── config ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MainConfig:
    wandb_run_name: Optional[str] = "E-gemi"

    model: Optional[str] = "google/gemini-3.1-pro-preview"

    doc_ids: Optional[List[str]] = field(default_factory=lambda: ["13b_01", "13b_02", "13b_03", "13b_04"])
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    test_dir: Path = Path("data/testData")
    test_results_dir: Path = Path("data/testResults")
    gold_path: Path = Path("data/predictions/training13b.json")

    temperature: float = 0.0
    topP: Optional[float] = None
    topK: Optional[int] = None
    presencePenalty: Optional[float] = None
    frequencyPenalty: Optional[float] = None
    thinkingLevel: Optional[str] = None

    maxOutputTokens: int = 8192

    request_timeout_s: float = 1000.0
    concurrency: int = 20           # 全域最大同時 API 呼叫數

    max_retries: int = 2
    backoff_base_s: float = 0.8
    backoff_cap_s: float = 20.0

    only_yesno: bool = True
    max_chars_per_snippet: Optional[int] = None

    missing_as: YesNo = "no"        # 某條 snippet 解析失敗時算的票

    system_prompt_path: Optional[Path] = None
    user_template_path: Optional[Path] = None

    wandb: bool = True
    wandb_project: str = "yesno"
    wandb_entity: Optional[str] = "bioasq"
    wandb_group: Optional[str] = "CSHS"
    wandb_mode: Optional[str] = "online"
    wandb_table_max_chars: int = 100000


@dataclass(frozen=True)
class PromptSpec:
    system: str = DEFAULT_SYSTEM_PROMPT
    user_template: str = SINGLE_SNIPPET_USER_TEMPLATE


# ─── data classes ─────────────────────────────────────────────────────────────

@dataclass
class SnippetVote:
    """單一 snippet 問 LLM 的結果。"""
    snippet_index: int          # 0-based，對應原始 snippet 順序
    snippet_text: str
    result: Optional[YesNo]     # None = 解析失敗
    raw_content: str
    thinking_text: str
    status: GenerationStatus
    latency_ms: int
    retry_count: int
    error_text: str
    finish_reason: str


@dataclass
class SingleSnippetRecord:
    """一道題的所有 snippet votes + majority 結果。"""
    qid: str
    qtype: str
    body: str
    total_snippets: int

    snippet_votes: List[SnippetVote]

    effective_votes: List[YesNo]   # None 替換為 missing_as
    yes_count: int
    no_count: int
    final: YesNo                   # majority 結果
    all_agree: bool


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
    avg_snippets_per_q: float       # 平均每題 snippet 數（投票數）
    agreement_rate: float           # 全票一致的題目比例


# ─── helpers ──────────────────────────────────────────────────────────────────

def normalize_qtype(qtype: str) -> str:
    qt = (qtype or "").strip().lower()
    return "yesno" if qt in {"yes/no", "yesno", "yes-no", "yn"} else qt


def coerce_yesno(val: Any) -> Optional[YesNo]:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().lower().strip(" .,:;!\"'")
        if s == "yes":
            return "yes"
        if s == "no":
            return "no"
    if isinstance(val, list) and val:
        first = val[0]
        if isinstance(first, list) and first:
            first = first[0]
        return coerce_yesno(first)
    return None


def _coerce_yesno_from_text(text: str) -> Optional[YesNo]:
    if not text:
        return None
    lines = [l.strip().lower() for l in text.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1].strip(" .,:;!\"'")
        if last == "yes":
            return "yes"
        if last == "no":
            return "no"
    matches = re.findall(r"\b(yes|no)\b", text.lower())
    return matches[-1] if matches else None  # type: ignore


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


def derive_doc_id(input_path: Path) -> str:
    return derive_doc_id_from_phase_filename(input_path) or input_path.stem


def derive_result_filename(doc_id: str) -> str:
    return f"result_{doc_id}_single_snippet.json"


def discover_existing_doc_ids(test_dir: Path) -> List[str]:
    out: List[str] = []
    for p in sorted(test_dir.glob("phaseB_*b_*.json")):
        did = derive_doc_id_from_phase_filename(p)
        if did:
            out.append(did)
    return sorted(set(out))


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
    snippets = q.get("snippets", [])
    return [
        s.get("text", "")
        for s in snippets
        if isinstance(s, dict) and s.get("text")
    ]


def ids_in_test_file(test_path: Path) -> List[str]:
    return [
        str(q.get("id", "")).strip()
        for q in load_questions(test_path)
        if normalize_qtype(str(q.get("type", ""))) == "yesno"
        and str(q.get("id", "")).strip()
    ]


def load_prompt_spec(cfg: MainConfig) -> PromptSpec:
    spec = PromptSpec()
    system = spec.system
    user_template = spec.user_template
    if cfg.system_prompt_path and cfg.system_prompt_path.exists():
        system = cfg.system_prompt_path.read_text(encoding="utf-8").strip()
    if cfg.user_template_path and cfg.user_template_path.exists():
        user_template = cfg.user_template_path.read_text(encoding="utf-8").strip()
    return PromptSpec(system=system, user_template=user_template)


def map_yesno_labels(questions: Sequence[dict], *, exact_answer_key: str = "exact_answer") -> Dict[str, YesNo]:
    out: Dict[str, YesNo] = {}
    for q in questions:
        if normalize_qtype(str(q.get("type", ""))) != "yesno":
            continue
        qid = str(q.get("id", "")).strip()
        if not qid:
            continue
        label = coerce_yesno(q.get(exact_answer_key))
        if label is not None:
            out[qid] = label
    return out


def map_gold_labels(gold_path: Path) -> Dict[str, YesNo]:
    return map_yesno_labels(load_questions(gold_path))


# ─── API client (identical to main.py) ────────────────────────────────────────

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

    async def __aexit__(self, exc_type, exc, tb) -> None:
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
            raise RuntimeError("Client not started.")
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


def build_payload(cfg: MainConfig, messages: List[dict]) -> dict:
    payload: Dict[str, Any] = {
        "model": cfg.model,
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
        rd = msg.get("reasoning_details")
        if isinstance(rd, list):
            parts: List[str] = []
            for item in rd:
                if isinstance(item, dict):
                    t = item.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
            thinking = "\n\n".join(parts)
    return content, thinking, finish_reason, native_finish_reason


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
        # sleep 在 sem 外，不佔 slot
        sleep_s = min(backoff_cap_s, backoff_base_s * (2 ** attempt)) * (0.8 + 0.4 * random.random())
        await asyncio.sleep(sleep_s)

    latency_ms = int((time.perf_counter() - start_all) * 1000)
    return {}, max_retries, latency_ms, last_status, last_err


# ─── core: single-snippet voting ──────────────────────────────────────────────

async def _ask_one_snippet(
    client: OpenRouterClient,
    cfg: MainConfig,
    prompt_spec: PromptSpec,
    sem: asyncio.Semaphore,
    q_body: str,
    snippet_text: str,
    snippet_index: int,
) -> SnippetVote:
    """
    用單一 snippet 問 LLM，回傳 SnippetVote。
    sem 只在 API 呼叫時持有，sleep/backoff 在 sem 外。
    """
    # 截斷
    txt = snippet_text
    if cfg.max_chars_per_snippet is not None and len(txt) > cfg.max_chars_per_snippet:
        txt = txt[:cfg.max_chars_per_snippet].rstrip() + "…"

    user_content = prompt_spec.user_template.format(
        q_body=q_body,
        snippet_text=txt,
    ).strip()

    messages = [
        {"role": "system", "content": prompt_spec.system.strip()},
        {"role": "user",   "content": user_content},
    ]
    payload = build_payload(cfg, messages)

    async with sem:
        resp_json, retry_count, latency_ms, status, err_text = await retry_chat(
            client,
            payload=payload,
            max_retries=cfg.max_retries,
            backoff_base_s=cfg.backoff_base_s,
            backoff_cap_s=cfg.backoff_cap_s,
        )

    content, thinking, finish_reason, _ = extract_content_and_thinking(resp_json)

    result: Optional[YesNo] = None
    if status == "ok":
        result = _coerce_yesno_from_text(content)
        if result is None:
            status = "parse_error"

    return SnippetVote(
        snippet_index=snippet_index,
        snippet_text=snippet_text,
        result=result,
        raw_content=content.strip(),
        thinking_text=thinking.strip(),
        status=status,
        latency_ms=latency_ms,
        retry_count=retry_count,
        error_text=err_text.strip(),
        finish_reason=finish_reason.strip(),
    )


async def _process_one_question(
    client: OpenRouterClient,
    cfg: MainConfig,
    prompt_spec: PromptSpec,
    sem: asyncio.Semaphore,
    q: dict,
) -> SingleSnippetRecord:
    """
    一道題目：對每條 snippet 各發一次 API call（全部並發），
    收集結果後 majority vote。
    """
    qid   = str(q.get("id", "")).strip()
    qtype = normalize_qtype(str(q.get("type", "")))
    body  = str(q.get("body", "")).strip()
    snippets = collect_snippet_texts(q)

    if not snippets:
        # 沒有 snippet → 直接用 missing_as
        return SingleSnippetRecord(
            qid=qid, qtype=qtype, body=body, total_snippets=0,
            snippet_votes=[],
            effective_votes=[],
            yes_count=0, no_count=0,
            final=cfg.missing_as,
            all_agree=True,
        )

    # 所有 snippet 並發
    tasks = [
        _ask_one_snippet(
            client, cfg, prompt_spec, sem,
            q_body=body,
            snippet_text=snip,
            snippet_index=idx,
        )
        for idx, snip in enumerate(snippets)
    ]
    snippet_votes: List[SnippetVote] = list(await asyncio.gather(*tasks))

    # majority vote
    effective: List[YesNo] = [
        sv.result if sv.result is not None else cfg.missing_as
        for sv in snippet_votes
    ]
    counter = Counter(effective)
    yes_c = counter.get("yes", 0)
    no_c  = counter.get("no",  0)
    final: YesNo = counter.most_common(1)[0][0]

    return SingleSnippetRecord(
        qid=qid,
        qtype=qtype,
        body=body,
        total_snippets=len(snippets),
        snippet_votes=snippet_votes,
        effective_votes=effective,
        yes_count=yes_c,
        no_count=no_c,
        final=final,
        all_agree=(len(set(effective)) == 1),
    )


# ─── file-level generation ────────────────────────────────────────────────────

class AsyncGenerator:
    def __init__(
        self,
        cfg: MainConfig,
        prompt_spec: PromptSpec,
        client: OpenRouterClient,
        sem: asyncio.Semaphore,
    ):
        self.cfg = cfg
        self.prompt_spec = prompt_spec
        self.client = client
        self.sem = sem

    async def generate_file(
        self, input_path: Path
    ) -> Tuple[dict, List[SingleSnippetRecord]]:
        raw = load_json_file(input_path)
        if not isinstance(raw, dict):
            return {"questions": []}, []

        raw_questions = raw.get("questions")
        if not isinstance(raw_questions, list):
            return {"questions": []}, []

        questions: List[dict] = [q for q in raw_questions if isinstance(q, dict)]
        work = [q for q in questions
                if (not self.cfg.only_yesno)
                or normalize_qtype(str(q.get("type", ""))) == "yesno"]

        # 每道題一個 task（題目內部 snippet 也全部並發）
        tasks = [
            asyncio.create_task(
                _process_one_question(
                    self.client, self.cfg, self.prompt_spec, self.sem, q
                )
            )
            for q in work
        ]

        records: List[SingleSnippetRecord] = []
        pbar = tqdm(total=len(tasks), desc=input_path.name)
        for coro in asyncio.as_completed(tasks):
            records.append(await coro)
            pbar.update(1)
        pbar.close()

        by_id: Dict[str, SingleSnippetRecord] = {r.qid: r for r in records}

        out_questions: List[dict] = []
        for q in questions:
            q_out = dict(q)
            qid = str(q_out.get("id", "")).strip()
            qt  = normalize_qtype(str(q_out.get("type", "")))
            q_out["ideal_answer"] = normalize_ideal_answer_for_submission(
                q_out.get("ideal_answer")
            )
            if (not self.cfg.only_yesno) or (qt == "yesno"):
                rec = by_id.get(qid)
                q_out["exact_answer"] = rec.final if rec else cfg.missing_as
            out_questions.append(q_out)

        return {"questions": out_questions}, records


# ─── metrics ──────────────────────────────────────────────────────────────────

def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2.0 * p * r, p + r)


def compute_metrics(
    gold: Dict[str, YesNo],
    records: List[SingleSnippetRecord],
    *,
    missing_as: YesNo = "no",
) -> Metrics:
    pred_map = {r.qid: r.final for r in records}
    agree_map = {r.qid: r.all_agree for r in records}
    snip_map  = {r.qid: r.total_snippets for r in records}

    tp = fp = fn_c = tn = missing = 0
    for qid, g in gold.items():
        p = pred_map.get(qid)
        if p is None:
            missing += 1
            p = missing_as
        if   g == "yes" and p == "yes": tp += 1
        elif g == "no"  and p == "yes": fp += 1
        elif g == "yes" and p == "no":  fn_c += 1
        else:                            tn += 1

    n = len(gold)
    acc   = _safe_div(tp + tn, n)
    p_yes = _safe_div(tp, tp + fp)
    r_yes = _safe_div(tp, tp + fn_c)
    f1y   = _f1(p_yes, r_yes)
    p_no  = _safe_div(tn, tn + fn_c)
    r_no  = _safe_div(tn, tn + fp)
    f1n   = _f1(p_no, r_no)

    agree_rate = _safe_div(
        sum(1 for qid in gold if agree_map.get(qid, False)), n
    )
    avg_snip = _safe_div(
        sum(snip_map.get(qid, 0) for qid in gold), n
    )

    return Metrics(
        n=n, missing_pred=missing,
        tp=tp, fp=fp, fn=fn_c, tn=tn,
        accuracy=acc,
        precision_yes=p_yes, recall_yes=r_yes, f1_yes=f1y,
        precision_no=p_no,   recall_no=r_no,   f1_no=f1n,
        maF1=0.5 * (f1y + f1n),
        avg_snippets_per_q=avg_snip,
        agreement_rate=agree_rate,
    )


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def print_metrics(title: str, m: Metrics) -> None:
    print(f"\n=== {title} ===")
    print(f"N={m.n} | missing_pred={m.missing_pred}")
    print(f"Confusion (YES+): TP={m.tp} FP={m.fp} FN={m.fn} TN={m.tn}")
    print(f"accuracy={_fmt(m.accuracy)}  maF1={_fmt(m.maF1)}")
    print(f"YES: P={_fmt(m.precision_yes)} R={_fmt(m.recall_yes)} F1={_fmt(m.f1_yes)}")
    print(f"NO : P={_fmt(m.precision_no)}  R={_fmt(m.recall_no)}  F1={_fmt(m.f1_no)}")
    print(f"avg snippets/q={_fmt(m.avg_snippets_per_q)}  "
          f"agreement_rate={_fmt(m.agreement_rate)}")


def metrics_to_dict(m: Metrics) -> Dict[str, Any]:
    data = asdict(m)
    for k, v in list(data.items()):
        if isinstance(v, float):
            data[k] = float(f"{v:.6f}")
    return data


# ─── W&B ──────────────────────────────────────────────────────────────────────

class WandbLogger:
    def __init__(self, cfg: MainConfig):
        self.cfg = cfg
        self._run = None

    def start(self, run_config: Dict[str, Any]) -> None:
        if not self.cfg.wandb:
            return
        if wandb is None:
            raise RuntimeError("wandb not installed but WandbLogger enabled.")
        init_kwargs: Dict[str, Any] = {
            "project": self.cfg.wandb_project,
            "config": run_config,
        }
        if self.cfg.wandb_entity:    init_kwargs["entity"] = self.cfg.wandb_entity
        if self.cfg.wandb_run_name:  init_kwargs["name"]   = self.cfg.wandb_run_name
        if self.cfg.wandb_group:     init_kwargs["group"]  = self.cfg.wandb_group
        if self.cfg.wandb_mode:      init_kwargs["mode"]   = self.cfg.wandb_mode
        self._run = wandb.init(**init_kwargs)

    def finish(self) -> None:
        if self._run and wandb is not None:
            wandb.finish()
            self._run = None

    def _trunc(self, v: Any, max_c: int) -> str:
        t = "" if v is None else str(v)
        return t if len(t) <= max_c else t[:max(0, max_c - 1)] + "…"

    def log_prompt(self, *, system_prompt: str, user_template: str) -> None:
        if not self._run or wandb is None:
            return
        wandb.config.update(
            {"prompt/system": system_prompt,
             "prompt/user_template": user_template},
            allow_val_change=True,
        )

    def log_questions_table(
        self,
        gold_map: Dict[str, YesNo],
        all_records: List[SingleSnippetRecord],
        doc_id_map: Dict[str, str],
        prompt_spec: PromptSpec,
    ) -> None:
        if not self._run or wandb is None:
            return
        mc = self.cfg.wandb_table_max_chars
        columns = [
            # 題目基本資訊
            "doc_id", "id", "body",
            "total_snippets",
            # 結果
            "gold_answer", "final_answer", "correct",
            # voting 摘要
            "votes", "yes_count", "no_count", "all_agree",
            # 每條 snippet 的投票細節（純文字，方便 debug）
            "snippet_vote_details",
            # 設定
            "model", "temperature",
            # prompt（方便 diff）
            "system_prompt", "user_template",
        ]
        table = wandb.Table(columns=columns)
        for r in all_records:
            gold = gold_map.get(r.qid)

            # 每條 snippet 的投票細節：[0]=yes [1]=no ...
            vote_details = " | ".join(
                f"[{sv.snippet_index}]={sv.result or 'FAIL'}"
                f"(lat={sv.latency_ms}ms)"
                for sv in sorted(r.snippet_votes, key=lambda x: x.snippet_index)
            )

            table.add_data(
                doc_id_map.get(r.qid, ""),
                r.qid,
                self._trunc(r.body, mc),
                r.total_snippets,
                gold,
                r.final,
                (gold is not None) and (r.final == gold),
                str(r.effective_votes),
                r.yes_count,
                r.no_count,
                r.all_agree,
                self._trunc(vote_details, mc),
                self.cfg.model or "",
                float(self.cfg.temperature),
                self._trunc(prompt_spec.system, mc),
                self._trunc(prompt_spec.user_template, mc),
            )
        wandb.log({"questions": table})

    def log_doc_metrics_table(
        self,
        per_doc: Dict[str, Metrics],
        *,
        table_name: str = "per_doc_metrics",
    ) -> None:
        if not self._run or wandb is None:
            return
        columns = [
            "doc_id", "n", "missing_pred",
            "accuracy", "maF1",
            "precision_yes", "recall_yes", "f1_yes",
            "precision_no",  "recall_no",  "f1_no",
            "avg_snippets_per_q", "agreement_rate",
        ]
        table = wandb.Table(columns=columns)
        for doc_id, m in sorted(per_doc.items()):
            table.add_data(
                doc_id, m.n, m.missing_pred,
                m.accuracy, m.maF1,
                m.precision_yes, m.recall_yes, m.f1_yes,
                m.precision_no,  m.recall_no,  m.f1_no,
                m.avg_snippets_per_q, m.agreement_rate,
            )
        wandb.log({table_name: table})

    def log_overall_metrics(self, metrics: Dict[str, Any]) -> None:
        if not self._run or wandb is None:
            return
        keep = [
            "accuracy", "maF1",
            "precision_yes", "recall_yes", "f1_yes",
            "precision_no",  "recall_no",  "f1_no",
            "avg_snippets_per_q", "agreement_rate",
        ]
        wandb.log({f"overall/{k}": metrics[k] for k in keep if k in metrics})


# ─── evaluation helpers ───────────────────────────────────────────────────────

def run_evaluation(
    cfg: MainConfig,
    test_paths: List[Path],
    all_records: List[SingleSnippetRecord],
) -> Tuple[Dict[str, Metrics], Metrics]:
    gold = map_gold_labels(cfg.gold_path)
    if not gold:
        raise RuntimeError(f"No gold labels found in {cfg.gold_path}")

    per_doc: Dict[str, Metrics] = {}
    union_ids: Set[str] = set()

    for tp in test_paths:
        did  = derive_doc_id_from_phase_filename(tp) or tp.stem
        ids  = [qid for qid in ids_in_test_file(tp) if qid in gold]
        union_ids |= set(ids)
        doc_recs = [r for r in all_records if r.qid in set(ids)]
        per_doc[did] = compute_metrics(
            {qid: gold[qid] for qid in ids},
            doc_recs,
            missing_as=cfg.missing_as,
        )

    overall = compute_metrics(
        {qid: gold[qid] for qid in union_ids},
        [r for r in all_records if r.qid in union_ids],
        missing_as=cfg.missing_as,
    )
    return per_doc, overall


def build_doc_metrics_rows(per_doc: Dict[str, Metrics]) -> List[Dict[str, Any]]:
    rows = []
    for doc_id, m in sorted(per_doc.items()):
        d = asdict(m)
        d["doc_id"] = doc_id
        rows.append(d)
    return rows


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = MainConfig()

    if not cfg.gold_path.exists():
        raise SystemExit(f"Gold file not found: {cfg.gold_path}")

    candidate_ids = (
        discover_existing_doc_ids(cfg.test_dir)
        if cfg.doc_ids is None
        else cfg.doc_ids
    )
    test_paths = resolve_test_paths(cfg.test_dir, candidate_ids)
    if not test_paths:
        raise SystemExit("No test files found. Check test_dir and doc_ids.")

    prompt_spec = load_prompt_spec(cfg)

    # ── run generation ────────────────────────────────────────────────────────
    all_records: List[SingleSnippetRecord] = []
    result_paths: List[Path] = []
    doc_id_map: Dict[str, str] = {}   # qid → doc_id

    async def _run_all() -> None:
        load_dotenv()
        api_key = os.getenv(cfg.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key: {cfg.api_key_env}")

        resolved_base_url = (
            os.getenv("OPENROUTER_BASE_URL", cfg.base_url).strip() or cfg.base_url
        )
        resolved_model = (
            os.getenv("OPENROUTER_MODEL", cfg.model or "").strip() or (cfg.model or "")
        )
        fields_dict = {
            f.name: getattr(cfg, f.name)
            for f in cfg.__dataclass_fields__.values()  # type: ignore[attr-defined]
        }
        runtime_cfg = MainConfig(**{
            **fields_dict,
            "base_url": resolved_base_url,
            "model": resolved_model,
        })

        runtime_cfg.test_results_dir.mkdir(parents=True, exist_ok=True)

        # 全域共用 semaphore：控制跨題目跨 snippet 的總 API 並發數
        sem = asyncio.Semaphore(max(1, runtime_cfg.concurrency))

        http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip() or None
        app_title    = os.getenv("OPENROUTER_APP_TITLE",    "").strip() or None

        async with OpenRouterClient(
            api_key=api_key,
            base_url=runtime_cfg.base_url,
            http_referer=http_referer,
            app_title=app_title,
            timeout_s=runtime_cfg.request_timeout_s,
        ) as client:
            generator = AsyncGenerator(runtime_cfg, prompt_spec, client, sem)

            for input_path in test_paths:
                doc_id = derive_doc_id(input_path)
                submission_json, records = await generator.generate_file(input_path)

                out_path = runtime_cfg.test_results_dir / derive_result_filename(doc_id)
                safe_json_dump(out_path, submission_json)
                result_paths.append(out_path)

                for r in records:
                    doc_id_map[r.qid] = doc_id
                all_records.extend(records)

    asyncio.run(_run_all())

    # ── evaluate ──────────────────────────────────────────────────────────────
    per_doc_metrics, overall_metrics = run_evaluation(cfg, test_paths, all_records)
    for did in sorted(per_doc_metrics.keys()):
        print_metrics(f"DOC {did}", per_doc_metrics[did])
    print_metrics("OVERALL", overall_metrics)

    # ── save vote detail json ─────────────────────────────────────────────────
    safe_json_dump(
        cfg.test_results_dir / "vote_details_single_snippet.json",
        [
            {
                "qid":             r.qid,
                "total_snippets":  r.total_snippets,
                "effective_votes": r.effective_votes,
                "yes_count":       r.yes_count,
                "no_count":        r.no_count,
                "final":           r.final,
                "all_agree":       r.all_agree,
                "snippet_votes": [
                    {
                        "snippet_index": sv.snippet_index,
                        "result":        sv.result,
                        "status":        sv.status,
                        "latency_ms":    sv.latency_ms,
                        "retry_count":   sv.retry_count,
                        "snippet_text":  sv.snippet_text[:200],
                        "raw_content":   sv.raw_content[:300],
                    }
                    for sv in sorted(r.snippet_votes, key=lambda x: x.snippet_index)
                ],
            }
            for r in all_records
        ],
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    if cfg.wandb:
        if wandb is None:
            print("[WARN] wandb not installed. Skipping W&B logging.")
        else:
            wb = WandbLogger(cfg)
            run_config = {
                "model":                cfg.model,
                "temperature":          cfg.temperature,
                "topP":                 cfg.topP,
                "topK":                 cfg.topK,
                "concurrency":          cfg.concurrency,
                "max_retries":          cfg.max_retries,
                "only_yesno":           cfg.only_yesno,
                "max_chars_per_snippet": cfg.max_chars_per_snippet,
                "missing_as":           cfg.missing_as,
                "test_dir":             str(cfg.test_dir),
                "gold_path":            str(cfg.gold_path),
                "docs":                 candidate_ids,
                "experiment":           "single_snippet_majority_vote",
            }
            wb.start(run_config)
            wb.log_prompt(
                system_prompt=prompt_spec.system,
                user_template=prompt_spec.user_template,
            )
            wb.log_questions_table(
                gold_map=map_gold_labels(cfg.gold_path),
                all_records=all_records,
                doc_id_map=doc_id_map,
                prompt_spec=prompt_spec,
            )
            wb.log_doc_metrics_table(per_doc_metrics)
            wb.log_overall_metrics(metrics_to_dict(overall_metrics))
            wb.finish()

    print("\n[single_snippet.py] Wrote result files:")
    for rp in result_paths:
        print(" -", rp)


if __name__ == "__main__":
    main()