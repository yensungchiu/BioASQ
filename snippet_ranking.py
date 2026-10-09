"""
cosine ranking snippets
可以選前幾個 是否附上分數
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
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

DEFAULT_SYSTEM_PROMPT = """You are a critical biomedical reviewer. Answer Yes/No questions only when the provided snippets contain direct and unambiguous evidence. If snippets conflict with each other, are only tangentially related to the question, or do not clearly support a yes answer, answer no. Your response must consist of exactly and only the single word "yes" or "no". Do not include any reasoning, explanation, punctuation, or extra characters.
"""

DEFAULT_USER_TEMPLATE = """### INSTRUCTIONS:
1. Read the question and each numbered snippet carefully.
2. For each relevant snippet, note its number [N] and what evidence it provides.
3. Based only on the snippets, reason toward your final answer.
4. Do not use outside knowledge.
5. Your response must consist of exactly and only the single word "yes" or "no". Do not include any reasoning, explanation, punctuation, or extra characters.

### QUESTION:
{q_body}

### SNIPPETS:
{snippets_block}

### SNIPPET ANALYSIS AND ANSWER:
"""

PROMPT_NO_SNIPPETS_TEXT = "(No snippets provided.)"
PROMPT_SNIPPET_INDEX_TEMPLATE = "[{rank}]"
PROMPT_SNIPPET_WITH_SCORE_TEMPLATE = "[{rank}] (similarity={score})"
PROMPT_SCORE_DECIMALS_DEFAULT = 4


YesNo = Literal["yes", "no"]
GenerationStatus = Literal["ok", "api_error", "timeout", "parse_error"]


@dataclass(frozen=True)
class MainConfig:
    wandb_run_name: Optional[str] = "B-show_scr"
    model: Optional[str] = "anthropic/claude-opus-4.6"
    prompt_include_snippet_scores: bool = True
    snippet_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_snippets: Optional[int] = None  # None = use all after ranking
    
    prompt_score_decimals: int = PROMPT_SCORE_DECIMALS_DEFAULT

    doc_ids: Optional[List[str]] = field(default_factory=lambda: ["12b_01", "12b_02", "12b_03", "12b_04", "13b_01", "13b_02", "13b_03", "13b_04"])
    
    # snippet reranking
    snippet_rerank_enabled: bool = True
    
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
    concurrency: int = 20

    max_retries: int = 1
    backoff_base_s: float = 0.8
    backoff_cap_s: float = 20.0

    only_yesno: bool = True
    max_chars_per_snippet: Optional[int] = None

    missing_as: YesNo = "no"

    system_prompt_path: Optional[Path] = None
    user_template_path: Optional[Path] = None

    wandb: bool = True
    wandb_project: str = "yesno"
    wandb_entity: Optional[str] = "bioasq"
    wandb_group: Optional[str] = "CSHS"
    wandb_mode: Optional[str] = "online"
    wandb_table_max_chars: int = 100000

    # prompt formatting controls


@dataclass(frozen=True)
class PromptSpec:
    system: str = DEFAULT_SYSTEM_PROMPT
    user_template: str = DEFAULT_USER_TEMPLATE


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
class PredRecord:
    qid: str
    qtype: str
    body: str
    snippet_count: int

    pred_exact: YesNo
    generation_status: GenerationStatus
    retry_count: int
    latency_ms: int

    raw_content: str
    thinking_text: str
    finish_reason: str
    native_finish_reason: str
    error_text: str

    model: str
    temperature: float
    topP: Optional[float]
    topK: Optional[int]
    maxOutputTokens: int
    presencePenalty: Optional[float]
    frequencyPenalty: Optional[float]
    thinkingLevel: Optional[str]


@dataclass
class RankedSnippet:
    rank: int
    original_index: int
    text: str
    score: Optional[float]


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
    return f"result_{doc_id}.json"


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

    texts: List[str] = []
    for snippet in snippets:
        text = snippet.get("text", None)
        if text is None:
            continue
        texts.append(str(text))

    return texts


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


class SnippetRanker:
    def __init__(self, model_name: str):
        if SentenceTransformer is None:
            raise RuntimeError(
                "Snippet reranking needs sentence-transformers. "
                "Please run: pip install sentence-transformers numpy"
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def rank(
        self,
        question: str,
        snippets: Sequence[str],
        *,
        top_k: Optional[int] = None,
    ) -> List[RankedSnippet]:
        if not snippets:
            return []

        if len(snippets) == 1:
            ranked = [
                RankedSnippet(
                    rank=1,
                    original_index=0,
                    text=str(snippets[0]),
                    score=1.0,
                )
            ]
            if top_k is not None:
                ranked = ranked[: max(0, int(top_k))]
            return ranked

        texts = [question, *snippets]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        q_emb = embeddings[0]
        snippet_embs = embeddings[1:]

        scored: List[Tuple[float, int, str]] = []
        for idx, (snippet, emb) in enumerate(zip(snippets, snippet_embs)):
            score = cosine_similarity(q_emb, emb)
            scored.append((score, idx, str(snippet)))

        scored.sort(key=lambda x: (-x[0], x[1]))

        ranked: List[RankedSnippet] = []
        for rank_idx, (score, original_idx, snippet_text) in enumerate(scored, start=1):
            ranked.append(
                RankedSnippet(
                    rank=rank_idx,
                    original_index=original_idx,
                    text=snippet_text,
                    score=score,
                )
            )

        if top_k is not None:
            ranked = ranked[: max(0, int(top_k))]
            for i, item in enumerate(ranked, start=1):
                item.rank = i

        return ranked


def rerank_snippets(
    q: dict,
    *,
    cfg: MainConfig,
    snippet_ranker: Optional[SnippetRanker],
) -> List[RankedSnippet]:
    snippet_texts = collect_snippet_texts(q)
    if not snippet_texts:
        return []

    if not cfg.snippet_rerank_enabled:
        limited = snippet_texts
        if cfg.top_k_snippets is not None:
            limited = limited[: max(0, int(cfg.top_k_snippets))]
        return [
            RankedSnippet(
                rank=i,
                original_index=i - 1,
                text=text,
                score=None,
            )
            for i, text in enumerate(limited, start=1)
        ]

    if snippet_ranker is None:
        raise RuntimeError("snippet_rerank_enabled=True but no SnippetRanker was provided.")

    q_body = str(q.get("body", "")).strip()
    return snippet_ranker.rank(
        q_body,
        snippet_texts,
        top_k=cfg.top_k_snippets,
    )


def format_score(score: Optional[float], decimals: int) -> str:
    if score is None:
        return "NA"
    return f"{score:.{decimals}f}"


def build_snippets_block(
    ranked_snippets: Sequence[RankedSnippet],
    *,
    max_chars_per_snippet: Optional[int],
    include_scores: bool,
    score_decimals: int,
) -> str:
    if not ranked_snippets:
        return PROMPT_NO_SNIPPETS_TEXT

    parts: List[str] = []
    for item in ranked_snippets:
        txt = item.text
        if max_chars_per_snippet is not None and len(txt) > max_chars_per_snippet:
            txt = txt[:max_chars_per_snippet].rstrip() + "…"

        if include_scores:
            prefix = PROMPT_SNIPPET_WITH_SCORE_TEMPLATE.format(
                rank=item.rank,
                score=format_score(item.score, score_decimals),
            )
        else:
            prefix = PROMPT_SNIPPET_INDEX_TEMPLATE.format(rank=item.rank)

        parts.append(f"{prefix}: {txt}")

    return "\n".join(parts)


def build_user_prompt_for_question(
    q: dict,
    *,
    prompt_spec: PromptSpec,
    max_chars_per_snippet: Optional[int],
    ranked_snippets: Optional[Sequence[RankedSnippet]] = None,
    include_scores: bool = False,
    score_decimals: int = PROMPT_SCORE_DECIMALS_DEFAULT,
) -> str:
    q_body = str(q.get("body", "")).strip()
    if ranked_snippets is None:
        ranked_snippets = [
            RankedSnippet(rank=i, original_index=i - 1, text=text, score=None)
            for i, text in enumerate(collect_snippet_texts(q), start=1)
        ]

    snippets_block = build_snippets_block(
        ranked_snippets,
        max_chars_per_snippet=max_chars_per_snippet,
        include_scores=include_scores,
        score_decimals=score_decimals,
    )

    return prompt_spec.user_template.format(
        q_body=q_body,
        snippets_block=snippets_block,
    ).strip()


def build_snippet_scores_text(
    ranked_snippets: Sequence[RankedSnippet],
    *,
    score_decimals: int,
    max_chars_per_snippet: Optional[int],
) -> str:
    if not ranked_snippets:
        return PROMPT_NO_SNIPPETS_TEXT

    rows: List[str] = []
    for item in ranked_snippets:
        txt = item.text
        if max_chars_per_snippet is not None and len(txt) > max_chars_per_snippet:
            txt = txt[:max_chars_per_snippet].rstrip() + "…"
        rows.append(
            f"[{item.rank}] orig_idx={item.original_index + 1} "
            f"similarity={format_score(item.score, score_decimals)} :: {txt}"
        )
    return "\n".join(rows)


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


def map_pred_labels_from_result(result_path: Path) -> Dict[str, YesNo]:
    return map_yesno_labels(load_questions(result_path))


def ids_in_test_file(test_path: Path) -> List[str]:
    ids: List[str] = []
    for q in load_questions(test_path):
        if normalize_qtype(str(q.get("type", ""))) != "yesno":
            continue
        qid = str(q.get("id", "")).strip()
        if qid:
            ids.append(qid)
    return ids


def _coerce_yesno_from_text(text: str) -> Optional[YesNo]:
    if not text:
        return None

    lines = [line.strip().lower() for line in text.strip().splitlines() if line.strip()]
    if lines:
        last_line = lines[-1].strip(" .,:;!\"'")
        if last_line == "yes":
            return "yes"
        if last_line == "no":
            return "no"

    matches = re.findall(r"\b(yes|no)\b", text.lower())
    if matches:
        return "yes" if matches[-1] == "yes" else "no"
    return None


class PromptBuilder:
    def __init__(
        self,
        spec: PromptSpec,
        cfg: MainConfig,
        snippet_ranker: Optional[SnippetRanker] = None,
    ):
        self.spec = spec
        self.cfg = cfg
        self.snippet_ranker = snippet_ranker

    def build_messages(self, q: dict) -> List[dict]:
        ranked_snippets = rerank_snippets(
            q,
            cfg=self.cfg,
            snippet_ranker=self.snippet_ranker,
        )

        return [
            {"role": "system", "content": self.spec.system.strip()},
            {
                "role": "user",
                "content": build_user_prompt_for_question(
                    q,
                    prompt_spec=self.spec,
                    max_chars_per_snippet=self.cfg.max_chars_per_snippet,
                    ranked_snippets=ranked_snippets,
                    include_scores=self.cfg.prompt_include_snippet_scores,
                    score_decimals=self.cfg.prompt_score_decimals,
                ),
            },
        ]


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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers

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


class AsyncGenerator:
    def __init__(self, cfg: MainConfig, prompt: PromptBuilder, client: OpenRouterClient):
        self.cfg = cfg
        self.prompt = prompt
        self.client = client
        self._sem = asyncio.Semaphore(max(1, cfg.concurrency))

    async def _generate_one(self, q: dict) -> PredRecord:
        qid = str(q.get("id", "")).strip()
        qtype = normalize_qtype(str(q.get("type", "")))
        body = str(q.get("body", "")).strip()

        ranked_snippets = rerank_snippets(
            q,
            cfg=self.cfg,
            snippet_ranker=self.prompt.snippet_ranker,
        )
        snippet_count = len(ranked_snippets)

        messages = self.prompt.build_messages(q)
        payload = build_payload(self.cfg, messages)

        async with self._sem:
            resp_json, retry_count, latency_ms, status, err_text = await retry_chat(
                self.client,
                payload=payload,
                max_retries=self.cfg.max_retries,
                backoff_base_s=self.cfg.backoff_base_s,
                backoff_cap_s=self.cfg.backoff_cap_s,
            )

        content, thinking, finish_reason, native_finish_reason = extract_content_and_thinking(resp_json)

        pred: YesNo = "no"
        if status == "ok":
            yn = _coerce_yesno_from_text(content)
            if yn is None:
                status = "parse_error"
                pred = "no"
            else:
                pred = yn

        return PredRecord(
            qid=qid,
            qtype=qtype,
            body=body,
            snippet_count=snippet_count,
            pred_exact=pred,
            generation_status=status,
            retry_count=retry_count,
            latency_ms=latency_ms,
            raw_content=(content or "").strip(),
            thinking_text=(thinking or "").strip(),
            finish_reason=(finish_reason or "").strip(),
            native_finish_reason=(native_finish_reason or "").strip(),
            error_text=(err_text or "").strip(),
            model=self.cfg.model or "",
            temperature=float(self.cfg.temperature),
            topP=self.cfg.topP,
            topK=self.cfg.topK,
            maxOutputTokens=int(self.cfg.maxOutputTokens),
            presencePenalty=self.cfg.presencePenalty,
            frequencyPenalty=self.cfg.frequencyPenalty,
            thinkingLevel=self.cfg.thinkingLevel,
        )

    async def generate_file(self, input_path: Path) -> Tuple[dict, List[PredRecord]]:
        raw = load_json_file(input_path)
        if not isinstance(raw, dict):
            return {"questions": []}, []

        raw_questions = raw.get("questions")
        if not isinstance(raw_questions, list):
            return {"questions": []}, []

        questions: List[dict] = [q for q in raw_questions if isinstance(q, dict)]
        if not questions:
            return {"questions": []}, []

        work: List[dict] = []
        for q in questions:
            qt = normalize_qtype(str(q.get("type", "")))
            if self.cfg.only_yesno and qt != "yesno":
                continue
            work.append(q)

        tasks = [asyncio.create_task(self._generate_one(q)) for q in work]

        preds: List[PredRecord] = []
        pbar = tqdm(total=len(tasks), desc=input_path.name)
        try:
            for coro in asyncio.as_completed(tasks):
                preds.append(await coro)
                pbar.update(1)
        finally:
            pbar.close()

        pred_by_id: Dict[str, PredRecord] = {p.qid: p for p in preds if p.qid}

        out_questions: List[dict] = []
        for q in questions:
            q_out = dict(q)
            qid = str(q_out.get("id", "")).strip()
            qt = normalize_qtype(str(q_out.get("type", "")))

            q_out["ideal_answer"] = normalize_ideal_answer_for_submission(q_out.get("ideal_answer"))

            if (not self.cfg.only_yesno) or (qt == "yesno"):
                p = pred_by_id.get(qid)
                q_out["exact_answer"] = p.pred_exact if p else "no"

            out_questions.append(q_out)

        return {"questions": out_questions}, preds


async def generate_for_files(
    input_paths: Sequence[Path],
    *,
    cfg: MainConfig,
    prompt_spec: Optional[PromptSpec] = None,
) -> Tuple[List[Path], List[PredRecord]]:
    load_dotenv()

    api_key = os.getenv(cfg.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Put {cfg.api_key_env}=... in your .env")

    resolved_base_url = os.getenv("OPENROUTER_BASE_URL", cfg.base_url).strip() or cfg.base_url
    resolved_model = os.getenv("OPENROUTER_MODEL", cfg.model or "").strip() or (cfg.model or "")
    http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip() or None
    app_title = os.getenv("OPENROUTER_APP_TITLE", "").strip() or None

    runtime_cfg = MainConfig(**{**cfg.__dict__, "base_url": resolved_base_url, "model": resolved_model})
    prompt_spec = prompt_spec or PromptSpec()

    snippet_ranker: Optional[SnippetRanker] = None
    if runtime_cfg.snippet_rerank_enabled:
        snippet_ranker = SnippetRanker(runtime_cfg.snippet_embedding_model)

    prompt_builder = PromptBuilder(prompt_spec, runtime_cfg, snippet_ranker=snippet_ranker)

    runtime_cfg.test_results_dir.mkdir(parents=True, exist_ok=True)

    all_preds: List[PredRecord] = []
    result_paths: List[Path] = []

    async with OpenRouterClient(
        api_key=api_key,
        base_url=runtime_cfg.base_url,
        http_referer=http_referer,
        app_title=app_title,
        timeout_s=runtime_cfg.request_timeout_s,
    ) as client:
        generator = AsyncGenerator(runtime_cfg, prompt_builder, client)

        for input_path in input_paths:
            doc_id = derive_doc_id(input_path)
            submission_json, preds = await generator.generate_file(input_path)

            out_path = runtime_cfg.test_results_dir / derive_result_filename(doc_id)
            safe_json_dump(out_path, submission_json)

            result_paths.append(out_path)
            all_preds.extend(preds)

    return result_paths, all_preds


async def run_generation(
    cfg: MainConfig,
    test_paths: List[Path],
    prompt_spec: PromptSpec,
) -> Tuple[List[Path], List[PredRecord]]:
    return await generate_for_files(test_paths, cfg=cfg, prompt_spec=prompt_spec)


class WandbLogger:
    def __init__(self, cfg: MainConfig):
        self.cfg = cfg
        self._run = None

    def start(self, run_config: Dict[str, Any]) -> None:
        if not self.cfg.wandb:
            return
        if wandb is None:
            raise RuntimeError("wandb is not installed, but WandbLogger is enabled.")

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

    def log_prompt(self, *, system_prompt: str, user_template: str) -> None:
        if not self._run or wandb is None:
            return
        wandb.config.update(
            {
                "prompt/system": system_prompt,
                "prompt/user_template": user_template,
            },
            allow_val_change=True,
        )

    def _truncate(self, value: Any, max_chars: int) -> str:
        text = "" if value is None else str(value)
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 1)] + "…"

    def log_questions_table(
        self,
        rows: Sequence[Dict[str, Any]],
        *,
        max_text_chars: int = 4000,
        table_name: str = "questions",
    ) -> None:
        if not self._run or wandb is None:
            return

        columns = [
            "doc_id",
            "id",
            "body",
            "snippet_count",
            "gold_answer",
            "system_answer",
            "correct",
            "generation_status",
            "system_prompt",
            "user_prompt",
            "snippet_scores",
            "thinking_text",
            "raw_output",
            "finish_reason",
            "temperature",
            "model",
            "error_text",
            "retry_count",
        ]

        table = wandb.Table(columns=columns)
        for r in rows:
            table.add_data(
                r.get("doc_id", ""),
                r.get("id", ""),
                self._truncate(r.get("body", ""), max_text_chars),
                int(r.get("snippet_count", 0) or 0),
                r.get("gold_answer", None),
                r.get("system_answer", None),
                bool(r.get("correct", False)),
                r.get("generation_status", ""),
                self._truncate(r.get("system_prompt", ""), max_text_chars),
                self._truncate(r.get("user_prompt", ""), max_text_chars),
                self._truncate(r.get("snippet_scores", ""), max_text_chars),
                self._truncate(r.get("thinking_text", ""), max_text_chars),
                self._truncate(r.get("raw_output", ""), max_text_chars),
                r.get("finish_reason", ""),
                r.get("temperature", None),
                r.get("model", ""),
                self._truncate(r.get("error_text", ""), max_text_chars),
                r.get("retry_count", None),
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
            "doc_id",
            "n",
            "missing_pred",
            "accuracy",
            "maF1",
            "precision_yes",
            "recall_yes",
            "f1_yes",
            "precision_no",
            "recall_no",
            "f1_no",
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

    def log_overall_metrics(self, metrics: Dict[str, Any]) -> None:
        if not self._run or wandb is None:
            return
        wandb.log({f"overall/{k}": v for k, v in metrics.items()})


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2.0 * p * r, p + r)


def compute_metrics(gold: Dict[str, YesNo], pred: Dict[str, YesNo], *, missing_as: YesNo = "no") -> Metrics:
    tp = fp = fn = tn = 0
    missing = 0

    for qid, g in gold.items():
        p = pred.get(qid)
        if p is None:
            missing += 1
            p = missing_as

        if g == "yes" and p == "yes":
            tp += 1
        elif g == "no" and p == "yes":
            fp += 1
        elif g == "yes" and p == "no":
            fn += 1
        elif g == "no" and p == "no":
            tn += 1

    n = len(gold)
    acc = _safe_div(tp + tn, n)

    p_yes = _safe_div(tp, tp + fp)
    r_yes = _safe_div(tp, tp + fn)
    f1_yes = _f1(p_yes, r_yes)

    p_no = _safe_div(tn, tn + fn)
    r_no = _safe_div(tn, tn + fp)
    f1_no = _f1(p_no, r_no)

    return Metrics(
        n=n,
        missing_pred=missing,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        accuracy=acc,
        precision_yes=p_yes,
        recall_yes=r_yes,
        f1_yes=f1_yes,
        precision_no=p_no,
        recall_no=r_no,
        f1_no=f1_no,
        maF1=0.5 * (f1_yes + f1_no),
    )


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def print_metrics(title: str, m: Metrics) -> None:
    print(f"\n=== {title} ===")
    print(f"N={m.n} | missing_pred={m.missing_pred}")
    print(f"Confusion (YES positive): TP={m.tp} FP={m.fp} FN={m.fn} TN={m.tn}")
    print(f"accuracy={_fmt(m.accuracy)}  maF1={_fmt(m.maF1)}")
    print(f"YES: P={_fmt(m.precision_yes)} R={_fmt(m.recall_yes)} F1={_fmt(m.f1_yes)}")
    print(f"NO : P={_fmt(m.precision_no)}  R={_fmt(m.recall_no)}  F1={_fmt(m.f1_no)}")


def metrics_to_dict(m: Metrics) -> Dict[str, Any]:
    data = asdict(m)
    for k, v in list(data.items()):
        if isinstance(v, float):
            data[k] = float(f"{v:.6f}")
    return data


def build_questions_rows(
    *,
    cfg: MainConfig,
    test_paths: List[Path],
    preds: List[PredRecord],
    prompt_spec: PromptSpec,
) -> List[Dict[str, Any]]:
    gold_map = map_gold_labels(cfg.gold_path)
    pred_by_id: Dict[str, PredRecord] = {p.qid: p for p in preds if p.qid}
    rows: List[Dict[str, Any]] = []

    snippet_ranker: Optional[SnippetRanker] = None
    if cfg.snippet_rerank_enabled:
        snippet_ranker = SnippetRanker(cfg.snippet_embedding_model)

    for tp in test_paths:
        did = derive_doc_id_from_phase_filename(tp) or tp.stem
        for q in load_questions(tp):
            if normalize_qtype(str(q.get("type", ""))) != "yesno":
                continue
            qid = str(q.get("id", "")).strip()
            if not qid:
                continue

            gold_answer = gold_map.get(qid, None)
            pred = pred_by_id.get(qid)
            system_answer = pred.pred_exact if pred else None

            ranked_snippets = rerank_snippets(
                q,
                cfg=cfg,
                snippet_ranker=snippet_ranker,
            )

            rows.append(
                {
                    "doc_id": did,
                    "id": qid,
                    "body": str(q.get("body", "")).strip(),
                    "snippet_count": pred.snippet_count if pred else len(ranked_snippets),
                    "gold_answer": gold_answer,
                    "system_answer": system_answer,
                    "correct": (gold_answer is not None) and (system_answer == gold_answer),
                    "generation_status": pred.generation_status if pred else "api_error",
                    "system_prompt": prompt_spec.system,
                    "user_prompt": build_user_prompt_for_question(
                        q,
                        prompt_spec=prompt_spec,
                        max_chars_per_snippet=cfg.max_chars_per_snippet,
                        ranked_snippets=ranked_snippets,
                        include_scores=cfg.prompt_include_snippet_scores,
                        score_decimals=cfg.prompt_score_decimals,
                    ),
                    "snippet_scores": build_snippet_scores_text(
                        ranked_snippets,
                        score_decimals=cfg.prompt_score_decimals,
                        max_chars_per_snippet=cfg.max_chars_per_snippet,
                    ),
                    "thinking_text": pred.thinking_text if pred else "",
                    "raw_output": pred.raw_content if pred else "",
                    "finish_reason": pred.finish_reason if pred else "",
                    "temperature": pred.temperature if pred else cfg.temperature,
                    "model": pred.model if pred else (cfg.model or ""),
                    "error_text": pred.error_text if pred else "",
                    "retry_count": pred.retry_count if pred else None,
                }
            )

    return rows


def build_doc_metrics_rows(per_doc: Dict[str, Metrics]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for doc_id, m in sorted(per_doc.items(), key=lambda x: x[0]):
        rows.append(
            {
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
            }
        )
    return rows


def run_evaluation_from_preds(
    *,
    cfg: MainConfig,
    test_paths: List[Path],
    preds: List[PredRecord],
) -> Tuple[Dict[str, Metrics], Metrics]:
    gold = map_gold_labels(cfg.gold_path)
    if not gold:
        raise RuntimeError(f"No gold yes/no labels found in: {cfg.gold_path}")

    pred_map: Dict[str, YesNo] = {}
    for p in preds:
        if p.qtype == "yesno" and p.qid:
            pred_map[p.qid] = p.pred_exact

    per_doc: Dict[str, Metrics] = {}
    union_ids: Set[str] = set()

    for tp in test_paths:
        did = derive_doc_id_from_phase_filename(tp) or tp.stem
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


def main() -> None:
    cfg = MainConfig()

    if not cfg.gold_path.exists():
        raise SystemExit(f"Gold file not found: {cfg.gold_path}")

    candidate_ids = discover_existing_doc_ids(cfg.test_dir) if cfg.doc_ids is None else cfg.doc_ids
    test_paths = resolve_test_paths(cfg.test_dir, candidate_ids)
    if not test_paths:
        raise SystemExit("No test files found. Check test_dir and doc_ids in MainConfig.")

    prompt_spec = load_prompt_spec(cfg)
    result_paths, all_preds = asyncio.run(run_generation(cfg, test_paths, prompt_spec))

    per_doc_metrics, overall_metrics = run_evaluation_from_preds(cfg=cfg, test_paths=test_paths, preds=all_preds)
    for did in sorted(per_doc_metrics.keys()):
        print_metrics(f"DOC {did}", per_doc_metrics[did])
    print_metrics("OVERALL", overall_metrics)

    if cfg.wandb:
        wb = WandbLogger(cfg)

        run_config = {
            "model": cfg.model or os.getenv("OPENROUTER_MODEL") or MainConfig().model,
            "temperature": cfg.temperature,
            "topP": cfg.topP,
            "topK": cfg.topK,
            "presencePenalty": cfg.presencePenalty,
            "frequencyPenalty": cfg.frequencyPenalty,
            "thinkingLevel": cfg.thinkingLevel,
            "concurrency": cfg.concurrency,
            "request_timeout_s": cfg.request_timeout_s,
            "max_retries": cfg.max_retries,
            "backoff_base_s": cfg.backoff_base_s,
            "backoff_cap_s": cfg.backoff_cap_s,
            "only_yesno": cfg.only_yesno,
            "max_chars_per_snippet": cfg.max_chars_per_snippet,
            "missing_as": cfg.missing_as,
            "test_dir": str(cfg.test_dir),
            "test_results_dir": str(cfg.test_results_dir),
            "gold_path": str(cfg.gold_path),
            "docs": candidate_ids,
            "snippet_rerank_enabled": cfg.snippet_rerank_enabled,
            "snippet_embedding_model": cfg.snippet_embedding_model,
            "top_k_snippets": cfg.top_k_snippets,
            "prompt_include_snippet_scores": cfg.prompt_include_snippet_scores,
            "prompt_score_decimals": cfg.prompt_score_decimals,
        }

        wb.start(run_config)
        wb.log_prompt(system_prompt=prompt_spec.system, user_template=prompt_spec.user_template)
        wb.log_questions_table(
            build_questions_rows(cfg=cfg, test_paths=test_paths, preds=all_preds, prompt_spec=prompt_spec),
            max_text_chars=cfg.wandb_table_max_chars,
            table_name="questions",
        )
        wb.log_doc_metrics_table(build_doc_metrics_rows(per_doc_metrics), table_name="per_doc_metrics")

        overall_dict = metrics_to_dict(overall_metrics)
        keep_keys = [
            "accuracy",
            "maF1",
            "precision_yes",
            "recall_yes",
            "f1_yes",
            "precision_no",
            "recall_no",
            "f1_no",
        ]
        wb.log_overall_metrics({k: overall_dict[k] for k in keep_keys if k in overall_dict})
        wb.finish()

    print("\n[main.py] Wrote result files:")
    for rp in result_paths:
        print(" -", rp)


if __name__ == "__main__":
    main()