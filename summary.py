from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None  # type: ignore

DEFAULT_SYSTEM_PROMPT = """You are an expert in the medical texts summarization. Answer the given question with a single paragraph text and your answer should be based strictly on the provided context snippets. You should generate your response in at most 2-3 sentences (30-50 words).
"""

DEFAULT_USER_TEMPLATE = """### INSTRUCTIONS:
1. Read the through evaluation rubric and understand the goal
2. Read the question and snippets carefully.
3. Answer using only the provided snippets.
4. Produce a single-paragraph summary in at most 2-3 sentences.
5. Target 30-50 words.
6. Do not use outside knowledge.
7. Do not add bullets, headers, or explanations.

### EVALUATION RUBRIC
1-5 points per each
#Information Recall : All the necessary information is reported.
#Information Precision : No irrelevant information is reported.
#Information Repetition : The answer does not repeat the same information multiple times.
#Readability : The answer is easily readable and fluent.

### QUESTION:
{q_body}

### SNIPPETS:
{snippets_block}

### SUMMARY:
"""

QuestionType = Literal["summary"]
GenerationStatus = Literal["ok", "api_error", "timeout", "parse_error"]


@dataclass(frozen=True)
class MainConfig:
    wandb_run_name: Optional[str] = "14b_04-5"
    model: Optional[str] = "google/gemini-3.1-pro-preview"
    doc_ids: Optional[List[str]] = field(default_factory=lambda: ["14b_04"])

    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    test_dir: Path = Path("data/testData")
    test_results_dir: Path = Path("data/testResults")
    temperature: float = 0.0
    topP: Optional[float] = None
    topK: Optional[int] = None
    presencePenalty: Optional[float] = None
    frequencyPenalty: Optional[float] = None
    thinkingLevel: Optional[str] = None

    maxOutputTokens: int = 8192

    request_timeout_s: float = 1000.0
    concurrency: int = 25

    max_retries: int = 2
    backoff_base_s: float = 0.8
    backoff_cap_s: float = 20.0

    only_summary: bool = True
    max_chars_per_snippet: Optional[int] = None

    system_prompt_path: Optional[Path] = None
    user_template_path: Optional[Path] = None

    wandb: bool = True
    wandb_project: str = "bioasq-summary"
    wandb_entity: Optional[str] = "bioasq"
    wandb_group: Optional[str] = "CSHS"
    wandb_mode: Optional[str] = "online"
    wandb_table_max_chars: int = 100000


@dataclass(frozen=True)
class PromptSpec:
    system: str = DEFAULT_SYSTEM_PROMPT
    user_template: str = DEFAULT_USER_TEMPLATE


@dataclass
class PredRecord:
    qid: str
    qtype: str
    body: str
    snippet_count: int

    pred_ideal: str
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


def normalize_qtype(qtype: str) -> str:
    return (qtype or "").strip().lower()


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



def collect_snippet_texts(q: dict) -> List[str]:
    snippets = q.get("snippets", [])
    texts: List[str] = []
    for snippet in snippets:
        if not isinstance(snippet, dict):
            continue
        text = snippet.get("text")
        if text is None:
            continue
        texts.append(str(text))
    return texts



def build_user_prompt_for_question(
    q: dict,
    *,
    prompt_spec: PromptSpec,
    max_chars_per_snippet: Optional[int],
) -> str:
    q_body = str(q.get("body", "")).strip()
    snippet_texts = collect_snippet_texts(q)

    parts: List[str] = []
    for idx, txt in enumerate(snippet_texts, start=1):
        if max_chars_per_snippet is not None and len(txt) > max_chars_per_snippet:
            txt = txt[:max_chars_per_snippet].rstrip() + "…"
        parts.append(f"[{idx}]: {txt}")

    snippets_block = "\n".join(parts) if parts else "(No snippets provided.)"

    return prompt_spec.user_template.format(
        q_body=q_body,
        snippets_block=snippets_block,
    ).strip()



def load_prompt_spec(cfg: MainConfig) -> PromptSpec:
    spec = PromptSpec()
    system = spec.system
    user_template = spec.user_template

    if cfg.system_prompt_path and cfg.system_prompt_path.exists():
        system = cfg.system_prompt_path.read_text(encoding="utf-8").strip()
    if cfg.user_template_path and cfg.user_template_path.exists():
        user_template = cfg.user_template_path.read_text(encoding="utf-8").strip()

    return PromptSpec(system=system, user_template=user_template)



def _normalize_summary_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n")]
    paragraphs = [p for p in paragraphs if p]
    return " ".join(paragraphs).strip()


class PromptBuilder:
    def __init__(self, spec: PromptSpec, cfg: MainConfig):
        self.spec = spec
        self.cfg = cfg

    def build_messages(self, q: dict) -> List[dict]:
        return [
            {"role": "system", "content": self.spec.system.strip()},
            {
                "role": "user",
                "content": build_user_prompt_for_question(
                    q,
                    prompt_spec=self.spec,
                    max_chars_per_snippet=self.cfg.max_chars_per_snippet,
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
        snippet_count = len(collect_snippet_texts(q))

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

        pred_ideal = ""
        if status == "ok":
            pred_ideal = _normalize_summary_text(content)
            if not pred_ideal:
                status = "parse_error"

        return PredRecord(
            qid=qid,
            qtype=qtype,
            body=body,
            snippet_count=snippet_count,
            pred_ideal=pred_ideal,
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
            if self.cfg.only_summary and qt != "summary":
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

            if (not self.cfg.only_summary) or (qt == "summary"):
                p = pred_by_id.get(qid)
                q_out["ideal_answer"] = p.pred_ideal if p else ""

            if qt == "summary":
                q_out.pop("exact_answer", None)

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
    prompt_builder = PromptBuilder(prompt_spec, runtime_cfg)

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
            "generation_status",
            "system_prompt",
            "user_prompt",
            "thinking_text",
            "generated_ideal_answer",
            "raw_output",
            "finish_reason",
            "latency_ms",
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
                r.get("generation_status", ""),
                self._truncate(r.get("system_prompt", ""), max_text_chars),
                self._truncate(r.get("user_prompt", ""), max_text_chars),
                self._truncate(r.get("thinking_text", ""), max_text_chars),
                self._truncate(r.get("generated_ideal_answer", ""), max_text_chars),
                self._truncate(r.get("raw_output", ""), max_text_chars),
                r.get("finish_reason", ""),
                int(r.get("latency_ms", 0) or 0),
                r.get("temperature", None),
                r.get("model", ""),
                self._truncate(r.get("error_text", ""), max_text_chars),
                r.get("retry_count", None),
            )

        wandb.log({table_name: table})

    def log_run_summary(self, rows: Sequence[Dict[str, Any]]) -> None:
        if not self._run or wandb is None:
            return
        total = len(rows)
        ok = sum(1 for r in rows if r.get("generation_status") == "ok")
        parse_error = sum(1 for r in rows if r.get("generation_status") == "parse_error")
        api_error = sum(1 for r in rows if r.get("generation_status") == "api_error")
        timeout = sum(1 for r in rows if r.get("generation_status") == "timeout")
        avg_latency_ms = (
            sum(int(r.get("latency_ms", 0) or 0) for r in rows) / total if total else 0.0
        )
        wandb.log(
            {
                "run/total_questions": total,
                "run/ok": ok,
                "run/parse_error": parse_error,
                "run/api_error": api_error,
                "run/timeout": timeout,
                "run/avg_latency_ms": avg_latency_ms,
            }
        )


async def run_generation(
    cfg: MainConfig,
    test_paths: List[Path],
    prompt_spec: PromptSpec,
) -> Tuple[List[Path], List[PredRecord]]:
    return await generate_for_files(test_paths, cfg=cfg, prompt_spec=prompt_spec)



def build_questions_rows(
    *,
    cfg: MainConfig,
    test_paths: List[Path],
    preds: List[PredRecord],
    prompt_spec: PromptSpec,
) -> List[Dict[str, Any]]:
    pred_by_id: Dict[str, PredRecord] = {p.qid: p for p in preds if p.qid}
    rows: List[Dict[str, Any]] = []

    for tp in test_paths:
        did = derive_doc_id_from_phase_filename(tp) or tp.stem
        for q in load_questions(tp):
            if normalize_qtype(str(q.get("type", ""))) != "summary":
                continue
            qid = str(q.get("id", "")).strip()
            if not qid:
                continue

            pred = pred_by_id.get(qid)

            rows.append(
                {
                    "doc_id": did,
                    "id": qid,
                    "body": str(q.get("body", "")).strip(),
                    "snippet_count": pred.snippet_count if pred else len(collect_snippet_texts(q)),
                    "generation_status": pred.generation_status if pred else "api_error",
                    "system_prompt": prompt_spec.system,
                    "user_prompt": build_user_prompt_for_question(
                        q,
                        prompt_spec=prompt_spec,
                        max_chars_per_snippet=cfg.max_chars_per_snippet,
                    ),
                    "thinking_text": pred.thinking_text if pred else "",
                    "generated_ideal_answer": pred.pred_ideal if pred else "",
                    "raw_output": pred.raw_content if pred else "",
                    "finish_reason": pred.finish_reason if pred else "",
                    "latency_ms": pred.latency_ms if pred else 0,
                    "temperature": pred.temperature if pred else cfg.temperature,
                    "model": pred.model if pred else (cfg.model or ""),
                    "error_text": pred.error_text if pred else "",
                    "retry_count": pred.retry_count if pred else None,
                }
            )

    return rows



def main() -> None:
    cfg = MainConfig()

    candidate_ids = discover_existing_doc_ids(cfg.test_dir) if cfg.doc_ids is None else cfg.doc_ids
    test_paths = resolve_test_paths(cfg.test_dir, candidate_ids)
    if not test_paths:
        raise SystemExit("No test files found. Check test_dir and doc_ids in MainConfig.")

    prompt_spec = load_prompt_spec(cfg)
    result_paths, all_preds = asyncio.run(run_generation(cfg, test_paths, prompt_spec))

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
            "only_summary": cfg.only_summary,
            "max_chars_per_snippet": cfg.max_chars_per_snippet,
            "test_dir": str(cfg.test_dir),
            "test_results_dir": str(cfg.test_results_dir),
            "docs": candidate_ids,
        }

        wb.start(run_config)
        wb.log_prompt(system_prompt=prompt_spec.system, user_template=prompt_spec.user_template)
        question_rows = build_questions_rows(cfg=cfg, test_paths=test_paths, preds=all_preds, prompt_spec=prompt_spec)
        wb.log_questions_table(
            question_rows,
            max_text_chars=cfg.wandb_table_max_chars,
            table_name="questions",
        )
        wb.log_run_summary(question_rows)
        wb.finish()

    print("\n[main_summary.py] Wrote result files:")
    for rp in result_paths:
        print(" -", rp)


if __name__ == "__main__":
    main()
