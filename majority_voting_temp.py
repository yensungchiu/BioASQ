"""
majority.py — Majority Voting Experiments

mode="random_order"  → 3a: shuffle snippets N 次，majority vote（temperature 固定 0）
mode="temperature"   → 3b: temperature > 0，跑 N 次，majority vote

修正項目：
  1. sleep 移到 sem 外面，retry 時不佔用 semaphore slot
  2. 每票獨立 rng（seed + vote_index），確保可重現
  3. 完整 W&B logging（questions table / per-doc table / overall metrics）
  4. per-doc metrics 分開顯示
  5. backoff_cap_s
  6. 詳細 SingleVote（latency / error_text / status）
  7. normalize_ideal_answer 正確處理 list
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None  # type: ignore

YesNo = Literal["yes", "no"]
GenerationStatus = Literal["ok", "api_error", "timeout", "parse_error"]

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


@dataclass
class ExpConfig:
    wandb_run_name: Optional[str] = "D-tmp1"

    # ── 實驗模式 ──────────────────────────────────────────────
    mode: str = "temperature"   # "random_order" | "temperature"
    n_votes: int = 3             # 每題投票幾次（建議奇數，避免平票）
    temperature: float = 1    # mode="temperature" 才用；mode="random_order" 自動為 0.0

    # ── 模型 ──────────────────────────────────────────────────
    model: str = "anthropic/claude-opus-4.6"

    # ── 資料路徑 ──────────────────────────────────────────────
    doc_ids: List[str] = field(default_factory=lambda: ["12b_01", "12b_02", "12b_03", "12b_04", "13b_01", "13b_02", "13b_03", "13b_04", "11b_03", "11b_04"])
    test_dir: Path = Path("data/testData")
    test_results_dir: Path = Path("data/testResults")
    gold_path: Path = Path("data/predictions/training13b.json")

    # ── API ───────────────────────────────────────────────────
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    maxOutputTokens: int = 8192
    request_timeout_s: float = 1000.0
    concurrency: int = 10        # 同時最多幾個 API 呼叫（跨所有題目所有票）
    max_retries: int = 2
    backoff_base_s: float = 0.8
    backoff_cap_s: float = 20.0

    # ── 其他 ──────────────────────────────────────────────────
    max_chars_per_snippet: Optional[int] = None
    missing_as: YesNo = "no"
    random_seed: int = 42        # 每票 seed = random_seed + vote_index，確保可重現

    # ── W&B ───────────────────────────────────────────────────
    wandb: bool = True
    wandb_project: str = "yesno"
    wandb_entity: Optional[str] = "bioasq"
    wandb_group: Optional[str] = "CSHS"
    wandb_mode: Optional[str] = "online"
    wandb_table_max_chars: int = 100000


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_questions(path: Path) -> List[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return [q for q in obj.get("questions", []) if isinstance(q, dict)]


def safe_json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize_qtype(qt: str) -> str:
    qt = (qt or "").strip().lower()
    return "yesno" if qt in {"yes/no", "yesno", "yes-no", "yn"} else qt


def coerce_yesno(val: Any) -> Optional[YesNo]:
    if isinstance(val, str):
        s = val.strip().lower().strip(" .,:;!\"'")
        if s == "yes":
            return "yes"
        if s == "no":
            return "no"
    return None


def _from_text(text: str) -> Optional[YesNo]:
    """解析 yes/no：優先看最後一行，fallback 用 regex。"""
    if not text:
        return None
    for line in reversed(text.strip().splitlines()):
        w = line.strip().lower().strip(" .,:;!\"'")
        if w in ("yes", "no"):
            return w  # type: ignore
    m = re.findall(r"\b(yes|no)\b", text.lower())
    return m[-1] if m else None  # type: ignore


def derive_doc_id(path: Path) -> str:
    m = re.search(r"(?:phaseB_|PhaseB_)?(\d{1,2}b_\d{2})", path.stem)
    return m.group(1) if m else path.stem


def collect_snippets(q: dict) -> List[str]:
    return [s.get("text", "") for s in q.get("snippets", [])
            if isinstance(s, dict) and s.get("text")]


def ids_in_file(path: Path) -> List[str]:
    return [
        str(q.get("id", "")).strip()
        for q in load_questions(path)
        if normalize_qtype(str(q.get("type", ""))) == "yesno"
        and str(q.get("id", "")).strip()
    ]


def normalize_ideal_answer(value: Any) -> str:
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


def map_gold(p: Path) -> Dict[str, YesNo]:
    out: Dict[str, YesNo] = {}
    for q in load_questions(p):
        if normalize_qtype(str(q.get("type", ""))) != "yesno":
            continue
        qid = str(q.get("id", "")).strip()
        label = coerce_yesno(q.get("exact_answer"))
        if qid and label:
            out[qid] = label
    return out


def build_block(
    snippets: List[str],
    max_chars: Optional[int],
    shuffle: bool,
    rng: random.Random,
) -> str:
    texts = list(snippets)
    if shuffle:
        rng.shuffle(texts)
    parts = []
    for i, txt in enumerate(texts, 1):
        if max_chars and len(txt) > max_chars:
            txt = txt[:max_chars].rstrip() + "…"
        parts.append(f"[{i}]: {txt}")
    return "\n".join(parts) if parts else "(No snippets provided.)"


# ─── API client ───────────────────────────────────────────────────────────────

class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str, timeout_s: float):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)

    async def __aenter__(self) -> "OpenRouterClient":
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, *_) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def complete(self, payload: dict) -> str:
        assert self._session is not None, "Client not started"
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with self._session.post(url, headers=headers, json=payload) as resp:
            raw = await resp.text()
            if resp.status >= 400:
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info, history=resp.history,
                    status=resp.status, message=raw[:500], headers=resp.headers,
                )
            choices = json.loads(raw).get("choices") or []
            return str((choices[0].get("message") or {}).get("content") or "")


# ─── voting ───────────────────────────────────────────────────────────────────

@dataclass
class SingleVote:
    vote_index: int
    result: Optional[YesNo]
    raw_content: str
    status: GenerationStatus
    latency_ms: int
    error_text: str


@dataclass
class VoteRecord:
    qid: str
    body: str
    snippet_count: int
    votes: List[Optional[YesNo]]   # 每票原始結果（None = 解析失敗）
    effective_votes: List[YesNo]   # None 替換為 missing_as
    final: YesNo                   # majority 結果
    all_agree: bool                # 所有票一致
    single_votes: List[SingleVote] # 每票詳細資訊


async def one_vote(
    client: OpenRouterClient,
    cfg: ExpConfig,
    q: dict,
    sem: asyncio.Semaphore,
    vote_index: int,
) -> SingleVote:
    """
    單次 API 呼叫。
    - 每票獨立 rng（seed + vote_index），shuffle 結果可重現
    - sleep 在 sem 外面，retry 時不佔用 semaphore slot
    """
    temp = 0.0 if cfg.mode == "random_order" else cfg.temperature

    # 每票獨立 seed：相同 vote_index 永遠產生相同 shuffle
    rng = random.Random(cfg.random_seed + vote_index)
    block = build_block(
        collect_snippets(q),
        cfg.max_chars_per_snippet,
        shuffle=(cfg.mode == "random_order"),
        rng=rng,
    )

    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": DEFAULT_USER_TEMPLATE.format(
                q_body=str(q.get("body", "")).strip(),
                snippets_block=block,
            ).strip()},
        ],
        "temperature": temp,
        "max_tokens": cfg.maxOutputTokens,
    }

    last_err = ""
    last_status: GenerationStatus = "api_error"
    start_all = time.perf_counter()

    for attempt in range(cfg.max_retries + 1):
        try:
            t0 = time.perf_counter()
            async with sem:                         # ← 只在 API 呼叫時持有 sem
                raw = await client.complete(payload)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            yn = _from_text(raw)
            if yn is None:
                return SingleVote(
                    vote_index=vote_index, result=None, raw_content=raw,
                    status="parse_error", latency_ms=latency_ms,
                    error_text="parse_error: no yes/no found",
                )
            return SingleVote(
                vote_index=vote_index, result=yn, raw_content=raw,
                status="ok", latency_ms=latency_ms, error_text="",
            )

        except asyncio.TimeoutError:
            last_status = "timeout"
            last_err = "TIMEOUT"
        except aiohttp.ClientResponseError as e:
            last_status = "api_error"
            last_err = f"HTTP {e.status}: {str(e)[:300]}"
        except aiohttp.ClientError as e:
            last_status = "api_error"
            last_err = f"CLIENT_ERROR: {repr(e)[:300]}"
        except Exception as e:
            last_status = "api_error"
            last_err = f"UNEXPECTED: {repr(e)[:300]}"

        if attempt >= cfg.max_retries:
            break
        # ← sleep 在 sem 外面，不佔 slot
        sleep_s = min(cfg.backoff_cap_s, cfg.backoff_base_s * (2 ** attempt))
        await asyncio.sleep(sleep_s)

    latency_ms = int((time.perf_counter() - start_all) * 1000)
    return SingleVote(
        vote_index=vote_index, result=None, raw_content="",
        status=last_status, latency_ms=latency_ms, error_text=last_err,
    )


async def vote_q(
    client: OpenRouterClient,
    cfg: ExpConfig,
    q: dict,
    sem: asyncio.Semaphore,
) -> VoteRecord:
    """一道題的所有 N 票並發執行，最後 majority vote。"""
    qid = str(q.get("id", "")).strip()
    body = str(q.get("body", "")).strip()
    snippet_count = len(collect_snippets(q))

    single_votes: List[SingleVote] = list(await asyncio.gather(*[
        one_vote(client, cfg, q, sem, vote_index=i)
        for i in range(cfg.n_votes)
    ]))

    raw_votes: List[Optional[YesNo]] = [sv.result for sv in single_votes]
    effective: List[YesNo] = [
        v if v is not None else cfg.missing_as for v in raw_votes
    ]

    # majority vote，平票時 yes 優先（Counter.most_common 在 Python 3.7+ 保留插入順序）
    counter = Counter(effective)
    if counter["yes"] >= counter["no"]:
        final: YesNo = "yes" if counter["yes"] > 0 else cfg.missing_as
    else:
        final = "no"
    # 更精確：直接取最多票的
    final = counter.most_common(1)[0][0]

    return VoteRecord(
        qid=qid,
        body=body,
        snippet_count=snippet_count,
        votes=raw_votes,
        effective_votes=effective,
        final=final,
        all_agree=(len(set(effective)) == 1),
        single_votes=single_votes,
    )


# ─── metrics ──────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    n: int
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
    agreement_rate: float   # majority voting 特有：全票一致比例


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2.0 * p * r, p + r)


def compute_metrics(
    gold: Dict[str, YesNo],
    records: List[VoteRecord],
    *,
    missing_as: YesNo = "no",
) -> Metrics:
    pred = {r.qid: r.final for r in records}
    agree_map = {r.qid: r.all_agree for r in records}

    tp = fp = fn_c = tn = 0
    for qid, g in gold.items():
        p = pred.get(qid, missing_as)
        if g == "yes" and p == "yes":
            tp += 1
        elif g == "no" and p == "yes":
            fp += 1
        elif g == "yes" and p == "no":
            fn_c += 1
        else:
            tn += 1

    n = len(gold)
    acc = _safe_div(tp + tn, n)
    p_yes = _safe_div(tp, tp + fp)
    r_yes = _safe_div(tp, tp + fn_c)
    f1_yes = _f1(p_yes, r_yes)
    p_no = _safe_div(tn, tn + fn_c)
    r_no = _safe_div(tn, tn + fp)
    f1_no = _f1(p_no, r_no)
    agree_rate = _safe_div(
        sum(1 for qid in gold if agree_map.get(qid, False)), n
    )

    return Metrics(
        n=n, tp=tp, fp=fp, fn=fn_c, tn=tn,
        accuracy=acc,
        precision_yes=p_yes, recall_yes=r_yes, f1_yes=f1_yes,
        precision_no=p_no, recall_no=r_no, f1_no=f1_no,
        maF1=0.5 * (f1_yes + f1_no),
        agreement_rate=agree_rate,
    )


def print_metrics(title: str, m: Metrics, cfg: ExpConfig) -> None:
    eff_temp = cfg.temperature if cfg.mode == "temperature" else 0.0
    print(f"\n=== {title} | mode={cfg.mode} | n_votes={cfg.n_votes} | temp={eff_temp} ===")
    print(f"N={m.n}  accuracy={m.accuracy:.4f}  maF1={m.maF1:.4f}")
    print(f"Confusion: TP={m.tp} FP={m.fp} FN={m.fn} TN={m.tn}")
    print(f"YES: P={m.precision_yes:.4f} R={m.recall_yes:.4f} F1={m.f1_yes:.4f}")
    print(f"NO : P={m.precision_no:.4f} R={m.recall_no:.4f} F1={m.f1_no:.4f}")
    print(f"Agreement rate (all votes same): {m.agreement_rate:.4f}")


# ─── W&B ──────────────────────────────────────────────────────────────────────

class WandbLogger:
    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self._run = None

    def start(self, run_config: Dict[str, Any]) -> None:
        if not self.cfg.wandb or wandb is None:
            return
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

    def _trunc(self, v: Any, max_c: int) -> str:
        t = "" if v is None else str(v)
        return t if len(t) <= max_c else t[:max(0, max_c - 1)] + "…"

    def log_prompt(self, system: str, user_template: str) -> None:
        if not self._run or wandb is None:
            return
        wandb.config.update(
            {"prompt/system": system, "prompt/user_template": user_template},
            allow_val_change=True,
        )

    def log_questions_table(
        self,
        gold_map: Dict[str, YesNo],
        all_records: List[VoteRecord],
        doc_id_map: Dict[str, str],
        cfg: ExpConfig,
    ) -> None:
        if not self._run or wandb is None:
            return
        mc = cfg.wandb_table_max_chars
        eff_temp = cfg.temperature if cfg.mode == "temperature" else 0.0
        columns = [
            "doc_id", "id", "body", "snippet_count",
            "gold_answer", "final_answer", "correct",
            "votes", "all_agree", "yes_count", "no_count",
            "mode", "n_votes", "temperature", "model",
        ]
        table = wandb.Table(columns=columns)
        for r in all_records:
            gold = gold_map.get(r.qid)
            yes_c = r.effective_votes.count("yes")
            no_c = r.effective_votes.count("no")
            table.add_data(
                doc_id_map.get(r.qid, ""),
                r.qid,
                self._trunc(r.body, mc),
                r.snippet_count,
                gold,
                r.final,
                (gold is not None) and (r.final == gold),
                str(r.effective_votes),
                r.all_agree,
                yes_c,
                no_c,
                cfg.mode,
                cfg.n_votes,
                eff_temp,
                cfg.model,
            )
        wandb.log({"questions": table})

    def log_per_doc_table(self, per_doc: Dict[str, Metrics]) -> None:
        if not self._run or wandb is None:
            return
        columns = [
            "doc_id", "n", "accuracy", "maF1",
            "f1_yes", "f1_no", "agreement_rate",
        ]
        table = wandb.Table(columns=columns)
        for doc_id, m in sorted(per_doc.items()):
            table.add_data(
                doc_id, m.n, m.accuracy, m.maF1,
                m.f1_yes, m.f1_no, m.agreement_rate,
            )
        wandb.log({"per_doc_metrics": table})

    def log_overall_metrics(self, m: Metrics) -> None:
        if not self._run or wandb is None:
            return
        wandb.log({
            "overall/accuracy": m.accuracy,
            "overall/maF1": m.maF1,
            "overall/f1_yes": m.f1_yes,
            "overall/f1_no": m.f1_no,
            "overall/precision_yes": m.precision_yes,
            "overall/recall_yes": m.recall_yes,
            "overall/precision_no": m.precision_no,
            "overall/recall_no": m.recall_no,
            "overall/agreement_rate": m.agreement_rate,
        })


# ─── file processing ──────────────────────────────────────────────────────────

async def process_file(
    path: Path,
    cfg: ExpConfig,
    client: OpenRouterClient,
    sem: asyncio.Semaphore,
) -> Tuple[Path, List[VoteRecord]]:
    questions = load_questions(path)
    yesno_qs = [q for q in questions
                if normalize_qtype(str(q.get("type", ""))) == "yesno"]

    tasks = [asyncio.create_task(vote_q(client, cfg, q, sem)) for q in yesno_qs]
    records: List[VoteRecord] = []
    pbar = tqdm(total=len(tasks), desc=path.name)
    for coro in asyncio.as_completed(tasks):
        records.append(await coro)
        pbar.update(1)
    pbar.close()

    by_id = {r.qid: r for r in records}
    out_qs = []
    for q in questions:
        q_out = dict(q)
        qid = str(q_out.get("id", "")).strip()
        qt = normalize_qtype(str(q_out.get("type", "")))
        q_out["ideal_answer"] = normalize_ideal_answer(q_out.get("ideal_answer"))
        if qt == "yesno":
            rec = by_id.get(qid)
            q_out["exact_answer"] = rec.final if rec else cfg.missing_as
        out_qs.append(q_out)

    doc_id = derive_doc_id(path)
    out_path = (
        cfg.test_results_dir
        / f"result_{doc_id}_{cfg.mode}_n{cfg.n_votes}.json"
    )
    safe_json_dump(out_path, {"questions": out_qs})
    return out_path, records


# ─── main ─────────────────────────────────────────────────────────────────────

async def main_async(cfg: ExpConfig) -> None:
    load_dotenv()
    api_key = os.getenv(cfg.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing {cfg.api_key_env} in .env")

    resolved_model = os.getenv("OPENROUTER_MODEL", cfg.model).strip() or cfg.model
    resolved_base_url = (
        os.getenv("OPENROUTER_BASE_URL", cfg.base_url).strip() or cfg.base_url
    )

    test_paths = [
        cfg.test_dir / f"phaseB_{did}.json"
        for did in cfg.doc_ids
        if (cfg.test_dir / f"phaseB_{did}.json").exists()
    ]
    if not test_paths:
        raise SystemExit("No test files found. Check test_dir and doc_ids.")

    gold = map_gold(cfg.gold_path) if cfg.gold_path.exists() else {}
    cfg.test_results_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[VoteRecord] = []
    doc_id_map: Dict[str, str] = {}   # qid → doc_id（for W&B table）

    # 全域共用一個 semaphore，控制總 API 呼叫並發數
    sem = asyncio.Semaphore(cfg.concurrency)

    async with OpenRouterClient(api_key, resolved_base_url, cfg.request_timeout_s) as client:
        for tp in test_paths:
            out_path, recs = await process_file(tp, cfg, client, sem)
            doc_id = derive_doc_id(tp)
            for r in recs:
                doc_id_map[r.qid] = doc_id
            all_records.extend(recs)
            print(f"[OK] {out_path}")

    # ── metrics ──────────────────────────────────────────────────────────────
    per_doc: Dict[str, Metrics] = {}
    union_ids: Set[str] = set()

    for tp in test_paths:
        doc_id = derive_doc_id(tp)
        ids = [qid for qid in ids_in_file(tp) if qid in gold]
        union_ids |= set(ids)
        doc_recs = [r for r in all_records if r.qid in set(ids)]
        per_doc[doc_id] = compute_metrics(
            {qid: gold[qid] for qid in ids},
            doc_recs,
            missing_as=cfg.missing_as,
        )

    overall = compute_metrics(
        {qid: gold[qid] for qid in union_ids},
        [r for r in all_records if r.qid in union_ids],
        missing_as=cfg.missing_as,
    )

    for doc_id, m in sorted(per_doc.items()):
        print_metrics(f"DOC {doc_id}", m, cfg)
    print_metrics("OVERALL", overall, cfg)

    # ── save vote details json ────────────────────────────────────────────────
    safe_json_dump(
        cfg.test_results_dir / f"vote_details_{cfg.mode}_n{cfg.n_votes}.json",
        [
            {
                "qid": r.qid,
                "votes": r.effective_votes,
                "final": r.final,
                "all_agree": r.all_agree,
                "yes_count": r.effective_votes.count("yes"),
                "no_count": r.effective_votes.count("no"),
                "gold": gold.get(r.qid),
                "correct": r.final == gold.get(r.qid),
            }
            for r in all_records
        ],
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    if cfg.wandb:
        if wandb is None:
            print("[WARN] wandb not installed. Skipping W&B logging.")
        else:
            eff_temp = cfg.temperature if cfg.mode == "temperature" else 0.0
            run_config = {
                "mode": cfg.mode,
                "n_votes": cfg.n_votes,
                "temperature": eff_temp,
                "model": resolved_model,
                "doc_ids": cfg.doc_ids,
                "concurrency": cfg.concurrency,
                "max_retries": cfg.max_retries,
                "backoff_base_s": cfg.backoff_base_s,
                "backoff_cap_s": cfg.backoff_cap_s,
                "random_seed": cfg.random_seed,
                "missing_as": cfg.missing_as,
                "max_chars_per_snippet": cfg.max_chars_per_snippet,
            }
            wb = WandbLogger(cfg)
            wb.start(run_config)
            wb.log_prompt(DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE)
            wb.log_questions_table(gold, all_records, doc_id_map, cfg)
            wb.log_per_doc_table(per_doc)
            wb.log_overall_metrics(overall)
            wb.finish()

    print("\n[majority.py] Done.")


def main() -> None:
    cfg = ExpConfig()
    eff_temp = cfg.temperature if cfg.mode == "temperature" else 0.0
    print(
        f"Mode={cfg.mode} | n_votes={cfg.n_votes} | "
        f"temperature={eff_temp} | model={cfg.model}"
    )
    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()