"""
exp_textgrad.py  —  BioASQ Yes/No Prompt Optimization via TextGrad
===================================================================
Completely standalone. Does NOT import main.py.

All tunable parameters are in the CONFIG BLOCK below (lines ~40-130).
Edit only that section; do not touch anything below the divider line.

Pipeline
--------
1. Load yes/no questions from gold_path  →  split train / val
2. Baseline evaluation (async, concurrent)
3. Each epoch:
   a. Async concurrent inference on train batch  →  build text loss
   b. Yes/No-specific TextGrad gradient step
   c. Optimizer rewrites system_prompt and/or user_template
   d. Async concurrent evaluation on val set
   e. Log everything to W&B
4. Save best prompts + summary JSON

Install
-------
pip install textgrad openai aiohttp python-dotenv tqdm wandb

Run
---
"""

from __future__ import annotations

# ── API ───────────────────────────────────────────────────────────────────
BASE_URL       = "https://openrouter.ai/api/v1"
API_KEY_ENV    = "OPENROUTER_API_KEY"          # env-var name holding your key

# ── Models ────────────────────────────────────────────────────────────────
INFERENCE_MODEL         = "anthropic/claude-sonnet-4.6"   # answers yes/no questions
GRADIENT_ENGINE_MODEL   = "anthropic/claude-sonnet-4.6"   # rewrites prompts

# ── Initial prompts (TextGrad starts from these and improves them) ────────
INITIAL_SYSTEM_PROMPT = """You are a critical biomedical reviewer. Answer Yes/No questions only when the provided snippets contain direct and unambiguous evidence. If snippets conflict with each other, are only tangentially related to the question, or do not clearly support a yes answer, answer no. Your response must consist of exactly and only the single word "yes" or "no". Do not include any reasoning, explanation, punctuation, or extra characters.
"""

INITIAL_USER_TEMPLATE = """### INSTRUCTIONS:
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
# NOTE: {q_body} and {snippets_block} are required placeholders — do not remove.

# ── Data ──────────────────────────────────────────────────────────────────
GOLD_PATH      = "data/predictions/training13b.json"   # labelled yes/no questions


WANDB_RUN_NAME   = "textgrad"            # None = auto-generated
# ── Optimization hyperparameters ──────────────────────────────────────────
N_EPOCHS            = 0     # number of optimization iterations
TRAIN_BATCH_SIZE    = 20    # questions sampled per gradient step
VAL_SIZE            = 80    # questions held out for validation each epoch
RANDOM_SEED         = 42

# ── What to optimize ──────────────────────────────────────────────────────
OPTIMIZE_SYSTEM_PROMPT   = False   # optimize the system prompt
OPTIMIZE_USER_TEMPLATE   = False   # optimize the user message template

# ── Inference settings ────────────────────────────────────────────────────
INFERENCE_TEMPERATURE    = 0.0    # 0.0 = deterministic answers
INFERENCE_MAX_TOKENS     = 1024
INFERENCE_TIMEOUT_S      = 120.0
INFERENCE_CONCURRENCY    = 20     # max simultaneous async API calls

# ── Gradient engine settings ──────────────────────────────────────────────
GRADIENT_TEMPERATURE     = 0.7    # higher = more diverse critiques
GRADIENT_MAX_TOKENS      = 2048

# ── Output ────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "textgrad_output"
LOG_FILE     = "optimization_log.jsonl"

# ── Weights & Biases ──────────────────────────────────────────────────────
WANDB_ENABLED    = True
WANDB_PROJECT    = "yesno"
WANDB_ENTITY     = "bioasq"        # your W&B team/username
WANDB_GROUP      = "textgrad"
WANDB_MODE       = "online"        # "online" | "offline" | "disabled"

import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import aiohttp
from dotenv import load_dotenv

load_dotenv()

# ── TextGrad ──────────────────────────────────────────────────────────────
try:
    import textgrad as tg
    from textgrad.variable import Variable
    from textgrad.optimizer import TextualGradientDescent
except ImportError:
    sys.exit("[ERROR] Run: pip install textgrad")

# ── OpenAI SDK (OpenRouter-compatible) ────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    sys.exit("[ERROR] Run: pip install openai>=1.0")

# ── W&B ───────────────────────────────────────────────────────────────────
try:
    import wandb as _wandb
    _WANDB_OK = True
except ImportError:
    _wandb = None       # type: ignore
    _WANDB_OK = False


# ══════════════════════════════════════════════════════════════════════════
#  Types
# ══════════════════════════════════════════════════════════════════════════
YesNo = Literal["yes", "no"]


# ══════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Metrics:
    n:            int
    missing_pred: int
    tp: int; fp: int; fn: int; tn: int
    accuracy:      float
    precision_yes: float; recall_yes: float; f1_yes: float
    precision_no:  float; recall_no:  float; f1_no:  float
    maF1: float


# ══════════════════════════════════════════════════════════════════════════
#  Core utilities  (self-contained — no import from main.py)
# ══════════════════════════════════════════════════════════════════════════
def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0

def _f1(p: float, r: float) -> float:
    return _safe_div(2.0 * p * r, p + r)

def normalize_qtype(qtype: str) -> str:
    qt = (qtype or "").strip().lower()
    return "yesno" if qt in {"yes/no", "yesno", "yes-no", "yn"} else qt

def coerce_yesno(val: Any) -> Optional[YesNo]:
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().lower().strip(" .,:;!\"'")
        if s == "yes": return "yes"
        if s == "no":  return "no"
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
    lines = [l.strip().lower() for l in text.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1].strip(" .,:;!\"'")
        if last == "yes": return "yes"
        if last == "no":  return "no"
    matches = re.findall(r"\b(yes|no)\b", text.lower())
    if matches:
        return "yes" if matches[-1] == "yes" else "no"
    return None

def load_questions(path: Path) -> List[dict]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(obj, dict):
        return []
    qs = obj.get("questions")
    if not isinstance(qs, list):
        return []
    return [q for q in qs if isinstance(q, dict)]

def map_gold_labels(path: Path) -> Dict[str, YesNo]:
    out: Dict[str, YesNo] = {}
    for q in load_questions(path):
        if normalize_qtype(str(q.get("type", ""))) != "yesno":
            continue
        qid = str(q.get("id", "")).strip()
        if not qid:
            continue
        label = coerce_yesno(q.get("exact_answer"))
        if label is not None:
            out[qid] = label
    return out

def build_user_content(q: dict, user_template: str) -> str:
    """Fill {q_body} and {snippets_block} into the user template."""
    q_body = str(q.get("body", "")).strip()
    snippets = q.get("snippets", [])
    parts = [f"[{i+1}]: {s['text']}" for i, s in enumerate(snippets) if s.get("text")]
    snippets_block = "\n".join(parts) if parts else "(No snippets provided.)"
    return user_template.format(q_body=q_body, snippets_block=snippets_block).strip()

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
        elif g == "no"  and p == "no":  tn += 1
    n      = len(gold)
    acc    = _safe_div(tp + tn, n)
    p_yes  = _safe_div(tp, tp + fp);  r_yes = _safe_div(tp, tp + fn)
    p_no   = _safe_div(tn, tn + fn);  r_no  = _safe_div(tn, tn + fp)
    f1_yes = _f1(p_yes, r_yes);       f1_no = _f1(p_no, r_no)
    return Metrics(
        n=n, missing_pred=missing,
        tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=acc,
        precision_yes=p_yes, recall_yes=r_yes, f1_yes=f1_yes,
        precision_no=p_no,   recall_no=r_no,   f1_no=f1_no,
        maF1=0.5 * (f1_yes + f1_no),
    )

def _append_log(log_path: Path, record: Dict[str, Any]) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════
#  W&B Logger
# ══════════════════════════════════════════════════════════════════════════
class WandbLogger:
    def __init__(self) -> None:
        self._run = None
        self.enabled = WANDB_ENABLED and _WANDB_OK

    def start(self, run_config: Dict[str, Any]) -> None:
        if not self.enabled:
            if WANDB_ENABLED and not _WANDB_OK:
                print("[WARN] wandb not installed — run: pip install wandb", file=sys.stderr)
            return
        kw: Dict[str, Any] = {
            "project": WANDB_PROJECT,
            "config":  run_config,
            "mode":    WANDB_MODE,
            "group":   WANDB_GROUP,
        }
        if WANDB_ENTITY:    kw["entity"] = WANDB_ENTITY
        if WANDB_RUN_NAME:  kw["name"]   = WANDB_RUN_NAME
        self._run = _wandb.init(**kw)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self._run:
            _wandb.log(metrics, step=step)

    def log_prompt_table(self, history: List[Dict[str, Any]]) -> None:
        if not self.enabled or not self._run:
            return
        cols = ["epoch", "val_maF1", "val_acc", "val_f1_yes", "val_f1_no",
                "train_maF1", "improved", "system_prompt", "user_template"]
        table = _wandb.Table(columns=cols)
        for r in history:
            table.add_data(
                r.get("epoch", 0),
                round(r.get("val_maF1",   0.0), 6),
                round(r.get("val_accuracy",0.0), 6),
                round(r.get("val_f1_yes", 0.0), 6),
                round(r.get("val_f1_no",  0.0), 6),
                round(r.get("train_maF1", 0.0), 6),
                bool(r.get("improved", False)),
                r.get("system_prompt", "")[:4000],
                r.get("user_template", "")[:4000],
            )
        _wandb.log({"prompt_evolution": table})

    def finish(self) -> None:
        if self.enabled and self._run:
            _wandb.finish()
            self._run = None


# ══════════════════════════════════════════════════════════════════════════
#  OpenRouter → TextGrad engine bridge
# ══════════════════════════════════════════════════════════════════════════
class OpenRouterTextGradEngine(tg.engine.EngineLM):
    """Wraps OpenRouter (OpenAI-compatible) as a TextGrad EngineLM.
    Used only for synchronous gradient generation and prompt rewriting."""

    def __init__(self) -> None:
        api_key = os.getenv(API_KEY_ENV, "").strip()
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)

    def generate(self, prompt: str | list, system_prompt: str = "", **kwargs) -> str:
        messages: List[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            messages.extend(prompt)
        try:
            resp = self.client.chat.completions.create(
                model=GRADIENT_ENGINE_MODEL,
                messages=messages,
                temperature=GRADIENT_TEMPERATURE,
                max_tokens=kwargs.get("max_tokens", GRADIENT_MAX_TOKENS),
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [WARN] Gradient engine error: {e}", file=sys.stderr)
            return ""

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
#  Async inference layer
# ══════════════════════════════════════════════════════════════════════════
async def _async_call_once(
    session: aiohttp.ClientSession,
    *,
    api_key: str,
    system: str,
    user_content: str,
) -> str:
    url = BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": INFERENCE_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        "temperature": INFERENCE_TEMPERATURE,
        "max_tokens":  INFERENCE_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    try:
        async with session.post(
            url, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=INFERENCE_TIMEOUT_S),
        ) as resp:
            if resp.status >= 400:
                txt = await resp.text()
                print(f"  [WARN] HTTP {resp.status}: {txt[:200]}", file=sys.stderr)
                return ""
            data = await resp.json()
            choices = data.get("choices") or []
            return str((choices[0].get("message") or {}).get("content") or "") if choices else ""
    except asyncio.TimeoutError:
        print("  [WARN] Inference timeout", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"  [WARN] Async inference error: {e}", file=sys.stderr)
        return ""


async def async_predict_batch(
    questions: List[dict],
    *,
    api_key: str,
    system: str,
    user_template: str,
) -> List[Tuple[str, Optional[YesNo], str]]:
    """Fire all questions concurrently. Returns list of (qid, prediction, raw_text)."""
    sem = asyncio.Semaphore(INFERENCE_CONCURRENCY)

    async def _one(q: dict) -> Tuple[str, Optional[YesNo], str]:
        qid          = str(q.get("id", "")).strip()
        user_content = build_user_content(q, user_template)
        async with sem:
            async with aiohttp.ClientSession() as session:
                raw = await _async_call_once(
                    session, api_key=api_key,
                    system=system, user_content=user_content,
                )
        return qid, _coerce_yesno_from_text(raw), raw

    return list(await asyncio.gather(*[asyncio.create_task(_one(q)) for q in questions]))


# ══════════════════════════════════════════════════════════════════════════
#  Loss builder  (async forward pass over a train batch)
# ══════════════════════════════════════════════════════════════════════════
async def build_loss_async(
    batch: List[dict],
    gold_map: Dict[str, YesNo],
    system: str,
    user_template: str,
    api_key: str,
) -> Tuple[str, float, Dict[str, Any]]:
    """Returns (loss_text, macro_f1, stats_dict)."""
    t0      = time.perf_counter()
    results = await async_predict_batch(batch, api_key=api_key,
                                        system=system, user_template=user_template)
    elapsed = time.perf_counter() - t0

    pred_map: Dict[str, YesNo] = {}
    error_cases: List[str] = []
    correct_cases: List[str] = []
    missed_yes = missed_no = 0

    q_by_id = {str(q.get("id", "")).strip(): q for q in batch}

    for qid, pred, raw in results:
        gold = gold_map.get(qid)
        if gold is None:
            continue
        eff = pred or "no"
        pred_map[qid] = eff

        q = q_by_id.get(qid, {})
        snippet_texts   = [s.get("text", "") for s in q.get("snippets", [])]
        snippet_summary = " | ".join(snippet_texts[:2])[:300]
        q_body          = str(q.get("body", ""))[:200]

        if eff != gold:
            error_cases.append(
                f"  WRONG  QID={qid}\n"
                f"  Question: {q_body}\n"
                f"  Snippets (first 2): {snippet_summary}\n"
                f"  Gold={gold}  Predicted={eff}  raw='{raw[:120].strip()}'\n"
            )
            if gold == "yes": missed_yes += 1
            else:             missed_no  += 1
        else:
            correct_cases.append(f"  OK  QID={qid}  Gold={gold}")

    gold_batch = {qid: gold_map[qid] for qid in pred_map if qid in gold_map}
    m          = compute_metrics(gold_batch, pred_map)

    loss_parts = [
        "=== Batch Evaluation Results ===",
        f"Macro-F1 : {m.maF1:.4f}   Accuracy : {m.accuracy:.4f}",
        f"YES  P={m.precision_yes:.4f}  R={m.recall_yes:.4f}  F1={m.f1_yes:.4f}",
        f"NO   P={m.precision_no:.4f}  R={m.recall_no:.4f}  F1={m.f1_no:.4f}",
        f"Correct={len(correct_cases)}  Wrong={len(error_cases)}  "
        f"missed_yes={missed_yes}  missed_no={missed_no}  Total={len(batch)}",
        f"Elapsed: {elapsed:.1f}s",
        "",
    ]
    if error_cases:
        loss_parts.append("--- WRONG predictions ---")
        loss_parts.extend(error_cases[:12])

    stats = {
        "maF1": m.maF1, "accuracy": m.accuracy,
        "f1_yes": m.f1_yes, "f1_no": m.f1_no,
        "precision_yes": m.precision_yes, "recall_yes": m.recall_yes,
        "precision_no":  m.precision_no,  "recall_no":  m.recall_no,
        "n_correct": len(correct_cases), "n_wrong": len(error_cases),
        "missed_yes": missed_yes, "missed_no": missed_no,
        "elapsed_s": round(elapsed, 2),
    }
    return "\n".join(loss_parts), m.maF1, stats


# ══════════════════════════════════════════════════════════════════════════
#  Val evaluator  (async)
# ══════════════════════════════════════════════════════════════════════════
async def evaluate_val_async(
    val_questions: List[dict],
    gold_map: Dict[str, YesNo],
    system: str,
    user_template: str,
    api_key: str,
) -> Dict[str, float]:
    results = await async_predict_batch(
        val_questions, api_key=api_key, system=system, user_template=user_template
    )
    pred_map: Dict[str, YesNo] = {
        qid: (pred or "no")
        for qid, pred, _ in results
        if qid in gold_map
    }
    gold_val = {qid: gold_map[qid] for qid in pred_map}
    if not gold_val:
        return {"maF1": 0.0, "accuracy": 0.0, "f1_yes": 0.0, "f1_no": 0.0,
                "precision_yes": 0.0, "recall_yes": 0.0,
                "precision_no":  0.0, "recall_no":  0.0}
    m = compute_metrics(gold_val, pred_map)
    return {
        "maF1": m.maF1, "accuracy": m.accuracy,
        "f1_yes": m.f1_yes, "f1_no": m.f1_no,
        "precision_yes": m.precision_yes, "recall_yes": m.recall_yes,
        "precision_no":  m.precision_no,  "recall_no":  m.recall_no,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Yes/No-specific TextGrad gradient + update prompts
# ══════════════════════════════════════════════════════════════════════════
_GRADIENT_SYSTEM = """\
You are an expert NLP researcher specialising in biomedical yes/no question answering.
Your job: critique a prompt used for binary Yes/No classification and suggest concrete improvements.

Always analyse these Yes/No-specific failure modes:
1. CLASS BIAS — Does the prompt cause the model to systematically favour "yes" or "no"?
   Compare missed_yes vs missed_no. If one is much larger, the prompt is biased.
2. EVIDENCE CONFLICT — Does the prompt tell the model what to do when snippets contradict each other?
   Missing guidance here causes inconsistent answers.
3. FORMAT ROBUSTNESS — The model MUST output exactly "yes" or "no" on the very last line, nothing else.
   If format failures exist, the prompt output instruction needs strengthening.
4. REASONING CHAIN — Does the prompt require step-by-step reasoning before the final answer?
   A clear chain reduces hallucination.
5. SNIPPET WEIGHTING — Does the prompt guide how to handle multiple, potentially conflicting snippets?

Output: a concrete critique followed by SPECIFIC rewrite instructions.
Do NOT rewrite the entire prompt yourself — describe exactly what should change and why.
"""

_GRADIENT_USER = """\
Current prompt  ({role}):
\"\"\"
{current_value}
\"\"\"

Batch evaluation:
\"\"\"
{loss_text}
\"\"\"

Key numbers:
  Macro-F1   = {maF1:.4f}   (maximise this)
  missed_yes = {missed_yes}  (gold=yes but predicted=no  → model biased toward NO)
  missed_no  = {missed_no}  (gold=no  but predicted=yes  → model biased toward YES)

Task:
1. Identify the PRIMARY failure mode from the list above.
2. Quote the exact phrase in the current prompt that caused or failed to prevent this error.
3. Give a concrete rewrite instruction (e.g. "After step 2, add: …").
4. Placeholders {{q_body}} and {{snippets_block}} MUST be kept if present in this prompt.

Be specific. Generic advice like "improve clarity" is not acceptable.
"""

_UPDATE_SYSTEM = """\
You are an expert prompt engineer for biomedical yes/no question answering.
Rewrite the given prompt following the improvement instructions exactly.

Hard rules:
- Output ONLY the rewritten prompt text — no preamble, no explanation, no markdown fences.
- Preserve ALL template placeholders verbatim: {q_body} and {snippets_block}.
- The user template must still end with an instruction to output exactly "yes" or "no" on the last line.
- Keep the prompt concise and directive.
"""

_UPDATE_USER = """\
Original prompt:
\"\"\"
{current_value}
\"\"\"

Improvement instructions:
\"\"\"
{gradient_text}
\"\"\"

Write the improved prompt now. Output only the new prompt text, nothing else.
"""


def _generate_gradient(engine: OpenRouterTextGradEngine, param: Variable,
                        loss_text: str, stats: Dict[str, Any]) -> str:
    msg = _GRADIENT_USER.format(
        role=param.role_description,
        current_value=param.value,
        loss_text=loss_text,
        maF1=stats["maF1"],
        missed_yes=stats["missed_yes"],
        missed_no=stats["missed_no"],
    )
    return engine.generate(msg, system_prompt=_GRADIENT_SYSTEM)


def _apply_gradient(engine: OpenRouterTextGradEngine,
                    param: Variable, gradient_text: str) -> str:
    msg = _UPDATE_USER.format(
        current_value=param.value,
        gradient_text=gradient_text,
    )
    return engine.generate(msg, system_prompt=_UPDATE_SYSTEM)


def run_yesno_textgrad_step(
    engine: OpenRouterTextGradEngine,
    params: List[Variable],
    loss_text: str,
    stats: Dict[str, Any],
) -> None:
    """Generate a yes/no-specific gradient and rewrite each prompt variable."""
    for param in params:
        print(f"    Gradient ← {param.role_description[:60]} …")
        gradient_text = _generate_gradient(engine, param, loss_text, stats)

        print(f"    Applying gradient …")
        new_value = _apply_gradient(engine, param, gradient_text)

        if new_value.strip():
            param.gradients = {Variable(
                gradient_text, requires_grad=False,
                role_description=f"gradient for {param.role_description}",
            )}
            param.set_value(new_value.strip())
        else:
            print("    [WARN] Empty update — keeping current value.", file=sys.stderr)


def _ensure_placeholders(template: str) -> str:
    """Revert to initial template if required placeholders are missing."""
    missing = [p for p in ("{q_body}", "{snippets_block}") if p not in template]
    if missing:
        print(f"  [WARN] Placeholders {missing} missing — reverting to initial template.",
              file=sys.stderr)
        return INITIAL_USER_TEMPLATE
    return template


# ══════════════════════════════════════════════════════════════════════════
#  Main optimization loop
# ══════════════════════════════════════════════════════════════════════════
async def _main_async() -> None:
    rng = random.Random(RANDOM_SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / LOG_FILE

    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        sys.exit(f"[ERROR] Environment variable {API_KEY_ENV} is not set.")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("[1/5] Loading data …")
    gold_path = Path(GOLD_PATH)
    if not gold_path.exists():
        sys.exit(f"[ERROR] gold_path not found: {gold_path}")

    all_qs   = load_questions(gold_path)
    yesno_qs = [q for q in all_qs if normalize_qtype(str(q.get("type", ""))) == "yesno"]
    gold_map = map_gold_labels(gold_path)

    yes_n = sum(1 for v in gold_map.values() if v == "yes")
    no_n  = sum(1 for v in gold_map.values() if v == "no")
    print(f"  {len(yesno_qs)} yes/no questions  |  gold: yes={yes_n}, no={no_n}, "
          f"yes_ratio={yes_n/max(1, yes_n+no_n):.1%}")

    val_size = VAL_SIZE
    if len(yesno_qs) < val_size + TRAIN_BATCH_SIZE:
        val_size = max(5, len(yesno_qs) // 4)
        print(f"  [WARN] Small dataset — adjusting val_size to {val_size}")

    rng.shuffle(yesno_qs)
    val_questions = yesno_qs[:val_size]
    train_pool    = yesno_qs[val_size:]
    print(f"  Train pool: {len(train_pool)}   Val: {len(val_questions)}")

    # ── 2. W&B init ───────────────────────────────────────────────────────
    wb = WandbLogger()
    wb.start({
        "n_epochs":               N_EPOCHS,
        "train_batch_size":       TRAIN_BATCH_SIZE,
        "val_size":               val_size,
        "concurrency":            INFERENCE_CONCURRENCY,
        "inference_model":        INFERENCE_MODEL,
        "gradient_engine_model":  GRADIENT_ENGINE_MODEL,
        "inference_temperature":  INFERENCE_TEMPERATURE,
        "gradient_temperature":   GRADIENT_TEMPERATURE,
        "inference_max_tokens":   INFERENCE_MAX_TOKENS,
        "optimize_system_prompt": OPTIMIZE_SYSTEM_PROMPT,
        "optimize_user_template": OPTIMIZE_USER_TEMPLATE,
        "gold_path":              str(gold_path),
        "random_seed":            RANDOM_SEED,
        "gold_yes_ratio":         yes_n / max(1, yes_n + no_n),
        "initial_system_prompt":  INITIAL_SYSTEM_PROMPT,
        "initial_user_template":  INITIAL_USER_TEMPLATE,
    })

    # ── 3. TextGrad setup ─────────────────────────────────────────────────
    print("[2/5] Initializing TextGrad engine …")
    engine = OpenRouterTextGradEngine()
    tg.set_backward_engine(engine, override=True)

    system_var = Variable(
        INITIAL_SYSTEM_PROMPT,
        requires_grad=OPTIMIZE_SYSTEM_PROMPT,
        role_description=(
            "system prompt for biomedical yes/no QA — "
            "controls reasoning strategy and output format"
        ),
    )
    template_var = Variable(
        INITIAL_USER_TEMPLATE,
        requires_grad=OPTIMIZE_USER_TEMPLATE,
        role_description=(
            "user message template for biomedical yes/no QA — "
            "contains {q_body} and {snippets_block} placeholders that MUST be preserved"
        ),
    )

    params: List[Variable] = []
    if OPTIMIZE_SYSTEM_PROMPT:  params.append(system_var)
    if OPTIMIZE_USER_TEMPLATE:  params.append(template_var)
    if not params:
        sys.exit("[ERROR] Both OPTIMIZE_SYSTEM_PROMPT and OPTIMIZE_USER_TEMPLATE are False.")

    # ── 4. Baseline ───────────────────────────────────────────────────────
    print("[3/5] Baseline evaluation …")
    base_val = await evaluate_val_async(
        val_questions, gold_map,
        INITIAL_SYSTEM_PROMPT, INITIAL_USER_TEMPLATE, api_key,
    )
    print(f"  Baseline → maF1={base_val['maF1']:.4f}  acc={base_val['accuracy']:.4f}  "
          f"F1_yes={base_val['f1_yes']:.4f}  F1_no={base_val['f1_no']:.4f}")

    wb.log({"val/maF1": base_val["maF1"], "val/accuracy": base_val["accuracy"],
            "val/f1_yes": base_val["f1_yes"], "val/f1_no": base_val["f1_no"],
            "epoch": 0})

    best_maF1   = base_val["maF1"]
    best_system = INITIAL_SYSTEM_PROMPT
    best_tmpl   = INITIAL_USER_TEMPLATE
    history: List[Dict[str, Any]] = []

    baseline_rec = {
        "epoch": 0, "phase": "baseline",
        "system_prompt":  INITIAL_SYSTEM_PROMPT,
        "user_template":  INITIAL_USER_TEMPLATE,
        "train_maF1": 0.0,
        **{f"val_{k}": v for k, v in base_val.items()},
        "improved": False,
    }
    history.append(baseline_rec)
    _append_log(log_path, baseline_rec)

    # ── 5. Optimization loop ──────────────────────────────────────────────
    print(f"[4/5] Starting {N_EPOCHS} optimization epochs …\n")

    for epoch in range(1, N_EPOCHS + 1):
        print(f"\n{'═'*58}")
        print(f"  Epoch {epoch} / {N_EPOCHS}")
        print(f"{'═'*58}")

        batch            = rng.sample(train_pool, min(TRAIN_BATCH_SIZE, len(train_pool)))
        current_system   = system_var.value
        current_template = template_var.value

        # [a] Async forward pass
        print(f"  [a] Async inference on {len(batch)} questions …")
        loss_text, train_maF1, train_stats = await build_loss_async(
            batch, gold_map, current_system, current_template, api_key
        )
        print(f"       maF1={train_maF1:.4f}  missed_yes={train_stats['missed_yes']}  "
              f"missed_no={train_stats['missed_no']}  elapsed={train_stats['elapsed_s']:.1f}s")

        # [b] Yes/No-specific TextGrad backward
        print("  [b] TextGrad backward (yes/no gradients) …")
        try:
            run_yesno_textgrad_step(engine, params, loss_text, train_stats)
        except Exception as e:
            print(f"  [WARN] TextGrad step failed: {e}", file=sys.stderr)
            continue

        # Retrieve and validate updated prompts
        new_system = system_var.value   if OPTIMIZE_SYSTEM_PROMPT else INITIAL_SYSTEM_PROMPT
        new_tmpl   = template_var.value if OPTIMIZE_USER_TEMPLATE  else INITIAL_USER_TEMPLATE
        new_tmpl   = _ensure_placeholders(new_tmpl)

        print(f"\n  Updated system prompt (first 220 chars):\n    {new_system[:220].strip()}\n")
        print(f"  Updated user template (first 220 chars):\n    {new_tmpl[:220].strip()}\n")

        # [c] Async val evaluation
        print("  [c] Async val evaluation …")
        val_m = await evaluate_val_async(
            val_questions, gold_map, new_system, new_tmpl, api_key
        )
        print(f"       maF1={val_m['maF1']:.4f}  acc={val_m['accuracy']:.4f}  "
              f"F1_yes={val_m['f1_yes']:.4f}  F1_no={val_m['f1_no']:.4f}")

        improved = val_m["maF1"] > best_maF1
        if improved:
            best_maF1   = val_m["maF1"]
            best_system = new_system
            best_tmpl   = new_tmpl
            print(f"  ★  New best val maF1 = {best_maF1:.4f}")

        # W&B per-epoch log
        wb.log({
            "train/maF1":        train_maF1,
            "train/f1_yes":      train_stats["f1_yes"],
            "train/f1_no":       train_stats["f1_no"],
            "train/missed_yes":  train_stats["missed_yes"],
            "train/missed_no":   train_stats["missed_no"],
            "train/elapsed_s":   train_stats["elapsed_s"],
            "val/maF1":          val_m["maF1"],
            "val/accuracy":      val_m["accuracy"],
            "val/f1_yes":        val_m["f1_yes"],
            "val/f1_no":         val_m["f1_no"],
            "val/precision_yes": val_m["precision_yes"],
            "val/recall_yes":    val_m["recall_yes"],
            "val/precision_no":  val_m["precision_no"],
            "val/recall_no":     val_m["recall_no"],
            "best_val_maF1":     best_maF1,
            "improved":          int(improved),
            "epoch":             epoch,
        }, step=epoch)

        rec = {
            "epoch":          epoch,
            "train_maF1":     train_maF1,
            "train_stats":    train_stats,
            "improved":       improved,
            "system_prompt":  new_system,
            "user_template":  new_tmpl,
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(rec)
        _append_log(log_path, rec)

    # ── 6. Save results ───────────────────────────────────────────────────
    print("\n[5/5] Saving results …")

    best_sys_path  = output_dir / "best_system_prompt.txt"
    best_tmpl_path = output_dir / "best_user_template.txt"
    best_sys_path.write_text(best_system, encoding="utf-8")
    best_tmpl_path.write_text(best_tmpl,  encoding="utf-8")

    (output_dir / "summary.json").write_text(json.dumps({
        "baseline_val_maF1":   base_val["maF1"],
        "best_val_maF1":       best_maF1,
        "improvement":         round(best_maF1 - base_val["maF1"], 6),
        "n_epochs":            N_EPOCHS,
        "train_batch_size":    TRAIN_BATCH_SIZE,
        "val_size":            val_size,
        "concurrency":         INFERENCE_CONCURRENCY,
        "inference_model":     INFERENCE_MODEL,
        "gradient_model":      GRADIENT_ENGINE_MODEL,
        "best_system_prompt":  best_system,
        "best_user_template":  best_tmpl,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    wb.log_prompt_table(history)
    wb.finish()

    print(f"\n{'═'*60}")
    print(f"  Baseline val maF1 : {base_val['maF1']:.4f}")
    print(f"  Best val maF1     : {best_maF1:.4f}   ({best_maF1 - base_val['maF1']:+.4f})")
    print(f"  best_system_prompt → {best_sys_path}")
    print(f"  best_user_template → {best_tmpl_path}")
    print(f"  optimization log   → {log_path}")
    print(f"{'═'*60}")
    print("\nTo use these prompts in main.py, set in MainConfig:")
    print(f"  system_prompt_path = Path('{best_sys_path}')")
    print(f"  user_template_path = Path('{best_tmpl_path}')")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()