"""
exp_dspy.py — DSPy 3.x 自動 Prompt 優化（BioASQ Yes/No）
適用版本：dspy >= 3.0

════════════════════════════════════════════════════════════
  ██  所有可調整參數集中在第一區塊，直接修改即可  ██
════════════════════════════════════════════════════════════

【評估指標說明】
  BioASQ 官方指標（BioASQ6 起）是 macro-F1（maF1）：
    F1_yes + F1_no，各自用 TP/FP/FN 計算，再取平均
  accuracy 僅供參考，優化目標以 maF1 為主

【MIPRO_AUTO 與 NUM_TRIALS 的關係】
  MIPRO_AUTO = "light"/"medium"/"heavy"  → dspy 自動決定 trials，NUM_TRIALS 被忽略
  MIPRO_AUTO = None                      → 手動模式，NUM_TRIALS 生效

【async 並行推理】
  ASYNC_CONCURRENCY 控制 evaluate_on_set 的並行 API 請求數
  DSPy MIPROv2 內部仍使用 NUM_THREADS 做多執行緒優化
"""

from __future__ import annotations

# ── 模型設定 ─────────────────────────────────────────────────
MODEL       = "openrouter/anthropic/claude-sonnet-4-6"
BASE_URL    = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
TEMPERATURE = 0.0
MAX_TOKENS  = 4096          # yesno 不需要很長的輸出

# ── 資料設定 ──────────────────────────────────────────────────
GOLD_PATH          = "data/predictions/training13b.json"
N_TRAIN            = 100
N_DEV              = 50
N_TEST             = 100
RANDOM_SEED        = 42
FILTER_NO_SNIPPETS = True

# ── DSPy MIPROv2 優化設定 ────────────────────────────────────
MAX_BOOTSTRAPPED_DEMOS = 0
MAX_LABELED_DEMOS      = 0

# MIPRO_AUTO = "light"/"medium"/"heavy" 時，NUM_TRIALS 會被忽略
# MIPRO_AUTO = None 時，NUM_TRIALS 才會傳給 compile()
MIPRO_AUTO  = "light"   # "light" / "medium" / "heavy" / None
NUM_TRIALS  = 20         # 只在 MIPRO_AUTO=None 時生效

NUM_THREADS      = 8     # MIPROv2 內部優化用的執行緒數
ASYNC_CONCURRENCY = 20   # evaluate_on_set async 並行請求數

# ── 實驗模式 ─────────────────────────────────────────────────
RUN_BASELINE   = True
RUN_OPTIMIZE   = True
RUN_FINAL_TEST = True
SAVE_RESULTS   = True

# ── 輸出設定 ──────────────────────────────────────────────────
OUTPUT_DIR                   = "dspy_output"
OPTIMIZED_SYSTEM_PROMPT_FILE = "optimized_system.txt"
OPTIMIZED_USER_TEMPLATE_FILE = "optimized_user.txt"
RESULTS_FILE                 = "results.json"

# ── WandB 設定 ────────────────────────────────────────────────
WANDB_ENABLED  = True
WANDB_PROJECT  = "yesno"
WANDB_ENTITY   = "bioasq"
WANDB_GROUP    = "CSHS"
WANDB_MODE     = "online"   # "online" / "offline" / "disabled"
WANDB_RUN_NAME = "dsH1-demo0-lightpy"   # None = 自動命名（時間戳）

# ════════════════════════════════════════════════════════════
#  以下為實作（一般不需要修改）
# ════════════════════════════════════════════════════════════

import asyncio
import json
import os
import random
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    import wandb as _wandb_module
except ImportError:
    _wandb_module = None  # type: ignore


# ══════════════════════════════════════════════════════════════
#  資料載入工具
# ══════════════════════════════════════════════════════════════

def load_questions(p: Path) -> List[dict]:
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [q for q in raw.get("questions", []) if isinstance(q, dict)]


def normalize_qtype(qt: str) -> str:
    qt = (qt or "").strip().lower()
    return "yesno" if qt in {"yes/no", "yesno", "yes-no", "yn"} else qt


def coerce_yesno(val: Any) -> Optional[str]:
    """把 exact_answer 正規化成 'yes'/'no'，支援 list 格式。"""
    if isinstance(val, list):
        val = val[0] if val else ""
        if isinstance(val, list):
            val = val[0] if val else ""
    if isinstance(val, str):
        s = val.strip().lower().strip(" .,:;!\"'")
        if s in ("yes", "no"):
            return s
    return None


def _from_text(text: str) -> Optional[str]:
    """從自由文字末尾抽取 yes/no，只看最後 3 行。"""
    if not text:
        return None
    for line in reversed(text.strip().splitlines()[-3:]):
        w = line.strip().lower().strip(" .,:;!\"'")
        if w in ("yes", "no"):
            return w
    m = re.findall(r"\b(yes|no)\b", text.lower())
    return m[-1] if m else None


def build_snippets_block(q: dict) -> str:
    parts = []
    for i, s in enumerate(q.get("snippets", []), 1):
        if isinstance(s, dict) and s.get("text"):
            parts.append(f"[{i}]: {s['text'].strip()}")
    return "\n".join(parts) if parts else ""


def load_yesno_examples(gold_path: str, filter_no_snippets: bool = True) -> List[dict]:
    p = Path(gold_path)
    if not p.exists():
        raise FileNotFoundError(f"找不到資料檔案：{p.resolve()}")

    examples, skipped_label, skipped_snippets = [], 0, 0
    for q in load_questions(p):
        if normalize_qtype(str(q.get("type", ""))) != "yesno":
            continue
        label = coerce_yesno(q.get("exact_answer"))
        if not label:
            skipped_label += 1
            continue
        snippets = build_snippets_block(q)
        if filter_no_snippets and not snippets:
            skipped_snippets += 1
            continue
        examples.append({
            "question": str(q.get("body", "")).strip(),
            "snippets": snippets,
            "answer":   label,
        })

    print(f"  載入 yes/no 題目：{len(examples)} 題"
          f"（跳過無標籤：{skipped_label}，跳過無snippets：{skipped_snippets}）")
    yes_cnt = sum(1 for e in examples if e["answer"] == "yes")
    no_cnt  = len(examples) - yes_cnt
    print(f"  分布：yes={yes_cnt} ({yes_cnt/len(examples)*100:.1f}%)"
          f"  no={no_cnt} ({no_cnt/len(examples)*100:.1f}%)")
    return examples


def build_splits(
    examples: List[dict], n_train: int, n_dev: int, n_test: int, seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    total = n_train + n_dev + n_test
    if len(examples) < total:
        print(f"  [WARN] 資料只有 {len(examples)} 題，不足 {total}，自動縮減。")
        r = len(examples) / total
        n_train = int(n_train * r)
        n_dev   = int(n_dev   * r)
        n_test  = len(examples) - n_train - n_dev

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_dev],
        shuffled[n_train + n_dev:n_train + n_dev + n_test],
    )


def to_dspy_examples(raw: List[dict]) -> List[Any]:
    import dspy
    return [
        dspy.Example(question=e["question"], snippets=e["snippets"], answer=e["answer"])
        .with_inputs("question", "snippets")
        for e in raw
    ]


# ══════════════════════════════════════════════════════════════
#  BioASQ Metrics（maF1 = 官方指標）
# ══════════════════════════════════════════════════════════════

@dataclass
class YesNoMetrics:
    n: int
    correct: int
    tp: int   # yes gold & yes pred
    fp: int   # no  gold & yes pred
    fn: int   # yes gold & no  pred
    tn: int   # no  gold & no  pred
    invalid: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def precision_yes(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall_yes(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1_yes(self) -> float:
        p, r = self.precision_yes, self.recall_yes
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def precision_no(self) -> float:
        d = self.tn + self.fn
        return self.tn / d if d else 0.0

    @property
    def recall_no(self) -> float:
        d = self.tn + self.fp
        return self.tn / d if d else 0.0

    @property
    def f1_no(self) -> float:
        p, r = self.precision_no, self.recall_no
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def maF1(self) -> float:
        return 0.5 * (self.f1_yes + self.f1_no)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 6),
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision_yes": round(self.precision_yes, 6),
            "recall_yes":    round(self.recall_yes, 6),
            "f1_yes":        round(self.f1_yes, 6),
            "precision_no":  round(self.precision_no, 6),
            "recall_no":     round(self.recall_no, 6),
            "f1_no":         round(self.f1_no, 6),
            "maF1":          round(self.maF1, 6),
            "invalid_preds": self.invalid,
        }


def compute_yesno_metrics(
    gold_answers: List[str],
    pred_answers: List[str],
) -> YesNoMetrics:
    """計算完整的 BioASQ yes/no 指標，包括官方 maF1。"""
    assert len(gold_answers) == len(pred_answers)
    tp = fp = fn = tn = correct = invalid = 0
    for g, p in zip(gold_answers, pred_answers):
        if p not in ("yes", "no"):
            invalid += 1
            p = "no"  # missing 當 no 計
        if g == "yes" and p == "yes":
            tp += 1; correct += 1
        elif g == "no" and p == "yes":
            fp += 1
        elif g == "yes" and p == "no":
            fn += 1
        elif g == "no" and p == "no":
            tn += 1; correct += 1
    return YesNoMetrics(
        n=len(gold_answers), correct=correct,
        tp=tp, fp=fp, fn=fn, tn=tn, invalid=invalid,
    )


def maf1_metric(example, prediction, trace=None) -> float:
    """
    DSPy metric：回傳 maF1 貢獻值（0.0 ~ 1.0 之間的 partial credit）。
    MIPROv2 以此最大化。

    注意：MIPROv2 在每個 example 層級計算 metric，
    所以這裡回傳對 maF1 有意義的 binary（正確=1.0，錯誤=0.0）。
    整體 maF1 在 evaluate_on_set 中用 YesNoMetrics 計算。
    """
    gold = str(example.answer).strip().lower()
    pred_raw = str(getattr(prediction, "answer", "") or "").strip().lower().strip(" .,:;!\"'")
    if pred_raw not in ("yes", "no"):
        pred_raw = _from_text(str(getattr(prediction, "reasoning", "") or "")) or "no"
    return float(gold == pred_raw)


# ══════════════════════════════════════════════════════════════
#  DSPy Module — 專屬 BioASQ Yes/No Signature
# ══════════════════════════════════════════════════════════════

def build_module() -> Any:
    import dspy

    class BioASQYesNo(dspy.Signature):
        """You are an expert biomedical AI assistant specialising in evidence-based
        clinical and biological question answering.

        TASK: Answer the yes/no question using ONLY the provided literature snippets.
        Follow these rules strictly:
          1. Read every snippet carefully for evidence that is directly relevant.
          2. Reason step-by-step: cite which snippet(s) support or contradict each claim.
          3. Do NOT rely on prior knowledge—only the snippets count as evidence.
          4. When evidence conflicts or is insufficient, the answer is 'no'.
          5. Your final answer MUST be exactly the single word 'yes' or 'no' (nothing else).

        The official BioASQ evaluation metric is macro-F1 (average of F1_yes and F1_no),
        so precision and recall for BOTH classes matter equally."""

        question: str = dspy.InputField(
            desc="The biomedical yes/no question to answer"
        )
        snippets: str = dspy.InputField(
            desc="Numbered PubMed literature snippets (may be empty if none provided)"
        )
        reasoning: str = dspy.OutputField(
            desc=(
                "Step-by-step reasoning grounded strictly in the snippets. "
                "Cite snippet numbers. Acknowledge conflicts or missing evidence explicitly."
            )
        )
        answer: str = dspy.OutputField(
            desc="Final answer: exactly the single word 'yes' or 'no' — nothing else."
        )

    class BioASQYesNoPredictor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.cot = dspy.ChainOfThought(BioASQYesNo)

        def forward(self, question: str, snippets: str) -> Any:
            pred = self.cot(question=question, snippets=snippets)
            # 正規化 answer
            raw = str(getattr(pred, "answer", "") or "").strip().lower().strip(" .,:;!\"'")
            if raw not in ("yes", "no"):
                raw = _from_text(str(getattr(pred, "reasoning", "") or "")) or "no"
            return dspy.Prediction(
                reasoning=getattr(pred, "reasoning", ""),
                answer=raw,
            )

    return BioASQYesNoPredictor()


# ══════════════════════════════════════════════════════════════
#  DSPy 設定
# ══════════════════════════════════════════════════════════════

def setup_dspy() -> None:
    import dspy
    load_dotenv()
    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        raise SystemExit(
            f"[ERROR] 找不到 {API_KEY_ENV}，請在 .env 設定：\n  {API_KEY_ENV}=sk-or-xxxx"
        )

    lm = dspy.LM(
        model=MODEL,
        api_base=BASE_URL,
        api_key=api_key,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        cache=False,
    )
    dspy.configure(lm=lm)
    print(f"  DSPy {dspy.__version__} LM 設定完成：{MODEL}")


# ══════════════════════════════════════════════════════════════
#  Async 並行評估
# ══════════════════════════════════════════════════════════════

async def _predict_one_async(
    module: Any,
    ex: Any,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, str, float]:
    """在 executor 中執行 DSPy 同步 forward，並用 semaphore 限制並行數。
    回傳 (gold, pred, latency_sec)。
    """
    loop = asyncio.get_event_loop()
    async with semaphore:
        t0 = time.perf_counter()
        pred = await loop.run_in_executor(
            None,
            lambda: module(question=ex.question, snippets=ex.snippets),
        )
        latency = time.perf_counter() - t0

    pred_raw = str(getattr(pred, "answer", "") or "").strip().lower().strip(" .,:;!\"'")
    if pred_raw not in ("yes", "no"):
        pred_raw = _from_text(str(getattr(pred, "reasoning", "") or "")) or "no"

    gold = str(ex.answer).strip().lower()
    return gold, pred_raw, latency


async def _evaluate_on_set_async(
    module: Any,
    dataset: List[Any],
    label: str,
    concurrency: int,
) -> Dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    t_wall = time.perf_counter()

    tasks = [
        asyncio.create_task(_predict_one_async(module, ex, semaphore))
        for ex in dataset
    ]

    gold_list: List[str] = []
    pred_list: List[str] = []
    latencies: List[float] = []

    # tqdm 需要在 async 環境包一層
    try:
        from tqdm.asyncio import tqdm as atqdm
        results_iter = atqdm.gather(*tasks, desc=label)
        results = await results_iter
    except ImportError:
        results = await asyncio.gather(*tasks)

    for gold, pred, lat in results:
        gold_list.append(gold)
        pred_list.append(pred)
        latencies.append(lat)

    elapsed = time.perf_counter() - t_wall
    metrics = compute_yesno_metrics(gold_list, pred_list)
    d = metrics.to_dict()
    d.update({
        "label":       label,
        "elapsed_sec": round(elapsed, 1),
        "avg_latency_ms": round(1000 * sum(latencies) / len(latencies), 1) if latencies else 0,
    })

    _print_metrics(label, metrics, elapsed)
    return d


def evaluate_on_set(
    module: Any,
    dataset: List[Any],
    label: str,
    concurrency: int = ASYNC_CONCURRENCY,
) -> Dict[str, Any]:
    """同步入口，內部用 asyncio 並行發送請求。"""
    return asyncio.run(_evaluate_on_set_async(module, dataset, label, concurrency))


def _print_metrics(label: str, m: YesNoMetrics, elapsed: float) -> None:
    print(f"\n  [{label}] n={m.n}  acc={m.accuracy:.4f} ({m.correct}/{m.n})"
          f"  maF1={m.maF1:.4f}  ({elapsed:.1f}s)")
    print(f"    YES: P={m.precision_yes:.4f} R={m.recall_yes:.4f} F1={m.f1_yes:.4f}"
          f"  (TP={m.tp} FP={m.fp} FN={m.fn})")
    print(f"    NO : P={m.precision_no:.4f} R={m.recall_no:.4f} F1={m.f1_no:.4f}"
          f"  (TN={m.tn})")
    print(f"    invalid_preds={m.invalid}")


# ══════════════════════════════════════════════════════════════
#  WandB 整合
# ══════════════════════════════════════════════════════════════

class WandbLogger:
    def __init__(self):
        self._run = None

    def start(self, run_config: Dict[str, Any]) -> None:
        if not WANDB_ENABLED or _wandb_module is None:
            return
        init_kwargs: Dict[str, Any] = {
            "project": WANDB_PROJECT,
            "config":  run_config,
        }
        if WANDB_ENTITY:
            init_kwargs["entity"] = WANDB_ENTITY
        if WANDB_GROUP:
            init_kwargs["group"] = WANDB_GROUP
        if WANDB_MODE:
            init_kwargs["mode"] = WANDB_MODE
        if WANDB_RUN_NAME:
            init_kwargs["name"] = WANDB_RUN_NAME
        self._run = _wandb_module.init(**init_kwargs)
        print(f"  WandB run 已啟動：{self._run.name if self._run else '—'}")

    def _active(self) -> bool:
        return WANDB_ENABLED and _wandb_module is not None and self._run is not None

    def log_config(self, extra: Dict[str, Any]) -> None:
        if not self._active():
            return
        _wandb_module.config.update(extra, allow_val_change=True)

    def log_metrics(self, metrics_dict: Dict[str, Any], prefix: str = "") -> None:
        """記錄一組指標。prefix 例如 'baseline/dev', 'optimized/test'。"""
        if not self._active():
            return
        log_data = {}
        for k, v in metrics_dict.items():
            if isinstance(v, (int, float)):
                key = f"{prefix}/{k}" if prefix else k
                log_data[key] = v
        _wandb_module.log(log_data)

    def log_prompts(self, system: str, user_template: str) -> None:
        if not self._active():
            return
        _wandb_module.config.update(
            {"prompt/system": system, "prompt/user_template": user_template},
            allow_val_change=True,
        )

    def log_examples_table(
        self,
        examples: List[dict],
        gold_list: List[str],
        pred_list: List[str],
        label: str,
    ) -> None:
        """記錄每題 gold/pred 的 W&B Table。"""
        if not self._active():
            return
        table = _wandb_module.Table(
            columns=["label", "question", "gold_answer", "pred_answer", "correct", "snippets"]
        )
        for ex, gold, pred in zip(examples, gold_list, pred_list):
            table.add_data(
                label,
                ex.get("question", "")[:2000],
                gold,
                pred,
                gold == pred,
                ex.get("snippets", "")[:2000],
            )
        _wandb_module.log({f"predictions/{label}": table})

    def log_comparison_table(self, all_results: Dict[str, Any]) -> None:
        """記錄 baseline vs optimized 對比 Table。"""
        if not self._active():
            return
        table = _wandb_module.Table(
            columns=["mode", "split", "n", "accuracy", "maF1",
                     "f1_yes", "f1_no", "precision_yes", "recall_yes",
                     "precision_no", "recall_no", "invalid_preds", "elapsed_sec"]
        )
        for mode_key in ("baseline", "optimized"):
            if mode_key not in all_results:
                continue
            r = all_results[mode_key]
            for split in ("dev", "test"):
                if split not in r:
                    continue
                s = r[split]
                table.add_data(
                    mode_key, split,
                    s.get("n", 0),
                    s.get("accuracy", 0),
                    s.get("maF1", 0),
                    s.get("f1_yes", 0),
                    s.get("f1_no", 0),
                    s.get("precision_yes", 0),
                    s.get("recall_yes", 0),
                    s.get("precision_no", 0),
                    s.get("recall_no", 0),
                    s.get("invalid_preds", 0),
                    s.get("elapsed_sec", 0),
                )
        _wandb_module.log({"summary/comparison_table": table})

    def finish(self) -> None:
        if self._active() and _wandb_module is not None:
            _wandb_module.finish()
            self._run = None


# ══════════════════════════════════════════════════════════════
#  Baseline & Optimize
# ══════════════════════════════════════════════════════════════

def run_baseline(
    train_raw: List[dict],
    dev_raw: List[dict],
    test_raw: List[dict],
    wb: WandbLogger,
) -> Dict[str, Any]:
    print("\n" + "=" * 55)
    print("  BASELINE（無 few-shot，無優化）")
    print("=" * 55)
    module  = build_module()
    results: Dict[str, Any] = {"mode": "baseline"}

    dev_res = evaluate_on_set(module, to_dspy_examples(dev_raw), "baseline_dev")
    results["dev"] = dev_res
    wb.log_metrics(dev_res, prefix="baseline/dev")

    if RUN_FINAL_TEST and test_raw:
        test_res = evaluate_on_set(module, to_dspy_examples(test_raw), "baseline_test")
        results["test"] = test_res
        wb.log_metrics(test_res, prefix="baseline/test")

    return results


def run_optimize(
    train_raw: List[dict],
    dev_raw: List[dict],
    test_raw: List[dict],
    wb: WandbLogger,
) -> Tuple[Any, Dict[str, Any]]:
    import dspy

    print("\n" + "=" * 55)
    print("  MIPROv2 優化（目標：maF1）")
    print("=" * 55)

    trainset = to_dspy_examples(train_raw)
    devset   = to_dspy_examples(dev_raw)
    module   = build_module()

    optimizer = dspy.MIPROv2(
        metric=maf1_metric,
        auto=MIPRO_AUTO,
        num_threads=NUM_THREADS,
        max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=MAX_LABELED_DEMOS,
    )

    compile_kwargs: Dict[str, Any] = {
        "trainset": trainset,
        "valset":   devset,
        "requires_permission_to_run": False,
    }
    if MIPRO_AUTO is None:
        compile_kwargs["num_trials"] = NUM_TRIALS
        print(f"  手動模式：num_trials={NUM_TRIALS}"
              f"  bootstrap_demos={MAX_BOOTSTRAPPED_DEMOS}  threads={NUM_THREADS}")
    else:
        print(f"  auto={MIPRO_AUTO}（trials 由 dspy 自動決定）"
              f"  bootstrap_demos={MAX_BOOTSTRAPPED_DEMOS}  threads={NUM_THREADS}")

    t0 = time.time()
    optimized = optimizer.compile(module, **compile_kwargs)
    elapsed = time.time() - t0
    print(f"\n  優化完成，耗時 {elapsed:.0f} 秒")

    wb.log_metrics({"optimize_elapsed_sec": round(elapsed, 1)}, prefix="optimize")

    results: Dict[str, Any] = {"mode": "optimized", "optimize_elapsed_sec": round(elapsed, 1)}

    dev_res = evaluate_on_set(optimized, devset, "optimized_dev")
    results["dev"] = dev_res
    wb.log_metrics(dev_res, prefix="optimized/dev")

    if RUN_FINAL_TEST and test_raw:
        test_res = evaluate_on_set(optimized, to_dspy_examples(test_raw), "optimized_test")
        results["test"] = test_res
        wb.log_metrics(test_res, prefix="optimized/test")

    return optimized, results


# ══════════════════════════════════════════════════════════════
#  儲存 — prompt 提取（兼容 dspy 3.x 結構）
# ══════════════════════════════════════════════════════════════

def save_optimized_module(optimized_module: Any) -> None:
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    prog_path = out / "optimized_program.json"
    optimized_module.save(str(prog_path))
    print(f"\n  已儲存優化程式：{prog_path}")

    _extract_and_save_prompts(optimized_module, out)


def _extract_and_save_prompts(optimized_module: Any, out: Path) -> None:
    """
    dspy 3.x 結構：
      module.cot.predict.signature.instructions
      module.cot.predict.demos
    """
    try:
        predict      = optimized_module.cot.predict
        sig          = predict.signature
        instructions = str(getattr(sig, "instructions", "") or "").strip()
        demos        = list(getattr(predict, "demos", []) or [])

        # ── System prompt ──────────────────────────────────
        system = instructions if instructions else (
            "You are an expert biomedical AI assistant. "
            "Answer yes/no questions step-by-step based strictly on the provided snippets. "
            "When evidence conflicts or is insufficient, the answer is 'no'. "
            "The official BioASQ metric is macro-F1; aim for balanced precision and recall "
            "on both 'yes' and 'no' classes."
        )
        (out / OPTIMIZED_SYSTEM_PROMPT_FILE).write_text(system, encoding="utf-8")
        print(f"  已儲存 system prompt：{out / OPTIMIZED_SYSTEM_PROMPT_FILE}")

        # ── Few-shot demos → user template ─────────────────
        demos_text = ""
        if demos:
            parts = []
            for i, demo in enumerate(demos, 1):
                q = getattr(demo, "question",  "")
                s = getattr(demo, "snippets",  "")
                r = getattr(demo, "reasoning", "")
                a = getattr(demo, "answer",    "")
                parts.append(
                    f"### EXAMPLE {i}\n"
                    f"QUESTION:\n{q}\n\n"
                    f"SNIPPETS:\n{s}\n\n"
                    f"REASONING:\n{r}\n\n"
                    f"ANSWER: {a}"
                )
            demos_text = "\n\n".join(parts)
            print(f"  包含 {len(demos)} 個 few-shot demos")

        if demos_text:
            user_template = (
                f"{demos_text}\n\n"
                "─────────────────────────────────────\n"
                "Now answer the following question:\n\n"
                "### QUESTION:\n{q_body}\n\n"
                "### SNIPPETS:\n{snippets_block}\n\n"
                "### REASONING AND EXACT ANSWER:"
            )
        else:
            user_template = (
                "### INSTRUCTIONS:\n"
                "1. Read the question and snippets carefully.\n"
                "2. Write down your step-by-step reasoning based only on the snippets.\n"
                "3. Do not use outside knowledge.\n"
                "4. When evidence conflicts or is insufficient, the answer is 'no'.\n"
                "5. End your response with exactly: yes or no\n\n"
                "### QUESTION:\n{q_body}\n\n"
                "### SNIPPETS:\n{snippets_block}\n\n"
                "### REASONING AND EXACT ANSWER:"
            )
        (out / OPTIMIZED_USER_TEMPLATE_FILE).write_text(user_template, encoding="utf-8")
        print(f"  已儲存 user template：{out / OPTIMIZED_USER_TEMPLATE_FILE}")

    except Exception as e:
        print(f"  [WARN] 無法自動抽取 prompt：{e}")
        traceback.print_exc()


def save_results(all_results: Dict[str, Any]) -> None:
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / RESULTS_FILE
    path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  已儲存實驗結果：{path}")


# ══════════════════════════════════════════════════════════════
#  Summary 列印
# ══════════════════════════════════════════════════════════════

def print_summary(all_results: Dict[str, Any]) -> None:
    print("\n" + "=" * 62)
    print("  實驗結果總覽（BioASQ 官方指標：maF1）")
    print("=" * 62)
    hdr = f"  {'模式':<22}  {'集合':<14}  {'accuracy':>8}  {'maF1':>8}  {'F1_yes':>7}  {'F1_no':>7}"
    print(hdr)
    print("  " + "-" * 60)
    for mode_key in ("baseline", "optimized"):
        if mode_key not in all_results:
            continue
        r = all_results[mode_key]
        for split in ("dev", "test"):
            if split not in r:
                continue
            s = r[split]
            print(f"  {mode_key:<22}  {split:<14}  "
                  f"{s.get('accuracy', 0):>8.4f}  "
                  f"{s.get('maF1', 0):>8.4f}  "
                  f"{s.get('f1_yes', 0):>7.4f}  "
                  f"{s.get('f1_no', 0):>7.4f}")
    print()


# ══════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 62)
    print("  DSPy Yes/No Prompt 優化（BioASQ）— maF1 版")
    print("=" * 62)
    print(f"  模型：          {MODEL}")
    print(f"  資料：          {GOLD_PATH}")
    print(f"  Train/Dev/Test：{N_TRAIN}/{N_DEV}/{N_TEST}")
    print(f"  Async並行：     {ASYNC_CONCURRENCY}  DSPy threads：{NUM_THREADS}")
    if MIPRO_AUTO is None:
        print(f"  Trials：        {NUM_TRIALS}（手動，auto=None）")
    else:
        print(f"  auto={MIPRO_AUTO}（trials 由 dspy 自動決定，NUM_TRIALS 被忽略）")
    print(f"  Demos：         bootstrap={MAX_BOOTSTRAPPED_DEMOS}  labeled={MAX_LABELED_DEMOS}")
    print(f"  WandB：         {'啟用' if WANDB_ENABLED else '停用'}"
          f"  project={WANDB_PROJECT}  group={WANDB_GROUP}")
    print()

    # ── 載入資料 ──────────────────────────────────────────────
    print("【1/4】載入資料")
    examples = load_yesno_examples(GOLD_PATH, FILTER_NO_SNIPPETS)
    train_raw, dev_raw, test_raw = build_splits(
        examples, N_TRAIN, N_DEV, N_TEST, RANDOM_SEED
    )
    print(f"  Train={len(train_raw)}  Dev={len(dev_raw)}  Test={len(test_raw)}")

    # ── DSPy 設定 ─────────────────────────────────────────────
    print("\n【2/4】設定 DSPy")
    setup_dspy()

    # ── WandB 設定 ────────────────────────────────────────────
    wb = WandbLogger()
    run_config: Dict[str, Any] = {
        "model":                  MODEL,
        "temperature":            TEMPERATURE,
        "max_tokens":             MAX_TOKENS,
        "n_train":                N_TRAIN,
        "n_dev":                  N_DEV,
        "n_test":                 N_TEST,
        "random_seed":            RANDOM_SEED,
        "filter_no_snippets":     FILTER_NO_SNIPPETS,
        "mipro_auto":             MIPRO_AUTO,
        "num_trials":             NUM_TRIALS if MIPRO_AUTO is None else "auto",
        "num_threads":            NUM_THREADS,
        "async_concurrency":      ASYNC_CONCURRENCY,
        "max_bootstrapped_demos": MAX_BOOTSTRAPPED_DEMOS,
        "max_labeled_demos":      MAX_LABELED_DEMOS,
        "run_baseline":           RUN_BASELINE,
        "run_optimize":           RUN_OPTIMIZE,
        "run_final_test":         RUN_FINAL_TEST,
        "gold_path":              GOLD_PATH,
        "output_dir":             OUTPUT_DIR,
        "metric":                 "maF1_binary_per_example",
    }
    wb.start(run_config)

    all_results: Dict[str, Any] = {"config": run_config}

    # ── Baseline ──────────────────────────────────────────────
    if RUN_BASELINE:
        print("\n【3/4】Baseline 評估")
        try:
            all_results["baseline"] = run_baseline(train_raw, dev_raw, test_raw, wb)
        except Exception as e:
            print(f"  [WARN] Baseline 失敗：{e}")
            traceback.print_exc()
    else:
        print("\n【3/4】略過 Baseline（RUN_BASELINE=False）")

    # ── Optimize ──────────────────────────────────────────────
    optimized_module: Any = None
    if RUN_OPTIMIZE:
        print("\n【4/4】MIPROv2 優化")
        try:
            optimized_module, opt_results = run_optimize(
                train_raw, dev_raw, test_raw, wb
            )
            all_results["optimized"] = opt_results

            # 抽取並記錄最終 prompt
            _out = Path(OUTPUT_DIR)
            _out.mkdir(parents=True, exist_ok=True)
            _extract_and_save_prompts(optimized_module, _out)
            sys_text = (_out / OPTIMIZED_SYSTEM_PROMPT_FILE).read_text(encoding="utf-8")
            usr_text = (_out / OPTIMIZED_USER_TEMPLATE_FILE).read_text(encoding="utf-8")
            wb.log_prompts(sys_text, usr_text)

            save_optimized_module(optimized_module)
        except Exception as e:
            print(f"  [ERROR] 優化失敗：{e}")
            traceback.print_exc()
    else:
        print("\n【4/4】略過優化（RUN_OPTIMIZE=False）")

    # ── WandB 彙整 ────────────────────────────────────────────
    wb.log_comparison_table(all_results)

    # 記錄最終最佳 maF1（優化 > baseline，dev 優先）
    best_maf1 = 0.0
    for mode in ("optimized", "baseline"):
        if mode in all_results:
            for split in ("test", "dev"):
                v = all_results[mode].get(split, {}).get("maF1", 0.0)
                if isinstance(v, float) and v > best_maf1:
                    best_maf1 = v
    wb.log_metrics({"best_maF1": best_maf1})
    wb.finish()

    # ── 列印總覽 ──────────────────────────────────────────────
    print_summary(all_results)
    if SAVE_RESULTS:
        save_results(all_results)

    print("完成。")
    if RUN_OPTIMIZE and optimized_module is not None:
        print(f"\n下一步：")
        print(f"  1. 查看 {OUTPUT_DIR}/{OPTIMIZED_SYSTEM_PROMPT_FILE}")
        print(f"  2. 查看 {OUTPUT_DIR}/{OPTIMIZED_USER_TEMPLATE_FILE}")
        print(f"  3. 在 main.py 的 MainConfig 設定：")
        print(f"       system_prompt_path = Path('{OUTPUT_DIR}/{OPTIMIZED_SYSTEM_PROMPT_FILE}')")
        print(f"       user_template_path = Path('{OUTPUT_DIR}/{OPTIMIZED_USER_TEMPLATE_FILE}')")


if __name__ == "__main__":
    main()