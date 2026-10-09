"""
document.py — BioASQ Yes/No experiment runner (main.py + PubMed abstract enrichment)

Modify only ExperimentConfig at the top of this file, then run:
    python document.py
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

try:
    import wandb
except Exception:
    wandb = None  # type: ignore

@dataclass(frozen=True)
class ExperimentConfig:

    # ── Experiment name (shown in W&B) ────────────────────────────────────────
    wandb_run_name: Optional[str] = "G-sni_fir"

    # ── Model ─────────────────────────────────────────────────────────────────
    model: Optional[str] = "anthropic/claude-opus-4.6"

    # ── Data paths ────────────────────────────────────────────────────────────
    doc_ids: Optional[List[str]] = field(default_factory=lambda: ["12b_01", "12b_02", "12b_03", "12b_04", "13b_01", "13b_02", "13b_03"])
    use_abstracts: bool = True
    use_snippets: bool = True
    # True  → abstracts placed before snippets in the prompt
    # False → abstracts placed after snippets
    abstracts_first: bool = False

    # True  → add === SNIPPETS === / === ABSTRACTS === section headers
    use_section_headers: bool = True

    test_dir:         Path = Path("data/testData")
    test_results_dir: Path = Path("data/testResults")
    gold_path:        Path = Path("data/predictions/training13b.json")

    # ── OpenRouter ────────────────────────────────────────────────────────────
    base_url:    str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"

    # ── Generation params ─────────────────────────────────────────────────────
    temperature:      float           = 0.0
    topP:             Optional[float] = None
    topK:             Optional[int]   = None
    presencePenalty:  Optional[float] = None
    frequencyPenalty: Optional[float] = None
    thinkingLevel:    Optional[str]   = None
    maxOutputTokens:  int             = 16384

    # ── Concurrency / retry ───────────────────────────────────────────────────
    request_timeout_s: float = 2000.0
    concurrency:       int   = 20
    max_retries:       int   = 2
    backoff_base_s:    float = 0.8
    backoff_cap_s:     float = 20.0

    # ── Filtering ─────────────────────────────────────────────────────────────
    only_yesno:            bool          = True
    max_chars_per_snippet: Optional[int] = None
    missing_as:            str           = "no"

    # ── Custom prompt files (None = use built-in defaults) ────────────────────
    system_prompt_path: Optional[Path] = None
    user_template_path: Optional[Path] = None

    # ── PubMed abstract enrichment ────────────────────────────────────────────
    # True  → fetch PubMed abstracts and add to context before generation
    # False → use only BioASQ snippets (original behaviour)
    

    # Disk cache for fetched abstracts (avoids re-fetching same PMID)
    abstract_cache_dir: Path = Path("data/abstract_cache")

    # NCBI API key env var name (.env: NCBI_API_KEY=xxx)
    # Without key: 3 req/s   |   With key: 10 req/s
    # Free signup: https://www.ncbi.nlm.nih.gov/account/
    ncbi_api_key_env: str = "NCBI_API_KEY"

    # ★ Required by NCBI policy – set your email to avoid blocks ★
    ncbi_tool:  str = "bioasq_yesno"
    ncbi_email: str = "11335007@st.chjhs.tp.edu.tw"

    # Max simultaneous NCBI connections (no key: ≤ 3; with key: ≤ 10)
    ncbi_concurrency: int   = 3
    ncbi_timeout_s:   float = 30.0
    ncbi_batch_size:  int   = 10   # PMIDs per request (NCBI recommends ≤ 20)

    # Max chars per abstract (None = no limit)
    max_chars_per_abstract: Optional[int] = None

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_enabled:         bool          = True
    wandb_project:         str           = "yesno"
    wandb_entity:          Optional[str] = "bioasq"
    wandb_group:           Optional[str] = "CSHS"
    wandb_mode:            Optional[str] = "online"
    wandb_table_max_chars: int           = 100000


# =============================================================================
# Default prompts
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are a critical biomedical reviewer. Answer Yes/No questions only when the provided snippets and abstracts contain direct and unambiguous evidence. If snippets or abstracts conflict with each other, are only tangentially related to the question, or do not clearly support a yes answer, answer no. Your response must consist of exactly and only the single word "yes" or "no". Do not include any reasoning, explanation, punctuation, or extra characters.
"""

DEFAULT_USER_TEMPLATE = """### INSTRUCTIONS:
1. Read the question and each numbered snippet  and abstracts carefully.
2. For each relevant snippet  and abstracts, note its number [N] and what evidence it provides.
3. Based only on the snippets  and abstracts , reason toward your final answer.
4. Do not use outside knowledge.
5. Your response must consist of exactly and only the single word "yes" or "no". Do not include any reasoning, explanation, punctuation, or extra characters.

### QUESTION:
{q_body}

### SNIPPETS:
{snippets_block}

### ANSWER:
"""

# =============================================================================
# Type aliases
# =============================================================================

YesNo            = Literal["yes", "no"]
GenerationStatus = Literal["ok", "api_error", "timeout", "parse_error"]


# =============================================================================
# PART 1 — PubMed Abstract Fetcher
#
# API reference: https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EFetch
#
# Correct parameters for PubMed XML:
#   db=pubmed  &  id=<pmid1,pmid2,...>  &  retmode=xml  &  rettype=abstract
#
# XML response contains <PubmedArticle> elements, each with:
#   <PMID>15858239</PMID>
#   <AbstractText>Full abstract text here...</AbstractText>
# =============================================================================

_NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_RPS_NO_KEY      = 3.0
_RPS_WITH_KEY    = 10.0

_PMID_RE_OLD = re.compile(r"/pubmed/(\d+)",                     re.IGNORECASE)
_PMID_RE_NEW = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.IGNORECASE)


def pmid_from_url(url: str) -> Optional[str]:
    """Extract PMID from any PubMed URL variant."""
    m = _PMID_RE_OLD.search(url) or _PMID_RE_NEW.search(url)
    return m.group(1) if m else None


def pmids_from_question(q: dict) -> List[str]:
    """Return ordered deduplicated PMIDs from a question's 'documents' field."""
    seen: set = set()
    out:  List[str] = []
    for url in q.get("documents", []):
        pid = pmid_from_url(str(url))
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


# ── Disk cache ────────────────────────────────────────────────────────────────

class AbstractCache:
    """One .txt file per PMID under cache_dir; in-memory layer on top."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, str] = {}

    def get(self, pmid: str) -> Optional[str]:
        if pmid in self._mem:
            return self._mem[pmid]
        p = self._fpath(pmid)
        if p.exists():
            text = p.read_text(encoding="utf-8").strip()
            self._mem[pmid] = text
            return text
        return None

    def set(self, pmid: str, text: str) -> None:
        self._mem[pmid] = text
        self._fpath(pmid).write_text(text, encoding="utf-8")

    def has(self, pmid: str) -> bool:
        return pmid in self._mem or self._fpath(pmid).exists()

    def stats(self) -> Dict[str, int]:
        return {"cached_files": len(list(self.cache_dir.glob("*.txt")))}

    def _fpath(self, pmid: str) -> Path:
        return self.cache_dir / f"{pmid}.txt"


# ── FetchResult ───────────────────────────────────────────────────────────────

class FetchResult:
    __slots__ = ("pmid", "abstract", "success", "error")

    def __init__(
        self, pmid: str, abstract: str,
        success: bool = True, error: str = ""
    ) -> None:
        self.pmid     = pmid
        self.abstract = abstract
        self.success  = success
        self.error    = error

    def __repr__(self) -> str:
        tag = "OK" if self.success else f"ERR({self.error[:60]})"
        return f"FetchResult({self.pmid}, {tag}, len={len(self.abstract)})"


# ── XML parser ────────────────────────────────────────────────────────────────

def _parse_efetch_xml(xml_text: str, pmids: Sequence[str]) -> List[FetchResult]:
    """Parse PubMed XML response (retmode=xml, rettype=abstract).

    XML structure per article:
        <PubmedArticle>
          <MedlineCitation>
            <PMID>15858239</PMID>
            <Article>
              <ArticleTitle>...</ArticleTitle>
              <Abstract>
                <AbstractText>Full abstract here...</AbstractText>
                <!-- or structured: -->
                <AbstractText Label="BACKGROUND">...</AbstractText>
                <AbstractText Label="RESULTS">...</AbstractText>
              </Abstract>
            </Article>
          </MedlineCitation>
        </PubmedArticle>
    """
    pmid_to_abstract: Dict[str, str] = {}

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        err = f"XML parse error: {e}"
        print(f"[NCBI] {err}", file=sys.stderr)
        return [FetchResult(p, "", False, err) for p in pmids]

    for article in root.iter("PubmedArticle"):
        # Get PMID
        pmid_el = article.find(".//MedlineCitation/PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text.strip()

        # Collect all AbstractText elements (handles structured abstracts)
        abstract_parts: List[str] = []
        for ab_el in article.findall(".//Abstract/AbstractText"):
            label = ab_el.get("Label", "")
            text  = (ab_el.text or "").strip()
            # Also include tail text that may appear in mixed content
            if not text:
                # Try itertext for mixed content nodes
                text = "".join(ab_el.itertext()).strip()
            if not text:
                continue
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)

        if abstract_parts:
            pmid_to_abstract[pmid] = " ".join(abstract_parts)

    results: List[FetchResult] = []
    for pmid in pmids:
        ab = pmid_to_abstract.get(pmid, "")
        if ab:
            results.append(FetchResult(pmid, ab, True))
        else:
            results.append(FetchResult(
                pmid, "", False,
                f"PMID {pmid}: no abstract in XML response "
                f"(article may not have an abstract, or PMID not found)"
            ))
    return results


# ── Async NCBI Fetcher ────────────────────────────────────────────────────────

class NCBIFetcher:
    """Rate-limited async NCBI EFetch client.

    Uses retmode=xml for reliable structured parsing.
    Includes tool + email params as required by NCBI policy.

    Rate limiting uses asyncio.Lock so only ONE request fires at a time
    during the rate-check window, preventing the 429 bursts that happen
    when multiple coroutines all read the same _last_t simultaneously.

    On HTTP 429, automatically backs off and retries up to _max_429_retries.
    """

    _max_429_retries = 5
    _429_base_wait   = 2.0
    _429_max_wait    = 60.0

    def __init__(self, cfg: ExperimentConfig) -> None:
        self._api_key  = os.getenv(cfg.ncbi_api_key_env, "").strip() or None
        self._tool     = cfg.ncbi_tool
        self._email    = cfg.ncbi_email
        self._timeout  = aiohttp.ClientTimeout(total=cfg.ncbi_timeout_s)
        self._rps      = _RPS_WITH_KEY if self._api_key else _RPS_NO_KEY
        self._min_gap  = 1.0 / self._rps
        self._last_t   = 0.0
        self._batch    = cfg.ncbi_batch_size
        # asyncio.Lock serialises the rate-check section so only one request
        # can read+write _last_t at a time (fixes the 429 burst bug)
        self._rate_lock: asyncio.Lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "NCBIFetcher":
        self._session = aiohttp.ClientSession(
            timeout=self._timeout,
            headers={
                "User-Agent": (
                    f"python-aiohttp/bioasq-yesno (contact: {self._email})"
                ),
                "Accept": "application/xml, text/xml, */*",
            },
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def _get_xml(self, pmids: Sequence[str]) -> str:
        """One EFetch call returning PubMed XML for the given PMIDs.

        Uses a Lock so the rate-check is truly serialised: each caller waits
        for the previous one to record its send time before proceeding.
        """
        params: Dict[str, str] = {
            "db":      "pubmed",
            "id":      ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "tool":    self._tool,
            "email":   self._email,
        }
        if self._api_key:
            params["api_key"] = self._api_key

        # Serialised rate gate
        async with self._rate_lock:
            now  = time.monotonic()
            wait = self._min_gap - (now - self._last_t)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_t = time.monotonic()

        # HTTP call outside the lock (allows network I/O overlap)
        assert self._session is not None
        async with self._session.get(_NCBI_EFETCH_URL, params=params) as resp:
            status = resp.status
            text   = await resp.text(encoding="utf-8", errors="replace")
            if status != 200:
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=status,
                    message=f"HTTP {status}: {text[:300]}",
                    headers=resp.headers,
                )
            return text

    async def _fetch_batch(self, pmids: Sequence[str]) -> List[FetchResult]:
        """Fetch one batch with automatic 429 retry + exponential backoff."""
        for attempt in range(self._max_429_retries + 1):
            try:
                xml_text = await self._get_xml(pmids)
                return _parse_efetch_xml(xml_text, list(pmids))

            except aiohttp.ClientResponseError as exc:
                if exc.status == 429:
                    wait = min(
                        self._429_max_wait,
                        self._429_base_wait * (2 ** attempt),
                    )
                    print(
                        f"[NCBI] 429 rate-limit (attempt {attempt + 1}/"
                        f"{self._max_429_retries + 1}), "
                        f"backing off {wait:.1f}s ...",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait)
                    continue

                err = f"HTTP {exc.status}: {exc.message[:200]}"
                print(f"[NCBI] Fetch error: {err}", file=sys.stderr)
                return [FetchResult(p, "", False, err) for p in pmids]

            except asyncio.TimeoutError:
                err = "NCBI request timed out"
                print(f"[NCBI] {err}", file=sys.stderr)
                return [FetchResult(p, "", False, err) for p in pmids]

            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                print(f"[NCBI] Unexpected fetch error: {err}", file=sys.stderr)
                return [FetchResult(p, "", False, err) for p in pmids]

        err = f"429 rate-limit: gave up after {self._max_429_retries} retries"
        print(f"[NCBI] {err}", file=sys.stderr)
        return [FetchResult(p, "", False, err) for p in pmids]

    async def fetch_all(
        self,
        pmids: Sequence[str],
        cache: Optional[AbstractCache] = None,
    ) -> Dict[str, FetchResult]:
        """Fetch all PMIDs, serving cache hits first."""
        results:  Dict[str, FetchResult] = {}
        to_fetch: List[str] = []

        for pmid in pmids:
            if cache and cache.has(pmid):
                text = cache.get(pmid) or ""
                results[pmid] = FetchResult(pmid, text, bool(text))
            else:
                to_fetch.append(pmid)

        if not to_fetch:
            return results

        batches = [
            to_fetch[i: i + self._batch]
            for i in range(0, len(to_fetch), self._batch)
        ]
        tasks = [asyncio.create_task(self._fetch_batch(b)) for b in batches]

        for coro in asyncio.as_completed(tasks):
            for fr in await coro:
                results[fr.pmid] = fr
                if cache and fr.success and fr.abstract:
                    cache.set(fr.pmid, fr.abstract)

        return results


# ── Enrichment ────────────────────────────────────────────────────────────────

async def enrich_questions_with_abstracts(
    questions: Sequence[dict],
    cfg: ExperimentConfig,
    cache: AbstractCache,
) -> List[dict]:
    """Fetch PubMed abstracts for all document URLs in the question list.

    Adds to each question dict:
        q["_abstracts"]  : List[str]        — formatted abstract strings
        q["_fetch_meta"] : Dict[str, dict]  — pmid -> {success, error}

    Original fields (body, snippets, documents, ...) are untouched.
    """
    all_pmids: List[str] = []
    seen: set = set()
    for q in questions:
        for pid in pmids_from_question(q):
            if pid not in seen:
                seen.add(pid)
                all_pmids.append(pid)

    if not all_pmids:
        print("[NCBI] No PMIDs found in questions.", file=sys.stderr)
        return list(questions)

    n_cached = sum(1 for p in all_pmids if cache.has(p))
    print(
        f"[NCBI] {len(all_pmids)} unique PMIDs | "
        f"{n_cached} cached | {len(all_pmids) - n_cached} to fetch",
        file=sys.stderr,
    )
    if cfg.ncbi_email == "your_email@example.com":
        print(
            "[NCBI] WARNING: ncbi_email is still the placeholder value. "
            "Set ExperimentConfig.ncbi_email to your real address to avoid NCBI blocks.",
            file=sys.stderr,
        )

    async with NCBIFetcher(cfg) as fetcher:
        pmid_results = await fetcher.fetch_all(all_pmids, cache=cache)

    # Print first few errors for diagnosis
    errors = [(p, fr.error) for p, fr in pmid_results.items() if not fr.success]
    if errors:
        print(f"[NCBI] {len(errors)} failed PMIDs. First 3 errors:", file=sys.stderr)
        for pmid, err in errors[:3]:
            print(f"  PMID {pmid}: {err}", file=sys.stderr)

    max_chars = cfg.max_chars_per_abstract
    for q in questions:
        abstracts: List[str] = []
        meta: Dict[str, dict] = {}
        for pmid in pmids_from_question(q):
            fr = pmid_results.get(pmid)
            if fr and fr.success and fr.abstract:
                text = fr.abstract.strip()
                if max_chars and len(text) > max_chars:
                    text = text[:max_chars].rstrip() + "..."
                abstracts.append(f"[PMID:{pmid}]\n{text}")
            meta[pmid] = {
                "success": fr.success if fr else False,
                "error":   fr.error   if fr else "not fetched",
            }
        q["_abstracts"]  = abstracts
        q["_fetch_meta"] = meta

    n_ok   = sum(1 for fr in pmid_results.values() if fr.success and fr.abstract)
    n_fail = len(all_pmids) - n_ok
    print(f"[NCBI] Done: {n_ok} OK | {n_fail} failed", file=sys.stderr)
    return list(questions)


# =============================================================================
# PART 2 — BioASQ Generation Pipeline  (original main.py logic)
# =============================================================================

@dataclass(frozen=True)
class PromptSpec:
    system:        str = DEFAULT_SYSTEM_PROMPT
    user_template: str = DEFAULT_USER_TEMPLATE


@dataclass
class Metrics:
    n:             int
    missing_pred:  int
    tp:  int;  fp:  int;  fn:  int;  tn:  int
    accuracy:      float
    precision_yes: float;  recall_yes: float;  f1_yes: float
    precision_no:  float;  recall_no:  float;  f1_no:  float
    maF1:          float


@dataclass
class PredRecord:
    qid:               str
    qtype:             str
    body:              str
    snippet_count:     int
    abstract_count:    int
    pred_exact:        YesNo
    generation_status: GenerationStatus
    retry_count:       int
    latency_ms:        int
    raw_content:       str
    thinking_text:     str
    finish_reason:     str
    native_finish_reason: str
    error_text:        str
    user_prompt:       str   # actual prompt sent to the model (includes abstracts)
    model:             str
    temperature:       float
    topP:              Optional[float]
    topK:              Optional[int]
    maxOutputTokens:   int
    presencePenalty:   Optional[float]
    frequencyPenalty:  Optional[float]
    thinkingLevel:     Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    if value is None: return ""
    if isinstance(value, str): return value.strip()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""
    return str(value).strip()


def collect_snippet_texts(q: dict) -> List[str]:
    """Return original BioASQ snippets only (unchanged from main.py)."""
    texts: List[str] = []
    for snippet in q.get("snippets", []):
        text = snippet.get("text", None)
        if text is None:
            continue
        texts.append(text)
    return texts


def build_snippets_block(q: dict, cfg: ExperimentConfig) -> str:
    """Build the {snippets_block} string for the prompt.

    Combines BioASQ snippets and PubMed abstracts according to cfg settings.
    Section headers are added when use_section_headers=True and both sources exist.
    """
    # BioASQ snippets
    snip_parts: List[str] = []
    if cfg.use_snippets:
        for snip in q.get("snippets", []):
            text = snip.get("text") or ""
            if not text:
                continue
            max_s = cfg.max_chars_per_snippet
            if max_s and len(text) > max_s:
                text = text[:max_s].rstrip() + "..."
            snip_parts.append(text)

    # PubMed abstracts
    ab_parts: List[str] = []
    if cfg.use_abstracts:
        max_a = cfg.max_chars_per_abstract
        for ab in q.get("_abstracts", []):
            if not ab:
                continue
            if max_a and len(ab) > max_a:
                ab = ab[:max_a].rstrip() + "..."
            ab_parts.append(ab)

    if not snip_parts and not ab_parts:
        return "(No snippets provided.)"

    parts: List[str] = []
    both  = bool(snip_parts) and bool(ab_parts)

    def _add_snippets() -> None:
        if not snip_parts:
            return
        if cfg.use_section_headers and both:
            parts.append("=== SNIPPETS ===")
        for i, t in enumerate(snip_parts, 1):
            parts.append(f"[{i}]: {t}")

    def _add_abstracts() -> None:
        if not ab_parts:
            return
        if cfg.use_section_headers and both:
            parts.append("=== ABSTRACTS ===")
        for i, a in enumerate(ab_parts, 1):
            parts.append(f"[A{i}]: {a}")

    if cfg.abstracts_first:
        _add_abstracts()
        if parts and snip_parts:
            parts.append("")
        _add_snippets()
    else:
        _add_snippets()
        if parts and ab_parts:
            parts.append("")
        _add_abstracts()

    return "\n".join(parts)


def build_user_prompt_for_question(
    q: dict,
    *,
    prompt_spec: PromptSpec,
    cfg: ExperimentConfig,
) -> str:
    return prompt_spec.user_template.format(
        q_body=str(q.get("body", "")).strip(),
        snippets_block=build_snippets_block(q, cfg),
    ).strip()


def load_prompt_spec(cfg: ExperimentConfig) -> PromptSpec:
    spec          = PromptSpec()
    system        = spec.system
    user_template = spec.user_template
    if cfg.system_prompt_path and cfg.system_prompt_path.exists():
        system = cfg.system_prompt_path.read_text(encoding="utf-8").strip()
    if cfg.user_template_path and cfg.user_template_path.exists():
        user_template = cfg.user_template_path.read_text(encoding="utf-8").strip()
    return PromptSpec(system=system, user_template=user_template)


def map_yesno_labels(
    questions: Sequence[dict],
    *,
    exact_answer_key: str = "exact_answer",
) -> Dict[str, YesNo]:
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
        last = lines[-1].strip(" .,:;!\"'")
        if last == "yes": return "yes"
        if last == "no":  return "no"
    matches = re.findall(r"\b(yes|no)\b", text.lower())
    if matches:
        return "yes" if matches[-1] == "yes" else "no"
    return None


# ── PromptBuilder ─────────────────────────────────────────────────────────────

class PromptBuilder:
    def __init__(self, spec: PromptSpec, cfg: ExperimentConfig) -> None:
        self.spec = spec
        self.cfg  = cfg

    def build_messages(self, q: dict) -> List[dict]:
        return [
            {"role": "system", "content": self.spec.system.strip()},
            {"role": "user",   "content": build_user_prompt_for_question(
                q, prompt_spec=self.spec, cfg=self.cfg)},
        ]


# ── OpenRouter client ─────────────────────────────────────────────────────────

class OpenRouterClient:
    def __init__(
        self, api_key: str, base_url: str,
        *, http_referer: Optional[str] = None,
        app_title:   Optional[str] = None,
        timeout_s:   float = 90.0,
    ) -> None:
        self.api_key      = api_key
        self.base_url     = base_url.rstrip("/")
        self.timeout      = aiohttp.ClientTimeout(total=timeout_s)
        self.http_referer = http_referer
        self.app_title    = app_title
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
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        if self.http_referer: h["HTTP-Referer"] = self.http_referer
        if self.app_title:    h["X-Title"]      = self.app_title
        return h

    async def chat_completions(self, *, payload: dict) -> dict:
        if self._session is None:
            raise RuntimeError("Client not started.")
        url = f"{self.base_url}/chat/completions"
        async with self._session.post(url, headers=self._headers(), json=payload) as resp:
            raw_text = await resp.text()
            if resp.status >= 400:
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info, history=resp.history,
                    status=resp.status, message=raw_text[:5000], headers=resp.headers)
            try:
                return json.loads(raw_text)
            except Exception as e:
                raise RuntimeError(f"Non-JSON response: {raw_text[:2000]}") from e


def build_payload(cfg: ExperimentConfig, messages: List[dict]) -> dict:
    payload: Dict[str, Any] = {
        "model":       cfg.model,
        "messages":    messages,
        "temperature": cfg.temperature,
        "max_tokens":  int(cfg.maxOutputTokens),
        "tool_choice": "none",
    }
    if cfg.topP             is not None: payload["top_p"]             = cfg.topP
    if cfg.topK             is not None: payload["top_k"]             = cfg.topK
    if cfg.presencePenalty  is not None: payload["presence_penalty"]  = cfg.presencePenalty
    if cfg.frequencyPenalty is not None: payload["frequency_penalty"] = cfg.frequencyPenalty
    if cfg.thinkingLevel    is not None: payload["thinkingLevel"]     = cfg.thinkingLevel
    return payload


def extract_content_and_thinking(resp_json: dict) -> Tuple[str, str, str, str]:
    choices = resp_json.get("choices") or []
    if not choices:
        return "", "", "", ""
    c0            = choices[0] or {}
    finish_reason = str(c0.get("finish_reason") or "")
    native_finish = str(c0.get("native_finish_reason") or "")
    msg           = c0.get("message") if isinstance(c0.get("message"), dict) else {}
    msg           = msg or {}
    content       = str(msg.get("content") or "")
    thinking      = str(msg.get("reasoning") or "")
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
    return content, thinking, finish_reason, native_finish


async def retry_chat(
    client: OpenRouterClient,
    *,
    payload: dict,
    max_retries: int,
    backoff_base_s: float,
    backoff_cap_s: float,
) -> Tuple[dict, int, int, GenerationStatus, str]:
    start_all  = time.perf_counter()
    last_err   = ""
    last_status: GenerationStatus = "api_error"

    for attempt in range(max_retries + 1):
        try:
            t0         = time.perf_counter()
            resp_json  = await client.chat_completions(payload=payload)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return resp_json, attempt, latency_ms, "ok", ""
        except asyncio.TimeoutError:
            last_status, last_err = "timeout", "TIMEOUT"
        except aiohttp.ClientResponseError as e:
            last_status = "api_error"
            last_err    = f"HTTP_ERROR status={e.status} msg={str(e)[:2000]}"
        except aiohttp.ClientError as e:
            last_status, last_err = "api_error", f"CLIENT_ERROR {repr(e)[:2000]}"
        except Exception as e:
            last_status, last_err = "api_error", f"UNEXPECTED_ERROR {repr(e)[:2000]}"

        if attempt >= max_retries:
            break
        sleep_s = (
            min(backoff_cap_s, backoff_base_s * (2 ** attempt))
            * (0.8 + 0.4 * random.random())
        )
        await asyncio.sleep(sleep_s)

    latency_ms = int((time.perf_counter() - start_all) * 1000)
    return {}, max_retries, latency_ms, last_status, last_err


# ── AsyncGenerator ────────────────────────────────────────────────────────────

class AsyncGenerator:
    def __init__(
        self,
        cfg:            ExperimentConfig,
        prompt:         PromptBuilder,
        client:         OpenRouterClient,
        abstract_cache: Optional[AbstractCache] = None,
    ) -> None:
        self.cfg            = cfg
        self.prompt         = prompt
        self.client         = client
        self.abstract_cache = abstract_cache
        self._sem           = asyncio.Semaphore(max(1, cfg.concurrency))

    async def _generate_one(self, q: dict) -> PredRecord:
        qid            = str(q.get("id", "")).strip()
        qtype          = normalize_qtype(str(q.get("type", "")))
        body           = str(q.get("body", "")).strip()
        snippet_count  = len(collect_snippet_texts(q))
        abstract_count = len(q.get("_abstracts", []))

        messages    = self.prompt.build_messages(q)
        user_prompt = messages[1]["content"] if len(messages) > 1 else ""
        payload     = build_payload(self.cfg, messages)

        async with self._sem:
            resp_json, retry_count, latency_ms, status, err_text = await retry_chat(
                self.client, payload=payload,
                max_retries=self.cfg.max_retries,
                backoff_base_s=self.cfg.backoff_base_s,
                backoff_cap_s=self.cfg.backoff_cap_s,
            )

        content, thinking, finish_reason, native_finish = (
            extract_content_and_thinking(resp_json)
        )

        pred: YesNo = "no"
        if status == "ok":
            yn = _coerce_yesno_from_text(content)
            if yn is None:
                status = "parse_error"
                pred   = "no"
            else:
                pred = yn

        return PredRecord(
            qid=qid, qtype=qtype, body=body,
            snippet_count=snippet_count, abstract_count=abstract_count,
            pred_exact=pred, generation_status=status,
            retry_count=retry_count, latency_ms=latency_ms,
            user_prompt=user_prompt,
            raw_content=(content or "").strip(),
            thinking_text=(thinking or "").strip(),
            finish_reason=(finish_reason or "").strip(),
            native_finish_reason=(native_finish or "").strip(),
            error_text=(err_text or "").strip(),
            model=self.cfg.model or "",
            temperature=float(self.cfg.temperature),
            topP=self.cfg.topP, topK=self.cfg.topK,
            maxOutputTokens=int(self.cfg.maxOutputTokens),
            presencePenalty=self.cfg.presencePenalty,
            frequencyPenalty=self.cfg.frequencyPenalty,
            thinkingLevel=self.cfg.thinkingLevel,
        )

    async def generate_file(
        self, input_path: Path
    ) -> Tuple[dict, List[PredRecord]]:
        raw = load_json_file(input_path)
        if not isinstance(raw, dict):
            return {"questions": []}, []
        raw_questions = raw.get("questions")
        if not isinstance(raw_questions, list):
            return {"questions": []}, []

        questions: List[dict] = [q for q in raw_questions if isinstance(q, dict)]
        if not questions:
            return {"questions": []}, []

        # Filter by question type
        work: List[dict] = []
        for q in questions:
            qt = normalize_qtype(str(q.get("type", "")))
            if self.cfg.only_yesno and qt != "yesno":
                continue
            work.append(q)

        # ── PubMed abstract enrichment ──────────────────────────────────────
        if self.cfg.use_abstracts and work and self.abstract_cache is not None:
            print(f"[NCBI] {input_path.name}: fetching abstracts...", file=sys.stderr)
            work = await enrich_questions_with_abstracts(
                work, self.cfg, self.abstract_cache
            )

        # ── Parallel generation ─────────────────────────────────────────────
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
            # Remove internal fetch fields from submission JSON
            q_out.pop("_abstracts",  None)
            q_out.pop("_fetch_meta", None)

            qid = str(q_out.get("id", "")).strip()
            qt  = normalize_qtype(str(q_out.get("type", "")))
            q_out["ideal_answer"] = normalize_ideal_answer_for_submission(
                q_out.get("ideal_answer")
            )
            if (not self.cfg.only_yesno) or (qt == "yesno"):
                p = pred_by_id.get(qid)
                q_out["exact_answer"] = p.pred_exact if p else "no"
            out_questions.append(q_out)

        return {"questions": out_questions}, preds


# ── generate_for_files ────────────────────────────────────────────────────────

async def generate_for_files(
    input_paths: Sequence[Path],
    *,
    cfg:         ExperimentConfig,
    prompt_spec: Optional[PromptSpec] = None,
) -> Tuple[List[Path], List[PredRecord]]:
    load_dotenv()

    api_key = os.getenv(cfg.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Put {cfg.api_key_env}=... in .env")

    resolved_base_url = (
        os.getenv("OPENROUTER_BASE_URL", cfg.base_url).strip() or cfg.base_url
    )
    resolved_model = (
        os.getenv("OPENROUTER_MODEL", cfg.model or "").strip() or (cfg.model or "")
    )
    http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip() or None
    app_title    = os.getenv("OPENROUTER_APP_TITLE",    "").strip() or None

    runtime_cfg = ExperimentConfig(**{
        **cfg.__dict__,
        "base_url": resolved_base_url,
        "model":    resolved_model,
    })

    prompt_spec    = prompt_spec or load_prompt_spec(runtime_cfg)
    prompt_builder = PromptBuilder(prompt_spec, runtime_cfg)

    abstract_cache: Optional[AbstractCache] = None
    if runtime_cfg.use_abstracts:
        abstract_cache = AbstractCache(runtime_cfg.abstract_cache_dir)
        print(
            f"[NCBI] cache dir: {runtime_cfg.abstract_cache_dir} "
            f"| {abstract_cache.stats()}",
            file=sys.stderr,
        )

    runtime_cfg.test_results_dir.mkdir(parents=True, exist_ok=True)

    all_preds:    List[PredRecord] = []
    result_paths: List[Path]       = []

    async with OpenRouterClient(
        api_key=api_key, base_url=runtime_cfg.base_url,
        http_referer=http_referer, app_title=app_title,
        timeout_s=runtime_cfg.request_timeout_s,
    ) as client:
        generator = AsyncGenerator(
            runtime_cfg, prompt_builder, client, abstract_cache
        )

        for input_path in input_paths:
            doc_id = derive_doc_id(input_path)
            submission_json, preds = await generator.generate_file(input_path)

            out_path = runtime_cfg.test_results_dir / derive_result_filename(doc_id)
            safe_json_dump(out_path, submission_json)
            result_paths.append(out_path)
            all_preds.extend(preds)

    return result_paths, all_preds


# =============================================================================
# PART 3 — Metrics
# =============================================================================

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
        elif g == "no"  and p == "no":  tn += 1

    n      = len(gold)
    acc    = _safe_div(tp + tn, n)
    p_yes  = _safe_div(tp, tp + fp);  r_yes = _safe_div(tp, tp + fn)
    p_no   = _safe_div(tn, tn + fn);  r_no  = _safe_div(tn, tn + fp)
    f1_yes = _f1(p_yes, r_yes);       f1_no  = _f1(p_no,  r_no)
    return Metrics(
        n=n, missing_pred=missing, tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=acc,
        precision_yes=p_yes, recall_yes=r_yes, f1_yes=f1_yes,
        precision_no=p_no,   recall_no=r_no,   f1_no=f1_no,
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


def run_evaluation_from_preds(
    *,
    cfg:        ExperimentConfig,
    test_paths: List[Path],
    preds:      List[PredRecord],
) -> Tuple[Dict[str, Metrics], Metrics]:
    gold = map_gold_labels(cfg.gold_path)
    if not gold:
        raise RuntimeError(f"No gold yes/no labels found in: {cfg.gold_path}")

    pred_map: Dict[str, YesNo] = {
        p.qid: p.pred_exact
        for p in preds if p.qtype == "yesno" and p.qid
    }

    per_doc:   Dict[str, Metrics] = {}
    union_ids: Set[str]           = set()

    for tp in test_paths:
        did     = derive_doc_id_from_phase_filename(tp) or tp.stem
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


# =============================================================================
# PART 4 — W&B Logger
# =============================================================================

def build_doc_metrics_rows(per_doc: Dict[str, Metrics]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for doc_id, m in sorted(per_doc.items()):
        rows.append({
            "doc_id": doc_id, "n": m.n, "missing_pred": m.missing_pred,
            "accuracy": m.accuracy,    "maF1": m.maF1,
            "precision_yes": m.precision_yes, "recall_yes": m.recall_yes,
            "f1_yes": m.f1_yes,
            "precision_no":  m.precision_no,  "recall_no":  m.recall_no,
            "f1_no":  m.f1_no,
        })
    return rows


def build_questions_rows(
    *,
    cfg:         ExperimentConfig,
    test_paths:  List[Path],
    preds:       List[PredRecord],
    prompt_spec: PromptSpec,
) -> List[Dict[str, Any]]:
    gold_map   = map_gold_labels(cfg.gold_path)
    pred_by_id = {p.qid: p for p in preds if p.qid}
    rows: List[Dict[str, Any]] = []

    for tp in test_paths:
        did = derive_doc_id_from_phase_filename(tp) or tp.stem
        for q in load_questions(tp):
            if normalize_qtype(str(q.get("type", ""))) != "yesno":
                continue
            qid = str(q.get("id", "")).strip()
            if not qid:
                continue

            gold_answer   = gold_map.get(qid, None)
            pred          = pred_by_id.get(qid)
            system_answer = pred.pred_exact if pred else None

            rows.append({
                "doc_id":            did,
                "id":                qid,
                "body":              str(q.get("body", "")).strip(),
                "snippet_count":     pred.snippet_count  if pred else len(collect_snippet_texts(q)),
                "abstract_count":    pred.abstract_count if pred else 0,
                "gold_answer":       gold_answer,
                "system_answer":     system_answer,
                "correct":           (gold_answer is not None) and (system_answer == gold_answer),
                "generation_status": pred.generation_status if pred else "api_error",
                "system_prompt":     prompt_spec.system,
                "user_prompt":       (
                                         pred.user_prompt if pred and pred.user_prompt
                                         else build_user_prompt_for_question(
                                             q, prompt_spec=prompt_spec, cfg=cfg)
                                     ),
                "thinking_text":     pred.thinking_text if pred else "",
                "raw_output":        pred.raw_content   if pred else "",
                "finish_reason":     pred.finish_reason if pred else "",
                "temperature":       pred.temperature   if pred else cfg.temperature,
                "model":             pred.model         if pred else (cfg.model or ""),
                "error_text":        pred.error_text    if pred else "",
                "retry_count":       pred.retry_count   if pred else None,
            })
    return rows


class WandbLogger:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg  = cfg
        self._run = None

    def start(self, run_config: Dict[str, Any]) -> None:
        if not self.cfg.wandb_enabled or wandb is None:
            return
        init_kwargs: Dict[str, Any] = {
            "project": self.cfg.wandb_project,
            "config":  run_config,
        }
        if self.cfg.wandb_entity:   init_kwargs["entity"] = self.cfg.wandb_entity
        if self.cfg.wandb_run_name: init_kwargs["name"]   = self.cfg.wandb_run_name
        if self.cfg.wandb_group:    init_kwargs["group"]  = self.cfg.wandb_group
        if self.cfg.wandb_mode:     init_kwargs["mode"]   = self.cfg.wandb_mode
        self._run = wandb.init(**init_kwargs)

    def finish(self) -> None:
        if self._run and wandb is not None:
            wandb.finish()
            self._run = None

    def log_prompt(self, *, system_prompt: str, user_template: str) -> None:
        if not self._run or wandb is None:
            return
        wandb.config.update(
            {"prompt/system": system_prompt, "prompt/user_template": user_template},
            allow_val_change=True,
        )

    def _trunc(self, value: Any, max_chars: int) -> str:
        text = "" if value is None else str(value)
        return text if len(text) <= max_chars else text[:max(0, max_chars - 1)] + "..."

    def log_questions_table(
        self,
        rows: Sequence[Dict[str, Any]],
        *,
        max_text_chars: int = 4000,
        table_name:     str = "questions",
    ) -> None:
        if not self._run or wandb is None:
            return
        columns = [
            "doc_id", "id", "body", "snippet_count", "abstract_count",
            "gold_answer", "system_answer", "correct", "generation_status",
            "system_prompt", "user_prompt", "thinking_text", "raw_output",
            "finish_reason", "temperature", "model", "error_text", "retry_count",
        ]
        table = wandb.Table(columns=columns)
        for r in rows:
            table.add_data(
                r.get("doc_id", ""),
                r.get("id", ""),
                self._trunc(r.get("body", ""),          max_text_chars),
                int(r.get("snippet_count",  0) or 0),
                int(r.get("abstract_count", 0) or 0),
                r.get("gold_answer",   None),
                r.get("system_answer", None),
                bool(r.get("correct", False)),
                r.get("generation_status", ""),
                self._trunc(r.get("system_prompt", ""), max_text_chars),
                self._trunc(r.get("user_prompt",   ""), max_text_chars),
                self._trunc(r.get("thinking_text", ""), max_text_chars),
                self._trunc(r.get("raw_output",    ""), max_text_chars),
                r.get("finish_reason", ""),
                r.get("temperature", None),
                r.get("model", ""),
                self._trunc(r.get("error_text", ""),    max_text_chars),
                r.get("retry_count", None),
            )
        wandb.log({table_name: table})

    def log_doc_metrics_table(
        self,
        rows:       Sequence[Dict[str, Any]],
        *,
        table_name: str = "per_doc_metrics",
    ) -> None:
        if not self._run or wandb is None:
            return
        columns = [
            "doc_id", "n", "missing_pred", "accuracy", "maF1",
            "precision_yes", "recall_yes", "f1_yes",
            "precision_no",  "recall_no",  "f1_no",
        ]
        table = wandb.Table(columns=columns)
        for r in rows:
            table.add_data(
                r.get("doc_id", ""),
                int(r.get("n",            0) or 0),
                int(r.get("missing_pred", 0) or 0),
                float(r.get("accuracy",      0.0) or 0.0),
                float(r.get("maF1",          0.0) or 0.0),
                float(r.get("precision_yes", 0.0) or 0.0),
                float(r.get("recall_yes",    0.0) or 0.0),
                float(r.get("f1_yes",        0.0) or 0.0),
                float(r.get("precision_no",  0.0) or 0.0),
                float(r.get("recall_no",     0.0) or 0.0),
                float(r.get("f1_no",         0.0) or 0.0),
            )
        wandb.log({table_name: table})

    def log_overall_metrics(self, metrics: Dict[str, Any]) -> None:
        if not self._run or wandb is None:
            return
        wandb.log({f"overall/{k}": v for k, v in metrics.items()})

    def log_abstract_stats(self, preds: List[PredRecord]) -> None:
        """Log abstract fetch statistics to W&B."""
        if not self._run or wandb is None:
            return
        yp = [p for p in preds if p.qtype == "yesno"]
        if not yp:
            return
        total_ab = sum(p.abstract_count for p in yp)
        with_ab  = sum(1 for p in yp if p.abstract_count > 0)
        wandb.log({
            "abstracts/total_fetched":        total_ab,
            "abstracts/questions_with_ab":    with_ab,
            "abstracts/questions_without_ab": len(yp) - with_ab,
            "abstracts/avg_per_question":     total_ab / len(yp) if yp else 0,
        })


# =============================================================================
# PART 5 — main
# =============================================================================

def main() -> None:
    # ★  Change ExperimentConfig above to modify experiment settings  ★
    cfg = ExperimentConfig()

    if not cfg.gold_path.exists():
        raise SystemExit(f"Gold file not found: {cfg.gold_path}")

    candidate_ids = (
        discover_existing_doc_ids(cfg.test_dir)
        if cfg.doc_ids is None else cfg.doc_ids
    )
    test_paths = resolve_test_paths(cfg.test_dir, candidate_ids)
    if not test_paths:
        raise SystemExit(
            "No test files found. Check test_dir and doc_ids in ExperimentConfig."
        )

    prompt_spec = load_prompt_spec(cfg)

    print(f"\n[config] model             : {cfg.model}")
    print(f"[config] doc_ids           : {candidate_ids}")
    print(f"[config] use_abstracts     : {cfg.use_abstracts}")
    if cfg.use_abstracts:
        print(f"[config] ncbi_email        : {cfg.ncbi_email}")
        print(f"[config] abstracts_first   : {cfg.abstracts_first}")
        print(f"[config] use_section_hdrs  : {cfg.use_section_headers}")
        print(f"[config] max_chars_per_ab  : {cfg.max_chars_per_abstract}")

    # ── Generation ────────────────────────────────────────────────────────────
    result_paths, all_preds = asyncio.run(
        generate_for_files(test_paths, cfg=cfg, prompt_spec=prompt_spec)
    )

    # ── Evaluation ───────────────────────────────────────────────────────────
    per_doc_metrics, overall_metrics = run_evaluation_from_preds(
        cfg=cfg, test_paths=test_paths, preds=all_preds,
    )
    for did in sorted(per_doc_metrics.keys()):
        print_metrics(f"DOC {did}", per_doc_metrics[did])
    print_metrics("OVERALL", overall_metrics)

    # ── W&B ──────────────────────────────────────────────────────────────────
    if cfg.wandb_enabled:
        if wandb is None:
            print("[WARN] wandb not installed; skipping W&B logging.", file=sys.stderr)
        else:
            wb = WandbLogger(cfg)
            run_config = {
                "model":                  cfg.model,
                "temperature":            cfg.temperature,
                "topP":                   cfg.topP,
                "topK":                   cfg.topK,
                "presencePenalty":        cfg.presencePenalty,
                "frequencyPenalty":       cfg.frequencyPenalty,
                "thinkingLevel":          cfg.thinkingLevel,
                "concurrency":            cfg.concurrency,
                "request_timeout_s":      cfg.request_timeout_s,
                "max_retries":            cfg.max_retries,
                "only_yesno":             cfg.only_yesno,
                "max_chars_per_snippet":  cfg.max_chars_per_snippet,
                "missing_as":             cfg.missing_as,
                "test_dir":               str(cfg.test_dir),
                "gold_path":              str(cfg.gold_path),
                "docs":                   candidate_ids,
                "use_abstracts":          cfg.use_abstracts,
                "abstracts_first":        cfg.abstracts_first,
                "use_section_headers":    cfg.use_section_headers,
                "max_chars_per_abstract": cfg.max_chars_per_abstract,
                "ncbi_concurrency":       cfg.ncbi_concurrency,
                "ncbi_batch_size":        cfg.ncbi_batch_size,
            }
            wb.start(run_config)
            wb.log_prompt(
                system_prompt=prompt_spec.system,
                user_template=prompt_spec.user_template,
            )
            wb.log_questions_table(
                build_questions_rows(
                    cfg=cfg, test_paths=test_paths,
                    preds=all_preds, prompt_spec=prompt_spec,
                ),
                max_text_chars=cfg.wandb_table_max_chars,
                table_name="questions",
            )
            wb.log_doc_metrics_table(
                build_doc_metrics_rows(per_doc_metrics),
                table_name="per_doc_metrics",
            )
            wb.log_abstract_stats(all_preds)

            overall_dict = metrics_to_dict(overall_metrics)
            keep_keys = [
                "accuracy", "maF1",
                "precision_yes", "recall_yes", "f1_yes",
                "precision_no",  "recall_no",  "f1_no",
            ]
            wb.log_overall_metrics(
                {k: overall_dict[k] for k in keep_keys if k in overall_dict}
            )
            wb.finish()

    print("\n[done] Result files:")
    for rp in result_paths:
        print(" -", rp)


if __name__ == "__main__":
    main()