# BioASQ — Team CSA-IISR

This repository contains the code of **CSA-IISR** for the BioASQ challenge (Task 14b, Phase B – question answering from gold snippets).

The `yesno/` folder holds the yes/no prompt templates; the Python scripts in the repository root run the experiments.

## Scripts

| File | Description |
|------|-------------|
| `main.py` | Baseline async runner for yes/no questions (LLM via OpenRouter, W&B logging, macro-F1 evaluation). |
| `document.py` | `main.py` plus PubMed abstract enrichment (NCBI E-utilities) in addition to the BioASQ snippets. |
| `single_snippet.py` | Snippet-by-snippet prediction, then majority vote over all snippets. |
| `majority_voting.py` | Majority voting over *N* runs, with shuffled snippet order (`random_order`) or temperature > 0 (`temperature`). |
| `majority_voting_temp.py` | Variant of `majority_voting.py`. |
| `snippet_ranking.py` | Orders snippets by cosine similarity to the question. |
| `in_context_ranking.py` | Snippet ranking plus retrieved in-context examples (ChromaDB). |
| `self_feedback.py` | 3-turn self-feedback pipeline: generate → critique → finalize. |
| `experiment_dspy.py` | Automatic prompt optimization with DSPy (MIPRO). |
| `experiment_textgrad.py` | Prompt optimization with TextGrad. |
| `summary.py` | Runner for summary-type questions. |
| `get_wandb.py` | Fetches a run's history from Weights & Biases. |

## Setup

```bash
pip install aiohttp python-dotenv tqdm wandb dspy textgrad chromadb
```

Create a `.env` file in the repository root (it is not tracked):

```
OPENROUTER_API_KEY=...
NCBI_API_KEY=...        # optional, used by document.py
WANDB_API_KEY=...       # optional
```

## Usage

Each script keeps its tunable parameters in a config block at the top of the file. Edit that block, then run the script, for example:

```bash
python main.py
python document.py
python self_feedback.py
```

BioASQ data (`phaseB_*.json`) is expected under `data/` and is not included in this repository.

## Team

**CSA-IISR**
