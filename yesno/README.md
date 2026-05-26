# Yes/No Prompt Templates

Prompts used in the BioASQ 14b Phase B yes/no question answering experiments.
All prompts were evaluated on BioASQ 11b–13b development data unless noted otherwise.
Primary backbone: Claude Opus 4.6 via OpenRouter, temperature = 0.

---

## Files

| File | Description | Macro-F1 |
|------|-------------|----------|
| `baseline_prompt.md` | Chain-of-thought output; used as default for all experiments | 0.9399 |
| `baseline_yesno_only.md` | Direct binary output; used for competition submissions | 0.9399 |
| `critical_reviewer.md` | Best prompt (A2-5): critical reviewer + snippet-analysis | 0.9523 |
| `dev3_refined.md` | Iterative refinement Dev-3; best on dev set but overfits | 0.9743 (dev) / 0.9667 (9b) |
| `sf_checklist.md` | Self-feedback Turn 2: default checklist critique (SF-2) | 0.9177 |
| `sf_devils_advocate.md` | Self-feedback Turn 2: devil's advocate critique (SF-3) | 0.8786 |

---

## Prompt Variables

- `{q_body}` — the question text
- `{snippets_block}` — concatenated gold snippets from BioASQ Phase B
