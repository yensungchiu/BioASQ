# sf_devils_advocate (Self-Feedback — Devil's Advocate Critique)

Used in: Self-feedback experiment (SF-3), Turn 2 critique prompt  
Model: Claude Opus 4.6  
Result: Macro-F1 = 0.8786 (baseline: 0.9258)  
Note: Most aggressive critique variant. Significantly degraded performance
      by overriding correct initial answers. Changed 6/102 predictions;
      2 became correct, 3 became wrong.

---

## Critique Prompt (Turn 2)

```
Assume for a moment that your current answer is WRONG.

Your task is to build the strongest possible case AGAINST your answer
using only the provided snippets.

Step 1 -- State the opposite answer (if you said "yes", argue for "no", and vice versa).
Step 2 -- Find every snippet that supports this opposite position. Quote relevant phrases.
Step 3 -- Identify weaknesses or gaps in your original reasoning that an opponent could exploit.
Step 4 -- After fully arguing the opposing case, decide: does this counter-argument change
          your answer, or does your original answer still hold? Explain why.

Be rigorous. A weak devil's advocate case means your original answer is robust.
A strong one means you should reconsider.
```
