# sf_checklist (Self-Feedback — Default Checklist Critique)

Used in: Self-feedback experiment (SF-2), Turn 2 critique prompt  
Model: Claude Opus 4.6  
Result: Macro-F1 = 0.9177 (baseline: 0.9258)  
Note: Applied after Turn 1 generates an initial answer.
      Changed 4/102 predictions; 3 became correct, 0 became wrong.
      Overall Macro-F1 still degraded due to no-bias shift.

---

## Critique Prompt (Turn 2)

```
Review your reasoning above. Check for:
1. Did you misread any snippet?
2. Are there snippets that CONTRADICT your answer?
3. Is the evidence clearly about the EXACT question asked, or only related?
4. Did any irrelevant snippet bias your reasoning?

Write a brief critique of your own reasoning. Identify any weaknesses.
```
