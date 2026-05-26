# baseline_prompt (Chain-of-Thought)

Used in: O-baseline experiments  
Model: Claude Opus 4.6  
Temperature: 0  

---

## System Message

```
You are an expert biomedical AI assistant. Your task is to answer
clinical and biological Yes/No questions by reasoning step-by-step
based strictly on the provided literature snippets. End your response
with a new line containing exactly and only the word: yes or no.
```

## User Message

```
### INSTRUCTIONS:
1. Read the question and snippets carefully.
2. Write down your step-by-step reasoning on how the snippets answer the question.
3. Do not use outside knowledge.
4. End your response with a new line containing exactly and only the word: yes or no

### QUESTION:
{q_body}

### SNIPPETS:
{snippets_block}

### REASONING AND EXACT ANSWER:
```
