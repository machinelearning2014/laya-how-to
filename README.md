# laya-how-to

A worked example and a field reference for [laya](https://github.com/NandhaKishorM/laya), a
non-autoregressive decision engine that answers typed questions about a piece of text in a
single forward pass. No text generation, so there is nothing to parse and nothing to hallucinate.

**Reference page:** https://machinelearning2014.github.io/laya-how-to/

## What's here

| File | What it is |
|---|---|
| `main.py` | Eight typed questions over one support message, exercising all three question types. Real output is in the comments. |
| `index.html` | The field reference: question types, result payloads, calibrated abstention, batching, typed output, and the gotchas. Served by GitHub Pages. |
| `.gitignore` | Keeps the virtualenv (745 MB) out of the repo. |

## Run the example

```bash
python3 -m venv .venv
.venv/bin/python -m pip install laya
.venv/bin/python main.py
```

The first run downloads a checkpoint (about 421M parameters for the English one) and caches it
under `~/.cache/huggingface`. Verified output:

```
billing
0.9875
1.7901
0.8429
1.6361
0.906
0.2352
0.373
english
```

## The three question types

```python
questions = {
    # choice: pick one label, get the full distribution back
    "department": {"type": "choice", "instructions": "Which department should handle `message`?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},

    # score: ordered levels of any length. Returns the expected level as a float, not an int.
    "urgency": {"type": "score", "instructions": "How urgent is `message`?",
                "criteria": ["not urgent", "soon", "blocking"]},

    # noul: yes/no under uncertainty. Always returns P(true) in 0..1.
    "churn_risk": {"type": "noul", "instructions": "Does `message` threaten to cancel or leave?"},
}
```

Two conventions worth adopting:

- **Name the state field in backticks** (`` `message` ``). Every preset in laya does this, and
  `laya.presets.state_field()` reads it back to learn where to put your text. It is what makes a
  question set portable to the CLI (`laya --questions`) and to `decide()`.
- **Gate on `answer_confidence`, never on the raw value.** For `noul`, a low number is a
  *confident no*, not uncertainty. In the example, `deadline_mentioned` returns `noul = 0.24`
  with `answer_confidence = 0.76` — the model is sure there is no deadline. The one genuinely
  uncertain answer is `sentiment` at `answer_confidence = 0.46`.

## Adding questions is nearly free

State and every question's options are packed into one sequence, with each option occupying its
own masked slot. One forward pass scores them all, so eight questions cost about what one costs.
Splitting them into eight separate `predict` calls would not.

## Related

- [laya on PyPI](https://pypi.org/project/laya/) · [model card](https://huggingface.co/convaiinnovations/laya) · [docs](https://nandhakishorm.github.io/laya/)
- The reference page covers `predict_batch`, `predict_long`, `shortlist`, `decide`, the CLI,
  serving over HTTP and MCP, hooks, and calibration.
