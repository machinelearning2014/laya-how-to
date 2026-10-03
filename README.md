# laya-how-to

An example tour and a field reference for [laya](https://github.com/NandhaKishorM/laya), a
non-autoregressive decision engine that answers typed questions about a piece of text in a
single forward pass. No text generation, so there is nothing to parse and nothing to hallucinate.

**Reference page:** https://machinelearning2014.github.io/laya-how-to/

## What's here

| File | What it is |
|---|---|
| `main.py` | A twelve-section tour of the in-process API, english checkpoint only. Every printed number is real output. |
| `index.html` | The field reference: question types, result payloads, calibrated abstention, batching, typed output, and the gotchas. Served by GitHub Pages. |
| `.gitignore` | Keeps the virtualenv (745 MB) out of the repo. |

## Run the example

```bash
python3 -m venv .venv
.venv/bin/python -m pip install laya
.venv/bin/python main.py
```

The first run downloads a checkpoint (about 421M parameters for the English one) and caches it
under `~/.cache/huggingface`. Nothing else downloads: the tour uses the english checkpoint
throughout, and shows the multilingual and typed-decisions routing decisions through `route()`,
which loads no weights.

It prints twelve sections. The first shows routing, and the second looks like this:

```
02. Question types: choice, score, noul
  choice -> billing of ['billing', 'technical', 'other']
  score  -> 1.7901 on a 0..2 scale (an expected level, not an index)
  noul   -> 0.8429 = P(true)
```

Three of the twelve sections print a result the model got wrong, on purpose — a schema-derived
score question, a relabelled `noul`, and a shortlist that drops the correct label. Each is
explained where it appears, because those rough edges are more useful to read than a tidy demo.

## Try it from the CLI

No Python needed for the fastest path. Routing loads no checkpoint at all:

```bash
laya "I was charged twice"      # -> english, ~40ms, nothing downloaded
```

Add `--predict` to answer. `--preset` picks a ready-made question set (`email`, `guard`,
`moderation`, `router`, `triage`); `--questions FILE` takes your own, as JSON:

```bash
laya "I was charged twice, refund me or I cancel" --preset triage --predict
```

```
intent      : refund (p=0.899)
is_urgent   : 0.154
frustration : 1.87
refund_requested: 0.909
churn_risk  : 0.089
```

Batch a file, or stdin, as JSONL:

```bash
laya --batch tickets.txt --predict --json
cat tickets.txt | laya --batch - --predict --json
```

`laya --help` lists every flag, and running it with no text gives an interactive REPL that
shares one Router across requests.

Four commands install with the package. `laya` and `laya-evals` need only core dependencies;
`laya-serve` needs `pip install "laya[serve]"` and `laya-mcp-server` needs
`pip install "laya[mcp]"` — without the extra they exit with a `ModuleNotFoundError`. The
reference page's CLI section documents every mode, including which features have no CLI
surface at all.

## Run it with no Hugging Face access

By default `Router()` pulls weights from the Hub once and caches them. To cut the Hub out
entirely — offline, air-gapped, or pinned to a copy you control — build the same three
checkpoints from this repo's GitHub Releases instead:

```bash
python fetch_model.py            # the complete mirror: all three checkpoints, ~2.2 GB
python fetch_model.py english    # or just one
python fetch_model.py --check    # re-verify the mirror; downloads nothing
python fetch_model.py --list     # what is available
```

[releases/tag/v2](https://github.com/machinelearning2014/laya-how-to/releases/tag/v2) carries a
complete copy of `convaiinnovations/laya` — all 38 files — as four archives plus a
`MANIFEST.json` recording the SHA-256 of every file:

| Asset | Contents | Size |
|---|---|---|
| `english-ckpt.tar.gz` | the english checkpoint | 778 MB |
| `multilingual-ckpt.tar.gz` | the multilingual checkpoint | 600 MB |
| `typed-decisions-ckpt.tar.gz` | the typed-decisions checkpoint | 778 MB |
| `repo-extras.tar.gz` | README, `.gitattributes`, the `rl_*` modules, `eval/`, `assets/` | 1.8 MB |

Four archives rather than one because GitHub caps a release asset at 2 GB. Extracting all of
them into one directory reproduces the upstream layout exactly, which is what the script
does — so the mirror is layout-compatible with the Hub repo itself:

```
~/.laya/mirror/model.safetensors        <- english sits at the root
~/.laya/mirror/multilingual/…
~/.laya/mirror/typed-decisions/…
~/.laya/mirror/{README.md,assets/,eval/,rl_*.py}
```

`main.py` picks it up automatically: it starts from the built-in checkpoint table and
overrides only the names it finds in the mirror, so routing to a checkpoint you have **not**
fetched still works. Section 01 prints the source of each. With the mirror in place,
`HF_HUB_OFFLINE=1 .venv/bin/python main.py` runs the whole tour — including the German
request that routes to `multilingual` — with no network and no populated Hub cache.

To point laya at a copy yourself:

```python
laya.load("~/.laya/mirror")                                  # english, at the root
Router(models={"english": "~/.laya/mirror",                  # keep routing, swap the storage
               "multilingual": ("~/.laya/mirror", "multilingual")})
```

`predict(..., model=...)` will **not** accept a path — that argument resolves strictly
against the registry (`english` / `multilingual` / `typed-decisions`), so use `models=` or
`load`. Every extracted file is verified against the manifest, not just the weights.

Release assets are **immutable**, so a new upstream revision means a new tag rather than an
overwrite — `build_release.py` rebuilds the archives deterministically (fixed mtimes, zeroed
ownership, no gzip timestamp), so a rebuild from the same source produces byte-identical
files and the recorded digests stay valid.

`fetch_model.py` follows the newest release by default, so publishing a new tag needs no code
change. Pin with `--tag v2` when you want a fixed revision.

## Refreshing the mirror when upstream moves

```bash
python check_updates.py --from-release    # is the published mirror still current?
```

It compares the Hub's `main` against the revision recorded in the manifest and, when they
differ, reports exactly what changed — without downloading 2.4 GB, because the Hub publishes a
content hash per file that the downloader records in its own sidecars. Exit code is `0` when
current, `1` when an update is available, so it works as a cron or CI check.

```
  built from: 55cf4c4ebb4ebe31…  (2026-10-03)
  now       : 7a1f9c02d3be4410…  (2026-11-02)
  UPDATE AVAILABLE
    modified  model.safetensors
    added     eval/results_v2.json
```

When it reports an update:

```bash
python -c "from huggingface_hub import snapshot_download as d; \
           d('convaiinnovations/laya', local_dir='laya-full', ignore_patterns=['.cache/*'])"
python build_release.py --source laya-full --out dist
gh release create v3 dist/*.tar.gz dist/MANIFEST.json --repo machinelearning2014/laya-how-to
```

Publish with a new tag, never an overwrite — assets are immutable. Existing mirrors keep
working; they are simply pinned to the revision they were built from, which is why the
manifest records `revision` rather than only file digests.

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
