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
git clone https://github.com/machinelearning2014/laya-how-to.git
cd laya-how-to
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
python fetch_model.py --all      # the complete mirror: all three checkpoints, ~2.2 GB
python fetch_model.py english    # or just one
python fetch_model.py --check    # re-verify the mirror; downloads nothing
python fetch_model.py --list     # what is available
```

`--all` is also the default when no names are given; it is spelled out here because it is the
explicit form to put in a script. Combining it with a name is an error rather than a silent
pick of one bundle.

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

### The `laya` command needs a wrapper

The installed CLI builds a bare `Router()`:

```python
cli.py   return laya.Router(device=args.device, preload=False)
```

That reads the built-in checkpoint table, so it always goes to the Hub. It consults no
environment variables, and its `--model` resolves strictly against the registry rather than
accepting a path, so there is **no supported way to point it at the mirror**. Offline it fails
even with a complete mirror on disk:

```
$ HF_HUB_OFFLINE=1 laya "I was charged twice" --predict
laya: could not run Laya (Cannot find an appropriate cached snapshot folder for the
specified revision on the local disk and outgoing traffic has been disabled…)
```

`laya_local.py` runs the real CLI with the mirror substituted in, so every flag, mode and
output format is unchanged:

```bash
python laya_local.py "I was charged twice" --preset triage --predict
python laya_local.py --batch tickets.txt --predict --json
python laya_local.py --model ml "hello"        # routing only, loads nothing
python laya_local.py                           # interactive
```

It reports which checkpoints it is using on stderr, and **refuses rather than silently
falling back to the network** if a future laya version builds its Router some other way — a
patch that quietly stopped applying would otherwise look like a working offline setup that
was secretly downloading. Verified against a direct API call with the same mirror: identical
values, and identical to the Hub-backed CLI.

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

The mirror is a **pinned snapshot**. It records the upstream revision it was built from, and
nothing polls for changes on its own — the release moves only when you move it. When
`convaiinnovations/laya` publishes a new revision, refreshing takes five steps.

### 1. Detect

```bash
python check_updates.py --from-release
```

Compares the Hub's current `main` against the revision recorded in the published manifest, and
reports exactly which files changed. It downloads nothing: the Hub publishes a content hash per
file (`lfs.sha256` for the weights, `blobId` for the small ones) and the downloader records the
same values in its sidecars. Exit `0` when current, `1` when an update is available, so it drops
into a cron job or CI gate unchanged.

```
  built from: 55cf4c4ebb4ebe31…  (2026-10-03)
  now       : 7a1f9c02d3be4410…  (2026-11-02)
  UPDATE AVAILABLE
    modified  model.safetensors
    added     eval/results_v2.json
```

If it says `up to date`, stop here.

### 2. Re-download from the Hub

```bash
python -c "from huggingface_hub import snapshot_download as d; \
           d('convaiinnovations/laya', local_dir='laya-full', ignore_patterns=['.cache/*'])"
```

Detection only says *something* moved; this is what says exactly what. It refreshes `laya-full/`
in place, re-fetching only the files whose etags changed.

### 3. Rebuild the assets

```bash
python build_release.py --source laya-full --out dist
```

Deterministic, and it records the new revision in `dist/MANIFEST.json`. Confirm the revision it
reports is the one you expect — the next step derives the tag from it.

### 4. Publish under a revision-derived tag

```bash
TAG="rev-$(python -c "import json;print(json.load(open('dist/MANIFEST.json'))['revision'][:12])")"

gh release create "$TAG" dist/*.tar.gz dist/MANIFEST.json \
  --repo machinelearning2014/laya-how-to \
  --title "laya checkpoints ($TAG)" \
  --notes "Mirror of convaiinnovations/laya at the revision above."
```

Deriving the tag from the revision rather than counting up makes a re-run idempotent: publishing
the same revision twice stops at "tag already exists" instead of creating a second release for a
change already published. Assets are **immutable**, so this must always be a new tag — never
`--clobber` onto an existing release.

**If it returns 404**, `gh` is authenticated as an account without admin on this repo. Reads
succeed either way because the repo is public, so only the write fails — and GitHub reports
insufficient permission as `404`, not `403`, which reads like a missing release. Use the owning
account's token for the write:

```bash
TOKEN=$(env -u GITHUB_TOKEN gh auth token --user machinelearning2014)
GH_TOKEN="$TOKEN" gh release create "$TAG" ...
```

### 5. Refresh your own mirror and verify

```bash
curl -sIL https://github.com/machinelearning2014/laya-how-to/releases/latest/download/MANIFEST.json | head -1
python fetch_model.py --all     # pulls the new revision into ~/.laya/mirror
python fetch_model.py --check   # verifies every file against the new manifest
```

`/releases/latest/download` follows the newest release, so `fetch_model.py` picks up the new tag
with no code change. Expect the affected bundles to re-download: a local mirror built from the
previous revision **fails verification against the new manifest** and reports `out of date`,
which is the intended behaviour rather than corruption.

### Keeping the release list bounded

Each revision adds ~2.16 GB permanently, and old releases do not delete themselves. Once the new
one verifies, prune:

```bash
gh release delete rev-<old-sha> --repo machinelearning2014/laya-how-to --yes --cleanup-tag
```

Keeping the two or three most recent leaves you something to fall back to without accumulating
indefinitely.

## Measure before you trust it: `guardrail_bench.py`

The claim this library rests on is that `answer_confidence` is calibrated enough to
threshold, so you can act on the answers it is sure about and abstain on the rest. That is
easy to assert and easy to falsify, so this measures it on labelled data:

```bash
python guardrail_bench.py                        # the bundled guardrail set
python guardrail_bench.py --only jailbreak       # one question
python guardrail_bench.py --data mine.jsonl      # your own rows
python guardrail_bench.py --questions q.json --preset guard
```

### The two input formats

They are independent, and it is worth being clear about which is which:

- **`--data`** supplies the examples and their correct answers.
- **`--questions`** defines what the model is asked.

**`--data`** — one JSON object per line. `text` is required; the label column is whichever
column holds 0/1 values, which is inferred and reported, so `--label` is only needed when a
file has several such columns:

```jsonl
{"text": "I was charged twice, please refund the duplicate.", "refund_request": 1}
{"text": "Where can I download my invoices?", "refund_request": 0}
{"text": "Refunds take 5-10 days, right? Just checking the policy.", "refund_request": 0, "note": "hard negative"}
```

**`--questions`** — the same question dict laya takes anywhere:

```json
{"refund_request": {"type": "noul",
                    "instructions": "Does `message` ask for money back or a charge reversed?"}}
```

Two conventions carry weight. A `note` starting with `"hard negative"` puts that row in its
own accuracy group — those are rows that merely *read* like a positive, and they are where
failures concentrate, so a set without them flatters the question. And the label and the
question **must describe the same decision**: point one at refunds and the other at
injections and every number is noise that reads like a bad model. The tool prints both in its
header and warns when accuracy cannot beat the majority-class baseline, but on a small set a
mismatched pair can clear that baseline by luck, so read the header rather than the warning.

`data/sample_rows.jsonl` and `data/sample_questions.json` are a matched pair in these
formats, in a different domain from the guardrail set. They exist to be **copied, not
quoted**: ten rows is not enough for a number to mean anything, and 8-of-10 correct clears a
60% baseline roughly one time in six.

It reports three things: accuracy per question, calibration (ECE before and after fitting a
temperature), and **the selective curve** — as the confidence threshold rises, what fraction
of inputs still gets an answer and how accurate that answer is. The curve is the actual
deployment trade, and accuracy alone hides it.

On the bundled 45-row adversarial set, the shipped `guard` preset scores 73.3% on `jailbreak`
at full coverage. Raising the threshold to 0.95 answers 69% of inputs at **90.3%** accuracy —
so abstention buys 17 points. It also over-triggers rather than under-detects: 35% false
positives against 18% false negatives, because benign text that merely borrows the vocabulary
("write a system prompt for my chatbot") reads as an attack.

That is the tool doing its job. Treat it as a harness to run on your own labelled rows before
you pick a threshold, not as a verdict on the preset.

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
