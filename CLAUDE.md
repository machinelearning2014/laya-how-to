# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A teaching repository for the `laya` decision engine (v0.3.24 in `.venv`). Two deliverables plus a
human-facing intro:

- `main.py` — a twelve-section tour of the in-process API, english checkpoint only.
- `index.html` — the field reference, and simultaneously the live GitHub Pages site.
- `README.md` — the public-facing intro for the repo.
- `fetch_model.py` — pulls a checkpoint from this repo's GitHub Releases instead of the Hub.
- `guardrail_bench.py` — measures accuracy, calibration and the selective curve on labelled rows.
- `local_models.py` — the shared checkpoint-source helper `main.py` and `guardrail_bench.py` import.
- `laya_local.py` — the `laya` CLI with the mirror substituted in; see the CLI note below.

**The installed `laya` command cannot be pointed at the mirror.** `cli.py` builds a bare
`Router()` and reads no environment variables, and its `--model` resolves strictly against the
registry instead of accepting a path. `LAYA_MODELS` exists but only `serve.py` reads it. So
anything that shells out to `laya` needs `laya_local.py`, which patches `laya.Router` before
calling the real `cli.main()` — the patch works because `cli.make_router` looks the attribute up
at call time, and the wrapper checks that assumption and exits rather than falling back to the
network if it ever stops holding.

These scripts are run by users on whatever `python3` they have, so keep them portable. All four
carry `from __future__ import annotations`, and `fetch_model.py` is stdlib-only on purpose.

**Do not add `choices=` to an argument with `nargs="*"`.** Before CPython 3.12.13, argparse calls
`_check_value(action, [])` when a positional gets no values, and `[] not in choices` raises
`argument names: invalid choice: []`. That made `fetch_model.py --all` fail on Python 3.10 while
working on 3.12.13 — reproducible only by running an older interpreter, which is why the venv
hides it. Validate the values by hand instead (`fetch_model.py` does, and a regression would be
invisible on this machine; test with `/usr/bin/python3` or a uv-managed 3.10).

There is no package manifest, no build step, and no test suite. Work here is either editing the
tour, editing the reference page, or verifying a claim about laya's behaviour against the
installed library.

## Commands

```bash
python3 -m venv .venv
.venv/bin/python -m pip install laya          # first run downloads a ~421M-param checkpoint
.venv/bin/python main.py                      # run the whole tour (~1 min, english checkpoint)
python3 -m http.server 8000                   # preview index.html at localhost:8000
```

Verify any claim about the library before writing it down, by running a snippet in the venv:

```bash
.venv/bin/python -c "import laya; print(laya.__version__)"
.venv/bin/python -c "from laya.router import Router; help(Router.predict)"
```

Checkpoint downloads are cached under `~/.cache/huggingface`, so repeat runs are offline and fast.
Suppress the noisy first-load warning with `warnings.filterwarnings('ignore')` when scripting.

## Architecture

**`main.py`** is a flat script: shared fixtures (`STATE`, `QUESTIONS`, a module-level `router`) at
the top, then one `show_*()` function per API area, called in order by `main()` behind an
`if __name__ == "__main__"` guard. That guard is load-bearing: it lets tests, notebooks, and other
tools `import main` to reuse the question set without triggering a full inference pass.

Keep it english-only. The multilingual and typed-decisions checkpoints are not cached, so importing
them here would turn a fast offline run into a ~750 MB download. Areas that need them are shown
through `route()` (which loads no weights) or listed in the final section as separate processes.

`model_sources()` builds the Router's checkpoint table by starting from `laya.DEFAULT_MODELS` and
overriding only the names present in `~/.laya/models/`. That matters: `Router(models=...)` replaces
the **whole** table, so passing only the locally fetched entries would break routing to every
checkpoint you had not fetched, with `ValueError: unknown model`. Keep the copy-and-override shape.

`fetch_model.py` verifies the SHA-256 of `model.safetensors` *inside* the archive rather than the
tarball, so repackaging does not invalidate the recorded digest. Build release tarballs with
`COPYFILE_DISABLE=1 tar czf …`, or macOS adds `._*` AppleDouble members to the archive.

## Hosting

Tag `v2` is a complete mirror of `convaiinnovations/laya` — all 38 files, 2.4 GB — as four release
assets plus a `MANIFEST.json`. `v1` (english only) is superseded and can be deleted.

`build_release.py` produces the assets deterministically: fixed mtimes, zeroed ownership, no gzip
timestamp, so a rebuild from the same source is byte-identical and the manifest digests stay valid.
`fetch_model.py` verifies **every** extracted file against `MANIFEST.json`, not just the weights.

The mirror reproduces the upstream layout, so `english` sits at the mirror root and the other two
are subfolders. That is why `model_sources()` maps them to `MIRROR` or `(MIRROR, subfolder)` — the
same `(repo, subfolder)` shape the built-in table uses, because the Hub repo packs all three
checkpoints that way too.

Release assets are immutable, so a new upstream revision needs a new tag and a re-upload rather than
an overwrite. `fetch_model.py` defaults to GitHub's `/releases/latest/download` redirect, so a new
tag needs no code change; `--tag` pins one.

`MANIFEST.json` records the upstream `revision` and a per-file `etag` (the content SHA-256 for LFS
files, the git blob id for small ones), read from the Hub downloader's `.cache/huggingface/download/
*.metadata` sidecars at build time. `check_updates.py` uses those to diff against the Hub's current
`main` without downloading anything: exit 0 current, 1 update available. If the sidecars are missing
the manifest has no revision and the checker can only report file-level differences.

**`index.html`** is deliberately self-contained — inline CSS and JS, no external scripts, no build
tooling, no image assets. The only external request is Google Fonts. Keep it that way; it is served
straight from the repository root by GitHub Pages, so any added dependency has to be reachable from
a static host.

The question dict is the unit of work in both files. One `predict` call packs the state and every
question's options into a single sequence with one masked slot per option, so a single forward pass
scores all of them. Adding a question is nearly free; splitting one call into several `predict`
calls multiplies the cost.

## Conventions to preserve

**Name the state field in backticks.** Every instruction reads `` `message` ``. This is the library's
own convention: `laya.presets.state_field()` scrapes the name back out to learn where to place the
text, which is what keeps a question set usable from the CLI (`laya --questions`) and `decide()`.
A set that names no field resolves to `None` and loses that portability.

**Everything printed is real observed output.** `main.py` prints live values rather than hardcoding
them, and the trailing comments in the question dict quote a real run. Rewording an instruction
shifts the numbers, so re-run `.venv/bin/python main.py` and update any quoted value in the same
change. The reference page quotes these numbers too, so it needs the same pass.

**Three tour sections deliberately show a wrong answer.** `decide()` returning `urgency: 0`,
relabelling moving a `noul` from 0.37 to 0.54, and shortlisting dropping the correct label. These
are documented findings, not bugs to fix — if an edit makes them disappear, check whether a claim
elsewhere on the reference page has gone stale.

**Gate on `answer_confidence`, not the raw value.** For `noul`, a low number is a *confident no*
rather than uncertainty. The same applies to `choice` and `score`, where the entropy `confidence`
drifts with option count and `answer_confidence` (max p) does not.

**`.venv/` is 745 MB and gitignored.** Never commit it or add it to the index.

## Laya facts that are easy to get wrong

- `Router(models=...)` is plural and constructor-only. The singular `model=` is a per-call argument
  on `predict`.
- Answer shapes: `choice` → `["choice"]`, `score` → `["score"]` (an *expected level* as a float, not
  an int), `noul` → `["noul"]` (always P(true), whatever the criteria or labels say).
- `usage["truncated"]` reports whether the state actually fit the token budget. Check it before
  concluding the model got something wrong on a long input.
- The English checkpoint logs a `RuntimeWarning` on load: its `choice:11+` temperatures are shipped
  out of range and get clamped, so confidence for `choice` questions with 11+ options is
  uncalibrated. Narrow choice questions are unaffected.
- `questions={}` skips inference entirely and returns empty answers with no forward pass.

## Deployment

`origin` is `https://github.com/machinelearning2014/laya-how-to`. GitHub Pages serves
`index.html` from the repository root of `main`, so committing and pushing to `main` publishes the
reference site automatically; the build takes roughly 35 seconds. If `index.html` is ever renamed,
the site root returns 404 and the page is only reachable at its full path.
