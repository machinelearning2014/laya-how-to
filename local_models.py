"""Resolve laya's checkpoint table, preferring a locally mirrored copy when one exists.

`fetch_model.py` builds `~/.laya/mirror` as a complete copy of the upstream Hub repo, in
the upstream layout: `english` at the root, `multilingual/` and `typed-decisions/` as
subfolders. That is deliberately the same `(repo, subfolder)` shape `laya.DEFAULT_MODELS`
already uses, so pointing the Router at a mirror is a substitution rather than a new
concept.

With no mirror present this returns `laya.DEFAULT_MODELS` unchanged, so a Router built
from it behaves exactly like a plain `Router()`.

    from local_models import model_sources
    router = Router(models=model_sources())

One caveat worth knowing before deploying: this **falls back silently** to the Hub when the
mirror is missing or partial. If you need to be sure you are offline, check the result
rather than trusting it:

    if not is_fully_local(model_sources()):
        raise SystemExit("mirror is incomplete; refusing to reach for the network")
"""
from __future__ import annotations

import os

import laya

MIRROR = os.path.expanduser("~/.laya/mirror")

# name -> subfolder inside the mirror. `english` sits at the root because that is where
# the upstream repo keeps it.
MIRROR_SUBFOLDERS = {"english": None, "multilingual": "multilingual",
                     "typed-decisions": "typed-decisions"}


def model_sources(mirror: str | None = None) -> dict:
    """Checkpoint table with mirrored checkpoints substituted in, where they exist."""
    mirror = mirror or MIRROR
    models = dict(laya.DEFAULT_MODELS)
    for name, sub in MIRROR_SUBFOLDERS.items():
        where = os.path.join(mirror, sub) if sub else mirror
        if os.path.isfile(os.path.join(where, "model.safetensors")):
            models[name] = (mirror, sub) if sub else mirror
    return models


def is_fully_local(models: dict) -> bool:
    """True when every checkpoint resolves to a local path rather than a Hub repo id."""
    return all(not isinstance(spec, tuple) or os.path.isabs(str(spec[0]))
               for spec in models.values())


def missing(models: dict) -> list[str]:
    return sorted(n for n, spec in models.items() if spec == laya.DEFAULT_MODELS[n])
