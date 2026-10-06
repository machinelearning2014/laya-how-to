#!/usr/bin/env python3
"""Run the `laya` command against the locally mirrored checkpoints.

The installed CLI builds a bare `Router()`:

    cli.py:147   return laya.Router(device=args.device, preload=False)

That reads the built-in checkpoint table, so it always reaches for the Hugging Face Hub.
Nothing in the CLI consults the environment, and `--model` resolves strictly against the
registry rather than accepting a path, so there is no supported way to point it at
`~/.laya/mirror`. Offline it fails with `LocalEntryNotFoundError` even when a complete
mirror is sitting on disk.

This wrapper substitutes `laya.Router` for one built with the mirror, which keeps every
flag, mode and output format of the real command without reimplementing any of it. If a
future laya builds its Router differently the patch would silently do nothing, so it checks
first and refuses rather than quietly falling back to the network.

    python laya_local.py "I was charged twice" --preset triage --predict
    python laya_local.py --batch tickets.txt --predict --json
    python laya_local.py                       # interactive

Every `laya` flag works, because this calls the real CLI. Build the mirror first:

    python fetch_model.py --all
"""
from __future__ import annotations

import inspect
import sys

import laya

from local_models import is_fully_local, missing, model_sources


def main() -> int:
    sources = model_sources()
    if is_fully_local(sources):
        print("laya_local: all checkpoints from the local mirror", file=sys.stderr)
    else:
        print(f"laya_local: mirror incomplete -- {', '.join(missing(sources))} will come "
              f"from the Hub, which needs network access", file=sys.stderr)

    from laya import cli

    # The patch works because make_router looks `laya.Router` up when it is called. If that
    # ever changes, the substitution is a no-op and the user silently gets a Hub download,
    # so verify the assumption instead of trusting it.
    source = inspect.getsource(cli.make_router)
    if "laya.Router(" not in source:
        raise SystemExit(
            "laya_local: this laya version does not build its Router via laya.Router(), "
            "so the mirror cannot be substituted.\n"
            f"  cli.make_router is now:\n    {source.strip()}\n"
            "  Use the mirror through main.py, or run online."
        )

    real_router = laya.Router

    def Router(*args, **kwargs):  # noqa: N802 -- stands in for the class of that name
        kwargs.setdefault("models", sources)
        return real_router(*args, **kwargs)

    laya.Router = Router
    return cli.main() or 0


if __name__ == "__main__":
    sys.exit(main())
