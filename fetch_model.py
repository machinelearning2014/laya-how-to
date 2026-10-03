#!/usr/bin/env python3
"""Fetch a laya checkpoint from this repository's GitHub Releases into a local directory.

`Router()` downloads weights from the Hugging Face Hub by default. This script pulls the
same weights from a GitHub Release asset instead, verifies them by SHA-256, and unpacks
them, so laya can run with no Hub access at all -- offline, air-gapped, or pinned to an
exact revision you control.

    python fetch_model.py english                    # download, verify, unpack
    python fetch_model.py english --dir models/en    # choose the destination
    python fetch_model.py english --check            # verify an existing copy, download nothing
    python fetch_model.py english --base-url URL     # fetch from somewhere else

Then point laya at the directory it produced:

    laya.load("~/.laya/models/english")              # direct
    Router(models={"english": "~/.laya/models/english"})   # keep routing, swap the storage

Note that `predict(..., model=...)` will NOT accept this path: the per-call argument
resolves strictly against the registry (english / multilingual / typed-decisions).
Use `models=` on the Router, or `laya.load`.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

REPO = "machinelearning2014/laya-how-to"
TAG = "v1"
DEFAULT_DIR = os.path.expanduser("~/.laya/models")

# One entry per checkpoint. `digest` is the SHA-256 of model.safetensors INSIDE the
# archive -- the same content hash Hugging Face uses for its blob filenames, and the
# value `laya.load(..., expected_sha256=...)` wants. Verifying the inner file rather
# than the tarball means repackaging the archive does not invalidate the check.
MODELS = {
    "english": {
        "asset": "english-ckpt.tar.gz",
        "digest": "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
        "size_mb": 743,
    },
    # Fill these in when the multilingual and typed-decisions assets are published.
    # "multilingual": {"asset": "multilingual-ckpt.tar.gz", "digest": "...", "size_mb": 571},
}


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def asset_url(base_url: str, asset: str) -> str:
    return f"{base_url.rstrip('/')}/{asset}"


def download(url: str, dest: str) -> None:
    """Stream to dest, reporting progress, with a clear error on HTTP failure."""
    try:
        with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            seen = 0
            step = 5  # report every 5%, so an 800 MB fetch prints ~20 lines rather than 800
            last = -step
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                seen += len(block)
                if not total:
                    continue
                pct = 100 * seen / total
                if pct - last >= step:
                    last = pct
                    sys.stdout.write(f"\r  downloading {seen/1e6:7.1f} / {total/1e6:.0f} MB  ({pct:5.1f}%)")
                    sys.stdout.flush()
            if total:
                sys.stdout.write(f"\r  downloaded  {seen/1e6:7.1f} / {total/1e6:.0f} MB  (100.0%)\n")
            sys.stdout.flush()
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"failed to fetch {url}\n  HTTP {e.code} {e.reason}\n"
            f"  If the release does not exist yet, create it first:\n"
            f"    gh release create {TAG} <asset> --repo {REPO}"
        ) from None
    except urllib.error.URLError as e:
        raise SystemExit(f"failed to fetch {url}\n  {e.reason}") from None


def extract(archive: str, target: str) -> None:
    """Unpack into target, refusing absolute paths and traversal."""
    os.makedirs(target, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # filter="data" rejects absolute paths, .. traversal, and unsafe links.
        tar.extractall(target, filter="data")


def verify(directory: str, digest: str) -> bool:
    weights = os.path.join(directory, "model.safetensors")
    if not os.path.isfile(weights):
        return False
    return sha256_file(weights) == digest


def fetch(name: str, dest_root: str, base_url: str, check_only: bool = False) -> str:
    spec = MODELS.get(name)
    if spec is None:
        raise SystemExit(f"unknown model {name!r}; known: {', '.join(MODELS) or '(none configured)'}")

    target = os.path.join(dest_root, name)

    if verify(target, spec["digest"]):
        print(f"  {name}: already present and verified at {target}")
        return target

    if check_only:
        raise SystemExit(f"  {name}: missing or failed verification in {target}")

    os.makedirs(dest_root, exist_ok=True)
    url = asset_url(base_url, spec["asset"])
    print(f"  {name}: fetching {url}  (~{spec['size_mb']} MB)")

    with tempfile.TemporaryDirectory(dir=dest_root) as tmp:
        archive = os.path.join(tmp, spec["asset"])
        download(url, archive)
        print("  verifying…")
        staged = os.path.join(tmp, "unpacked")
        extract(archive, staged)
        if not verify(staged, spec["digest"]):
            got = sha256_file(os.path.join(staged, "model.safetensors"))
            raise SystemExit(
                f"  checksum mismatch for {name}\n    expected {spec['digest']}\n    got      {got}"
            )
        # Replace only after the copy verifies, so a failed run cannot leave a half-written dir.
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(staged, target)

    print(f"  {name}: verified and unpacked to {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", default="english", help="checkpoint name (default: english)")
    parser.add_argument("--dir", default=DEFAULT_DIR, help=f"where to unpack (default: {DEFAULT_DIR})")
    parser.add_argument("--base-url", default=f"https://github.com/{REPO}/releases/download/{TAG}",
                        help="override the release URL, e.g. to serve assets locally")
    parser.add_argument("--check", action="store_true", help="verify an existing copy without downloading")
    parser.add_argument("--list", action="store_true", help="list the checkpoints this script knows")
    args = parser.parse_args()

    if args.list:
        for name, spec in MODELS.items():
            print(f"  {name:<14} {spec['asset']:<28} ~{spec['size_mb']} MB")
        return

    print(f"laya model fetch -> {args.dir}")
    path = fetch(args.model, args.dir, args.base_url, check_only=args.check)
    print(f"\nPoint laya at it:\n  laya.load({path!r})\n  Router(models={{{args.model!r}: {path!r}}})")


if __name__ == "__main__":
    main()
