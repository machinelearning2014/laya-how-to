#!/usr/bin/env python3
"""Fetch the laya checkpoints from this repository's GitHub Releases instead of Hugging Face.

`Router()` downloads weights from the Hub by default. This script builds the same three
checkpoints from GitHub Release assets, so laya can run with no Hub access at all --
offline, air-gapped, or pinned to a copy you control.

    python fetch_model.py                 # the complete mirror (all three checkpoints)
    python fetch_model.py english         # just one checkpoint
    python fetch_model.py --check         # verify the mirror, download nothing
    python fetch_model.py --list          # what is available

It extracts into one directory that mirrors the upstream Hub repo's layout, so the result
is layout-compatible with `convaiinnovations/laya` itself:

    ~/.laya/mirror/model.safetensors          <- english sits at the root
    ~/.laya/mirror/multilingual/…
    ~/.laya/mirror/typed-decisions/…
    ~/.laya/mirror/{README.md,assets/,eval/,rl_*.py}

Point laya at it:

    Router(models={
        "english":          "~/.laya/mirror",
        "multilingual":     ("~/.laya/mirror", "multilingual"),
        "typed-decisions":  ("~/.laya/mirror", "typed-decisions"),
    })

`main.py` does this automatically. Note that `predict(..., model=...)` will NOT accept a
path: the per-call argument resolves strictly against the registry (english /
multilingual / typed-decisions). Use `models=` on the Router, or `laya.load`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

REPO = "machinelearning2014/laya-how-to"
DEFAULT_TAG = "latest"  # GitHub's /releases/latest/download redirects to the newest release,
                        # so publishing a new tag needs no edit here. Pin with --tag for
                        # reproducibility.
DEFAULT_DIR = os.path.expanduser("~/.laya/mirror")
MANIFEST = "MANIFEST.json"


def release_url(tag: str) -> str:
    if tag == "latest":
        return f"https://github.com/{REPO}/releases/latest/download"
    return f"https://github.com/{REPO}/releases/download/{tag}"

# checkpoint name -> the archive holding it. "extras" is the repo's remaining files
# (README, .gitattributes, the rl_* modules, eval/ and assets/).
BUNDLES = {
    "english": "english-ckpt.tar.gz",
    "multilingual": "multilingual-ckpt.tar.gz",
    "typed-decisions": "typed-decisions-ckpt.tar.gz",
    "extras": "repo-extras.tar.gz",
}
ALL_BUNDLES = list(BUNDLES.values())


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: str, label: str = "") -> None:
    """Stream to dest, reporting every 5% so an 800 MB fetch prints ~20 lines."""
    try:
        with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            seen, last, step = 0, -5, 5
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
                    sys.stdout.write(f"\r  {label} {seen/1e6:7.1f} / {total/1e6:.0f} MB ({pct:5.1f}%)")
                    sys.stdout.flush()
            if total:
                sys.stdout.write(f"\r  {label} {seen/1e6:7.1f} / {total/1e6:.0f} MB (100.0%)\n")
            sys.stdout.flush()
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"failed to fetch {url}\n  HTTP {e.code} {e.reason}\n"
            f"  If the release does not exist:\n"
            f"    python build_release.py --source <full-download> --out dist\n"
            f"    gh release create <tag> dist/*.tar.gz dist/MANIFEST.json --repo {REPO}"
        ) from None
    except urllib.error.URLError as e:
        raise SystemExit(f"failed to fetch {url}\n  {e.reason}") from None


def load_manifest(base_url: str, cache_path: str) -> dict:
    """Read MANIFEST.json from the release (or from disk when checking offline)."""
    if not os.path.isfile(cache_path):
        print(f"  fetching {MANIFEST}")
        download(f"{base_url.rstrip('/')}/{MANIFEST}", cache_path, label="manifest")
    with open(cache_path) as f:
        return json.load(f)


def bundle_files(manifest: dict, bundle: str) -> list[str]:
    files = manifest.get("bundles", {}).get(bundle)
    if files is None:
        raise SystemExit(f"{MANIFEST} has no file list for {bundle}")
    return files


def bundle_state(root: str, manifest: dict, bundle: str) -> tuple[str, list[str]]:
    """('complete'|'partial'|'absent', [problems]) for one bundle against the manifest."""
    files = bundle_files(manifest, bundle)
    present = [f for f in files if os.path.isfile(os.path.join(root, f))]
    if not present:
        return "absent", []
    if len(present) < len(files):
        missing = [f for f in files if not os.path.isfile(os.path.join(root, f))]
        return "partial", missing
    # A digest mismatch usually means the release has moved on and this copy predates it,
    # which is ordinary when upstream publishes a revision rather than local corruption.
    bad = [f for f in files
           if sha256(os.path.join(root, f)) != manifest["files"].get(f)]
    return ("complete", []) if not bad else ("differs", bad)


def extract_verified(archive: str, staging: str, root: str, manifest: dict, bundle: str) -> None:
    """Unpack to staging, verify against the manifest, then move into the mirror."""
    os.makedirs(staging, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # filter="data" rejects absolute paths, .. traversal and unsafe links
        tar.extractall(staging, filter="data")

    for rel in bundle_files(manifest, bundle):
        staged = os.path.join(staging, rel)
        if not os.path.isfile(staged):
            raise SystemExit(f"  {bundle}: archive is missing {rel}")
        want = manifest["files"].get(rel)
        got = sha256(staged)
        if want and got != want:
            raise SystemExit(f"  {bundle}: checksum mismatch for {rel}\n    expected {want}\n    got      {got}")
        dest = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(staged, dest)


def fetch(names: list[str], root: str, base_url: str, check_only: bool) -> None:
    os.makedirs(root, exist_ok=True)
    manifest = load_manifest(base_url, os.path.join(root, MANIFEST))

    todo = []
    for name in names:
        bundle = BUNDLES[name]
        state, problems = bundle_state(root, manifest, bundle)
        if state == "complete":
            print(f"  {name:<17} already present and verified")
            continue
        if check_only:
            label = {"absent": "not fetched", "partial": "incomplete",
                     "differs": "out of date"}.get(state, state)
            detail = f" ({len(problems)} file(s))" if problems else ""
            print(f"  {name:<17} {label}{detail}")
            todo.append(name)
            continue
        todo.append(name)

    if check_only:
        if todo:
            raise SystemExit(f"\n{len(todo)} bundle(s) not verified: {', '.join(todo)}")
        print("\n  mirror verified against MANIFEST.json")
        return

    for name in todo:
        bundle = BUNDLES[name]
        meta = manifest["assets"].get(bundle, {})
        print(f"  {name:<17} fetching {bundle}  (~{meta.get('size', 0)/1e6:.0f} MB)")
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            archive = os.path.join(tmp, bundle)
            download(f"{base_url.rstrip('/')}/{bundle}", archive, label=f"{name:<17}")
            extract_verified(archive, os.path.join(tmp, "unpacked"), root, manifest, bundle)
        print(f"  {name:<17} verified ({meta.get('files', '?')} files)")

    print(f"\nMirror ready at {root}")
    print("Point laya at it:")
    print(f"  Router(models={{'english': {root!r},")
    print(f"                  'multilingual': ({root!r}, 'multilingual'),")
    print(f"                  'typed-decisions': ({root!r}, 'typed-decisions')}})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", choices=sorted(BUNDLES) + [], default=None,
                    help=f"checkpoints to fetch (default: all). One of: {', '.join(sorted(BUNDLES))}")
    ap.add_argument("--all", action="store_true", help="fetch every bundle (the default)")
    ap.add_argument("--dir", default=DEFAULT_DIR, help=f"mirror directory (default: {DEFAULT_DIR})")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help=f"release tag, or 'latest' (default: {DEFAULT_TAG})")
    ap.add_argument("--base-url", default=None, help="override the release URL entirely")
    ap.add_argument("--check", action="store_true", help="verify the mirror; download nothing")
    ap.add_argument("--list", action="store_true", help="list the bundles")
    args = ap.parse_args()

    base_url = args.base_url or release_url(args.tag)

    if args.list:
        print(f"  release: {release_url(args.tag)}")
        for name, bundle in BUNDLES.items():
            print(f"  {name:<17} {bundle}")
        return

    names = args.names or list(BUNDLES)
    print(f"laya mirror -> {args.dir}   (release: {args.tag})")
    fetch(names, args.dir, base_url, args.check)


if __name__ == "__main__":
    main()
