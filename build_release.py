#!/usr/bin/env python3
"""Package a complete copy of the laya checkpoints as GitHub Release assets.

A mirror lives or dies on being reproducible, so this writes deterministic archives
(fixed mtimes, zeroed ownership, no gzip timestamp) and records a MANIFEST.json of
every file's SHA-256 plus each archive's own digest. Re-running it against the same
source produces byte-identical output, so the manifest digests stay valid.

    # 1. download the complete repo from the Hub
    python -c "from huggingface_hub import snapshot_download as d; \\
               d('convaiinnovations/laya', local_dir='laya-full', ignore_patterns=['.cache/*'])"

    # 2. build the assets
    python build_release.py --source laya-full --out dist

    # 3. publish
    gh release create v2 dist/*.tar.gz dist/MANIFEST.json --repo OWNER/REPO

Why four archives and not one: GitHub caps a release asset at 2 GB, and the complete
repo is ~2.4 GB. Extracting all of them into one directory reconstructs the original
layout exactly.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tarfile
import time

# Extracting every bundle into one directory reproduces the Hub repo's layout:
# three checkpoints, each independently usable, plus the repo's remaining files.
BUNDLES = {
    "english-ckpt.tar.gz": ["model.safetensors", "rl_agent_config.json", "encoder", "tokenizer"],
    "multilingual-ckpt.tar.gz": ["multilingual"],
    "typed-decisions-ckpt.tar.gz": ["typed-decisions"],
    "repo-extras.tar.gz": ["README.md", ".gitattributes", "rl_common.py", "rl_agent_api.py",
                           "email_utils.py", "eval", "assets"],
}

SKIP_NAMES = {".cache", ".DS_Store"}
EPOCH = 315532800  # 1980-01-01, the earliest mtime a gzip/tar header represents cleanly


def included(path: str) -> bool:
    """Skip macOS AppleDouble sidecars and the downloader's own metadata."""
    return not (os.path.basename(path).startswith("._") or os.path.basename(path) in SKIP_NAMES)


def walk(source: str, entries: list[str]) -> list[tuple[str, str]]:
    """(absolute, archive-relative) pairs for every file under `entries`, sorted."""
    out = []
    for entry in entries:
        full = os.path.join(source, entry)
        if not os.path.exists(full):
            raise SystemExit(f"source is missing {entry!r} -- is --source a complete download?")
        if os.path.isfile(full):
            if included(full):
                out.append((full, entry))
            continue
        for root, dirs, files in os.walk(full):
            dirs[:] = sorted(d for d in dirs if included(os.path.join(root, d)))
            for name in sorted(files):
                p = os.path.join(root, name)
                if included(p):
                    out.append((p, os.path.relpath(p, source)))
    return sorted(out, key=lambda pair: pair[1])


def normalise(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    """Strip everything that varies between runs, so the archive digest is stable."""
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = ""
    tarinfo.mtime = EPOCH
    if tarinfo.isdir():
        tarinfo.mode = 0o755
    else:
        tarinfo.mode = 0o644
    return tarinfo


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build(source: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"source": os.path.abspath(source), "generated": time.strftime("%Y-%m-%d"),
                "files": {}, "assets": {}, "bundles": {}}

    for bundle, entries in BUNDLES.items():
        pairs = walk(source, entries)
        if not pairs:
            raise SystemExit(f"{bundle}: nothing to pack")
        dest = os.path.join(out_dir, bundle)
        # mtime=0 in the gzip header too, or the archive digest changes every run.
        with open(dest, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for full, rel in pairs:
                    tar.add(full, arcname=rel, filter=normalise)

        size = os.path.getsize(dest)
        manifest["assets"][bundle] = {"digest": sha256(dest), "size": size, "files": len(pairs)}
        # The per-bundle file list is what lets a fetcher verify an extraction without
        # having to know how the bundles were grouped.
        manifest["bundles"][bundle] = [rel for _, rel in pairs]
        for full, rel in pairs:
            manifest["files"][rel] = sha256(full)
        print(f"  {bundle:<30} {len(pairs):>2} files  {size/1e6:7.1f} MB  {manifest['assets'][bundle]['digest'][:16]}…")

    manifest["total_files"] = len(manifest["files"])
    manifest_path = os.path.join(out_dir, "MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  {'MANIFEST.json':<30} {manifest['total_files']:>2} files  {os.path.getsize(manifest_path)/1e6:7.3f} MB")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="directory holding the complete checkpoint repo")
    ap.add_argument("--out", default="dist", help="where to write the archives (default: dist)")
    args = ap.parse_args()
    if not os.path.isdir(args.source):
        raise SystemExit(f"no such directory: {args.source}")
    print(f"packaging {args.source} -> {args.out}")
    build(args.source, args.out)
    print("\nPublish with:\n  gh release create <tag> dist/*.tar.gz dist/MANIFEST.json --repo <owner>/<repo>")


if __name__ == "__main__":
    main()
