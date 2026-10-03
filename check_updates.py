#!/usr/bin/env python3
"""Check whether the upstream Hugging Face model has moved since the mirror was built.

The mirror pins a specific revision. This compares the Hub's current `main` against the
revision recorded in MANIFEST.json and, when they differ, reports exactly which files
changed -- without downloading 2.4 GB, because the Hub publishes a content hash per file
(`lfs.sha256` for the big weights, `blobId` for the small ones) and the downloader records
the same values in its sidecars.

    python check_updates.py                  # compare against a local dist/MANIFEST.json
    python check_updates.py --from-release   # compare against the published release
    python check_updates.py --json           # machine-readable

Exit codes: 0 = mirror is current, 1 = update available, 2 = could not check.

When it reports an update, refresh with:

    python -c "from huggingface_hub import snapshot_download as d; \\
               d('convaiinnovations/laya', local_dir='laya-full', ignore_patterns=['.cache/*'])"
    python build_release.py --source laya-full --out dist
    gh release create <new-tag> dist/*.tar.gz dist/MANIFEST.json --repo OWNER/REPO

Release assets are immutable, so an update always means a new tag -- never an overwrite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

REPO = "machinelearning2014/laya-how-to"
UPSTREAM = "convaiinnovations/laya"


def api_payload(repo: str) -> dict:
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    try:
        with urllib.request.urlopen(url) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"could not query {url}\n  HTTP {e.code} {e.reason}") from None
    except urllib.error.URLError as e:
        raise SystemExit(f"could not query {url}\n  {e.reason}") from None


def upstream_state(payload: dict) -> tuple[str, dict[str, str]]:
    """(revision, {path: content-hash}) for the Hub's current main."""
    etags = {}
    for s in payload.get("siblings") or []:
        # LFS objects carry a content SHA-256; small files only a git blob id. Both match
        # what the downloader writes to its .metadata sidecars.
        lfs = s.get("lfs") or {}
        digest = lfs.get("sha256") or s.get("blobId")
        if digest:
            etags[s["rfilename"]] = digest
    return payload.get("sha") or "", etags


def load_manifest(args) -> dict:
    path = args.manifest
    if args.from_release:
        url = (f"https://github.com/{REPO}/releases/download/{args.tag}/MANIFEST.json"
               if args.tag != "latest"
               else f"https://github.com/{REPO}/releases/latest/download/MANIFEST.json")
        try:
            with urllib.request.urlopen(url) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"could not fetch {url}\n  HTTP {e.code} {e.reason}") from None
    if not os.path.isfile(path):
        raise SystemExit(
            f"no manifest at {path}\n"
            f"  build one with: python build_release.py --source laya-full --out dist\n"
            f"  or compare against the release: python check_updates.py --from-release"
        )
    with open(path) as f:
        return json.load(f)


def compare(manifest: dict, revision: str, etags: dict[str, str]) -> dict:
    recorded = manifest.get("revision")
    local = manifest.get("etags") or {}
    changed = sorted(p for p in etags if p in local and local[p] != etags[p])
    added = sorted(p for p in etags if p not in local)
    removed = sorted(p for p in local if p not in etags)
    return {"recorded_revision": recorded, "upstream_revision": revision,
            "revision_changed": bool(recorded) and recorded != revision,
            "changed": changed, "added": added, "removed": removed,
            "up_to_date": (not added and not removed and not changed
                           and (not recorded or recorded == revision))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=os.path.join("dist", "MANIFEST.json"))
    ap.add_argument("--from-release", action="store_true",
                    help="read MANIFEST.json from the GitHub release instead of dist/")
    ap.add_argument("--tag", default="latest", help="release tag for --from-release")
    ap.add_argument("--repo", default=UPSTREAM, help="upstream Hub repo id")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest(args)
    payload = api_payload(args.repo)
    revision, etags = upstream_state(payload)
    result = compare(manifest, revision, etags)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["up_to_date"] else 1)

    rec = result["recorded_revision"] or "(not recorded)"
    print(f"  upstream : {args.repo}")
    print(f"  built from: {rec[:16]}{'…' if len(rec) > 16 else ''}  ({manifest.get('generated', '?')})")
    print(f"  now       : {revision[:16]}{'…' if len(revision) > 16 else ''}  "
          f"({payload.get('lastModified', '?')[:10]})")
    print(f"  files     : {len(etags)} upstream, {len(manifest.get('files', {}))} in the mirror")

    if result["up_to_date"]:
        print("\n  up to date -- nothing to refresh.")
        return

    if not result["revision_changed"] and not (result["added"] or result["removed"] or result["changed"]):
        print("\n  the revision hash moved but no file differs from the mirror.")
        return

    print("\n  UPDATE AVAILABLE")
    for label, key in (("modified", "changed"), ("added", "added"), ("removed", "removed")):
        for path in result[key]:
            print(f"    {label:<9} {path}")
    if result["changed"] and "model.safetensors" in result["changed"]:
        print("\n    note: a weights file changed, so the checkpoints must be re-fetched;"
              "\n          a mirror built from the old revision will keep working but is stale.")
    print("\n  refresh with:")
    print(f"    python -c \"from huggingface_hub import snapshot_download as d;"
          f" d('{args.repo}', local_dir='laya-full', ignore_patterns=['.cache/*'])\"")
    print("    python build_release.py --source laya-full --out dist")
    print(f"    gh release create <new-tag> dist/*.tar.gz dist/MANIFEST.json --repo {REPO}")
    sys.exit(1)


if __name__ == "__main__":
    main()
