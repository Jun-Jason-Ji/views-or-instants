"""Publish a new version of the Zenodo reproducibility archive, and report its DOI.

The record series is the one created for this project (concept DOI
10.5281/zenodo.22784740).  This script creates a new version of the latest
published record, replaces its file with the archive built by
`build_zenodo_archive_v110.py`, updates the metadata, and optionally publishes.

    python src/publish_zenodo_version.py --version 1.1.0                # dry run
    python src/publish_zenodo_version.py --version 1.1.0 --publish      # for real

Publishing is IRREVERSIBLE on Zenodo: a published version cannot be deleted, only
superseded by another version.  Nothing is published without --publish; without
it the script stops at the draft and prints the draft URL for inspection.

The token is read, in order, from --token-file, the ZENODO_TOKEN environment
variable, or %USERPROFILE%/.zenodo_token.  It needs the scopes
`deposit:write` and `deposit:actions`, and is never printed or written anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ZENODO = "https://zenodo.org/api"
CONCEPT_RECID = "22784740"
TIMEOUT = 120


def read_token(token_file: str | None) -> str:
    if token_file:
        return Path(token_file).read_text(encoding="utf-8").strip()
    if os.environ.get("ZENODO_TOKEN"):
        return os.environ["ZENODO_TOKEN"].strip()
    default = Path(os.path.expanduser("~")) / ".zenodo_token"
    if default.exists():
        return default.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "no Zenodo token found.  Create one at\n"
        "  https://zenodo.org/account/settings/applications/tokens/new/\n"
        "with the scopes deposit:write and deposit:actions, then save it to\n"
        f"  {default}\n"
        "or pass --token-file, or set ZENODO_TOKEN.")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def api(method: str, url: str, token: str, **kw):
    kw.setdefault("timeout", TIMEOUT)
    headers = kw.pop("headers", {})
    headers["Authorization"] = "Bearer %s" % token
    r = requests.request(method, url, headers=headers, **kw)
    if r.status_code >= 400:
        raise SystemExit("%s %s -> %d\n%s" % (method, url.split("?")[0], r.status_code, r.text[:800]))
    return r


def latest_published(token: str) -> dict:
    r = requests.get("%s/records/%s" % (ZENODO, CONCEPT_RECID), timeout=TIMEOUT)
    r.raise_for_status()
    rec = r.json()
    print("latest published: id %s  doi %s  version %s"
          % (rec["id"], rec["doi"], rec["metadata"].get("version")))
    return rec


def build_metadata(old_meta: dict, version: str, description_extra: str) -> dict:
    meta = {k: v for k, v in old_meta.items()
            if k in ("title", "creators", "keywords", "language", "access_right",
                     "license", "upload_type", "related_identifiers", "notes")}
    # the /records API omits the deposit-only field upload_type; the archive has always been "software"
    meta.setdefault("upload_type", "software")
    meta["version"] = version
    meta["publication_date"] = date.today().isoformat()
    desc = old_meta.get("description", "")
    if description_extra:
        desc = description_extra + "\n\n" + desc
    meta["description"] = desc
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.1.0")
    ap.add_argument("--archive", default=None,
                    help="default: paper/zenodo/view_instant_allocation_v<version>.zip")
    ap.add_argument("--notes-file", default=None,
                    help="text prepended to the Zenodo description for this version")
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--check-token", action="store_true",
                    help="only verify that the token authenticates, change nothing")
    ap.add_argument("--resume-draft", default=None,
                    help="id of an existing new-version draft left by an interrupted run; "
                         "reuse it instead of requesting another new version")
    ap.add_argument("--upload-timeout", type=float, default=3600.0,
                    help="seconds allowed for the archive upload (a 270 MB file can exceed the default)")
    ap.add_argument("--parts", nargs="+", default=None,
                    help="upload these files (split archive parts plus reassembly notes) instead "
                         "of the single archive; each part is retried and checksum-verified")
    ap.add_argument("--publish", action="store_true",
                    help="actually publish; without it the script stops at the draft")
    a = ap.parse_args()

    if a.check_token:
        token = read_token(a.token_file)
        deps = api("GET", "%s/deposit/depositions?size=3" % ZENODO, token).json()
        print("token OK: authenticates, %d deposition(s) visible" % len(deps))
        for d in deps:
            print("  %s  %s  %s" % (d.get("id"), d.get("state"),
                                    (d.get("title") or "")[:60]))
        return 0

    archive = Path(a.archive) if a.archive else \
        ROOT / ("paper/zenodo/view_instant_allocation_v%s.zip" % a.version)
    if not archive.exists():
        raise SystemExit("archive not found: %s (build it first)" % archive)
    token = read_token(a.token_file)

    digest = sha256(archive)
    print("archive: %s\n  size %.1f MB\n  sha256 %s"
          % (archive, archive.stat().st_size / 1e6, digest))

    rec = latest_published(token)
    dep_id = rec["id"]

    if a.resume_draft:
        draft_url = "%s/deposit/depositions/%s" % (ZENODO, a.resume_draft)
        print("resuming existing draft %s" % a.resume_draft)
    else:
        draft = api("POST", "%s/deposit/depositions/%s/actions/newversion" % (ZENODO, dep_id),
                    token).json()
        draft_url = draft["links"]["latest_draft"]
    draft = api("GET", draft_url, token).json()
    draft_id = draft["id"]
    print("draft: id %s  %s" % (draft_id, draft["links"].get("html", draft_url)))

    for f in draft.get("files", []):
        api("DELETE", "%s/deposit/depositions/%s/files/%s" % (ZENODO, draft_id, f["id"]), token)
        print("  removed inherited file %s" % f.get("filename", f.get("key")))

    bucket = draft["links"]["bucket"]
    if a.parts:
        import hashlib as _h
        import time as _t
        for part in [Path(x) for x in a.parts]:
            md5 = _h.md5(part.read_bytes()).hexdigest()
            for attempt in range(1, 5):
                try:
                    with open(part, "rb") as fh:
                        api("PUT", "%s/%s" % (bucket, part.name), token, data=fh, timeout=(60, a.upload_timeout))
                    break
                except Exception as e:  # noqa: BLE001
                    print("  %s attempt %d failed: %s" % (part.name, attempt, str(e)[:120]), flush=True)
                    if attempt == 4:
                        raise
                    _t.sleep(20 * attempt)
            files = api("GET", "%s/deposit/depositions/%s/files" % (ZENODO, draft_id), token).json()
            remote = [f for f in files if f.get("filename") == part.name]
            if not remote or remote[0].get("checksum") != md5:
                raise SystemExit("checksum mismatch for %s" % part.name)
            print("  uploaded %s (%.1f MB, md5 ok)" % (part.name, part.stat().st_size / 1e6), flush=True)
    else:
        _upload_single(api, bucket, archive, token, a, draft_id)
    notes = Path(a.notes_file).read_text(encoding="utf-8") if a.notes_file else ""
    return _finish(a, rec, draft, draft_id, draft_url, token, digest, archive, notes)


def _upload_single(api, bucket, archive, token, a, draft_id):
    print("  uploading %s ..." % archive.name)
    with open(archive, "rb") as fh:
        api("PUT", "%s/%s" % (bucket, archive.name), token, data=fh,
            timeout=(60, a.upload_timeout))
    print("  uploaded")
    files = api("GET", "%s/deposit/depositions/%s/files" % (ZENODO, draft_id), token).json()
    remote = [f for f in files if f.get("filename") == archive.name]
    import hashlib as _h
    md5 = _h.md5(archive.read_bytes()).hexdigest()
    if not remote or remote[0].get("checksum") != md5:
        raise SystemExit("uploaded file missing or checksum mismatch (remote %s, local md5 %s)"
                         % (remote[0].get("checksum") if remote else None, md5))
    print("  remote checksum matches (md5 %s)" % md5)



def _finish(a, rec, draft, draft_id, draft_url, token, digest, archive, notes):
    meta = build_metadata(rec["metadata"], a.version, notes)
    api("PUT", "%s/deposit/depositions/%s" % (ZENODO, draft_id), token,
        json={"metadata": meta}, headers={"Content-Type": "application/json"})
    print("  metadata updated (version %s, %s)" % (meta["version"], meta["publication_date"]))

    if not a.publish:
        print("\nDRY RUN: draft prepared but NOT published.")
        print("Inspect it at %s" % draft["links"].get("html", draft_url))
        print("Re-run with --publish to publish it.")
        return 0

    published = api("POST", "%s/deposit/depositions/%s/actions/publish" % (ZENODO, draft_id),
                    token).json()
    doi = published.get("doi") or published.get("metadata", {}).get("doi")
    print("\nPUBLISHED")
    print("  version DOI : %s" % doi)
    print("  concept DOI : 10.5281/zenodo.%s" % CONCEPT_RECID)
    print("  record      : %s" % published["links"].get("record_html", ""))
    out = ROOT / "paper/zenodo/published_v%s.json" % a.version
    out.write_text(json.dumps({"version": a.version, "doi": doi, "sha256": digest,
                               "archive": archive.name,
                               "concept_doi": "10.5281/zenodo.%s" % CONCEPT_RECID},
                              indent=1), encoding="utf-8")
    print("  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
