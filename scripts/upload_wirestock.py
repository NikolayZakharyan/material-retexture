#!/usr/bin/env python3
"""Upload finished result sets to Wirestock as project submissions.

For each set folder under ``assets/result_sets`` (containing ORIGINAL, TEXTURE
and RESULT images) this reproduces the flow captured in ``UPLOAD_SETS.md``:

    1. POST  /project-submissions                      -> new submission id
    2. for each image (ORIGINAL, TEXTURE, RESULT):
       a. GET   /project-submissions/{id}/media/upload-url?type=Image&fileExtension=..
       b. PUT   <presigned S3 url>  (with the returned x-amz-meta-* headers)
       c. GET   /project-submission-media/{mediaId}/status/done
    3. PUT   /project-submissions/{id}/save            -> map input/reference/output

The three images map to the submission fields:
    ORIGINAL -> inputImage      TEXTURE -> referenceImage      RESULT -> resultImage

Config comes from .env: WIRESTOCK_TOKEN, WIRESTOCK_PROJECT_ID, WIRESTOCK_API_BASE.

Usage:
    python3 scripts/upload_wirestock.py                 # every not-yet-uploaded set
    python3 scripts/upload_wirestock.py <set_folder>    # one specific set
    python3 scripts/upload_wirestock.py --dry-run       # show what would happen
    python3 scripts/upload_wirestock.py --force         # re-upload even if marked done
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "assets" / "result_sets"
UPLOADED_MARKER = ".uploaded"  # written into a set folder once submitted
DESCRIPTION_CSV = "description.csv"
INSTRUCTION_FIELD = "instruction"  # submission key Wirestock shows as "Instruction"

# image role -> filename prefix inside a set folder
ROLE_PREFIX = {"inputImage": "ORIGINAL", "referenceImage": "TEXTURE", "resultImage": "RESULT"}
CONTENT_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


class Wirestock:
    def __init__(self, base: str, token: str, project_id: str):
        self.base = base.rstrip("/")
        self.project_id = project_id
        self.api = httpx.Client(
            timeout=httpx.Timeout(180.0),
            headers={
                "authorization": f"Bearer {token}",
                "accept": "application/json, text/plain, */*",
                "origin": "https://wirestock.io",
                "referer": "https://wirestock.io/",
            },
        )
        # unauthenticated client for checking presigned CloudFront preview URLs
        self.anon = httpx.Client(timeout=30.0, follow_redirects=True)

    # --- step 1 -----------------------------------------------------------
    def create_submission(self) -> str:
        r = self.api.post(
            f"{self.base}/data-program/core/project-submissions",
            json={"projectId": self.project_id, "submission": {},
                  "media": [], "files": [], "releases": []},
        )
        r.raise_for_status()
        # the new id comes back in the Location header: project-submissions/<uuid>
        location = r.headers.get("location", "")
        sub_id = location.rstrip("/").split("/")[-1]
        if not sub_id:
            raise RuntimeError(f"No submission id in Location header: {location!r}")
        return sub_id

    # --- step 2a ----------------------------------------------------------
    def get_upload_url(self, sub_id: str, ext: str) -> tuple[str, dict[str, str], str]:
        r = self.api.get(
            f"{self.base}/data-program/core/project-submissions/{sub_id}/media/upload-url",
            params={"type": "Image", "fileExtension": ext},
        )
        r.raise_for_status()
        data = r.json()
        meta = {h["key"]: h["value"] for h in data["headers"]}
        media_id = meta["x-amz-meta-attribute-media-id"]
        return data["url"], meta, media_id

    # --- step 2b ----------------------------------------------------------
    def put_to_s3(self, url: str, meta: dict[str, str], path: Path, ext: str) -> None:
        headers = dict(meta)
        headers["content-type"] = CONTENT_TYPES.get(ext, "application/octet-stream")
        # a bare client (no auth header) — the presigned URL carries its own auth
        with httpx.Client(timeout=httpx.Timeout(300.0)) as s3:
            r = s3.put(url, content=path.read_bytes(), headers=headers)
            r.raise_for_status()

    # --- step 2c ----------------------------------------------------------
    def mark_done(self, media_id: str) -> None:
        r = self.api.get(
            f"{self.base}/data-program/core/project-submission-media/{media_id}/status/done"
        )
        r.raise_for_status()

    # --- step 3 -----------------------------------------------------------
    def save(self, sub_id: str, mapping: dict[str, str]) -> None:
        # projectId is required by SaveDraftProjectSubmissionRequest (400 without it)
        r = self.api.put(
            f"{self.base}/data-program/core/project-submissions/{sub_id}/save",
            json={"projectId": self.project_id, "submission": mapping},
        )
        r.raise_for_status()

    def get_submission(self, sub_id: str) -> dict:
        r = self.api.get(
            f"{self.base}/data-program/core/project-submissions/{sub_id}"
        )
        r.raise_for_status()
        return r.json()

    def _preview_ready(self, url: str) -> bool:
        """True if the preview/thumbnail URL actually serves an image."""
        try:
            r = self.anon.get(url, headers={"range": "bytes=0-0"})
            return r.status_code in (200, 206) and "image" in r.headers.get("content-type", "")
        except Exception:
            return False

    def ensure_previews(self, sub_id: str, media_ids: list[str],
                        attempts: int = 6, delay: float = 3.0) -> dict[str, bool]:
        """Poll each media's preview; re-trigger status/done for any that aren't
        ready yet. The preview pipeline is fired by status/done and runs async, so
        this guarantees a thumbnail exists before we report success. Returns
        {media_id: ready}."""
        pending = set(media_ids)
        ready: dict[str, bool] = {m: False for m in media_ids}
        for i in range(attempts):
            info = self.get_submission(sub_id)
            urls = {m.get("projectSubmissionMediaId"): (m.get("urls") or {})
                    for m in info.get("media", [])}
            for mid in list(pending):
                u = urls.get(mid, {}).get("preview") or urls.get(mid, {}).get("thumbnail")
                if u and self._preview_ready(u):
                    ready[mid] = True
                    pending.discard(mid)
            if not pending:
                break
            for mid in pending:      # nudge the laggards again
                try:
                    self.mark_done(mid)
                except Exception:
                    pass
            time.sleep(delay)
        return ready

    # --- step 4: finalize -------------------------------------------------
    def submit(self, sub_id: str, dest_status_id: str) -> None:
        """Move the draft to its next workflow status (the 'submit' transition).
        Without this the submission stays a draft and expires after a while."""
        r = self.api.post(
            f"{self.base}/data-program/core/project-submissions/{sub_id}/make-transition",
            json={"projectId": self.project_id, "destinationSubmissionStatusId": dest_status_id},
        )
        r.raise_for_status()

    def upload_image(self, sub_id: str, path: Path) -> str:
        ext = path.suffix.lstrip(".").lower()
        url, meta, media_id = self.get_upload_url(sub_id, ext)
        self.put_to_s3(url, meta, path, ext)
        self.mark_done(media_id)
        return media_id


def classify_set(folder: Path) -> dict[str, Path] | None:
    """Return {field: image_path} for a complete set, else None."""
    files = [p for p in folder.iterdir() if p.is_file()]
    roles: dict[str, Path] = {}
    for field, prefix in ROLE_PREFIX.items():
        matches = sorted(p for p in files
                         if p.stem.upper() == prefix and p.suffix.lower() != ".csv")
        # RESULT may be "RESULT" (exact); ORIGINAL/TEXTURE too. Fall back to prefix match.
        if not matches:
            matches = sorted(p for p in files if p.stem.upper().startswith(prefix)
                             and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        if matches:
            roles[field] = matches[0]
    return roles if len(roles) == 3 else None


def find_sets(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and classify_set(p))


def read_description(folder: Path) -> str:
    """Pull the 'description' column from the set's description.csv (the text we
    submit as the Instruction). Returns '' if absent."""
    csv_path = folder / DESCRIPTION_CSV
    if not csv_path.exists():
        return ""
    try:
        with open(csv_path, newline="") as f:
            row = next(csv.DictReader(f), {})
        return (row.get("description") or "").strip()
    except Exception:
        return ""


def upload_set(ws: Wirestock, folder: Path, dry_run: bool,
               instruction_override: str | None = None, submit: bool = False) -> str | None:
    roles = classify_set(folder)
    if not roles:
        print(f"  skip: not a complete set (need ORIGINAL, TEXTURE, RESULT)")
        return None

    order = ["inputImage", "referenceImage", "resultImage"]
    instruction = instruction_override if instruction_override is not None else read_description(folder)
    if dry_run:
        for field in order:
            print(f"    would upload {roles[field].name:<14} -> {field}")
        print(f"    instruction: {instruction or '(none)'}")
        print(f"    would create submission + save mapping + {'submit' if submit else 'leave as draft'}")
        return None

    sub_id = ws.create_submission()
    print(f"    submission {sub_id}")
    media_ids: dict[str, str] = {}
    for field in order:
        path = roles[field]
        media_ids[field] = ws.upload_image(sub_id, path)
        print(f"    uploaded {path.name:<14} -> {field} = {media_ids[field]}")

    # build the submission with the SAME key order a manual UI upload produces:
    # inputImage, referenceImage, instruction, resultImage (result is added last)
    mapping: dict[str, str] = {
        "inputImage": media_ids["inputImage"],
        "referenceImage": media_ids["referenceImage"],
    }
    if instruction:
        mapping[INSTRUCTION_FIELD] = instruction
        print(f"    instruction: {instruction}")
    mapping["resultImage"] = media_ids["resultImage"]

    ws.save(sub_id, mapping)

    # verify
    info = ws.get_submission(sub_id)
    sub = info.get("submission", {})
    n_media = len(info.get("media", []))
    ok = all(sub.get(f) == media_ids[f] for f in order)
    instr_ok = (not instruction) or sub.get(INSTRUCTION_FIELD) == instruction
    print(f"    saved. media={n_media}, mapping_ok={ok}, instruction_ok={instr_ok}")
    if not ok or not instr_ok:
        print(f"    WARNING: submission is {json.dumps(sub)}")

    # make sure every image's preview actually got generated (re-triggers
    # status/done for any laggards) so all three show a thumbnail in the UI
    image_media = [media_ids[f] for f in order]
    prev = ws.ensure_previews(sub_id, image_media)
    missing = [f for f in order if not prev.get(media_ids[f])]
    print(f"    previews ready: {sum(prev.values())}/{len(image_media)}"
          + (f"  MISSING: {missing}" if missing else ""))

    # finalize so the submission persists (drafts expire otherwise)
    if submit:
        transitions = info.get("workflowState", {}).get("transitions", [])
        if not transitions:
            print("    NOTE: no transition available; left as draft (may expire)")
        else:
            dest = transitions[0]["destinationSubmissionStatusId"]
            ws.submit(sub_id, dest)
            print(f"    submitted (status {dest})")

    return sub_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload result sets to Wirestock.")
    ap.add_argument("sets", nargs="*", type=Path,
                    help="set folder(s) to upload; default: all in assets/result_sets")
    ap.add_argument("--project", help="override WIRESTOCK_PROJECT_ID")
    ap.add_argument("--instruction", help="override the Instruction text for ALL sets "
                                          "(default: each set's description.csv)")
    ap.add_argument("--submit", action="store_true",
                    help="finalize (submit) each set after upload; DEFAULT is draft only")
    ap.add_argument("--dry-run", action="store_true", help="show plan, upload nothing")
    ap.add_argument("--force", action="store_true", help="upload even if already marked uploaded")
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("WIRESTOCK_TOKEN")
    base = os.getenv("WIRESTOCK_API_BASE", "https://api.wirestock.io")
    project_id = args.project or os.getenv("WIRESTOCK_PROJECT_ID")
    if not token or not project_id:
        sys.exit("Set WIRESTOCK_TOKEN and WIRESTOCK_PROJECT_ID in .env")

    if args.sets:
        sets = [p if p.is_absolute() else (PROJECT_ROOT / p) for p in args.sets]
    else:
        sets = find_sets(RESULTS_DIR)
    if not sets:
        sys.exit(f"No complete sets found in {RESULTS_DIR}")

    print(f"Project: {project_id}")
    print(f"Sets:    {len(sets)}\n")

    ws = None if args.dry_run else Wirestock(base, token, project_id)
    done = 0
    for folder in sets:
        marker = folder / UPLOADED_MARKER
        if marker.exists() and not args.force and not args.dry_run:
            print(f"- {folder.name}: already uploaded ({marker.read_text().strip()}), skipping")
            continue
        print(f"- {folder.name}")
        try:
            sub_id = upload_set(ws, folder, args.dry_run, args.instruction,
                                submit=args.submit)
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300]
            print(f"    FAILED {e.response.status_code}: {body}")
            continue
        except Exception as e:
            print(f"    FAILED: {e}")
            continue
        if sub_id:
            marker.write_text(sub_id + "\n")
            done += 1

    if not args.dry_run:
        print(f"\nDone. {done} set(s) submitted to Wirestock.")


if __name__ == "__main__":
    main()
