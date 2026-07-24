#!/usr/bin/env python3
"""Generate flat material-swatch textures with Google's Nano Banana 2 Lite.

Unlike ``nano_banana.py`` (which retextures a product photo), this is pure
text-to-image: it invents flat, full-frame material surfaces — the kind stored
in ``assets/textures`` and fed to the retexturing pipeline as the SECOND image.

Each swatch is a top-down, evenly-lit, edge-to-edge close-up of a single
material (no product, no background, no seams), saved as a 1024x1024 JPEG and
named by material type:
    leather -> leather_<n>.jpg
    fabric  -> textile_<n>.jpg
The <n> continues from the highest existing number of that prefix so nothing is
overwritten.

Usage:
    python3 scripts/gen_textures.py                 # the full built-in set
    python3 scripts/gen_textures.py --only leather  # only the leather swatches
    python3 scripts/gen_textures.py --only textile  # only the fabric swatches
    python3 scripts/gen_textures.py --add leather "oxblood pull-up leather, waxy sheen"
    python3 scripts/gen_textures.py --size 1024

Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the project-root .env file.
"""
from __future__ import annotations

import argparse
import io
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps
from google import genai
from google.genai import errors, types

# --- configuration ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXTURES_DIR = PROJECT_ROOT / "assets" / "textures"

# Nano Banana 2 Lite == Gemini 3.1 Flash Lite Image
MODEL_ID = os.getenv("NANO_BANANA_MODEL", "gemini-3.1-flash-lite-image")

SWATCH_SIZE = 1024  # output is a SWATCH_SIZE x SWATCH_SIZE square
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_API_ATTEMPTS = int(os.getenv("NANO_BANANA_MAX_ATTEMPTS", "5"))

# Filename prefix per material family. Add a family here to support it.
PREFIX = {"leather": "leather", "textile": "textile"}

# Wraps every material description into a flat, full-frame swatch instruction so
# the output fills the frame with just the surface — no product, edges or props.
SWATCH_TEMPLATE = (
    "A flat, full-frame, top-down photograph of {desc}. The material surface "
    "fills the ENTIRE square frame edge to edge, seen straight-on from directly "
    "above like a fabric/leather swatch. Evenly and softly lit with no harsh "
    "glare, sharp focus, fine natural macro detail and realistic weave/grain. "
    "Show ONLY the flat material surface — no product, no object, no folds into "
    "shape, no stitching, no seams, no hardware, no edges of the material, no "
    "background, no hands, no shadows of other objects. A single uniform "
    "material filling the whole image. "
    "Completely plain and generic: NO logos, brand names, trademarks, text, "
    "lettering, numbers, labels or watermarks anywhere."
)

# The built-in swatch set. (material_family, short material description.)
SWATCHES: list[tuple[str, str]] = [
    # --- leather ------------------------------------------------------------
    ("leather", "smooth full-grain tan cognac leather with a soft natural sheen"),
    ("leather", "dark brown distressed pebbled leather with a rugged grain"),
    ("leather", "matte camel-beige suede nubuck with a soft velvety nap"),
    ("leather", "deep oxblood burgundy saffiano leather with a fine cross-hatch grain"),
    ("leather", "black smooth napa leather with subtle natural creases"),
    # --- textile / fabric ---------------------------------------------------
    ("textile", "natural undyed beige linen fabric with a visible plain weave"),
    ("textile", "heather grey wool herringbone tweed fabric"),
    ("textile", "indigo blue cotton denim with a fine diagonal twill weave"),
    ("textile", "cream chunky cable-knit wool fabric"),
    ("textile", "emerald green cotton velvet with a soft plush pile"),
]
# ---------------------------------------------------------------------------


def call_with_retries(fn, what: str, attempts: int = MAX_API_ATTEMPTS):
    """Run *fn* (a no-arg API call), retrying transient server errors with
    exponential backoff. Non-retryable / final failures are re-raised."""
    import time
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code not in RETRYABLE_STATUS or attempt == attempts:
                raise
            print(f"    ({what}: {code} transient error, retry "
                  f"{attempt}/{attempts - 1} in {delay:.0f}s)", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def next_index(folder: Path, prefix: str) -> int:
    """Highest existing <prefix>_<n> in *folder*, plus one (1 if none exist)."""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    nums = [
        int(m.group(1))
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        and (m := pat.match(p.stem))
    ]
    return (max(nums) + 1) if nums else 1


def generate_swatch(client: genai.Client, desc: str, model: str,
                    size: int) -> Image.Image:
    """Text-to-image a single flat material swatch, returned as an exact
    ``size`` x ``size`` RGB image."""
    config = types.GenerateContentConfig(
        image_config=types.ImageConfig(aspect_ratio="1:1", image_size="1K"),
        temperature=1.2,
        seed=random.randint(0, 2_147_483_647),
    )
    prompt = SWATCH_TEMPLATE.format(desc=desc)
    response = call_with_retries(
        lambda: client.models.generate_content(
            model=model, contents=[prompt], config=config),
        what="swatch gen")

    candidates = getattr(response, "candidates", None)
    if not candidates:
        feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(f"No candidates returned. prompt_feedback={feedback}")

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        reason = getattr(candidates[0], "finish_reason", None)
        raise RuntimeError(f"Candidate had no content parts (finish_reason={reason}); "
                           "likely a filtered or empty response — retry.")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and inline.data:
            img = Image.open(io.BytesIO(inline.data)).convert("RGB")
            if img.size != (size, size):
                img = ImageOps.fit(img, (size, size), method=Image.LANCZOS)
            return img
    raise RuntimeError("The model returned no image for this swatch prompt.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate flat material-swatch textures with Nano Banana 2 Lite.")
    ap.add_argument("--only", choices=sorted(PREFIX),
                    help="generate only this material family (default: all)")
    ap.add_argument("--add", nargs=2, action="append", metavar=("FAMILY", "DESC"),
                    default=[],
                    help="add an extra swatch, e.g. --add leather 'waxy oxblood "
                         "pull-up leather'. Repeatable. Skips the built-in set.")
    ap.add_argument("--size", type=int, default=SWATCH_SIZE,
                    help=f"square pixel size of each swatch (default: {SWATCH_SIZE})")
    ap.add_argument("--model", default=MODEL_ID, help=f"model id (default: {MODEL_ID})")
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")

    if args.add:
        jobs = [(fam, desc) for fam, desc in args.add]
        bad = {fam for fam, _ in jobs} - set(PREFIX)
        if bad:
            sys.exit(f"Unknown --add family {sorted(bad)}; choose from {sorted(PREFIX)}")
    else:
        jobs = [(fam, desc) for fam, desc in SWATCHES
                if not args.only or fam == args.only]

    TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=api_key)

    # Reserve filenames up front so each family's numbering keeps incrementing
    # even within this run.
    counters = {fam: next_index(TEXTURES_DIR, PREFIX[fam]) for fam in PREFIX}

    print(f"Model: {args.model}")
    print(f"Generating {len(jobs)} swatch(es) at {args.size}x{args.size} "
          f"into {TEXTURES_DIR.relative_to(PROJECT_ROOT)}/\n")

    made: list[Path] = []
    for i, (fam, desc) in enumerate(jobs, start=1):
        name = f"{PREFIX[fam]}_{counters[fam]}.jpg"
        dest = TEXTURES_DIR / name
        print(f"  [{i}/{len(jobs)}] {name:<14} {desc} ... ", end="", flush=True)
        try:
            img = generate_swatch(client, desc, args.model, args.size)
        except Exception as e:
            print(f"FAILED\n    {e}")
            continue
        img.save(dest, "JPEG", quality=95)
        counters[fam] += 1
        made.append(dest)
        print("saved")

    print(f"\nDone. {len(made)} swatch(es) written to "
          f"{TEXTURES_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
