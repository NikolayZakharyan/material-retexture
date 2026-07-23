#!/usr/bin/env python3
"""Retexture a product photo with Google's Nano Banana 2 Lite.

Takes one image from ``assets/or_images`` (the "base") plus one random
texture from ``assets/textures``, asks Nano Banana 2 Lite
(Gemini 3.1 Flash Lite Image) to re-skin the main item in the base photo
with that texture, and writes the generated image to
``assets/result_sets``.

Usage:
    python3 scripts/nano_banana.py                      # random base + random texture
    python3 scripts/nano_banana.py --base 644_Gq1n..._1600.jpeg
    python3 scripts/nano_banana.py --texture leather_3.jpg
    python3 scripts/nano_banana.py -n 3                 # 3 random texture variants of one base
    python3 scripts/nano_banana.py --prompt "Make the sofa green velvet"
    python3 scripts/nano_banana.py --generate "woman cotton handbag on a table, photorealistic"
                                                        # invent a fresh 1024x1024 base, then retexture it

Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the project-root .env file.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage
from dotenv import load_dotenv
from PIL import Image, ImageOps
from google import genai
from google.genai import errors, types

# --- configuration ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OR_IMAGES_DIR = PROJECT_ROOT / "assets" / "or_images"
TEXTURES_DIR = PROJECT_ROOT / "assets" / "textures"
RESULTS_DIR = PROJECT_ROOT / "assets" / "result_sets"
# saved bank of text-to-image prompts used by `-g N` to invent base images
GEN_PROMPTS_FILE = PROJECT_ROOT / "assets" / "gen_prompts.txt"

# Nano Banana 2 Lite == Gemini 3.1 Flash Lite Image
MODEL_ID = os.getenv("NANO_BANANA_MODEL", "gemini-3.1-flash-lite-image")
# cheap multimodal model used to write the per-edit description (Instruction)
CAPTION_MODEL = os.getenv("NANO_BANANA_CAPTION_MODEL", "gemini-3.1-flash-lite")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_DIM = 1600  # longest edge sent to the API; keeps requests small & fast
GEN_BASE_SIZE = 1024  # default square size for text-to-image generated bases

# Transient API failures (rate limits, model overload, gateway blips) that are
# worth retrying rather than aborting the whole run over.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_API_ATTEMPTS = int(os.getenv("NANO_BANANA_MAX_ATTEMPTS", "5"))

# Appended to every prompt so nothing brand-identifying ends up in the output.
# Stock platforms (Wirestock) reject visible logos, trademarks and legible text,
# so we force plain, generic, unbranded products.
NO_BRANDING_CLAUSE = (
    "The product must be completely generic and unbranded: NO logos, brand "
    "names, trademarks, monograms, emblems, badges, brand tags, labels, "
    "printed text, lettering, numbers, slogans, barcodes, QR codes or "
    "watermarks anywhere in the image. Leave every surface plain."
)

PROMPT_TEMPLATE = (
    "You are given two images. The FIRST image is a product photo. The SECOND "
    "image is a material/texture reference ({texture}).\n\n"
    "TASK: Perform a LOCALIZED, MASKED edit. Change the pixels of ONE region "
    "ONLY — the surface of the main item (furniture, bag, garment, upholstery, "
    "shoes, etc.) — re-skinning it so it is made of the material shown in the "
    "SECOND image. Treat every pixel OUTSIDE that item as a locked, read-only "
    "layer that must be copied through untouched.\n\n"
    "ABSOLUTE RULE — BACKGROUND AND EVERYTHING ELSE IS FROZEN:\n"
    "Outside the item's silhouette, the output must be a byte-for-byte copy of "
    "the FIRST image. Do NOT re-render, re-paint, denoise, sharpen, relight, "
    "recolor, re-compress or 'improve' any of it. Zero pixels may differ there. "
    "If you are tempted to regenerate the whole frame, don't — only paint inside "
    "the item's outline and leave the rest exactly as received.\n\n"
    "Keep IDENTICAL and UNCHANGED (not a single pixel of difference):\n"
    "- the entire background: floor, walls, ground, sky, trees, and every other "
    "object or surface in the scene\n"
    "- any people, faces, hands, skin, hair, clothing (other than the target "
    "item) and poses\n"
    "- the camera angle, framing, crop, zoom, aspect ratio and resolution\n"
    "- the lighting, shadows, highlights, reflections, grain, blur, bokeh and "
    "overall color balance of the whole scene\n"
    "- the exact shape, size, position, proportions and outline of the item "
    "itself (only its SURFACE MATERIAL changes, never its geometry)\n\n"
    "Do not move, resize, restyle, add, remove or crop anything. Do not redraw "
    "the scene. Inside the item's outline, replace ONLY the material/texture "
    "with the reference material, wrapping it naturally over the item's existing "
    "form and following its original folds, seams, curves, lighting and shadows. "
    "The result must be indistinguishable from the original photograph in every "
    "area except the item's surface — as if only that one region had been "
    "repainted and the rest of the file was left completely intact.\n\n"
    "The new material must be plain: do NOT add or invent any logos, brand "
    "names, trademarks, monograms, emblems, badges, printed text, lettering, "
    "numbers, labels or watermarks on the re-skinned surface."
)
# ---------------------------------------------------------------------------


def list_images(folder: Path) -> list[Path]:
    """Image files directly inside *folder* (non-recursive)."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def pick_base(folder: Path, explicit: str | None) -> Path:
    """Choose a base image. Prefers the resized ``*_1600`` variants so we don't
    ship the multi-megabyte originals over the wire."""
    if explicit:
        p = folder / explicit if not os.path.isabs(explicit) else Path(explicit)
        if not p.exists():
            sys.exit(f"Base image not found: {p}")
        return p

    images = list_images(folder)
    if not images:
        sys.exit(f"No images found in {folder}")
    resized = [p for p in images if p.stem.endswith("_1600")]
    return random.choice(resized or images)


def unique_bases(folder: Path) -> list[Path]:
    """One image per subject, preferring the resized ``_1600`` variant so the
    same photo isn't processed twice (original + resized)."""
    groups: dict[str, Path] = {}
    for p in list_images(folder):
        key = p.stem[:-5] if p.stem.endswith("_1600") else p.stem
        if key not in groups or p.stem.endswith("_1600"):
            groups[key] = p
    return sorted(groups.values())


def pick_bases(folder: Path, count: int) -> list[Path]:
    """Pick *count* DISTINCT random base images (deduped, prefers ``_1600``).

    If *count* exceeds how many unique photos exist, every photo is used once."""
    available = unique_bases(folder)
    if not available:
        sys.exit(f"No images found in {folder}")
    if count >= len(available):
        if count > len(available):
            print(f"Only {len(available)} unique image(s) available; using all of them.")
        return available
    return random.sample(available, count)


def pick_texture(folder: Path, explicit: str | None) -> Path:
    if explicit:
        p = folder / explicit if not os.path.isabs(explicit) else Path(explicit)
        if not p.exists():
            sys.exit(f"Texture not found: {p}")
        return p

    textures = list_images(folder)
    if not textures:
        sys.exit(f"No textures found in {folder}")
    return random.choice(textures)


def load_image(path: Path, max_dim: int = MAX_DIM) -> Image.Image:
    """Open an image as RGB, fix EXIF rotation, and downscale so its longest
    edge is at most *max_dim*."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def slugify(text: str, maxlen: int = 40) -> str:
    """Turn a free-text prompt into a short, filename-safe slug."""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:maxlen].rstrip("_") or "image"


def load_gen_prompts(path: Path = GEN_PROMPTS_FILE) -> list[str]:
    """Read the saved base-image prompt bank (one prompt per line; blank lines
    and ``#`` comments ignored)."""
    if not path.exists():
        sys.exit(f"Prompt bank not found: {path}")
    prompts = [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        sys.exit(f"No prompts in {path}")
    return prompts


def pick_gen_prompts(count: int) -> list[str]:
    """Pick *count* generation prompts from the bank. Samples without repeats
    when possible; if *count* exceeds the bank size, cycles through with repeats."""
    bank = load_gen_prompts()
    if count <= len(bank):
        return random.sample(bank, count)
    picks = bank[:]                       # use each at least once
    picks += random.choices(bank, k=count - len(bank))
    random.shuffle(picks)
    return picks


def call_with_retries(fn, what: str, attempts: int = MAX_API_ATTEMPTS):
    """Run *fn* (a no-arg API call), retrying on transient server errors with
    exponential backoff. Non-retryable errors (e.g. 400/permission) and the
    final failed attempt are re-raised so the caller can handle/skip them."""
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


def generate_base_image(client: genai.Client, prompt: str, model: str,
                        size: int = GEN_BASE_SIZE) -> Image.Image:
    """Invent a brand-new base product photo from a *text* prompt with Nano
    Banana, returned as an RGB image of exactly ``size`` x ``size`` pixels.

    Unlike :func:`generate` (which edits an existing photo), this is pure
    text-to-image. We ask for a 1:1 / 1K image and, as a guarantee, fit the
    model's output to an exact square so downstream sizing is deterministic."""
    config = types.GenerateContentConfig(
        image_config=types.ImageConfig(aspect_ratio="1:1", image_size="1K"),
    )
    full_prompt = f"{prompt}\n\n{NO_BRANDING_CLAUSE}"
    response = call_with_retries(
        lambda: client.models.generate_content(
            model=model, contents=[full_prompt], config=config),
        what="base gen")

    candidates = getattr(response, "candidates", None)
    if not candidates:
        feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(f"No candidates returned. prompt_feedback={feedback}")

    for part in candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline and inline.data:
            img = Image.open(io.BytesIO(inline.data)).convert("RGB")
            if img.size != (size, size):
                img = ImageOps.fit(img, (size, size), method=Image.LANCZOS)
            return img
    raise RuntimeError("The model returned no image for the base-generation prompt.")


def texture_label(path: Path) -> str:
    """Human-ish description used inside the prompt, e.g. 'leather' from
    'leather_3.jpg'."""
    stem = path.stem.rsplit("_", 1)[0] if path.stem[-1].isdigit() else path.stem
    return stem.replace("_", " ")


def make_run_dir(base: Path, when: datetime) -> Path:
    """Create a fresh per-result folder named '<first 8 of base name>_<DD-MM_HH-MM-SS>'.

    ('/' and ':' from 'DD/MM hh:mm:ss' can't live in a folder name, so dashes are
    used here; the exact 'DD/MM hh:mm:ss' string is still written into the CSV.)"""
    name = f"{base.stem[:8]}_{when.strftime('%d-%m_%H-%M-%S')}"
    run_dir = RESULTS_DIR / name
    n = 2
    while run_dir.exists():  # avoid clobbering if two results land in the same second
        run_dir = RESULTS_DIR / f"{name}_{n}"
        n += 1
    run_dir.mkdir(parents=True)
    return run_dir


def decode_result(data: bytes, size: tuple[int, int]) -> Image.Image:
    """Decode the model's image bytes to an RGB image at exactly *size*.

    The model returns whatever size it likes, so we always resize to the
    original photo's ``(width, height)``."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return img


def item_mask(original: Image.Image, result: Image.Image,
              thresh: int = 15) -> np.ndarray:
    """Estimate which pixels belong to the re-textured item.

    A generative model repaints the whole frame, so the result differs from the
    original *everywhere* — but only slightly in the background and strongly on
    the item whose material was swapped. We threshold that difference, then clean
    it up to get a solid mask of just the item. Returns a float32 array in
    [0, 1] with feathered edges for a seamless paste."""
    a = np.asarray(original, dtype=np.int16)
    b = np.asarray(result, dtype=np.int16)
    diff = np.abs(a - b).max(axis=2)
    m = diff > thresh

    # 1. drop tiny background speckle (a light opening keeps thin item parts)
    m = ndimage.binary_opening(m, structure=np.ones((3, 3)), iterations=1)
    # 2. keep every sizable region. The item can be split into several pieces by
    #    things in front of it (e.g. a crossed arm over a shirt), so keep all
    #    blobs at least 2% as big as the largest — not just the single biggest.
    labels, n = ndimage.label(m)
    if n >= 1:
        sizes = ndimage.sum(m, labels, range(1, n + 1))
        min_area = max(sizes.max() * 0.02, 400)
        keep = np.nonzero(sizes >= min_area)[0] + 1
        m = np.isin(labels, keep)
    # 3. bridge the gaps between those pieces and fill interior holes, so the
    #    whole item becomes one solid mask with no leftover original patches.
    m = ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=12)
    m = ndimage.binary_fill_holes(m)
    # 4. grow by a couple of pixels to swallow thin slivers of the old material
    #    right at the item's edge.
    m = ndimage.binary_dilation(m, structure=np.ones((3, 3)), iterations=2)

    # feather the edge so the composite has no hard seam
    alpha = ndimage.gaussian_filter(m.astype(np.float32), sigma=2.0)
    return np.clip(alpha, 0.0, 1.0)


def composite_over_original(original: Image.Image, result: Image.Image
                            ) -> tuple[Image.Image, Image.Image]:
    """Keep the model's output only on the item; restore the ORIGINAL pixels
    everywhere else. Returns (composited_image, mask_image_for_debugging)."""
    alpha = item_mask(original, result)
    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(result, dtype=np.float32)
    blended = a * (1.0 - alpha[..., None]) + b * alpha[..., None]
    out = Image.fromarray(blended.round().astype(np.uint8), "RGB")
    mask_img = Image.fromarray((alpha * 255).astype(np.uint8), "L")
    return out, mask_img


def save_result_set(run_dir: Path, images: list[tuple[bytes, str]], base: Path,
                    texture: Path, material: str, description: str, model: str,
                    when: datetime, composite: bool = True,
                    debug_mask: bool = False, gen_prompt: str | None = None) -> None:
    """Fill *run_dir* with the 4 deliverables: original, texture, RESULT.jpg, CSV.

    When *composite* is True (default), each RESULT keeps the model's output only
    on the swapped item and restores the exact ORIGINAL pixels everywhere else,
    so the background has zero changes."""
    # 1. original image (copied verbatim from assets/or_images)
    shutil.copy2(base, run_dir / f"ORIGINAL{base.suffix.lower()}")
    # 2. texture image (copied verbatim from assets/textures)
    shutil.copy2(texture, run_dir / f"TEXTURE{texture.suffix.lower()}")
    # 3. the generated image(s) as RESULT.jpg (RESULT_2.jpg, ... for extras),
    #    resized to the original photo's exact dimensions
    original = ImageOps.exif_transpose(Image.open(base)).convert("RGB")
    for i, (data, mime) in enumerate(images, start=1):
        dest = run_dir / ("RESULT.jpg" if i == 1 else f"RESULT_{i}.jpg")
        result = decode_result(data, original.size)
        if composite:
            result, mask = composite_over_original(original, result)
            if debug_mask:
                mask.save(run_dir / (f"MASK.png" if i == 1 else f"MASK_{i}.png"))
        result.save(dest, "JPEG", quality=95)
    # 4. description of the change
    with open(run_dir / "description.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["datetime", "original_image", "texture_image",
                    "result_image", "material", "model", "description",
                    "gen_prompt"])
        w.writerow([when.strftime("%d/%m %H:%M:%S"), base.name, texture.name,
                    "RESULT.jpg", material, model, description,
                    gen_prompt or ""])


def generate(client: genai.Client, prompt: str, base_img: Image.Image,
             texture_img: Image.Image) -> tuple[list[tuple[bytes, str]], str]:
    """Call Nano Banana and return (list of (image_bytes, mime), any_text)."""
    response = call_with_retries(
        lambda: client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt, base_img, texture_img],
        ),
        what="retexture")

    candidates = getattr(response, "candidates", None)
    if not candidates:
        feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(f"No candidates returned. prompt_feedback={feedback}")

    images: list[tuple[bytes, str]] = []
    texts: list[str] = []
    for part in candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline and inline.data:
            images.append((inline.data, inline.mime_type or "image/png"))
        elif getattr(part, "text", None):
            texts.append(part.text)
    return images, "\n".join(texts)


def describe_edit(client: genai.Client, base_img: Image.Image,
                  texture_img: Image.Image, material: str, fallback: str) -> str:
    """Ask a fast vision model to write ONE sentence describing this specific edit:
    which item changed, and to what material and colour. Falls back to a generic
    line if the call fails."""
    prompt = (
        "You are writing a short caption for a product-image edit in a dataset.\n"
        "IMAGE 1 is the original product photo. IMAGE 2 is a material/texture swatch "
        f"that was applied to the main item in IMAGE 1 (material type: {material}).\n\n"
        "Write ONE short sentence (max ~16 words) that answers: which item was changed, "
        "and to what material and colour. Name the item specifically (e.g. sofa, handbag, "
        "sneakers, jacket, armchair) and give the texture's main colour with a common name "
        "(e.g. tan, dark brown, black, navy, cognac). Start with 'Changed' or 'Replaced'. "
        "Reply with ONLY the sentence — no quotes, no extra text."
    )
    try:
        resp = client.models.generate_content(
            model=CAPTION_MODEL, contents=[prompt, base_img, texture_img])
        text = (resp.text or "").strip().strip('"').splitlines()[0].strip()
        return text or fallback
    except Exception as e:
        print(f"    (caption failed: {e}; using generic description)")
        return fallback


def process_base(client: genai.Client, base_path: Path, args,
                 gen_prompt: str | None = None) -> list[Path]:
    """Generate --count result folders for a single base image.

    *gen_prompt* is the text-to-image prompt that invented this base (when it
    came from ``-g``), recorded in each result's CSV; ``None`` for real photos."""
    base_img = load_image(base_path)
    print(f"Base:  {base_path.name}  ({base_img.width}x{base_img.height})")

    made: list[Path] = []
    for i in range(1, args.count + 1):
        texture_path = pick_texture(TEXTURES_DIR, args.texture)
        texture_img = load_image(texture_path)
        material = texture_label(texture_path)
        prompt = args.prompt or PROMPT_TEMPLATE.format(texture=material)
        description = args.prompt or describe_edit(
            client, base_img, texture_img, material,
            fallback=f"Replaced the main item's surface material with the {material} texture")

        print(f"  [{i}/{args.count}] texture: {texture_path.name} ... ", end="", flush=True)
        try:
            images, text = generate(client, prompt, base_img, texture_img)
        except Exception as e:
            print(f"FAILED\n    {e}")
            continue

        if not images:
            print(f"no image returned. model said: {text or '(nothing)'}")
            continue

        when = datetime.now()
        run_dir = make_run_dir(base_path, when)
        save_result_set(run_dir, images, base_path, texture_path,
                        material, description, args.model, when,
                        composite=not args.no_composite, debug_mask=args.debug_mask,
                        gen_prompt=gen_prompt)
        made.append(run_dir)
        print(f"saved {run_dir.name}/")
        if text:
            print(f"    note: {text}")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description="Retexture a photo with Nano Banana 2 Lite.")
    ap.add_argument("--base", help="base image filename in assets/or_images (default: random)")
    ap.add_argument("--all", action="store_true",
                    help="process every image in assets/or_images (deduped, prefers _1600)")
    ap.add_argument("-i", "--images", type=int,
                    help="how many DIFFERENT random photos to process from "
                         "assets/or_images (default: 1). Use -n for variants per photo.")
    ap.add_argument("-g", "--generate", type=int, metavar="N",
                    help="invent N NEW base images with Nano Banana (using random "
                         "prompts from assets/gen_prompts.txt) instead of using "
                         "assets/or_images, then retexture them")
    ap.add_argument("--gen-prompt", metavar="PROMPT",
                    help="explicit text-to-image prompt for --generate (used for "
                         "all N images; overrides the saved prompt bank)")
    ap.add_argument("--gen-size", type=int, default=GEN_BASE_SIZE,
                    help=f"square pixel size for --generate bases (default: {GEN_BASE_SIZE})")
    ap.add_argument("--texture", help="texture filename in assets/textures (default: random)")
    ap.add_argument("--prompt", help="override the instruction sent to the model")
    ap.add_argument("--model", default=MODEL_ID, help=f"model id (default: {MODEL_ID})")
    ap.add_argument("-n", "--count", type=int, default=1,
                    help="results per base image; each uses a fresh random texture "
                         "(ignored if --texture is set)")
    ap.add_argument("--no-composite", action="store_true",
                    help="disable background compositing (keep the model's raw "
                         "output; the background will have minor pixel changes)")
    ap.add_argument("--debug-mask", action="store_true",
                    help="also save the item mask as MASK.png for inspection")
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")

    client = genai.Client(api_key=api_key)

    gen_prompt_map: dict[Path, str] = {}
    if args.generate is not None:
        if args.generate < 1:
            sys.exit("-g/--generate needs a count of at least 1")
        print(f"Model: {args.model}")
        if args.gen_prompt:
            prompts = [args.gen_prompt] * args.generate
        else:
            prompts = pick_gen_prompts(args.generate)

        OR_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        bases = []
        for j, gprompt in enumerate(prompts, start=1):
            print(f"[gen {j}/{args.generate}] {gprompt!r} ...", flush=True)
            try:
                gen_img = generate_base_image(client, gprompt, args.model, args.gen_size)
            except Exception as e:
                print(f"  FAILED: {e}\n  skipping this base.")
                continue
            when0 = datetime.now()
            stem = f"gen_{slugify(gprompt)}_{when0.strftime('%d-%m_%H-%M-%S')}"
            gen_path = OR_IMAGES_DIR / f"{stem}.jpg"
            k = 2
            while gen_path.exists():  # two bases in the same second -> unique name
                gen_path = OR_IMAGES_DIR / f"{stem}_{k}.jpg"
                k += 1
            gen_img.save(gen_path, "JPEG", quality=95)
            print(f"  saved base: {gen_path.name}  ({gen_img.width}x{gen_img.height})")
            bases.append(gen_path)
            gen_prompt_map[gen_path] = gprompt

        if not bases:
            sys.exit("All base-image generations failed; nothing to retexture.")

        # Retexturing needs a material reference; if none exist, stop after
        # producing the base(s) rather than crashing deeper in the pipeline.
        if not args.texture and not list_images(TEXTURES_DIR):
            print(f"\nNo textures in {TEXTURES_DIR.relative_to(PROJECT_ROOT)}/ — "
                  "saved the generated base(s) only.\nAdd a material image there "
                  "(then run process_images.py) to retexture them.")
            return
    elif args.all:
        bases = unique_bases(OR_IMAGES_DIR)
        if not bases:
            sys.exit(f"No images found in {OR_IMAGES_DIR}")
    elif args.base:
        bases = [pick_base(OR_IMAGES_DIR, args.base)]
    else:
        bases = pick_bases(OR_IMAGES_DIR, args.images or 1)

    if args.generate is None:
        print(f"Model: {args.model}")
    if len(bases) > 1 or args.count > 1:
        print(f"Batch: {len(bases)} image(s) x {args.count} texture(s) each")
    print()

    made: list[Path] = []
    for n, base_path in enumerate(bases, start=1):
        if len(bases) > 1:
            print(f"=== {n}/{len(bases)} ===")
        made += process_base(client, base_path, args,
                             gen_prompt=gen_prompt_map.get(base_path))
        print()

    print(f"Done. {len(made)} result folder(s) in {RESULTS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
