#!/usr/bin/env python3
"""Resize + center-crop all images in this folder to 1600x1600, save as JPG,
and rename them to leather_1.jpg, leather_2.jpg, ...

Usage:
    python3 process_images.py

Requires Pillow (`pip install Pillow`). Formats Pillow can't open (e.g. AVIF)
are converted first with ImageMagick (`convert`) if it is available.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageOps

# --- settings ---------------------------------------------------------------
SIZE = (1600, 1600)          # target width x height
PREFIX = "texture"           # output name prefix -> texture_1.jpg, ...
QUALITY = 99                 # JPEG quality
SOURCE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tiff"}
# ---------------------------------------------------------------------------

# optional AVIF support if the plugin is installed
try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

OUTPUT_RE = re.compile(rf"^{re.escape(PREFIX)}_\d+\.jpg$")


def load_rgb(src: Path) -> Image.Image:
    """Open an image as RGB, falling back to ImageMagick for formats Pillow
    can't decode (e.g. AVIF on older Pillow)."""
    try:
        with Image.open(src) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        if not shutil.which("convert"):
            raise
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            subprocess.run(
                ["convert", str(src), str(tmp_path)],
                check=True, capture_output=True,
            )
            with Image.open(tmp_path) as im:
                return im.convert("RGB")
        finally:
            tmp_path.unlink(missing_ok=True)


def main():
    folder = Path(__file__).resolve().parent
    script_name = Path(__file__).name

    # collect source images, skipping this script and our own outputs
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.name != script_name
        and p.suffix.lower() in SOURCE_EXTS
        and not OUTPUT_RE.match(p.name)
    )

    if not images:
        print("No images found.")
        return

    count = 0
    for src in images:
        try:
            im = load_rgb(src)
            im = ImageOps.fit(im, SIZE, method=Image.LANCZOS)  # fill + center-crop
            count += 1
            out = folder / f"{PREFIX}_{count}.jpg"
            im.save(out, "JPEG", quality=QUALITY)
            print(f"{src.name} -> {out.name}")
        except Exception as e:
            print(f"Skipped {src.name}: {e}")


if __name__ == "__main__":
    main()
