#!/usr/bin/env python3
"""Resize image(s) so the SHORTER side is 1600px, at maximum quality.

Aspect ratio is kept and images are only ever downscaled (never enlarged).
Each resized image is saved as "<name>_<size><ext>" and the original is then
DELETED (pass --keep to preserve it). JPEGs are written at high quality
(q=95, no chroma subsampling) with the ICC profile and EXIF preserved.

Usage:
    python3 resize.py                        # resize every image here, delete originals
    python3 resize.py image.jpg              # resize one image
    python3 resize.py *.jpg                  # resize many
    python3 resize.py photo.png -s 1200      # custom short-side target
    python3 resize.py --keep                 # keep the originals
"""
import argparse
from pathlib import Path
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
JPEG_QUALITY = 95  # Pillow's recommended max useful value


def is_resized_output(path: Path, size: int) -> bool:
    """True if this file looks like something we already produced (ends in _<size>)."""
    return path.stem.endswith(f"_{size}")


def find_images(folder: Path, size: int) -> list[Path]:
    """All image files in `folder`, skipping our own resized outputs."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not is_resized_output(p, size)
    )


def save_max_quality(img: Image.Image, out: Path, info: dict) -> None:
    """Save *img* to *out* at the highest sensible quality for its format,
    preserving colour profile / EXIF where possible."""
    ext = out.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        params = dict(quality=JPEG_QUALITY, subsampling=0, optimize=True)
        if info.get("icc_profile"):
            params["icc_profile"] = info["icc_profile"]
        if info.get("exif"):
            params["exif"] = info["exif"]
        img.save(out, **params)
    elif ext == ".webp":
        img.save(out, quality=JPEG_QUALITY, method=6)
    elif ext == ".png":
        img.save(out, optimize=True)  # PNG is lossless
    else:
        img.save(out)


def resize(path: Path, size: int, keep_original: bool = False) -> None:
    img = Image.open(path)
    short = min(img.width, img.height)
    if short <= size:
        print(f"skip  {path.name} (short side already {short}px)")
        return
    scale = size / short
    new_size = (round(img.width * scale), round(img.height * scale))
    resized = img.resize(new_size, Image.LANCZOS)
    out = path.with_name(f"{path.stem}_{size}{path.suffix}")
    save_max_quality(resized, out, img.info)
    img.close()
    print(f"saved {out.name} ({new_size[0]}x{new_size[1]}, short side {size}px)")
    # replace the original: delete it only once the resized file exists on disk
    if not keep_original and out.resolve() != path.resolve() and out.exists():
        path.unlink()
        print(f"deleted {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resize images so the shorter side is a target size, at max quality.")
    ap.add_argument("images", nargs="*", type=Path,
                    help="image file(s); if omitted, every image in the current folder")
    ap.add_argument("-s", "--size", type=int, default=1600,
                    help="target size for the SHORTER side (default 1600)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the original file (default: delete it after resizing)")
    args = ap.parse_args()

    images = args.images or find_images(Path.cwd(), args.size)
    if not images:
        print("No images found to resize.")
        return

    for path in images:
        try:
            resize(path, args.size, keep_original=args.keep)
        except Exception as e:
            print(f"error {path}: {e}")


if __name__ == "__main__":
    main()
