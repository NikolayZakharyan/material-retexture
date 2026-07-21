# Nano Banana 2 Lite — texture swap

Take a product photo, drop a random material texture onto its main item using
Google AI Studio's **Nano Banana 2 Lite** (`gemini-3.1-flash-lite-image`), and
save the result.

```
assets/
  or_images/     base product photos (input)
  textures/      material references, e.g. leather_1.jpg (input)
  result_sets/   generated images land here (output)
scripts/
  nano_banana.py the whole pipeline
```

## Setup

Already done in this repo:

* a virtualenv at `.venv` with `google-genai`, `python-dotenv`, `Pillow`
* `.env` holding `GEMINI_API_KEY`

To recreate the env elsewhere:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

`./run.sh` does **two steps in one run**: (1) generate the sets with Nano Banana,
then (2) upload them to Wirestock **as drafts** (see [Wirestock upload](#wirestock-upload)).
Nothing is ever submitted unless you add `--submit`.

```bash
./run.sh --all                    # EVERY photo in or_images -> generate -> upload as draft
./run.sh --images 5               # 5 DIFFERENT random photos, then upload as draft
./run.sh                          # 1 random base photo + random texture, then upload as draft
./run.sh --all -n 2               # every photo, 2 random-texture variants each
./run.sh -n 3                     # ONE base, 3 random-texture variants (same photo x3)
./run.sh --texture leather_3.jpg  # pick the texture
./run.sh --base 644_..._1600.jpeg # pick the base photo
./run.sh --prompt "Turn the chair into blue denim"
./run.sh --debug-mask             # also save MASK.png (the item region)
./run.sh --no-composite           # keep the model's raw output (background may drift)

# controlling step 2:
./run.sh --all --submit           # opt in to finalize/submit (OFF by default)
./run.sh --all --no-upload        # generate only, skip Wirestock entirely
```

**`-i` / `--images N` vs `-n` / `--count N`** — two different counts:

* `--images N` = how many **different photos** to process (distinct random picks
  from `or_images`). This is what you want for "do N images".
* `-n N` = how many **texture variants of each photo** (same photo, N times, each
  with a fresh random texture).

`--all` processes every image in `assets/or_images`, skipping duplicate
full-size originals when a `_1600` copy exists, so each photo is done once.

Or directly: `.venv/bin/python scripts/nano_banana.py [options]`.

## How it works

1. Picks a base image from `assets/or_images` (prefers the `_1600` resized
   variants so the huge originals aren't uploaded).
2. Picks one random texture from `assets/textures`.
3. Downscales both to ≤1600 px on the long edge.
4. Sends `[prompt, base_image, texture_image]` to Nano Banana 2 Lite asking it
   to re-skin the main item with the texture while keeping shape, lighting and
   background intact.
5. **Composites the result back over the original** so only the item changes.
   The model is generative and re-renders the whole frame, so its raw output has
   tiny differences everywhere. To guarantee a truly unchanged background, the
   script builds a mask of the item (from where the output differs most from the
   original), then keeps the model's pixels only inside that mask and restores
   the exact original pixels everywhere else. The RESULT is also resized to the
   original photo's exact dimensions. Disable with `--no-composite`.
6. Creates one folder per result under `assets/result_sets/` and fills it with 4 files.

## Output layout

Each run makes a folder named `<first 8 chars of base name>_<DD-MM_HH-MM-SS>`
(dashes because `/` and `:` aren't allowed in folder names — the exact
`DD/MM hh:mm:ss` value is kept inside the CSV):

```
result_sets/
  6549_qo0_20-07_16-55-00/
    ORIGINAL.jpg      copy of the base photo from or_images
    TEXTURE.jpg       copy of the texture from textures
    RESULT.jpg        the generated image
    description.csv   datetime, source files, material, model, what changed
```

The `description` is written per edit by a fast vision model (`gemini-3.1-flash-lite`)
that looks at the original photo + texture and names the item, material and colour
(e.g. *"Replaced the orange handbag with magenta leather."*). It becomes the
Wirestock **Instruction**. Override the models with `NANO_BANANA_MODEL` / `--model`
and `NANO_BANANA_CAPTION_MODEL`.

## Wirestock upload

`scripts/upload_wirestock.py` submits finished sets to a Wirestock project. It is
run automatically as step 2 of `./run.sh`, or on its own:

```bash
.venv/bin/python scripts/upload_wirestock.py            # upload all new sets AS DRAFTS
.venv/bin/python scripts/upload_wirestock.py --dry-run  # show the plan only
.venv/bin/python scripts/upload_wirestock.py <folder>   # one specific set
.venv/bin/python scripts/upload_wirestock.py --submit   # also finalize/submit (opt-in)
.venv/bin/python scripts/upload_wirestock.py --force    # re-upload an already-uploaded set
```

Per set it: creates a submission, uploads `ORIGINAL`→`inputImage`,
`TEXTURE`→`referenceImage`, `RESULT`→`resultImage`, sets the **Instruction** from
the CSV's `description` column, and saves it **as a draft**. It does NOT submit
unless you pass `--submit`.

* After uploading it **verifies each image's preview actually rendered** (the
  thumbnail pipeline is fired by `status/done` and runs async), re-triggering
  `status/done` for any that lag — so all three, including RESULT, show a preview
  in the UI. Look for `previews ready: 3/3` in the output.

* A `.uploaded` file (holding the submission id) is written into each set folder
  so re-runs skip it. Use `--force` to override.
* **Heads-up: drafts expire.** An uploaded-but-not-submitted submission disappears
  from the project after a while (~10–15 min in testing). Saving as draft is the
  default per your preference; run with `--submit` when you're ready to make them
  permanent.

Config lives in `.env`: `WIRESTOCK_TOKEN`, `WIRESTOCK_PROJECT_ID`,
`WIRESTOCK_API_BASE`. The bearer token expires eventually — refresh it from a
logged-in browser session when uploads start returning 401.

