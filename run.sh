#!/usr/bin/env bash
# Generate result sets with Nano Banana, then upload them to Wirestock as DRAFTS,
# all in one run. (Uploads are saved as drafts only — never submitted.)
#
#   ./run.sh --all              generate every photo in or_images, then upload as draft
#   ./run.sh --images 3         3 random photos, then upload as draft
#   ./run.sh --all --submit     upload AND finalize/submit (opt-in; off by default)
#   ./run.sh --all --no-upload  generate only, skip Wirestock entirely
#
# Flags consumed here: --no-upload (skip step 2) and --submit (finalize instead of
# leaving as a draft). Every other argument is forwarded to nano_banana.py.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

do_upload=1
upload_args=()
gen_args=()
for a in "$@"; do
  case "$a" in
    --no-upload) do_upload=0 ;;
    --submit)    upload_args+=("--submit") ;;
    *)           gen_args+=("$a") ;;
  esac
done

# 1. create the sets
if [ ${#gen_args[@]} -gt 0 ]; then
  "$PY" scripts/nano_banana.py "${gen_args[@]}"
else
  "$PY" scripts/nano_banana.py
fi

# 2. upload + submit the new sets (already-uploaded sets are skipped)
if [ "$do_upload" -eq 1 ]; then
  echo
  echo "=== Uploading sets to Wirestock ==="
  if [ ${#upload_args[@]} -gt 0 ]; then
    "$PY" scripts/upload_wirestock.py "${upload_args[@]}"
  else
    "$PY" scripts/upload_wirestock.py
  fi
fi
