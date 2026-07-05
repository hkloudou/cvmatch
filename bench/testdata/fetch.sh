#!/usr/bin/env bash
# Downloads real sample photographs from OpenCV's samples/data into this
# directory. The bench scenarios pick them up automatically; everything
# still works (with synthetic scenes only) when these files are absent.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://raw.githubusercontent.com/opencv/opencv/4.12.0/samples/data
for f in fruits.jpg baboon.jpg building.jpg graf1.png starry_night.jpg; do
  [ -f "$f" ] && { echo "have $f"; continue; }
  echo "fetching $f"
  curl -fsSL --retry 3 "$BASE/$f" -o "$f"
done
echo "sample images ready"
