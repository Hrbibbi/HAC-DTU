#!/bin/bash

if [ -z "${DSDIR:-}" ]; then
  {
    echo "Error: DSDIR is not set."
    echo
    echo "Set DSDIR to the dataset root directory, for example:"
    echo "  export DSDIR=\"/path/to/challenge_data\""
  } >&2
  exit 1
fi

if [ -e "$DSDIR" ] && [ ! -d "$DSDIR" ]; then
  {
    echo "Error: DSDIR points to a file, not a directory: $DSDIR"
    echo "Set DSDIR to a writable directory path and try again."
  } >&2
  exit 1
fi

if ! mkdir -p "$DSDIR"; then
  {
    echo "Error: Could not create or access DSDIR directory: $DSDIR"
    echo "Check path permissions and try again."
  } >&2
  exit 1
fi

OUTPUT_DIR="$(CDPATH= cd "$DSDIR" && pwd -P)"

# Ensure unzip exists
if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is not installed. Installing it now..."
  sudo apt update
  sudo apt install -y unzip
fi

ARCHIVE="$(CDPATH= cd "$DSDIR" && pwd -P)/challenge_data.zip"
OUTPUT_DIR="$(CDPATH= cd "$DSDIR" && pwd -P)/challenge_data"
echo "Challenge data will be downloaded to $OUTPUT_DIR"

URL="https://www.dropbox.com/scl/fo/gcqw0ffbt2xa6vfzlhu51/AESiKo8nIYUKRU_6agUcF1g?rlkey=4o0tiegpeiepxf9ikvnnzfh2s&e=1&dl=1"

echo "Downloading challenge data archive to $ARCHIVE..."
wget --show-progress --progress=bar:force:noscroll -O "$ARCHIVE" "$URL"

echo "Extracting $ARCHIVE to $OUTPUT_DIR (overwriting existing files)..."
unzip -o "$ARCHIVE" -d "$OUTPUT_DIR"

echo "Removing temporary archive $ARCHIVE..."
rm "$ARCHIVE"

echo "Challenge data is ready in $OUTPUT_DIR."