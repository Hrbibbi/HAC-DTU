#!/usr/bin/env bash
set -euo pipefail

input_folders=(
  "/work3/projects/HAC-DTU/challenge_data/AsteroidModel01_shape_public/Asteroid1_videos_simulation"
  "/work3/projects/HAC-DTU/challenge_data/AsteroidModel02_shape_public/Asteroid2_videos_simulation"
  "/work3/projects/HAC-DTU/challenge_data/AsteroidModel03_shape_public/Asteroid3_videos_simulation"
  "/work3/projects/HAC-DTU/challenge_data/AsteroidModel01_shape_public/Asteroid1_videos_real"
  "/work3/projects/HAC-DTU/challenge_data/AsteroidModel02_shape_public/Asteroid2_videos_real"
  "/work3/projects/HAC-DTU/challenge_data/AsteroidModel03_shape_public/Asteroid3_videos_real"
)

output_folders=(
  "/work3/projects/HAC-DTU/workspaces/jeppe/camera_perspective_inference/first_frames/model_1_simulation"
  "/work3/projects/HAC-DTU/workspaces/jeppe/camera_perspective_inference/first_frames/model_2_simulation"
  "/work3/projects/HAC-DTU/workspaces/jeppe/camera_perspective_inference/first_frames/model_3_simulation"
  "/work3/projects/HAC-DTU/workspaces/jeppe/camera_perspective_inference/first_frames/model_1_real"
  "/work3/projects/HAC-DTU/workspaces/jeppe/camera_perspective_inference/first_frames/model_2_real"
  "/work3/projects/HAC-DTU/workspaces/jeppe/camera_perspective_inference/first_frames/model_3_real"
)

if (( ${#input_folders[@]} != ${#output_folders[@]} )); then
  echo "Error: input_folders and output_folders must have the same number of entries." >&2
  exit 1
fi

for i in "${!input_folders[@]}"; do
  input_folder="${input_folders[$i]}"
  output_folder="${output_folders[$i]}"

  mkdir -p "$output_folder"

  printf 'Input folder: <%q>\n' "$input_folder"

  command find "$input_folder" -maxdepth 1 -type f -iname '*.mp4' -print0 |
  while IFS= read -r -d '' video; do
    # Guard against a tool returning a path without its initial slash.
    [[ "$video" == /* ]] || video="/$video"

    filename="$(basename "$video")"
    name="${filename%.*}"

    printf 'Processing: <%q>\n' "$video"

    ffmpeg -y -i "$video" -frames:v 1 -update 1 \
      "$output_folder/${name}_frame0.png"
  done
done