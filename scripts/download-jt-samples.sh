#!/usr/bin/env bash
set -euo pipefail

sample_dir="${JT_SAMPLE_DIR:-dist/jt-samples}"
base_url="https://d23lrj9wre6ddz.cloudfront.net/samples"
samples=(
  "Control_Cabinet_assm_LW.jt"
  "ElectricRazor_assm_LW.jt"
)

mkdir -p "$sample_dir"

for sample in "${samples[@]}"; do
  target="$sample_dir/$sample"
  url="$base_url/$sample"
  printf 'downloading %s\n' "$url"
  curl -L --fail --retry 3 --retry-delay 2 --show-error --output "$target" "$url"
done

printf 'downloaded JT samples to %s\n' "$sample_dir"
