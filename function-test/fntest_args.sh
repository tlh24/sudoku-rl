#!/bin/bash

# Find the base label and store other args
base_label=""
other_args=()
skip_next=false

for arg in "$@"; do
  if [[ "$skip_next" == true ]]; then
    base_label="$arg"
    skip_next=false
    continue
  fi
  if [[ "$arg" == "-l" ]]; then
    skip_next=true
  else
    other_args+=("$arg")
  fi
done

# Check if base_label was found
if [[ -z "$base_label" ]]; then
  echo "Error: -l <base_label> argument is required." >&2
  exit 1
fi

echo "Launching 4 parallel runs..."
python fntest_args.py "${other_args[@]}" -u 0 -l "${base_label}-1" &
python fntest_args.py "${other_args[@]}" -u 1 -l "${base_label}-2" &
python fntest_args.py "${other_args[@]}" -u 0 -l "${base_label}-3" &
python fntest_args.py "${other_args[@]}" -u 1 -l "${base_label}-4" &

# Wait for all background jobs launched by this script to finish
echo "Waiting for runs to complete..."
wait
echo "All runs finished."

exit 0
