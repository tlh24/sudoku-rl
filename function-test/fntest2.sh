#!/bin/bash

python fntest2.py -c -u 0 -l l1-5-7_1_adam -a &
python fntest2.py -c -u 1 -l l1-5-7_2_adam -a &
python fntest2.py -c -u 1 -l l1-5-7_3_adam -a &
python fntest2.py -c -u 0 -l dp-5-7_1_adam -a -d &
python fntest2.py -c -u 0 -l dp-5-7_2_adam -a -d &
python fntest2.py -c -u 1 -l dp-5-7_3_adam -a -d &

# Wait for all background jobs launched by this script to finish
echo "Waiting for runs to complete..."
wait
echo "All runs finished."

exit 0

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

# Run the 4 commands in parallel
echo "Launching 4 parallel runs..."
python fntest2.py "${other_args[@]}" -u 0 -l "${base_label}-5" &
python fntest2.py "${other_args[@]}" -u 0 -l "${base_label}-6" &
python fntest2.py "${other_args[@]}" -u 0 -l "${base_label}-7" &
python fntest2.py "${other_args[@]}" -u 0 -l "${base_label}-8" &

# Wait for all background jobs launched by this script to finish
echo "Waiting for runs to complete..."
wait
echo "All runs finished."

exit 0
