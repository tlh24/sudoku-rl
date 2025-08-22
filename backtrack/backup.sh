#!/usr/bin/bash

# Define the prefix of files to be moved
prefix="rrn_hard_backtrack_"

# Loop through files matching the prefix pattern
for file in ${prefix}*; do
	# Skip if no files match or if it's already a backup file
	[[ -e "$file" && "$file" != *"_backup_"* ]] || continue

	# Find the next available backup number
	counter=1
	while [[ -e "${file}_backup_${counter}" ]]; do
		((counter++))
	done

	# Move the file to the backup name with the found counter
	mv "$file" "${file}_backup_${counter}"
	echo "Moved $file to ${file}_backup_${counter}"
done
