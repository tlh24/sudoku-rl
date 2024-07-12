#!/bin/bash

# Array of batch sizes
batch_sizes=(32 48 64 80 96 128 160 192 224 256 320 384 448 512 640 768)
# batch_sizes=(32 48 )

# Number of times to run the script for each batch size
num_runs=16

# Array of heads and layers combinations
heads_layers=(
	"1 1"
	"2 1"
	"2 2"
	"4 4"
	"8 8"
)

# Function to run the script with specified parameters
run_script() {
	local batch_size=$1
	local gpu_id=$2
	local heads=$3
	local layers=$4
	echo "Run $run_id for batch size $batch_size, heads $heads, layers $layers on GPU $gpu_id"
	python test_gtrans2.py -m -b $batch_size -d $gpu_id --heads $heads --layers $layers
}

for batch_size in "${batch_sizes[@]}"; do
	echo "Running with batch size: $batch_size"
	
	run_script $batch_size 1 1 1 & # gpu, heads, layers
	run_script $batch_size 1 2 1 &
	run_script $batch_size 0 2 2 &
	run_script $batch_size 0 4 4 &
	
	# Wait for all 4 processes to finish before starting the next batch
	wait
	
	echo "Finished batch size: $batch_size"
	echo "------------------------"
done



echo "All runs completed."
