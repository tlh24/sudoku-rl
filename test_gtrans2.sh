#!/bin/bash

# Array of batch sizes
batch_sizes=(32 48 64 80 96 128 160 192 224 256 320 384 448 512 640 768)

# Number of times to run the script for each batch size
num_runs=12

# Function to run the script with specified parameters
run_script() {
  local batch_size=$1
  local run_id=$2
  local gpu_id=$3
  echo "Run $run_id for batch size $batch_size on GPU $gpu_id"
  python test_gtrans2.py -m -b $batch_size -d $gpu_id
}

# Loop over each batch size
for batch_size in "${batch_sizes[@]}"; do
  echo "Running with batch size: $batch_size"
  
  # Run the script num_runs times for the current batch size in parallel
  for ((i=1; i<=num_runs; i+=4)); do
    run_script $batch_size $i 0 &
    run_script $batch_size $((i+1)) 0 &
    run_script $batch_size $((i+2)) 1 &
    run_script $batch_size $((i+3)) 1 &
    
    # Wait for all 4 processes to finish before starting the next batch
    wait
  done
  
  echo "Finished batch size: $batch_size"
  echo "------------------------"
done

echo "All runs completed."
