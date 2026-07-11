#!/bin/bash
# Helper script to calculate optimal job array size

if [ ! -f "file_list.txt" ]; then
    echo "Error: file_list.txt not found!"
    echo "Run: python preprocess_ieeg_chunked.py --discover"
    exit 1
fi

N_FILES=$(wc -l < file_list.txt)
echo "Total files to process: ${N_FILES}"
echo ""

for BATCH_SIZE in 10 15 20 25; do
    N_JOBS=$(( (N_FILES + BATCH_SIZE - 1) / BATCH_SIZE ))  # Ceiling division
    TIME_PER_JOB=$(( BATCH_SIZE * 5 ))  # Assume 5 min per file
    echo "Batch size ${BATCH_SIZE}:"
    echo "  Jobs needed: ${N_JOBS}"
    echo "  Time per job: ~${TIME_PER_JOB} min"
    echo "  SLURM setting: --array=1-${N_JOBS}%${N_JOBS}"
    echo ""
done