#!/bin/bash

# Number of jobs to submit
NUM_JOBS=5
# Job script name.
JOB_SCRIPT="login_ql40s.sh"
# Check if the script exists.
if [[ ! -f $JOB_SCRIPT ]]; then
    echo "Error: $JOB_SCRIPT not found!"
    exit 1
fi
# Initialize previous job ID.
PREV_ID=""
for ii in $(seq 1 $NUM_JOBS); do
    if [[ -z "$PREV_ID" ]]; then
        # First job: submit normally.
        JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    else
        # Subsequent jobs: depend on previous one.
        JOB_ID=$(sbatch --dependency=afterok:$PREV_ID "$JOB_SCRIPT" | awk '{print $4}')
    fi
    echo "Submitted instance $ii as job $JOB_ID"
    PREV_ID=$JOB_ID
done
