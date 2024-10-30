#!/bin/bash
#SBATCH --job-name=PAC-H
#SBATCH --partition=defq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%j.output   
#SBATCH --error=errors/%j.error

# Number of maximum jobs per node
MAX_JOBS_PER_NODE=8

# Initialize an associative array to keep track of jobs per node
declare -A JOBS_PER_NODE

# Function to get the number of running jobs on a node
get_running_jobs() {
    local node=$1
    local num_jobs=$(squeue -h -w "${node}" -o "%j" | wc -l)
    echo "${num_jobs}"
}

# launch each PAC combination of {model, dataset, C_err, seed}
for i in {0..559}; do
    # Find the node with the least number of jobs
    least_busy_node=""
    least_busy_count=${MAX_JOBS_PER_NODE}
    for node in $(scontrol show hostnames); do
        if [ -z "${JOBS_PER_NODE[$node]}" ]; then
            JOBS_PER_NODE[$node]=0
        fi
        running_jobs=$(get_running_jobs "${node}")
        if [ "${running_jobs}" -lt "${least_busy_count}" ]; then
            least_busy_node="${node}"
            least_busy_count="${running_jobs}"
        fi
    done
    
    # Check if the least busy node has fewer than the maximum allowed jobs
    if [ "${least_busy_count}" -lt "${MAX_JOBS_PER_NODE}" ]; then
        srun --exclusive --ntasks 1 --mem-per-cpu=16G conda run -n afterburner python3 PAC_main.py "${i}" &
        JOBS_PER_NODE["${least_busy_node}"]=$(( JOBS_PER_NODE["${least_busy_node}"] + 1 ))
    else
        # If all nodes have reached the maximum allowed jobs, wait for a while and try again
        sleep 10
        ((i--))
    fi
done

wait