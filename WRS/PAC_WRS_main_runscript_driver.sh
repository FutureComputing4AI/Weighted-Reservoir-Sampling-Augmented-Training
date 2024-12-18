#!/bin/bash
#SBATCH --job-name=PAC-WRS
#SBATCH --partition=defq
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%j.output   
#SBATCH --error=errors/%j.error

# fire off the PAC trials
for i in {0..639}; do
   srun --exclusive --ntasks 1 --mem-per-cpu=32G conda run -n afterburner python3 PAC_WRS_main.py ${i} &
done
wait