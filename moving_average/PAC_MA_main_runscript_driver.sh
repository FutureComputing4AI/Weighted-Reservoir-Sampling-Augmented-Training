#!/bin/bash
#SBATCH --job-name=PAC-MA
#SBATCH --partition=defq
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%j.output   
#SBATCH --error=errors/%j.error

for i in {0..79}; do
   srun --exclusive --ntasks 1 --mem-per-cpu=32G conda run -n afterburner python3 PAC_MA_main.py ${i} &
done
wait