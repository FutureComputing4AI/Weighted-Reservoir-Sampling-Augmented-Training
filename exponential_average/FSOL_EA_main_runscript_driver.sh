#!/bin/bash
#SBATCH --job-name=FSOL-EA
#SBATCH --partition=defq
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%j.output   
#SBATCH --error=errors/%j.error

# fire off the FSOL-EA
for i in {0..79}; do
   srun --exclusive --ntasks 1 --mem-per-cpu=32G conda run -n afterburner python3 FSOL_EA_main.py ${i} &
done
wait