#!/bin/bash
#SBATCH --job-name=FSOL-TPK
#SBATCH --partition=defq
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%j.output   
#SBATCH --error=errors/%j.error

# fire off the FSOL top-K trials!
for i in {0..319}; do
   srun --exclusive --ntasks 1 --mem-per-cpu=32G conda run -n afterburner python3 FSOL_TOPK_main.py ${i} &
done
wait