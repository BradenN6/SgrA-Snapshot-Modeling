#!/bin/bash                                                                                                                                                                                      
#SBATCH --job-name=stacked_analysis                                                                                                                                                             
#SBATCH --output=logs/job_%A.out                                                                                                                                                              
#SBATCH --error=logs/job_%A.err                                                                                                                                                               
#SBATCH --ntasks=1                                                                                                                                                                              
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G                                                                                                                                                                                                                                                                                                                                            
#SBATCH --time=00-01:00:00                                                                                                                                                                       
#SBATCH --partition=blackhole                                                                                                                                                                       

module load intel
module load intelmpi

cd $SLURM_SUBMIT_DIR

julia_path="/n/holylfs05/LABS/bhi/Lab/narayan_lab/bnowicki/.julia/juliaup/julia-1.10.10+0.x64.linux.gnu/bin/julia"

echo "SLURM Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Task ID: $SLURM_ARRAY_TASK_ID"

echo "which julia: $(which julia)"
echo "Julia Path: $julia_path"
echo "Scan: $padded_scan_num"

# Run the Julia function                                                                                                                                                                         
#srun -n $SLURM_NTASKS                                                                                                                                                                           
$julia_path -e "include(\"analysis.jl\")"