#!/bin/bash                                                                                                                                                                                      
#SBATCH --job-name=stacking                                                                                                                                                           
#SBATCH --output=logs/job_%A.out                                                                                                                                                              
#SBATCH --error=logs/job_%A.err                                                                                                                                                               
#SBATCH --ntasks=1                                                                                                                                                                              
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3G                                                                                                                                                                                                                                                                                                                                            
#SBATCH --time=10-00:00:00                                                                                                                                                                       
#SBATCH --partition=blackhole                                                                                                                                                                       

module load intel
module load intelmpi

cd $SLURM_SUBMIT_DIR

julia_path="/n/holylfs05/LABS/bhi/Lab/narayan_lab/bnowicki/.julia/juliaup/julia-1.10.10+0.x64.linux.gnu/bin/julia"

#chain_path="/n/holylfs05/LABS/bhi/Lab/narayan_lab/bnowicki/Stacking/PolarizedStacker/krang_run/stacker_chain_SgrA.h5"
chain_path="/n/holylfs05/LABS/bhi/Lab/narayan_lab/bnowicki/Stacking/PolarizedStacker/run_grmhdtest3599/stacker_chain_grmhdtest3599.h5"
#prior_path="/n/holylfs05/LABS/bhi/Lab/narayan_lab/bnowicki/Stacking/PolarizedStacker/krang_run/priorKrang.txt"
prior_path="/n/holylfs05/LABS/bhi/Lab/narayan_lab/bnowicki/Stacking/PolarizedStacker/run_grmhdtest3599/priorKrang.txt"

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "which julia: $(which julia)"
echo "Julia Path: $julia_path"

# Run the Julia function                                                                                                                                                                         
#srun -n $SLURM_NTASKS
# TODO once it ends, restart and alter the pt checkpoint in process_pigeons                                                                                                                                                                           
$julia_path -t 24 main_krang_pigeons.jl --nrounds 14 --nchains 24 -o "krang_grmhdtest3599_stack_no-restrict_nrounds-14_ALL_s161" $chain_path $prior_path
