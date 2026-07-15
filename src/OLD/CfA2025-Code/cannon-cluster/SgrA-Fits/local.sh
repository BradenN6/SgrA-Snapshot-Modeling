#!/bin/bash                                                                                                                                                                                      
#SBATCH --job-name=snapshot_modeling                                                                                                                                                             
#SBATCH --output=logs/job_%A_%a.out                                                                                                                                                              
#SBATCH --error=logs/job_%A_%a.err                                                                                                                                                               
#SBATCH --array=0                                                                                                                                                                                                                                                                                                                                                               
#SBATCH --ntasks=1                                                                                                                                                                              
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G                                                                                                                                                                                                                                                                                                                                            
#SBATCH --time=00-02:00:00                                                                                                                                                                       
#SBATCH --partition=blackhole                                                                                                                                                                       

module load intel
module load intelmpi

cd $SLURM_SUBMIT_DIR

# Define the directory with .jls files                                                                                                                                                           
INPUT_DIR="data_comrade"

# List all matching .jls files into an array (sorted to maintain order)                                                                                                                          
#INPUT_FILES=($(ls ${INPUT_DIR}/hops_3599_SGRA_LO_netcal_LMTcal_normalized_10s_preprocessed_snapshot_120_noisefrac0.02_scan*.jls | sort))                                                        
INPUT_FILES=("${INPUT_DIR}/hops_3599_SGRA_LO_netcal_LMTcal_normalized_10s_preprocessed_snapshot_120_noisefrac0.02_scan157.jls")

# Select the input file for this array task                                                                                                                                                      
INPUT_FILE=${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}
julia_path="/n/holylfs05/LABS/bhi/Lab/narayan_lab/bnowicki/.julia/juliaup/julia-1.10.10+0.x64.linux.gnu/bin/julia"

scan_num=$(echo "$INPUT_FILE" | grep -oP 'scan\K[0-9]+')
padded_scan_num=$(printf "%03d" "$scan_num")
output_file="stdout-$padded_scan_num.txt"
err_file="stderr-$padded_scan_num.txt"

echo "SLURM Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using input file: $INPUT_FILE"

echo "which julia: $(which julia)"
echo "Julia Path: $julia_path"
echo "Using input file: $INPUT_FILE"
echo "Scan: $padded_scan_num"

# Run the Julia function                                                                                                                                                                         
#srun -n $SLURM_NTASKS                                                                                                                                                                           
#$julia_path -e "include(\"SgrAfits_grmhd.jl\"); SnapshotModeling.main([\"$INPUT_FILE\"])"
$julia_path -e "include(\"SgrAfits_figs.jl\"); SnapshotModeling.main([])"
