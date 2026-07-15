module load intel
module load intelmpi

# Define the directory with .jls files
INPUT_DIR="grmhd-truth/data_grmhdtest3599"

# List all matching .jls files into an array (sorted to maintain order)
#INPUT_FILES=($(ls ${INPUT_DIR}/hops_3599_SGRA_LO_netcal_LMTcal_normalized_10s_preprocessed_snapshot_120_noisefrac0.02_scan*.jls | sort))
INPUT_FILES=("${INPUT_DIR}/hops_3599_MAD_a-0.5_Rh40_i50_LO_preprocessed_snapshot_120_noisefrac0.02_scan102.jls")
	     
# Select the input file for this array task
#INPUT_FILE=${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}
julia_path="/n/holylfs05/LABS/bhi/Lab/narayan_lab/bnowicki/.julia/juliaup/julia-1.10.10+0.x64.linux.gnu/bin/julia"

echo "which julia: $(which julia)"
echo "Julia Path: $julia_path"

# Run the Julia function
counter=0

for INPUT_FILE in "${INPUT_FILES[@]}"; do
  scan_num=$(echo "$INPUT_FILE" | grep -oP 'scan\K[0-9]+')
  padded_scan_num=$(printf "%03d" "$scan_num")
  output_file="stdout-$padded_scan_num.txt"
  err_file="stderr-$padded_scan_num.txt"
  echo "Index: $counter"
  echo "Using input file: $INPUT_FILE"
  $julia_path --threads 4 -e "include(\"SgrAfits_grmhd.jl\"); SnapshotModeling.main([\"$INPUT_FILE\"])" > >(tee $output_file) 2> >(tee $err_file >&2)
  ((counter++))
done
