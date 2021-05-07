#!/bin/bash
##SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
##SBATCH --mail-user=samlg@mit.edu # mail to me!
#SBATCH -n 1           # 1 core
#SBATCH -t 0-05:00:00   # 5 hours
#SBATCH -J uq_reanalysis   # sensible name for the job
#SBATCH --output=logs/slurm_generic_%j.log   # Standard output and error log
#SBATCH -p sched_mit_ccoley
#SBATCH -w node1238
#SBATCH --mem=20000 # 10 gb
##SBATCH --mem=20000 # 20 gb


##SBATCH --gres=gpu:1 #1 gpu
##SBATCH --mem=20000  # 20 gb 
##SBATCH -p {Partition Name} # Partition with GPUs

# Use this to run generic scripts:
# sbatch --export=CMD="python my_python_script --my-arg" src/scripts/slurm_scripts/generic_slurm.sh

# Import module
source /etc/profile 
source /home/samlg/.bashrc

# Activate conda
# source {path}/miniconda3/etc/profile.d/conda.sh

# Activate right python version
# conda activate {conda_env}
conda activate hie

# Evaluate the passed in command... in this case, it should be python ... 
eval $CMD


