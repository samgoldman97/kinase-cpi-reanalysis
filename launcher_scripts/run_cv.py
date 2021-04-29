""" Pythonic launcher script to use slurm"""

import os
import re
import shutil 
import subprocess
import argparse
import tqdm
import time


def main(): 

    os.makedirs("logs", exist_ok=True)
    os.makedirs("target/log", exist_ok=True)

    # Define slurm script
    slurm_script = "launcher_scripts/generic_slurm.sh"
    experiment_name = "hie_uq_test"

    # Create sbatch string
    sbatch_args = f"--output=logs/{experiment_name}_%j.log --job-name uq_int"# --gres=gpu:1"

    # Create python string
    seed = 1
    cuda_visible = "CUDA_VISIBLE_DEVICES=\"\"" #""

    python_string = f"{cuda_visible} python bin/train_davis2011kinase.py mlper1 --seed {seed} >> target/log/train_davis2011kinase_mlper1.log"

    # Create command string to feed into sbatch
    cmd_str = f"sbatch --export=CMD=\"{python_string}\" {sbatch_args} {slurm_script}"
    subprocess.call(cmd_str, shell=True)

if __name__=="__main__": 
    main()


