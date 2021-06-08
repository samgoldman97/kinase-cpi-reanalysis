""" Pythonic launcher script to use slurm"""

import os
import re
import shutil 
import subprocess
import argparse
import tqdm
import time

exploit_models = ["hybrid", "mlper1", "ridgesplit", "ridgesplit_morgan",
                  "hybridsplit", 'mlper1split'] 

experiment_name = "uq_cv"
log_dir = "target/log"
slurm_script = "launcher_scripts/generic_slurm.sh"

def main(): 
    os.makedirs("logs", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create sbatch string
    sbatch_args = f"--output=logs/{experiment_name}_%j.log --job-name uq_int"# --gres=gpu:1"

    # Create python string
    cuda_visible = "CUDA_VISIBLE_DEVICES=\"\"" #""
    exploit_params = ""

    for seed in [0,1,2,3,4]:
        for exploit_model in exploit_models: 
            output_name = os.path.join(log_dir,
                                       f"train_davis2011kinase_{exploit_model}_{seed}.log")
            python_string = f"{cuda_visible} python bin/train_davis2011kinase.py  {exploit_model} {exploit_params} --seed {seed} >> {output_name}"

            # Create command string to feed into sbatch
            cmd_str = f"sbatch --export=CMD=\"{python_string}\" {sbatch_args} {slurm_script}"
            subprocess.call(cmd_str, shell=True)

if __name__=="__main__": 
    main()


