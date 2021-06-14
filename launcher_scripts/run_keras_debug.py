""" Pythonic launcher script for cross validation"""

import os
import re
import shutil 
import subprocess
import argparse
import tqdm
import time

# Target models
exploit_models = ["mlper1", "mlper1norm", "mlper1split", 
                  "mlper1splitnorm", "mlper1splitsklearn", 
                  "mlper1splitnormsklearn"]
exploit_models = ["mlper1norm"]

experiment_name = "uq_cv"
log_dir = "target/log_mlp_debug"
slurm_script = "launcher_scripts/generic_slurm.sh"

def main(run_local): 
    os.makedirs("logs", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create sbatch string
    sbatch_args = f"--output=logs/{experiment_name}_%j.log --job-name uq_int" # --gres=gpu:1"

    # Create python string
    cuda_visible = "CUDA_VISIBLE_DEVICES=\"\"" #""
    exploit_params = ""

    for seed in [0,1,2]:
        for exploit_model in exploit_models: 
            output_name = os.path.join(log_dir,
                                       f"train_davis2011kinase_{exploit_model}_{seed}.log")
            python_string = f"{cuda_visible} python bin/train_davis2011kinase.py  {exploit_model} {exploit_params} --seed {seed} >> {output_name}"
            if run_local: 
                cmd_str = python_string
            else:
                # Create command string to feed into sbatch
                cmd_str = f"sbatch --export=CMD=\"{python_string}\" {sbatch_args} {slurm_script}"

            # Uncomment this out to see the command strings
            print(cmd_str)
            subprocess.call(cmd_str, shell=True)

if __name__=="__main__": 
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", default=False, 
                        help="If true, run experiments locally")
    args = parser.parse_args()
    main(run_local = args.local)


