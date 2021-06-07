## Learning with Uncertainty for Biological Discovery and Design

This repository contains the analysis source code used in the paper ["Leveraging uncertainty in machine learning accelerates biological discovery and design"](https://www.cell.com/cell-systems/fulltext/S2405-4712(20)30364-1) by Brian Hie, Bryan Bryson, and Bonnie Berger (Cell Systems, 2020).




### SLG Notes 


#### Summary of changes: 

1. `README.md`: 
        - Added description of changes and instructions
2. `bin/cv.sh`: 
        - Added annotations to bin/cv.sh for model description
3. `bin/exploit.sh`:  
        - Added cuda visible devices to prevent gpu running and proper output
4. `bin/hybrid.py`: 
        - Added new args, `use_uq`, `predict_flat` to decide when to use uq and when to predict flat objs. 
        - Added reshape to y
5. `bin/mlp_ensemble.py`: 
        - Added split and normalize parameters for normalizing GP's and treating GP + split as different
        - `remove_duplicates` function added to aid in splitting
        - Add standard scaler to all input y values for fit
        - allow MSE loss 
6. `bin/plot_benchmark_cv.py`: 
        - Added support for experiments in 2021
        - Added new make figure tags for additional models
        - Modified color palette
7. `bin/plot_benchmark_lead.py`:        
        - Identical changes to `bin/plot_benchmark_cv.py`
8. `bin/plot_benchmark_lead_quad.py`: 
        - Identical changes to `bin/plot_benchmark_cv.py`
9. `bin/process_davis20111kinase.py`:
        - Add flag for using morgan fingeprrints
10. `bin/sklearn_single_task.py`: 
        - New file to contain generic sklearn models using different featurizers for both proteins and compounds 
11. `bin/train_davis2011kinase.py`: 
        - Add support for an sklearn backend  in mlp_ensemble
        - Add support for morgan fingerprint
        - Add new models mlper1norm, mlper1normsklearn, ridgesplit, ridgesplit_morgan, gpsplit, hybridsplit, mlper1split,
12. `launcher_scripts/generic_slurm.sh`: 
        - Generic launcher script
13. `launcher_scripts/run_cv.sh`: 
        - Script to run cv models
14. `launcher_scripts/run_exploit.sh`: 
        - Script to run iterative selection
15. `notebook/make_figs.ipynb`:
        - Notebook to make figures using strategies from `bin/plot_benchmark_*`


#### Installation directions

conda create --name hie python=3.6 tensorflow-gpu=1.15
pip install -r requirements.txt


#### Engaging 


Getting an interactive session:


`srun -N 1 -n 1 --gres=gpu:1 --time=1:00:00 --partition=sched_mit_ccoley --constraint=centos7 --mem-per-cpu=100000 --pty bash -in`

conda env: 

`conda activate hie`

Debugging this on MLMP `python bin/train_davis2011kinase.py mlper1 --seed $i >> train_davis2011kinase_mlper1.log`

Running with normalization: `CUDA_VISIBLE_DEVICES="" python bin/train_davis2011kinase.py mlper1norm`
Running with normalization + sklearn: `CUDA_VISIBLE_DEVICES="" python bin/train_davis2011kinase.py mlper1normsklearn >> train_davis2011kinase_mlper1normsklearn.log`

Running with ridge regr: `CUDA_VISIBLE_DEVICES="" python bin/train_davis2011kinase.py ridgesplit >> train_davis2011kinase_ridgesplit.log`
Running with ridge regr + morgan: `CUDA_VISIBLE_DEVICES="" python bin/train_davis2011kinase.py ridgesplit_morgan  >> train_davis2011kinase_ridgesplit_morgan.log`
Running on SLURM: `python slurm_scripts/run_cv.py`

Making plots: 
`python bin/plot_benchmark_cv.py`

*Running full experiments on engaging:* 

`launcher_scripts/run_cv.py`, launcher_scripts/exploit.py`


### Data

You can download the relevant datasets using the commands
```bash
wget http://cb.csail.mit.edu/cb/uncertainty-ml-mtb/data.tar.gz
tar xvf data.tar.gz
```
within the same directory as this repository.

### Dependencies

The major Python package requirements and their tested versions are in [requirements.txt](requirements.txt). These are the requirements for most of the experiments below, including for the GP-based models. These experiments were run with Python version 3.7.4 on Ubuntu 18.04.

For the Bayesian neural network experiments, we used the `edward` package (version 1.3.5) alongside `tensorflow` on a CPU (version 1.5.1) in a separate conda environment. These experiments used Python 3.6.10.

We also used the RDKit (version 2017.09.1) within its own separate conda environment with Python 3.6.10; download instructions can be found [here](https://www.rdkit.org/docs/Install.html).

### Compound-kinase affinity prediction experiments

#### Cross-validation experiments

The command for running the cross-validation experiments is
```bash
# Average case metrics.
bash bin/cv.sh
# Lead prioritization (all).
bash bin/exploit.sh
# Lead prioritization (separated by quadrant).
bash bin/quad.sh
```
which will launch the CV experiments for various models at different seeds implemented in `bin/train_davis2011kinase.py`.

#### Discovery experiments for validation

The command for running the prediction-based discovery experiments (to identify new candidate inhibitors in the ZINC/Cayman dataset) is
```bash
python bin/predict_davis2011kinase.py MODEL exploit N_CANDIDATES [TARGET] \
    > predict.log 2>&1
```
which will launch a prediction experiment for the `MODEL` (one of `gp`, `sparsehybrid`, or `mlper1` for the GP, MLP + GP, or MLP, respectively) to acquire `N_CANDIDATES` number of compounds. The `TARGET` argument is optional, but will restrict acquisition to a single protein target. For example, to acquire the top 100 compounds for PknB, the command is:
```bash
python bin/predict_davis2011kinase.py gp exploit 100 pknb > \
    gp_exploit100_pknb.log 2>&1
```

To incorporate a second round of prediction, you can also specify an additional text file argument at the command line, e.g.,
```bash
python bin/predict_davis2011kinase.py gp exploit 100 pknb data/prediction_results.txt \
    > gp_exploit100_pknb_round2.log 2>&1
```

#### Docking experiments

Docking experiments to validate generative designs selected by a GP, MLP + GP, and MLP can be launched by
```bash
bash bin/dock.sh
```
using the structure in `data/docking/`.

### Protein fitness experiments

Experiments testing out-of-distribution prediction of avGFP fluorescence can be launched by
```bash
bash bin/gfp.sh
```

### Gene imputation experiments

Experiments testing out-of-distribution imputation can be launched by
```bash
bash bin/dataset_norman2019_k562.sh
```

### Troubleshooting

- Changes in the sklearn API in later version may lead to [very different results](https://github.com/brianhie/uncertainty/issues/3) than reported in the paper. See [requirements.txt](requirements.txt) for a list of package version used in our experiments.
