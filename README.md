## Learning with Uncertainty for Biological Discovery and Design

This repository contains the analysis source code used in the paper ["Leveraging uncertainty in machine learning accelerates biological discovery and design"](https://www.cell.com/cell-systems/fulltext/S2405-4712(20)30364-1) by Brian Hie, Bryan Bryson, and Bonnie Berger (Cell Systems, 2020).

### Kinase-CPI-Reanalysis

This section reflects a set of notes summarizing changes to the code 

#### Summary of changes: 

1. `bin/hybrid.py`:    
        - Added new args, `use_uq`, `predict_flat` to decide when to use uq and when to predict flat objs.    
        - Added reshape to y   

2. `bin/mlp_ensemble.py`: 
        - Added split and normalize parameters for normalizing GP's and treating GP + split as different   
        - `remove_duplicates` function added to aid in splitting   
        - Add standard scaler to all input y values for fit    
        - allow MSE loss    

3. `bin/plot_benchmark_cv.py`, `bin/plot_benchmark_lead.py`, `bin/plot_benchmark_lead_quad.py`: 
        - Added support for experiments in 2021   
        - Added new make figure tags for additional models   
        - Modified color palette   

4. `bin/process_davis20111kinase.py`:   
        - Add flag for using morgan fingeprrints   

5. `bin/sklearn_single_task.py`:    
        - New file to contain generic sklearn models using different featurizers for both proteins and compounds   

6. `bin/train_davis2011kinase.py`:   
        - Add support for an sklearn backend  in mlp_ensemble
        - Add support for morgan fingerprint   
        - Add new models mlper1norm, mlper1normsklearn, ridgesplit, ridgesplit_morgan, gpsplit, hybridsplit, mlper1split,  

7. `launcher_scripts/`:
        - Python scripts to run cv and exploit experiments  
        - Additional slurm script  

8. `bin/make_figs.ipynb` and `bin/plot_reanalysis.py`:   
        - Notebook and equivalent script to make figures using strategies from `bin/plot_benchmark_*`


#### New models added

We have added several models to this repository:
1. `ridgesplit`: This represents ridge regression as implemented in Sklearn. Rather than train a single ridge regression model, a different model is trained specific to each protein task and each substrate task. For instance, if we want to predict the value of substrate *e*  and substrate *s*. If *s* has been seen in the training set but *e* has not, then a ridge regression model specific to substrate *s* will be used to predict the Kd for *e*. For the totally unobserved quadrant where neither *s* nor *e* has been seen, all models making predictions about enzymes are used to predict *e* and vice versa for *s*. These are averaged together to make a single prediction about the pair of *s* and *e*. 
2. `mlper1norm`: An MLP model with standard normalized target values.
3. `mlper1normsklearn`: An MLP model with standard normalized target values implemented in sklearn
4. `ridgesplit_morgan`: Equivalent to `ridgesplit` but using Morgan FP's.
5. `gpsplit`: Equivalent to `ridgesplit` but using gaussian processes
6. `hybridsplit`: Equivalent to `ridgesplit` but using hybrid models without MLP normalization  
7. `hybridsplitnorm`: Equivalent to `hybridsplit` but with normalization   
8. `hybridnorm`: Equivalent to `hybrid` but with normalization   
9. `hybridsplit`: Equivalent to `ridgesplit` but using hybrid MLP and GP models
10. `mlper1split`: Equivalent to `ridgesplit` but using an MLP model with keras. 
11. `mlper1splitnormsklearn`: Equivalent to `mlper1split` but using sklearn and normalization
12. `mlper1splitsklearn`: Equivalent to `mlper1split` but using sklearn and no normalization
13. `mlper1splitnorm`: Equivalent to `mlper1split` but using normalization


#### Installation directions

To create an enviornment on cluster: 

conda create --name hie python=3.6 tensorflow-gpu=1.15
pip install -r requirements.txt


#### Running experiments

Both benchmarking and exploitation experiments were run. 

CV experiments can be run directly using `launcher_scripts/run_cv.py` or `launcher_scripts/run_exploit.py` . These files rely on a slurm launch script, `generic_slurm.sh`. To run these commands locally, the `--local` can be used (i.e., `python launcher_scripts/run_cv.py --local`). This will generate logs for 5 different seeds of the experiments. 

In cross validation, we test CPI models as originally implemented (MLP + GP, MLP), original models without CPI (MLP + GP split norm, MLP split norm), and logistic regression models without CPI (ridge split, ridge split without figures). We also additionally run experiments with (MLP + GP norm, MLP norm) to show that normalization is more helpful when training split models joint models. 

In exploitation, we test MLP + GP and MLP CPI models against ridge split regression models.

#### Plotting results

After generating results with the scripts above (see *Running experiments* section) in `target/log/` and `iterate/log/`,  results can be plotted using the notebook `notebook/make_figs.ipynb`. Alternatively, this can be accomplished 

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
