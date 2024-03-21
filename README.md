Simplified version of S5 implementation from Smith, Warrington, Linderman 2023 that works on Princeton's Della Cluster.


## Requirements & Installation
To install and run the code on a cluster with a module system (in the virtual environment "s5-gpu-test"):

```
$ git clone https://github.com/rkp8000/s5-della.git
$ cd s5-della

$ module load anaconda3/2023.3
$ module load cudnn/cuda-11.x/8.2.0

$ export HF_DATASETS_CACHE="/scratch/gpfs/<your_netid>/cache_hf"

$ conda create -n s5-gpu-test python=3.11
$ conda activate s5-gpu-test

$ pip install -r requirements_gpu.txt

$ pip install jax "jaxlib==0.4.7+cuda11.cudnn82" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

$ pip install jax==0.4.10

$ conda install jupyter matplotlib numpy scipy scikit-learn
```

(There's probably a way to simplify this installation process, but this seems to work).

## Testing the installation

### Using Della's interactive JupyterLab interface

Go to mydella.princeton.edu --> Interactive Apps --> Jupyter to make a new session.

Make sure "Node type" is "mig" (small GPU). Set "Number of cores" to 1, "Memory allocated for the job, in GBs" as 4, "Anaconda3 version" as "custom", "Path to a custom..." as "/home/<your_netid>/.conda/envs/s5-gpu-test", "Custom environment module paths..." as "cudnn/cuda-11.x/8.2.0", "Module(s) to load..." as "anaconda3/2023.3", "How to handle conda environments..." as "Try installing ipykernel...". Then click Launch.

Navigate to s5-della and open the notebook 1_real_test.ipynb. If everything installed correctly, the first cell should output "Using gpu", and training on the classification and regression examples should be very fast. You can also test a version in which the inputs are short token sequences in 2_token_test.ipynb.

### Via slurm script

You can also test the code using the slurm system instead of the interactive Jupyter notebooks. The corresponding slurm files are 1_real_test.slurm and 2_token_test.slurm (and the eponymous Python files that they call).

Before running either slurm files make sure to replace <your_netid> with your actual net id in the line beginning with "export".

Then run

```
$ sbatch 1_real_test.slurm
```

or 

```
$ sbatch 2_token_test.slurm
```

# From the original authors:

**Simplified State Space Layers for Sequence Modeling**  
Jimmy T.H. Smith\*, Andrew Warrington\*, Scott Linderman  
International Conference on Learning Representations, 2023.  
Notable-top-5% (Oral).  
[arXiv](https://arxiv.org/abs/2208.04933)  
[OpenReview](https://openreview.net/forum?id=Ai8Hw3AXqks)

![](./docs/figures/pngs/s5-matrix-blocks.png)
<p style="text-align: center;">
Figure 1:  S5 uses a single multi-input, multi-output linear state-space model, coupled with non-linearities, to define a non-linear sequence-to-sequence transformation. Parallel scans are used for efficient offline processing. 
</p>


The S5 layer builds on the prior S4 work ([paper](https://arxiv.org/abs/2111.00396)). While it has departed considerably, this repository originally started off with much of the JAX implementation of S4 from the
Annotated S4 blog by Rush and Karamcheti (available [here](https://github.com/srush/annotated-s4)).

## Repository Structure
Directories and files that ship with GitHub repo:
```
s5/                    Source code for models, datasets, etc.
    dataloading.py          Dataloading functions.
    layers.py               Defines the S5 layer which wraps the S5 SSM with nonlinearity, norms, dropout, etc.
    seq_model.py            Defines deep sequence models that consist of stacks of S5 layers.
    ssm.py                  S5 SSM implementation.
    ssm_init.py             Helper functions for initializing the S5 SSM .
    train.py                Training loop code.
    train_helpers.py        Functions for optimization, training and evaluation steps.
    dataloaders/            Code mainly derived from S4 processing each dataset.
    utils/                  Range of utility functions.
bin/                    Shell scripts for downloading data and running example experiments.
requirements_gpu.txt    Requirements for running in GPU mode (installation can be highly system-dependent).
run_train.py            Training loop entrypoint.
```

## Citation
Please use the following when citing our work:
```
@inproceedings{
smith2023simplified,
title={Simplified State Space Layers for Sequence Modeling},
author={Jimmy T.H. Smith and Andrew Warrington and Scott Linderman},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=Ai8Hw3AXqks}
}
```

Please reach out if you have any questions.

-- The S5 authors.
