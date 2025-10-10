# Task-Level Insights from Eigenvalues across Sequence Models

[GitHub](https://github.com/IntelligentControlSystems/filter-sltmpc) | [Paper](https://www.sciencedirect.com/science/article/pii/S0005109825003607) | [Issues](https://github.com/IntelligentControlSystems/filter-sltmpc/issues)

*Code accompanying the paper:*

R. Rickenbach*, J. Trisovic*, A. Didier, J. Sieber, and M. N. Zeiligner, "Task-Level Insights from Eigenvalues across Sequence Models".

The paper is available on [arXiv](https://arxiv.org/abs/2406.12573).

This paper shows that eigenvalues offer a metric for memory retention and selective forgetting in sequence models, guiding model design choices to meet task requirements.
## Folder Structure
```

analysis/        # scripts for eigenvalue analysis 
configs/         # contains training config files
dataloaders/     # contains all task loaders
jax_helpers/     # contains all auxiliary jax functions
models/          # contains all models
launch.py        # entry point for execution
train.py         # training script, called by launch.py
README.md
requirements.txt
.gitignore

```

## Installation 

The working setup is shown below (output of `python -m pip freeze --user`):
```
causal-conv1d==1.3.0.post1
chex==0.1.89
datasets==2.20.0
dill==0.3.8
einops==0.8.0
etils==1.12.0
flax==0.8.1
fsspec==2024.5.0
huggingface-hub==0.27.1
humanize==4.9.0                    
importlib_resources==6.5.2
jupyter==1.1.1
jupyter-console==6.6.3
mamba-ssm==2.1.0
ml_dtypes==0.5.1
multiprocess==0.70.16
optax==0.2.4
orbax-checkpoint==0.6.3
pyarrow==17.0.0
pyarrow-hotfix==0.6
pydantic==2.4.2
pydantic_core==2.10.1
pytorch-warmup==0.1.1
regex==2024.5.15
safetensors==0.4.3
tensorstore==0.1.72
tokenizers==0.21.0
toolz==1.0.0
torchtext==0.18.0
transformers==4.49.0
xxhash==3.4.1
-e git+https://gitlab.ethz.ch/ics-group/projects/jerome-sieber/p2025a-seqmodinsights.git@cf56a8fa6618979a48fe2500cdf3cedb5b476d84#egg=zoology&subdirectory=zoology

```
*Note:* You might need to update huggingface-hub to newest version 

## Data Preparation

Data for all tasks except for ListOps is automatically downloaded.
For ListOps, the data can be downloaded [here](https://storage.googleapis.com/long-range-arena/lra_release.gz). Make sure to specify the path to the data location, by setting `data_dir` in the corresponding yaml configuration file. For other tasks, this
field determines where the data will be downloaded.

## Running Tasks

To train the specific model configuration on a particular task, run:

```
python launch.py --config [config-file].yaml 
```

Additional arguments can be provided, for example, `--sweep True` indicates that the specified configuration file is a sweep over certain hyperparameters, and by adding `--analysis_config iclr2026/[analysis-config-file].yaml`, it is indicated that the eigenvalues should be computed and stored on W&B, with the size of the evaluation batch specified in the .yaml file.

Specifically, if we want to train linear attention on MQAR, while sweeping over multiple learning rates and random seeds, as well as analyze the eigenvalues for each configuration, we run the following:

```
python launch.py --config iclr2026/sweep/mqar/seeds/mqar-lin-attention.yaml \
                 --sweep True --analysis_config iclr2026/analysis_configs/mqar_analysis_config.yaml"
```

## Checkpoints
If the trained model should be saved, set the save parameter in the .yaml configuration file with the path where the model should be saved.

## Logging
The training is logged via W&B. To successfully log the training add the following W&B login information to the .yaml configurations:
```
wandb:
  group: "..." # name of the group displayed in W&B 
  name: "..." # name of the run displayed in W&B
  key: "..." # your API key
  entity: "..." # name of the entity within your W&B profile where the runs should be saved
  project: "..." # name of the project 
```
Note that the result of the analysis script is saved to W&B if the information above is provided, otherwise it is saved locally in the path provided in the analysis configuration. Additionally, if the 
model is not saved, the eigenvalue analysis is not performed (it is skipped).

## Citation

If you find this method helpful, please cite our work:

```bib
@article{Rickenbach2025,
title = {Task-Level Insights from Eigenvalues across Sequence Models},
year = {2025},
url = {https://www.sciencedirect.com/science/article/pii/S0005109825003607},
author = {Rahel Rickenbach and Jelena Trisovic and Alexandre Didier and Jerome Sieber and  Melanie N. Zeilinger},
}
```