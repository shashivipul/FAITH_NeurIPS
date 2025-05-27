# FAITH

This repository contains the source codes for our paper.

# FAITH Framework - Code Overview

This repository contains the source code for the **FAITH** framework. Below is an overview of its structure and instructions for running experiments.

## Repository Structure



## Requirements

Ensure you have the following dependencies installed before running the project:

```bash
pip install numpy==1.21.6 scipy==1.7.3 torch==1.6.0 dgl==0.6.1 scikit-learn==1.0.2

```


## Running Experiments

### Transductive Setting
To run the FAITH framework in the transductive (0)/ inductive(1) setting with a **SAGE** teacher on the **Cora** dataset, use the following command:

```bash
python3 train.py --teacher SAGE --exp_setting 0 --data_mode 0

python3 train.py --teacher SAGE --exp_setting 1 --data_mode 0

```



