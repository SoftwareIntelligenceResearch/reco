# ReCo
The implementation of paper: ReCo: A Modular Neural Framework for Automatically Recommending Connections in Software Models

## Environment Setup
The conda environment can be created from the **ReCo.yml** with "conda create -f ReCo.yml"
## Data
The format of our labeled graph representation is as same as **dgl** official ones. For your convenience, our dataset can be download from  https://zenodo.org/records/10078103
If you want, you can create your own model datset.
## Train and Test
An example of the running configuration is _python run_exp.py --hiden_1=128 --hiden_2=128 --dataset_name=_py_parsed_bert_ --pca_dim=128 --encoder=sbert_
## Implementation
The main function of running experiments is in run_exp.py
The core functions are implemented in NLU_GNN.py

