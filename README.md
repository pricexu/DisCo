# DisCo

Implementaion of NeurIPS 24 "Discrete-state Continuous-time Diffusion for Graph Generation"

If you have any questions about this repo, feel free to drop an email to zhexu3@illinois.edu

## Requirements
We test this implementation for the following core packages:

python 3.9

graph-tool 2.45

numpy 1.25.0

pyemd 1.0.0

pytorch 1.13.1

rdkit 2023.9.4

torchmetrics 1.2.1

pyg 2.2.0

## Commands

For sbm, planar, community, and qm9:
1. python train_spectre.py --dataset sbm
2. python train_spectre.py --dataset planar
3. python train_spectre.py --dataset community
4. python train_mol.py --dataset qm9

For Moses and Guacamol, we first train (or we can load trained models) and save the generated graphs:

1. python train_mol.py --dataset moses
2. python train_mol.py --dataset guacamol

Then we test them using the files "moses_benchmark.py" and "guacamol_benchmark.py" with the correct saved smile file names.

## Note for graph-tool
The intallation of graph-tool ("conda install -c conda-forge graph-tool") might be time-consuming: https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions

## Note for orca
orca is a C++ software for counting motifs: https://file.biolab.si/biolab/supp/orca/orca.html

We got it from DiGress which is originally from GraphRNN: https://github.com/JiaxuanYou/graph-generation

The binary file already in repo works in Ubuntu.

## Acknowledgements

DiGress: https://github.com/cvignac/DiGress

tauLDR: https://github.com/andrew-cr/tauLDR
