# From Tensor Network Quantum States to Tensorial Recurrent Neural Networks

This repository contains modifications and experiments from Schuyler Moss.

Paper link: [arXiv:2206.12363](https://arxiv.org/abs/2206.12363) | [Phys. Rev. Research 5, L032001 (2023)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.L032001)

The code requires Python >= 3.8. Use `pip install -r requirements.txt` to install the dependencies. Currently it requires a custom branch of NetKet, and we are working on upstreaming it to the master branch.

`vmc.py` trains a network. It will automatically read checkpoints when doing the hierarchical initialization. `args_parser.py` contains all the configurations.
