# INR-MT
Codes for paper "Implicit Neural Representation for 2D Magnetotelluric Inversion: A Physics-Constrained Deep Learning Approach"

###### Main Scripts ######
·1.INR_main_blocks_TEM.py   # Run for two blocks model inversion
·2.INR_main_layer_TEM.py   # Run for layered model inversion

###### Core Modules ######
·iMT_modules.py
Neural network architecture and related modules.

·MT2D_secondary_direct_torch.py
2D magnetotelluric (MT) finite-difference (FD) forward modeling functions (secondary field formulation).

·torch_spsolve.py
A PyTorch-based utility for efficiently solving sparse linear systems on CPU or GPU, supporting batch operations.

###### Utilities ######
·utils.py
Utility functions for mesh generation and model extension.

·plot_functions2.py
Functions for plotting MT responses and inversion results.

###### Prediction ######
·predict.py
Load a trained neural network model and predict the final inversion results.

