# INR-MT
Codes for paper "Implicit Neural Representation for 2D Magnetotelluric Inversion: A Physics-Constrained Deep Learning Approach", which has been submitted to Computers & Geosciences.

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
Utility functions for mesh generationm, model extension and noise.

·plot_functions2.py
Functions for plotting MT responses and inversion results.

###### Prediction ######
·predict.py
Load a trained neural network model and predict the final inversion results.

###### Data ######
The datasets utilized are available for download at the following links:

Wang, Y., Gracia, X., Attias, E., Zhang, B., Xiong, F., Liu, J., Guo, Z., 2026a. Blocks model example dataset for paper "Implicit Neural Representation for 2D Magnetotelluric Inversion: A Physics-Constrained Deep Learning Approach" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18480852.
Wang, Y., Garcia, X., Attias, E., Zhang, B., Xiong, F., Liu, J., Guo, Z., 2026b. Layered model example dataset for paper "Implicit Neural Representation for 2D Magnetotelluric Inversion: A Physics-Constrained Deep Learning Approach" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18497702.
