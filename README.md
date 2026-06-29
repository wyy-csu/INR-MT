# INR-MT

**Code for:** *Implicit Neural Representation for 2D Magnetotelluric Inversion: A Physics-Constrained Deep Learning Approach*

This repository provides the Python implementation used for the synthetic 2D magnetotelluric (MT) inversion examples in the manuscript. The code implements an implicit neural representation (INR) model for the subsurface resistivity distribution and couples it with a physics-based 2D MT finite-difference forward solver in a differentiable PyTorch training workflow.

## Code Availability

- **Code name:** `INR-MT`
- **Repository:** <https://github.com/wyy-csu/INR-MT>
- **Programming language:** Python
- **License:** MIT License
- **Main purpose:** Reproduce the synthetic two-block and layered MT inversion examples reported in the manuscript.
- **Workflow included:** Yes. The released code includes the main differentiable inversion workflow used in the paper, including the INR model, 2D MT forward modeling, loss calculation for TE/TM apparent resistivity and phase data, backpropagation-based optimization, model checkpointing, plotting, and prediction/export of recovered models and responses.

## Repository Structure

```text
INR-MT/
├── 1.INR_main_blocks_TEM.py          # Main inversion script for the two-block synthetic model
├── 2.INR_main_layer_TEM.py           # Main inversion script for the layered synthetic model
├── predict.py                        # Load a trained checkpoint and export predicted models/responses
├── iMT_modules.py                    # INR neural network architecture
├── MT2D_secondary_direct_torch.py    # 2D MT finite-difference forward solver, secondary-field formulation
├── torch_spsolve.py                  # PyTorch sparse linear-system solver utility
├── utils.py                          # Mesh generation, model padding/extension, noise, logging utilities
├── plot_functions2.py                # Plotting functions for MT responses and inversion results
├── Results.zip                       # Example output files and trained-result artifacts
├── LICENSE
└── README.md
```

## Dependencies

The code was developed for Python 3 and uses the following main packages:

- `numpy`
- `scipy`
- `matplotlib`
- `torch`

A minimal installation can be created with:

```bash
conda create -n inr-mt python=3.10
conda activate inr-mt
pip install numpy scipy matplotlib torch
```

For GPU acceleration, install the PyTorch build matching your CUDA version by following the official PyTorch installation instructions: <https://pytorch.org/get-started/locally/>.

## Tested Environment

The scripts are written to run on CPU by default:

```python
device = torch.device("cpu")
```

The code can also be adapted for CUDA-enabled GPUs by changing this line in the main scripts and `predict.py`:

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Recommended tested-style environment for reproducibility:

- OS: Linux, macOS, or Windows with Python/Conda
- Python: 3.9 or newer
- PyTorch: 2.x recommended
- NumPy/SciPy/Matplotlib: recent stable versions
- Execution mode: CPU by default; CUDA GPU optional

## Hardware Requirements

The examples can run on CPU, but training is computationally intensive because each inversion iteration evaluates the MT forward response. A modern CPU is sufficient for reproducing the examples, while a CUDA-capable GPU can substantially reduce runtime if the solver configuration is adapted to GPU execution.

Suggested minimum:

- CPU: modern multi-core processor
- Memory: at least 8 GB RAM
- Disk: at least several GB free space for input data, checkpoints, figures, and exported predictions

Suggested for faster experiments:

- NVIDIA GPU with CUDA-compatible PyTorch
- 16 GB RAM or more

## Input Data

The synthetic datasets used in the manuscript are available from Zenodo:

1. **Two-block model dataset**  
   Wang, Y., Gracia, X., Attias, E., Zhang, B., Xiong, F., Liu, J., Guo, Z., 2026a. *Blocks model example dataset for paper "Implicit Neural Representation for 2D Magnetotelluric Inversion: A Physics-Constrained Deep Learning Approach"* [Data set]. Zenodo.  
   <https://doi.org/10.5281/zenodo.18480852>

2. **Layered model dataset**  
   Wang, Y., Garcia, X., Attias, E., Zhang, B., Xiong, F., Liu, J., Guo, Z., 2026b. *Layered model example dataset for paper "Implicit Neural Representation for 2D Magnetotelluric Inversion: A Physics-Constrained Deep Learning Approach"* [Data set]. Zenodo.  
   <https://doi.org/10.5281/zenodo.18497702>

After downloading and extracting the datasets, place them under a `data/` directory in the repository root. The scripts expect the following layout:

```text
INR-MT/
└── data/
    ├── blocks/
    │   └── mare2dem_data_-4-4-TEM/
    │       ├── distance.txt
    │       ├── Apres_mare2dem_TE.txt
    │       ├── Phase_mare2dem_TE.txt
    │       ├── Apres_mare2dem_TM.txt
    │       └── Phase_mare2dem_TM.txt
    └── layer/
        ├── model_true.txt
        └── mare2dem_data_-4-4-TEM/
            ├── distance.txt
            ├── Apres_mare2dem_TE.txt
            ├── Phase_mare2dem_TE.txt
            ├── Apres_mare2dem_TM.txt
            └── Phase_mare2dem_TM.txt
```

## Reproducing the Synthetic Experiments

Clone the repository:

```bash
git clone https://github.com/wyy-csu/INR-MT.git
cd INR-MT
```

Create and activate the environment:

```bash
conda create -n inr-mt python=3.10
conda activate inr-mt
pip install numpy scipy matplotlib torch
```

Download the Zenodo datasets listed above and arrange the files as described in **Input Data**.

Run the two-block synthetic inversion:

```bash
python 1.INR_main_blocks_TEM.py
```

Run the layered synthetic inversion:

```bash
python 2.INR_main_layer_TEM.py
```

By default, the scripts use:

- joint TE+TM mode: `mode = "TETM"`
- low-noise data: `noise = "low"`
- 1000 pre-training epochs
- up to 5000 main inversion epochs
- Adam optimizer with initial learning rate `1e-3`
- early stopping based on loss behavior

These settings can be changed based on the inversion tasks.

## Output Files

Each run creates a folder named according to the model and noise setting, for example:

```text
paper_model_two-blocks-noisy-low/TETM/Data_Only/
paper_model_layer-noisy-low/TETM/Data_Only/
```

Typical outputs include:

```text
Data/
├── train.log          # Training log
├── Loss.mat           # Loss history
└── lr.txt             # Learning-rate history

Nets_model/
└── checkpoint.pth     # Saved model checkpoint

Pic/
├── Rho/               # True and recovered resistivity images
├── Forward-TE/        # TE apparent resistivity/phase plots
├── Forward-TM/        # TM apparent resistivity/phase plots
├── Initial_model/     # Initial model figures
└── loss/ or loss&lr/  # Loss and learning-rate figures
```

`Results.zip` contains example result files and can be used to inspect the expected output structure.

## Prediction and Export

After training, use `predict.py` to load a saved checkpoint and export the recovered model and predicted MT responses:

```bash
python predict.py
```

The script expects a saved checkpoint named:

```text
paper_model_<model-name>/<mode>/Data_Only/Nets_model/checkpoint.pth
```

It writes exported text files to:

```text
paper_model_<model-name>/<mode>/Data_Only/prediction/
```

The exported files include:

- `predicted_rho.txt`
- `true_rho.txt`
- `predicted_Apres_TE.txt`
- `predicted_Phase_TE.txt`
- `predicted_Apres_TM.txt`
- `predicted_Phase_TM.txt`
- `true_Apres_TE.txt`
- `true_Phase_TE.txt`
- `true_Apres_TM.txt`
- `true_Phase_TM.txt`
- `frequency.txt`
- `receivers_locations.txt`
- `depths.txt`

## Notes on the Differentiable Inversion Workflow

The full inversion workflow released here consists of:

1. Generating or loading the synthetic resistivity model and MT data.
2. Padding/extending the computational mesh.
3. Representing the resistivity model with an INR neural network.
4. Evaluating the 2D MT forward response using the finite-difference solver.
5. Computing data misfit losses for TE and TM apparent resistivity and phase.
6. Updating INR parameters through PyTorch backpropagation.
7. Saving checkpoints, loss histories, figures, and predicted responses.

Thus, the repository includes the main code needed to reproduce the differentiable physics-constrained inversion examples in the manuscript.


## Citation

If you use this code, please cite the associated manuscript:

```text
Wang, Y., Gracia, X., Attias, E., Zhang, B., Xiong, F., Liu, J., Guo, Z.
Implicit Neural Representation for 2D Magnetotelluric Inversion:
A Physics-Constrained Deep Learning Approach.
Computers & Geosciences, submitted.
```
