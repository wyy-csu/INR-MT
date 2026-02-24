import torch
import numpy as np
import sys
import atexit

def kronecker(A, B, shape_A, shape_B, device):
    a_rows, a_cols = A.indices()
    a_values = A.values()
    b_rows, b_cols = B.indices()
    b_values = B.values()

    rows = (a_rows.unsqueeze(1) * shape_B[0] + b_rows.unsqueeze(0)).flatten()
    cols = (a_cols.unsqueeze(1) * shape_B[1] + b_cols.unsqueeze(0)).flatten()
    values = (a_values.unsqueeze(1) * b_values.unsqueeze(0)).flatten()

    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        values,
        (shape_A[0] * shape_B[0], shape_A[1] * shape_B[1]),
        device=device
    )


def construct_sparse_matrix(n, diag, off_diag, device):
    i = torch.arange(1, n - 1, device=device)
    rows = torch.cat([i, i, i])
    cols = torch.cat([i - 1, i, i + 1])
    values = torch.cat([
        torch.full_like(i, off_diag, dtype=torch.float64),
        torch.full_like(i, diag, dtype=torch.float64),
        torch.full_like(i, off_diag, dtype=torch.float64)
    ])
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n, n))


def SigPad(input_tensor, size_b, nza, device, mode='edge', background=2.5):
    """
    This function is used to pad the sigma model
    Args:
        - input_tensor:
                        the sigma model to be inverted (required by forward): [nz, nx],
                        for MT, it is the tensor of the initial model and requires_grad_(True),
                        for IRN and  IMT, this is not needed (num_vels can only be 1 in the case of IFWI).
        - wavelet:      segment_wavelet, input for RNN(fd), shape: [num_vels, len_tSeg] or [len_tSeg]
        - prev_state:   Contains initial wavefields for FD, in shape of [num_vels, ns, nz_pad, nx_pad]
        - curr_state:   Contains initial wavefields for FD, in shape of [num_vels, ns, nz_pad, nx_pad]
        - option:       =0 (default), averaging partitioning the input with segement_size.
                        =1, starting point for segments moving forward by step.
                            for even number segment_size: segment_size//2 step.
                            for odd number segment_size: segment_size//2+1 step.
                        =2, starting point for segments at always index=0.
                            For example, segments are:[0->segment_size, 0->2*segment_size, 0->3*segment_size, ...]
    """

    len_z_k, len_y_k = input_tensor.shape

    if mode == 'edge':

        modelbottom = torch.ones((size_b, len_y_k), device=device, dtype=torch.float32) * input_tensor[-1:, :]
        modelpadBottom = torch.cat([input_tensor, modelbottom], -2)  # padding on axis=2 (nz)

        modelleft = torch.ones((size_b + len_z_k, size_b), device=device, dtype=torch.float32) * modelpadBottom[:, :1]
        modelpadLeft = torch.cat([modelleft, modelpadBottom], -1)  # padding on axis=3 (nx)

        modelright = torch.ones((size_b + len_z_k, size_b), device=device, dtype=torch.float32) * modelpadBottom[:, -1:]
        modelpadRight = torch.cat([modelpadLeft, modelright], -1)  # padding on axis=3 (nx)

        modeltop = torch.ones((nza, len_y_k + 2 * size_b), device=device, dtype=torch.float32) * 9
        model = torch.cat([modeltop, modelpadRight], -2)  # padding on axis=2 (nz)

    elif mode == 'constant':

        modelbottom = torch.ones((size_b, len_y_k), device=device, dtype=torch.float32) * input_tensor[-1, -1]
        modelpadBottom = torch.cat([input_tensor, modelbottom], -2)  # padding on axis=2 (nz)

        modelleft = torch.ones((size_b + len_z_k, size_b), device=device, dtype=torch.float32) * modelpadBottom[-1, -1]
        modelpadLeft = torch.cat([modelleft, modelpadBottom], -1)  # padding on axis=3 (nx)

        modelright = torch.ones((size_b + len_z_k, size_b), device=device, dtype=torch.float32) * modelpadBottom[-1, -1]
        modelpadRight = torch.cat([modelpadLeft, modelright], -1)  # padding on axis=3 (nx)

        modeltop = torch.ones((nza, len_y_k + 2 * size_b), device=device, dtype=torch.float32) * 9
        model = torch.cat([modeltop, modelpadRight], -2)  # padding on axis=2 (nz)
        #
        # modelbottom = torch.ones((size_b, len_y_k), device=device, dtype=torch.float32) * np.log10(500)
        # modelpadBottom = torch.cat([input_tensor, modelbottom], -2)  # padding on axis=2 (nz)
        #
        # modelleft = torch.ones((size_b + len_z_k, size_b), device=device, dtype=torch.float32) * np.log10(500)
        # modelpadLeft = torch.cat([modelleft, modelpadBottom], -1)  # padding on axis=3 (nx)
        #
        # modelright = torch.ones((size_b + len_z_k, size_b), device=device, dtype=torch.float32) * np.log10(500)
        # modelpadRight = torch.cat([modelpadLeft, modelright], -1)  # padding on axis=3 (nx)
        #
        # modeltop = torch.ones((nza, len_y_k + 2 * size_b), device=device, dtype=torch.float32) * 9
        # model = torch.cat([modeltop, modelpadRight], -2)  # padding on axis=2 (nz)
        #


    elif mode == 'fixed':

        modelbottom = torch.ones((size_b, len_y_k), device=device, dtype=torch.float32) * background
        modelpadBottom = torch.cat([input_tensor, modelbottom], -2)  # padding on axis=2 (nz)

        modelleft = torch.ones((size_b + len_z_k, size_b), device=device, dtype=torch.float32) * background
        modelpadLeft = torch.cat([modelleft, modelpadBottom], -1)  # padding on axis=3 (nx)

        modelright = torch.ones((size_b + len_z_k, size_b), device=device, dtype=torch.float32) * background
        modelpadRight = torch.cat([modelpadLeft, modelright], -1)  # padding on axis=3 (nx)

        modeltop = torch.ones((nza, len_y_k + 2 * size_b), device=device, dtype=torch.float32) * 9
        model = torch.cat([modeltop, modelpadRight], -2)  # padding on axis=2 (nz)


    return model

def SigCor(input_model, dz_k, dx_k, nza, size_b, multiple_t, multiple_b, multiple_l, multiple_r):
    len_z_k, len_y_k = np.shape(input_model)
    z = len_z_k * dz_k


    z_air = -(np.logspace(np.log10(50e3), np.log10(50e3 + multiple_t * z), nza + 1) - 50e3)[::-1]

    zn0 = np.arange(0, len_z_k + 1) * dz_k

    z_b = np.logspace(np.log10(zn0[-1]), np.log10(multiple_b * zn0[-1]), size_b + 1)

    zn = np.concatenate((z_air[:-1], zn0, z_b[1:]))

    yn0 = np.arange(-len_y_k / 2, len_y_k / 2 + 1) * dx_k

    # expand non kernel domain
    y_l = -(np.logspace(np.log10(multiple_l * yn0[-1]), np.log10(yn0[-1]), size_b + 1))
    y_r = np.logspace(np.log10(yn0[-1]), np.log10(multiple_r * yn0[-1]), size_b + 1)
    yn = np.concatenate((y_l[:-1], yn0, y_r[1:]))

    ry = yn0

    return  zn, yn, zn0, yn0, ry


def add_noise2data(rho_clean, phs_clean, noise_level='low'):

    np.random.seed(42)
    n, ry_num = np.shape(rho_clean)

    if noise_level == 'none':
        # Minimal noise for numerical stability
        rho0 = rho_clean + np.random.randn(n, ry_num) * 0.001
        erho = np.ones((n, ry_num))  * 0.01
        phs0 = phs_clean + np.random.randn(n, ry_num) * 0.001
        ephs = np.ones((n, ry_num))  * np.deg2rad(0.5)
    elif noise_level == 'low':
        # Low noise (high quality data)
        rho0 = rho_clean + np.random.randn(n, ry_num) * 0.02
        erho = np.ones((n, ry_num))  * 0.02
        phs0 = phs_clean + np.random.randn(n, ry_num) * np.deg2rad(1.0)
        ephs = np.ones((n, ry_num))  * np.deg2rad(1.0)
    # elif noise_level == 'medium':
    #     # Medium noise (typical data)
    #     rho0 = rho_clean + np.random.randn(n, ry_num) * 0.03
    #     erho = np.ones((n, ry_num)) * 0.03
    #     phs0 = phs_clean + np.random.randn(n, ry_num) * np.deg2rad(2.0)
    #     ephs = np.ones((n, ry_num))  * np.deg2rad(2.0)
    elif noise_level == 'medium':
        # Medium noise (typical data)
        rho0 = rho_clean + np.random.randn(n, ry_num) * 0.05
        erho = np.ones((n, ry_num)) * 0.05
        phs0 = phs_clean + np.random.randn(n, ry_num) * np.deg2rad(4.0)
        ephs = np.ones((n, ry_num)) * np.deg2rad(4.0)

    else:
        raise ValueError(f"Unknown noise level: {noise_level}")

    return rho0, phs0, erho, ephs

class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w", buffering=1)  # 行缓冲
        self.stdout = sys.__stdout__

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

