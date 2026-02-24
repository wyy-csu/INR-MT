import torch.nn as nn
import matplotlib.pyplot as plt
from torch import sym_min
# from triton.language import dtype
import sys
import atexit
from plot_functions2 import *
from utils import *
import matplotlib.colors as colors
from MT2D_secondary_direct_torch import MT2DFD
import os
from time import perf_counter
from iMT_modules import IRN
import copy
import scipy.io as sio
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
seed_num = 10

torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed_num)

os.environ['PYTHONHASHSEED'] = str(seed_num)

loss_function = nn.MSELoss()
pretrain_option = True
mode = 'TETM'  ## 'TETM', 'TE' or 'TM'
noise = 'low'  ### 'low' or 'medium'
m_name = 'blocks'
model_name = f'two-{m_name}-noisy-{noise}-seed{seed_num}-0224'

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                              Create all folders                                                        """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

floder_pic = f'./paper_model_{model_name}/{mode}/Data_Only/Pic/'

Rho_image_filename = f'Rho'
Rho_directory = os.path.join(floder_pic, Rho_image_filename)
os.makedirs(Rho_directory, exist_ok=True)

forward_image_filename_TE = f'Forward-TE'
forward_directory_TE = os.path.join(floder_pic, forward_image_filename_TE)
os.makedirs(forward_directory_TE, exist_ok=True)

forward_image_filename_TM = f'Forward-TM'
forward_directory_TM = os.path.join(floder_pic, forward_image_filename_TM)
os.makedirs(forward_directory_TM, exist_ok=True)

initial_image_filename = f'Initial_model'
initial_directory = os.path.join(floder_pic, initial_image_filename)
os.makedirs(initial_directory, exist_ok=True)

line_1d_image_filename = f'Line_1D'
line_1d_directory = os.path.join(floder_pic, line_1d_image_filename)
os.makedirs(line_1d_directory, exist_ok=True)

floder = f'./paper_model_{model_name}/{mode}/Data_Only/'
data_filename = f'Data/'
data_path_directory = os.path.join(floder, data_filename)
os.makedirs(data_path_directory, exist_ok=True)

log_path = os.path.join(data_path_directory, "train.log")

model_filename = f'Nets_model/'
model_path_directory = os.path.join(floder, model_filename)
os.makedirs(model_path_directory, exist_ok=True)

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                        Synthetic resistivity model                                                     """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

SigModel = np.ones((40, 80)) * 500
SigModel[10:20, 10:25] = 50
SigModel[15:25, 55:70] = 50
SigModel = np.log10(SigModel)
SigModel = torch.tensor(SigModel).to(device)
dx = 50
dz = 40
nz, nx = SigModel.shape

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                        Create meshes, initial model                                                    """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

nza = 10  ## air layers
size_b = 5  ## boundary layers

# Extend factor
# mul_t = 10.0 # Top (Air)
# mul_b = 10.0 # Bottom
# mul_l = 4.0 # Left
# mul_r = 4.0 # Left

mul_t = 5.0 # Top (Air)
mul_b = 5.0 # Bottom
mul_l = 4.0 # Left
mul_r = 4.0 # Left

zn, yn, zn0, yn0, _ = SigCor(input_model=SigModel.cpu().detach().numpy(), dz_k=dz, dx_k=dx, nza=nza, size_b=size_b,
                             multiple_t=mul_t, multiple_b=mul_b, multiple_l=mul_l, multiple_r=mul_r)

sig_padded = SigPad(input_tensor=SigModel, size_b=size_b, nza=nza, device=device, mode='fixed', background=np.log10(500)).to(torch.float32)
sig_initial = torch.ones_like(SigModel).to(torch.float32).to(device) * np.log10(500)

sig_air = sig_padded[:nza, :].to(torch.float32)

Y, Z = np.meshgrid(yn0[None, :], zn0[:, None])

fig = plt.figure(figsize=(12, 12))
ax = plt.subplot(1, 1, 1)
h = ax.pcolormesh(Y/1e3, Z/1e3, SigModel.cpu().detach().numpy(),
                  vmin=0, vmax=4,
                  # norm=colors.LogNorm(vmin=-4, vmax=2),
                  edgecolors='white',
                  linewidths=0.5,
                  cmap='jet')
ax.invert_yaxis()
fig.colorbar(h)
image_path = os.path.join(Rho_directory, 'True_Rho.png')
fig.savefig(image_path)

zn = torch.tensor(zn).float().to(device)
yn = torch.tensor(yn).float().to(device)
zn0 = torch.tensor(zn0).float().to(device)
yn0 = torch.tensor(yn0).float().to(device)

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                    Load receivers locations and frequency                                              """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

ry = np.loadtxt(f'./data/{m_name}/mare2dem_data_-4-4-TEM/distance.txt')
ry = torch.tensor(ry).float().to(device)

freq = np.logspace(np.log10(1e-4), np.log10(1e4), 31)# dominant frequency of wavelet in Hz
freq = torch.tensor(freq).float().to(device)

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                          Initialize forward operator                                                   """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

forward_rnn = MT2DFD(nza, zn, yn, freq, ry)

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                          Load MT data and add noise                                                    """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

### load observation (noise-free)

Apres_truth_pure_TE = np.loadtxt(f'./data/{m_name}/mare2dem_data_-4-4-TEM/Apres_mare2dem_TE.txt', delimiter=',')
Apres_truth_pure_TE = torch.tensor(Apres_truth_pure_TE).float().to(device)
Phase_truth_pure_TE = np.loadtxt(f'./data/{m_name}/mare2dem_data_-4-4-TEM/Phase_mare2dem_TE.txt', delimiter=',')
Phase_truth_pure_TE = torch.tensor(Phase_truth_pure_TE).float().to(device)

Apres_truth_pure_TM = np.loadtxt(f'./data/{m_name}/mare2dem_data_-4-4-TEM/Apres_mare2dem_TM.txt', delimiter=',')
Apres_truth_pure_TM = torch.tensor(Apres_truth_pure_TM).float().to(device)
Phase_truth_pure_TM = np.loadtxt(f'./data/{m_name}/mare2dem_data_-4-4-TEM/Phase_mare2dem_TM.txt', delimiter=',')
Phase_truth_pure_TM = torch.tensor(Phase_truth_pure_TM).float().to(device)

### add noise

if noise == 'low':
    noise_level_apres = 0.02
    noise_level_phase = 1.0
elif noise == 'medium':
    noise_level_apres = 0.05
    noise_level_phase = 4.0

rho0_TE, phs0_TE, erho_TE, ephs_TE = add_noise2data(np.log10(Apres_truth_pure_TE.cpu().detach().numpy()), Phase_truth_pure_TE.cpu().detach().numpy() * np.pi / 180,
                                                    noise_level=noise)
rho0_TM, phs0_TM, erho_TM, ephs_TM = add_noise2data(np.log10(Apres_truth_pure_TM.cpu().detach().numpy()), Phase_truth_pure_TM.cpu().detach().numpy() * np.pi / 180,
                                                    noise_level=noise)

rho0_TE = torch.tensor(rho0_TE).float().to(device)
rho0_TM = torch.tensor(rho0_TM).float().to(device)
erho_TE = torch.tensor(erho_TE).float().to(device)
erho_TM = torch.tensor(erho_TM).float().to(device)

phs0_TE = torch.tensor(phs0_TE).float().to(device)
phs0_TM = torch.tensor(phs0_TM).float().to(device)
ephs_TE = torch.tensor(ephs_TE).float().to(device)
ephs_TM = torch.tensor(ephs_TM).float().to(device)

### plot observation
plot_Apres_phase_2D(rho0_TE, torch.zeros_like(rho0_TE), phs0_TE, torch.zeros_like(phs0_TE), forward_directory_TE, 'True_response', freq,
                    ry / 1e3,
                    figsize=[15, 6])
plot_Apres_phase_2D(rho0_TM, torch.zeros_like(rho0_TM), phs0_TM, torch.zeros_like(phs0_TM), forward_directory_TM, 'True_response', freq,
                    ry / 1e3,
                    figsize=[15, 6])


""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                              Training Parameters                                                       """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """


n_epochs_pretrain = 1000   ### epochs for pre-training stage
n_epochs = 5000   ### maximum epochs for main-training stage
log_interval = 5   ### calculate loss average and print log interval
model_save_interval = 100   ### save model interval
lr = 1e-3   ### initial learning rate

start_time = perf_counter()
total_loss = []
Apres_TE_loss = []
Apres_TM_loss = []
Phase_TE_loss = []
Phase_TM_loss = []
torch.autograd.set_detect_anomaly(True)

### MLP structure ###
neuron=[2, 256, 256, 256, 256, 256, 256, 1]
omega_0=30
prob=0.2
activation='tanh'
bias=True
dropout=False
outermost_linear=True
net = IRN(neuron=neuron, omega_0=omega_0, prob=prob, bias=bias, dropout=dropout,
          activation=activation, outermost_linear=outermost_linear).to(torch.float32).to(device)
print(net)
net.train()
params = net.parameters()
optimizer = torch.optim.Adam(lr=lr, params=params)
ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8,
                                                               patience=20, min_lr=1e-05)
torch.autograd.set_detect_anomaly(True)

## --------------Early Stopping------------------##

min_loss = float('inf')
count = 0   ### count for determine early stop (loss decay)
count1 = 0   ### count for determine early stop (loss below 1)
patience_es = 150  ### patience for early-stopping
patience_1 = 50
whether_es = False

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                                   NN inputs                                                           """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

z_middle = (zn0[:-1] + zn0[1:]) / 2
y_middle = (yn0[:-1] + yn0[1:]) / 2

z_input = (z_middle - z_middle.min()) / (z_middle.max() - z_middle.min())
y_input = (y_middle - y_middle.min()) / (y_middle.max() - y_middle.min())

Y_input, Z_input = torch.meshgrid(y_input, z_input, indexing='xy')
coords = torch.stack([Y_input, Z_input], dim=-1).to(device).to(torch.float32) # shape [nz, nx, 2]


""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                              Pre Training Stage                                                        """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
best_loss_epoch = 'final'
# model_path = os.path.join(model_path_directory, 'imt_5000.pth')
import re
files = [f for f in os.listdir(model_path_directory) if f.startswith('imt_') and f.endswith('.pth')]

# # 提取数字
# def extract_num(filename):
#     match = re.search(r'imt_(\d+)\.pth', filename)
#     return int(match.group(1)) if match else -1
#
# # 找到 num 最大的文件
# max_file = max(files, key=extract_num)
#
# print("Loading:", max_file)
#
# net.load_state_dict(torch.load(os.path.join(model_path_directory, max_file), map_location=device))
checkpoint = torch.load(os.path.join(model_path_directory, f'checkpoint.pth'), map_location=device)

net.load_state_dict(checkpoint['model_state_dict'])


# net.load_state_dict(torch.load('./imt_5000.pth', map_location=device))
net.eval()

sig_best, _ = net(coords)

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(1, 1, 1)
h = ax.pcolormesh(Y / 1e3, Z / 1e3,
                  sig_best[:, :, 0].cpu().detach().numpy(),
                  vmin=0, vmax=4,
                  edgecolors='white',
                  linewidths=0.5,
                  cmap='jet')
ax.invert_yaxis()
fig.colorbar(h)

image_path = os.path.join(Rho_directory, f'{best_loss_epoch}_Rho.png')
fig.savefig(image_path, bbox_inches='tight')
sig_best_padded = SigPad(input_tensor=sig_best[:, :, 0], size_b=size_b, nza=nza, device=device, mode='fixed', background=np.log10(500)).to(torch.float32)

Apres_best_TE, Phase_best_TE, Apres_best_TM, Phase_best_TM ,_ , _ = forward_rnn.mt2d(sig_best_padded, mode=mode)

loss_Apres_TE = loss_function(torch.log10(Apres_best_TE), rho0_TE) / (noise_level_apres ** 2)
loss_Phase_TE = loss_function(Phase_best_TE * torch.pi / 180, phs0_TE) / (np.deg2rad(noise_level_phase) ** 2)

loss_Apres_TM = loss_function(torch.log10(Apres_best_TM), rho0_TM) / (noise_level_apres ** 2)
loss_Phase_TM = loss_function(Phase_best_TM * torch.pi / 180, phs0_TM) / (np.deg2rad(noise_level_phase) ** 2)

loss = torch.sqrt(loss_Apres_TE + loss_Phase_TE + loss_Apres_TM + loss_Phase_TM) / 2

print(loss.item())

plot_Apres_phase_2D(rho0_TE, torch.log10(Apres_best_TE), phs0_TE * 180 / torch.pi, Phase_best_TE, forward_directory_TE, best_loss_epoch, freq, ry/1e3,
                        figsize=[15, 6])

plot_Apres_phase_2D(rho0_TM, torch.log10(Apres_best_TM), phs0_TM * 180 / torch.pi, Phase_best_TM, forward_directory_TM, best_loss_epoch, freq, ry/1e3,
                        figsize=[15, 6])

plot_Apres_phase_1D_TEM(rho0_TE.cpu().detach().numpy(), np.log10(Apres_best_TE.cpu().detach().numpy()), phs0_TE.cpu().detach().numpy() * 180 / np.pi, Phase_best_TE.cpu().detach().numpy(),
                        rho0_TM.cpu().detach().numpy(), np.log10(Apres_best_TM.cpu().detach().numpy()), phs0_TM.cpu().detach().numpy() * 180 / np.pi, Phase_best_TM.cpu().detach().numpy(),
                        erho_TE.cpu().detach().numpy(), erho_TM.cpu().detach().numpy(), ephs_TE.cpu().detach().numpy() * 180 / np.pi, ephs_TM.cpu().detach().numpy() * 180 / np.pi,
                        path=forward_directory_TM, epoch=best_loss_epoch,
                        fre=freq.cpu().detach().numpy(), file_name=None, x=10)


pred_data_save_path = f'./paper_model_{model_name}/{mode}/Data_Only/prediction'

os.makedirs(pred_data_save_path, exist_ok=True)


fig = plt.figure(figsize=(12, 8))


np.savetxt(os.path.join(pred_data_save_path, 'frequency.txt'), freq.cpu().detach().numpy())
np.savetxt(os.path.join(pred_data_save_path, 'receivers_locations.txt'), ry.cpu().detach().numpy())
np.savetxt(os.path.join(pred_data_save_path, 'depths.txt'), zn0.cpu().detach().numpy())

np.savetxt(os.path.join(pred_data_save_path, 'predicted_rho.txt'), sig_best[:, :, 0].cpu().detach().numpy())
np.savetxt(os.path.join(pred_data_save_path, 'true_rho.txt'), SigModel.cpu().detach().numpy())

np.savetxt(os.path.join(pred_data_save_path, 'predicted_Apres_TE.txt'), torch.log10(Apres_best_TE).cpu().detach().numpy())
np.savetxt(os.path.join(pred_data_save_path, 'true_Apres_TE.txt'), rho0_TE.cpu().detach().numpy())
np.savetxt(os.path.join(pred_data_save_path, 'predicted_Phase_TE.txt'), Phase_best_TE.cpu().detach().numpy())
np.savetxt(os.path.join(pred_data_save_path, 'true_Phase_TE.txt'), (phs0_TE * 180 / torch.pi).cpu().detach().numpy())

np.savetxt(os.path.join(pred_data_save_path, 'predicted_Apres_TM.txt'), torch.log10(Apres_best_TM).cpu().detach().numpy())
np.savetxt(os.path.join(pred_data_save_path, 'true_Apres_TM.txt'), rho0_TM.cpu().detach().numpy())
np.savetxt(os.path.join(pred_data_save_path, 'predicted_Phase_TM.txt'), Phase_best_TM.cpu().detach().numpy())
np.savetxt(os.path.join(pred_data_save_path, 'true_Phase_TM.txt'), (phs0_TM * 180 / torch.pi).cpu().detach().numpy())
