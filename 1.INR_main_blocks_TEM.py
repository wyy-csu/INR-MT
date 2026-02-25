import torch.nn as nn
import matplotlib.pyplot as plt
from torch import sym_min
# from triton.language import dtype

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
import sys
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
seed_num = 100

torch.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed_num)

print("Current working directory:", os.getcwd())

os.environ['PYTHONHASHSEED'] = str(seed_num)

loss_function = nn.MSELoss()
pretrain_option = True
mode = 'TETM'  ## 'TETM', 'TE' or 'TM'
noise = 'low'  ### 'low' or 'medium'
m_name = 'blocks'
model_name = f'two-{m_name}-noisy-{noise}'

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

loss_image_filename = f'loss'
loss_directory = os.path.join(floder_pic, loss_lr_image_filename)
os.makedirs(loss_lr_directory, exist_ok=True)

floder = f'./paper_model_{model_name}/{mode}/Data_Only/'
data_filename = f'Data/'
data_path_directory = os.path.join(floder, data_filename)
os.makedirs(data_path_directory, exist_ok=True)

log_path = os.path.join(data_path_directory, "train.log")

model_filename = f'Nets_model/'
model_path_directory = os.path.join(floder, model_filename)
os.makedirs(model_path_directory, exist_ok=True)
print(Rho_directory)

loss_path = os.path.join(data_path_directory, 'Loss.mat')
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

mul_t = 10.0 # Top (Air)
mul_b = 10.0 # Bottom
mul_l = 4.0 # Left
mul_r = 4.0 # Right


extend_mode = 'constant' # 'edge', 'constant', 'fixed'
extend_background = np.log10(500) # only applies when the 'fixed' mode is selected

zn, yn, zn0, yn0, _ = SigCor(input_model=SigModel.cpu().detach().numpy(), dz_k=dz, dx_k=dx, nza=nza, size_b=size_b,
                             multiple_t=mul_t, multiple_b=mul_b, multiple_l=mul_l, multiple_r=mul_r)

sig_padded = SigPad(input_tensor=SigModel, size_b=size_b, nza=nza, device=device, mode=extend_mode, background=extend_background).to(torch.float32)
sig_initial = torch.ones_like(SigModel).to(torch.float32).to(device) * np.log10(500)

sig_air = sig_padded[:nza, :].to(torch.float32)

Y, Z = np.meshgrid(yn0[None, :], zn0[:, None])

fig = plt.figure(figsize=(20, 10))
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
loss_threshold = 2

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
"""                                            Save parameters                                                             """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

sig_model_path = os.path.join(data_path_directory, 'SigModelPara.mat')
sio.savemat(sig_model_path, mdict={'zn': zn.cpu().numpy(),
                              'zn0': zn0.cpu().numpy(),
                              'yn': yn.cpu().numpy(),
                              'yn0': yn0.cpu().numpy(),
                              'ry': ry.cpu().numpy(),
                              'extend_f_t': mul_t,
                              'extend_f_b': mul_b,
                              'extend_f_l': mul_l,
                              'extend_f_r': mul_r,
                              'air_layer': nza,
                              'bound_layer': size_b,
                              'extend_mode': extend_mode,
                              'extend_value': extend_background,
                              'dx': dx,
                              'dz': dz
                              })

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                              Pre Training Stage                                                        """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

log_path = os.path.join(data_path_directory, "train.log")
tee = Tee(log_path)

sys.stdout = tee
sys.stderr = tee

atexit.register(tee.close)

for epoch in range(n_epochs_pretrain):
    optimizer.zero_grad()

    sig_input, _ = net(coords)

    loss = torch.sqrt(loss_function(sig_input[:, :, 0], sig_initial))
    loss.backward()

    optimizer.step()

    ax = plt.subplot(1, 1, 1)
    h = ax.pcolormesh(Y / 1e3, Z / 1e3,
                      sig_initial.cpu().detach().numpy(),
                      vmin=0, vmax=4,
                      edgecolors='white',
                      linewidths=0.5,
                      cmap='jet')
    ax.invert_yaxis()
    fig.colorbar(h)

    image_path = os.path.join(initial_directory, f'initial_Rho.png')
    fig.savefig(image_path, bbox_inches='tight')


    if (epoch + 1) % log_interval == 0:
        print(f'epoch {epoch + 1} | total_loss: {loss.item():.4f}')

        fig = plt.figure(figsize=(20, 10))


    if (epoch + 1) % model_save_interval == 0:
        fig = plt.figure(figsize=(20, 10))
        ax = plt.subplot(1, 1, 1)
        h = ax.pcolormesh(Y / 1e3, Z / 1e3,
                          sig_input[:, :, 0].cpu().detach().numpy(),
                          vmin=0, vmax=4,
                          edgecolors='white',
                          linewidths=0.5,
                          cmap='jet')
        ax.invert_yaxis()
        fig.colorbar(h)

        image_path = os.path.join(initial_directory, f'initial_Rho_learned.png')
        fig.savefig(image_path, bbox_inches='tight')

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                             Main Training Stage                                                        """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

Loss_record = 0
Loss_Average = [0, 0, 0, 0, 0]
lr_record = []
for epoch in range(n_epochs):
    optimizer.zero_grad()

    sig_input, _ = net(coords)
    sig_np = sig_input[:, :, 0].cpu().detach().numpy()
    sig_input = SigPad(input_tensor=sig_input[:, :, 0], size_b=size_b, nza=nza, device=device, mode=extend_mode, background=extend_background).to(torch.float32)

    Apres_pred_TE, Phase_pred_TE, Apres_pred_TM, Phase_pred_TM, _, _ = forward_rnn.mt2d(sig_input, mode=mode)

    loss_Apres_TE = loss_function(torch.log10(Apres_pred_TE), rho0_TE) / (noise_level_apres ** 2)
    loss_Phase_TE = loss_function(Phase_pred_TE * torch.pi / 180, phs0_TE) / (np.deg2rad(noise_level_phase) ** 2)

    loss_Apres_TM = loss_function(torch.log10(Apres_pred_TM), rho0_TM) / (noise_level_apres ** 2)
    loss_Phase_TM = loss_function(Phase_pred_TM * torch.pi / 180, phs0_TM) / (np.deg2rad(noise_level_phase) ** 2)

    loss = torch.sqrt(loss_Apres_TE + loss_Phase_TE + loss_Apres_TM + loss_Phase_TM)

    loss.backward()
    optimizer.step()

    #### Determining early stopping conditions by loss decay ####
    current_loss = loss.item()

    if current_loss < min_loss:
        min_loss = current_loss
        count = 0
    else:
        count += 1

    #### Determining early stopping conditions by loss decay ####
    if current_loss <= loss_threshold:
        count1 += 1
    else:
        count1 = 0

    if count == patience_es or count1 == patience_1:
        whether_es = True

    #### End ####

    Loss_record += current_loss
    Loss_Average[0] += current_loss
    Loss_Average[1] += loss_Apres_TE.item()
    Loss_Average[2] += loss_Phase_TE.item()
    Loss_Average[3] += loss_Apres_TM.item()
    Loss_Average[4] += loss_Phase_TM.item()
    lr_record.append(optimizer.param_groups[0]['lr'])

    if (epoch+1) % log_interval == 0:
        current_time = perf_counter()
        elapsed_time = current_time - start_time
        ReduceLROnPlateau.step(Loss_record / log_interval)
        learning_rate = optimizer.param_groups[0]['lr']
        result = [xx / log_interval for xx in Loss_Average]
        print(f'epoch {epoch + 1} | total_loss: {result[0]:.4f}, '
              f'Apres_loss_TE: {result[1]:.4f}, Phase_loss_TE: {result[2]:.4f}, '
              f'Apres_loss_TM: {result[3]:.4f}, Phase_loss_TM: {result[4]:.4f}, '
              f'learning_rate: {learning_rate:.5f}, es_loss: {count}/{patience_es}, es_threshold: {count1}/{patience_1}, elapsed_time: {elapsed_time:.2f}')

        Loss_record = 0
        Loss_Average = [0, 0, 0, 0, 0]


    if (epoch + 1) % model_save_interval == 0  or whether_es:
        best_loss_epoch = epoch + 1
        best_loss_model = copy.deepcopy(net.state_dict())
        # torch.save(best_loss_model, os.path.join(model_path_directory, f'imt_{best_loss_epoch}.pth'))
        torch.save({
            'epoch': best_loss_epoch,
            'model_state_dict': best_loss_model,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': ReduceLROnPlateau.state_dict(),
        }, os.path.join(model_path_directory, f'checkpoint.pth'))

        model_copy = copy.deepcopy(net).to(device)

        checkpoint = torch.load(os.path.join(model_path_directory, f'checkpoint.pth'), map_location=device)

        model_copy.load_state_dict(checkpoint['model_state_dict'])

        # model_path = os.path.join(model_path_directory, f'imt_{best_loss_epoch}.pth')
        # model_copy.load_state_dict(torch.load(model_path, map_location=device))
        model_copy.eval()

        with torch.no_grad():
            sig_best, _ = model_copy(coords)

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
            sig_best_padded = SigPad(input_tensor=sig_best[:, :, 0], size_b=size_b, nza=nza, device=device, mode=extend_mode, background=extend_background).to(torch.float32)

            Apres_best_TE, Phase_best_TE, Apres_best_TM, Phase_best_TM ,_ , _ = forward_rnn.mt2d(sig_best_padded, mode=mode)

            plot_Apres_phase_2D(rho0_TE, torch.log10(Apres_best_TE), phs0_TE * 180 / torch.pi, Phase_best_TE, forward_directory_TE, best_loss_epoch, freq, ry/1e3,
                                    figsize=[15, 6])

            plot_Apres_phase_2D(rho0_TM, torch.log10(Apres_best_TM), phs0_TM * 180 / torch.pi, Phase_best_TM, forward_directory_TM, best_loss_epoch, freq, ry/1e3,
                                    figsize=[15, 6])

            plot_Apres_phase_1D_TEM(rho0_TE.cpu().detach().numpy(), np.log10(Apres_best_TE.cpu().detach().numpy()), phs0_TE.cpu().detach().numpy() * 180 / np.pi, Phase_best_TE.cpu().detach().numpy(),
                                    rho0_TM.cpu().detach().numpy(), np.log10(Apres_best_TM.cpu().detach().numpy()), phs0_TM.cpu().detach().numpy() * 180 / np.pi, Phase_best_TM.cpu().detach().numpy(),
                                    erho_TE.cpu().detach().numpy(), erho_TM.cpu().detach().numpy(), ephs_TE.cpu().detach().numpy() * 180 / np.pi, ephs_TM.cpu().detach().numpy() * 180 / np.pi,
                                    path=forward_directory_TM, epoch=epoch,
                                    fre=freq.cpu().detach().numpy(), file_name=None, x=10)

        del model_copy
        torch.cuda.empty_cache()

    if whether_es:
        print('Early stopping at epoch:', epoch+1)
        break

    total_loss.append(loss.item())
    Apres_TE_loss.append(loss_Apres_TE.item())
    Phase_TE_loss.append(loss_Phase_TE.item())
    Apres_TM_loss.append(loss_Apres_TM.item())
    Phase_TM_loss.append(loss_Phase_TM.item())


""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
"""                                          Plot Loss curves and save lr                                                  """
""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

total_loss_values = np.array(total_loss).flatten()
Apres_TE_loss_values = np.array(Apres_TE_loss).flatten()
Phase_TE_loss_values = np.array(Phase_TE_loss).flatten()
Apres_TM_loss_values = np.array(Apres_TM_loss).flatten()
Phase_TM_loss_values = np.array(Phase_TM_loss).flatten()

plt.figure(figsize=(10, 5))
plt.plot(total_loss_values, label='Total Loss', color='red')
# plt.plot(Apres_TE_loss_values, label='Apres Loss-TE', color='blue')
# plt.plot(Phase_TE_loss_values, label='Phase Loss-TE', color='green')
# plt.plot(Apres_TM_loss_values, label='Apres Loss-TM', color='orange')
# plt.plot(Phase_TM_loss_values, label='Phase Loss-TM', color='purple')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid()
image_path_Loss = os.path.join(loss_lr_directory, f'Loss.png')
plt.savefig(image_path_Loss)
plt.close()


sio.savemat(loss_path, mdict={'Total_loss': total_loss_values,
                              'Apres_TE_loss': Apres_TE_loss_values,
                              'Phase_TE_loss': Phase_TE_loss_values,
                              'Apres_TM_loss': Apres_TM_loss_values,
                              'Phase_TM_loss': Phase_TM_loss_values
                              })

lr_path = os.path.join(data_path_directory, 'lr.txt')
np.savetxt(lr_path, lr_record)


