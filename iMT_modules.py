'''
Modules for Implicit Full Waveform Inversion (IFWI):
    where, 
    - IRN denotes a deep neural network (MLP) for subsurface models,
    - IFWI2D denotes a framework for FWI and IFWI.

Updates:
    - 

@Author: Jian Sun
-- Ocean University of China
-- Jan, 2022
'''

from __future__ import print_function
import os
import copy
import torch
import torch.utils.data
import numpy as np
from torchvision import transforms
from utils import *
import torch_spsolve
import torch.nn as nn

# from generator import gen_Segment2d
import matplotlib.pyplot as plt
from time import perf_counter
from plot_functions2 import plot_Apres_phase_record, plot_Apres_phase_1D
from MT2D_secondary_direct_torch import MT2DFD
import matplotlib.colors as colors
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return [repackage_hidden(v) for v in h]


#############################################################################################
# ##                            Implicit Neural Network Model                             ###
#############################################################################################
class IRN(torch.nn.Module):
    '''
    An Implicit Representation Network.
    
    neuron:             a list to define number of layers (length of the list) and number of neurons in each layer;
    activation:         'relu', 'tanh' or 'sine',
                        if 'sine' is chosen, then special initilizations are implemented (see SIREN paper).
    outermost_linear:   if True, then not activation function is applied on the last layer.
    '''
    def __init__(self, 
                 neuron=[2, 256, 256, 256, 256, 1], 
                 omega_0=30, 
                 prob=0.2,
                 bias=True, 
                 dropout=False,
                 outermost_linear=False,
                 activation='sine'):
        super(IRN, self).__init__()
        self.omega_0 = omega_0
        self.neuron = neuron
        self.d_flag = dropout
        self.outermost_linear = outermost_linear
        
        self.linear = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()

        # self.backgroung_value = nn.Parameter(torch.tensor([2.5]))


        for idx in range(len(neuron) - 1):
            self.linear.append(torch.nn.Linear(neuron[idx], neuron[idx + 1], bias=bias))
            if self.d_flag:
                self.dropout.append(torch.nn.Dropout(prob, inplace=True))
            
        if activation == 'relu':
            self.omega_0 = 1
            self.activation = torch.nn.ReLU()
        elif activation == 'tanh':
            self.omega_0 = 1
            self.activation = torch.nn.Tanh()
        else:
            self.activation = torch.sin
            self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            self.linear[0].weight.uniform_(-1 / self.neuron[0], 1 / self.neuron[0])
            for ix in range(1, len(self.linear)):
                self.linear[ix].weight.uniform_(-np.sqrt(6 / self.neuron[ix]) / self.omega_0, 
                                                 np.sqrt(6 / self.neuron[ix]) / self.omega_0)
            
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        feature = self.activation(self.omega_0 * self.linear[0](coords))
        for ilayer, layer in enumerate(self.linear[1:]):
            if self.d_flag:
                feature = self.dropout[ilayer](feature)
            feature = layer(feature)
            if not self.outermost_linear or ilayer + 2 < len(self.linear):
                feature = self.activation(self.omega_0 * feature)
        return feature, coords


#############################################################################################
# ##                       Implicit Full Waveform Inversion  Model                        ###
#############################################################################################
class IMT2D():
    """
    Implicit Full Waveform Inversion (for acoustic only) with an implicit repreesentation neural network.
    This module allows three types of network:
        - FWI:          A version full waveform inversion using RNN cell,
                        where each cell acts as a finite-difference operator, 
                        which takes the velocity (could be variable) as input and output shot gather.
        - IRN:          An implicit MLP neural network for image/velocity representation,
                        which takes coordinates as inputs, and output a (normalized) velocity model.
        - IFWI:         A two-step implicit full waveform inversion, including IRN + FWI(RNN),
                        coords -> [NN] -> {vel} -> [RNN] -> shot_pred.
    """
    def __init__(self, 
                 mean=2.6718, 
                 std=0.9738,
                 neuron=[2, 256, 256, 256, 256, 1], 
                 omega_0=30, 
                 prob=0.2,
                 activation='sine', 
                 bias=True, 
                 dropout=False,
                 outermost_linear=False,
                 zn=None,
                 xn=None,
                 nza=None,
                 ry=None,
                 # dz=None,
                 # dx=None,
                 frequency=[1],
                 rho_max=6000,
                 segy_truth_max=0,
                 segy_truth_min=0,
                 segment_size=100,
                 regularization="TV",
                 ref_model='Half_space',
                 dtype=torch.float32,
                 pretrained=None,
                 device='cpu',
                 netOpt='IMT',
                 mode='TE'):
        """
        Args:
            start_channel(int):     the number of channels for the first Conv layer in FCN.
            ns(int):                the number of shot gathers (in_channel for fcn2D class).
            nz(int):                the number of samples in depth for velocity model
            zs(tensor):             in shape of [num_vels, ns].
            xs(tensor):             in shape of [num_vels, ns].
            zr(tensor):             in shape of [num_vels, ns, nx], or an integer
            xr(tensor):             in shape of [num_vels, ns, nx]. or a list/tensor in length of nx.
            dz & dt (float):        represent grid and time interval respectively.
            npad(int):              is the number of grid padding to the velocity for absorbing boundaries.
            order(int);             the order of finite difference is used for wave propagation modeling.
            freeSurface(boolean):   True (default) or False for free surface option in forward modeling (RNN).
            netOpt(str):            one of the str in list ['RNN', 'FCN', 'PGNN'].
            vpadding:               the intention of vpadding is to fix the velocity in the surface layer (such as water layer), 
                                        to stablize the forward propagation.
                                        if it's a tensor, size: [num_grids_1st_layer, nx], 
                                            the first layer padding to velocity models.
                                        elif it's a tuple, (v0, num_grids_1st_layer)
                                        else: None(default), no padding.
        """
        super(IMT2D, self).__init__()
        assert netOpt in ('MT', 'IRN', 'IMT'), "Error: Only ['MT', 'IRN', 'IMT'] are supported."
        self.min_loss = float('inf')
        self.std = std
        self.mean = mean
        self.rho_max = rho_max  # self.vmax is in km/s
        self.dtype = dtype
        self.rho = torch.empty(1)  # this is for the predict and load_state Funcs in FWI mode.
        self.device = device
        self.netOpt = netOpt
        self.reg_op = regularization
        self.ref_op = ref_model
        self.segment_size = segment_size
        self.zn = zn
        self.xn = xn
        self.nza = nza
        self.ry = ry
        # self.dz = dz
        # self.dx = dx
        self.frequency = frequency
        self.segy_truth_max = segy_truth_max
        self.segy_truth_min = segy_truth_min
        self.loss_function = nn.MSELoss()
        self.mode = mode

        x = xn.cpu().detach().numpy()
        z = zn.cpu().detach().numpy()
        X, Z = np.meshgrid(x[None, :], z[:, None])
        X = torch.from_numpy(X).type(dtype=dtype).to(device)
        Z = torch.from_numpy(Z).type(dtype=dtype).to(device)
        self.coords = torch.stack([X, Z], dim=-1).to(device) # shape [nz, nx, 2]
        # self.transform = transforms.Normalize(mean=(mean), std=(std))

        if netOpt in ['MT', 'IMT', 'IRN']:
            rnn = MT2DFD(self.nza, self.zn, self.xn, self.frequency, self.ry)
            # self.rnn = rnn.type(dtype).to(device)
            self.rnn = rnn
        if netOpt in ['IRN', 'IMT']:
            vel_net = IRN(neuron=neuron, omega_0=omega_0, prob=prob, bias=bias, dropout=dropout, 
                          activation=activation, outermost_linear=outermost_linear)
            print(vel_net)
            self.vel_net = vel_net.type(dtype).to(device)
            if pretrained is not None:
                self.vel_net.load_state_dict(torch.load(pretrained, map_location=self.device))
        # if torch.cuda.device_count() > 1:
        #     print(torch.cuda.device_count(), " GPUs will be used!")
        #     # rnn  = torch.nn.DataParallel(rnn)
        #     vel_net = torch.nn.DataParallel(vel_net)
            
    # def predict(self, coords=None, resume_file_name=None, best=False, uncertainty=False, NoSim=100):
    #     """
    #     This function only works for IRN or IFWI, performs the predict process of the IRN(MLP).
    #
    #     coords: coordinates for velocity model, which will not be used in FWI mode.
    #     best:   flag for loading best_loss_model.
    #     uncertainty: flag for uncertainty analysis in IFWI or IRN mode, in which the dropout option will be activated.
    #     NoSim:  number of simulation will be performed for uncertainty analysis.
    #     """
    #     # assert self.netOpt in ['IRN', 'IFWI'], "Error: Only ['IRN', 'IFWI'] are supported in model.predict option."
    #     coords = self.coords if coords is None else coords
    #     if isinstance(resume_file_name, str):
    #         _, _, _, _, _, _ = self.load_state(resume_file_name, best)
    #     else:
    #         print("Failed loading save model, using current model for prediction.")
    #     if self.netOpt in ['IRN', 'IFWI']:
    #         self.vel_net.eval()
    #         with torch.no_grad():
    #             if uncertainty:
    #                 rho = []
    #                 for m in self.vel_net.modules():
    #                     if m.__class__.__name__.startswith('Dropout'):
    #                         m.train()
    #                 for ix in range(NoSim):
    #                     _vpred_, _ = self.vel_net(coords)
    #                     rho.append(_vpred_.detach())
    #                 rho = torch.cat(rho, axis=0)
    #             else:
    #                 rho, _ = self.vel_net(self.coords if coords is None else coords)
    #             rho = (rho * self.std + self.mean) * 1000
    #             # rho[rho < 1000] = 1000
    #             # rho[rho > self.vmax * 1000] = self.vmax * 1000
    #     else:
    #         rho = self.rho * 1000
    #     return rho, coords
    def predict(self, coords=None, uncertainty=False, NoSim=100, best_loss_epoch=0, segy_save_file_name='',
                Vp_save_file_name='', Model_save_file_name='', Initial_save_file_name='', Vp_min=0, Vp_max=1,
                real_Apres=None, real_Phase=None):
        """
        This function only works for IRN or IFWI, performs the predict process of the IRN(MLP).

        coords: coordinates for velocity model, which will not be used in FWI mode.
        best:   flag for loading best_loss_model.
        uncertainty: flag for uncertainty analysis in IFWI or IRN mode, in which the dropout option will be activated.
        NoSim:  number of simulation will be performed for uncertainty analysis.
        """
        # assert self.netOpt in ['IRN', 'IFWI'], "Error: Only ['IRN', 'IFWI'] are supported in model.predict option."
        coords = self.coords if coords is None else coords
        if self.netOpt == 'IRN':
            if isinstance(Initial_save_file_name, str):
                model_path = os.path.join(Initial_save_file_name, 'imt_pretrain_marmousi.pth')
                self.vel_net.load_state_dict(torch.load(model_path, map_location=self.device))
        elif self.netOpt == 'IMT':
            if isinstance(Model_save_file_name, str):
                model_path = os.path.join(Model_save_file_name, 'GS2DInvNet.pth')
                self.vel_net.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Failed loading save model, using current model for prediction.")

        if self.netOpt in ['IRN', 'IMT']:
            self.vel_net.eval()
            with torch.no_grad():
                if uncertainty:
                    rho = []
                    for m in self.vel_net.modules():
                        if m.__class__.__name__.startswith('Dropout'):
                            m.train()
                    for ix in range(NoSim):
                        _vpred_, _ = self.vel_net(coords)
                        rho.append(_vpred_.detach())
                    rho = torch.cat(rho, axis=0)
                else:
                    rho, _ = self.vel_net(self.coords if coords is None else coords)
                # rho = (rho * self.std + self.mean)
                # rho[rho < 1000] = 1000
                # rho[rho > self.vmax * 1000] = self.vmax * 1000
        else:
            rho = self.rho

        if self.netOpt == 'IRN':
            pic_name = 'Initial'
            Vp_save_file_name = Initial_save_file_name
            segy_save_file_name = Initial_save_file_name
        else:
            pic_name = best_loss_epoch + 1
        plt.figure(figsize=(10, 3))
        plt.imshow(rho.squeeze().cpu().detach().numpy(),
                   extent=[0, self.nx * self.dx / 1000, self.nz * self.dz / 1000, 0],
                   # cmap='jet',vmin=rho.min().item(), vmax=rho.max().item(), aspect=4)
                   cmap='jet', vmin=Vp_min, vmax=Vp_max, aspect=4)

        plt.xlabel('Distance (km)', fontsize=18)
        plt.ylabel('Depth (km)', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        cbar = plt.colorbar()
        cbar.set_label('Rho (ohm`m)', fontsize=16)
        # plt.gca().invert_yaxis()b
        plt.gca().set_aspect('auto')
        image_path = os.path.join(Vp_save_file_name, f'{pic_name}_rho.png')
        plt.savefig(image_path)
        # plt.show()
        plt.close()

        rho = rho[:, :, 0]
        apres_pred, phase_pred = self.rnn(rho=rho)

        plot_Apres_phase_record(real_Apres, apres_pred, real_Phase, phase_pred, segy_save_file_name, best_loss_epoch, figsize=[10, 10])
        plot_Apres_phase_1D(real_Apres, apres_pred, real_Phase, phase_pred, segy_save_file_name, best_loss_epoch, x=int(apres_pred.shape[0]/2), figsize=[10, 10])

        # imagesc(fig,
        #         segy_pred[:, ::2, :, :][:, :6, :, :].cpu().numpy().reshape(-1, 3, self.nz, self.nx),
        #         vmin=self.segy_truth_min,
        #         vmax=self.segy_truth_max,
        #         extent=[0, self.nx * self.dz / 1000, self.t.numpy().max(), 0],
        #         aspect=4,
        #         nRows_nCols=(2, 3),
        #         cmap='RdBu_r',  # seismic
        #         ylabel="Time (s)",
        #         xlabel="Position (km)",
        #         clabel="",
        #         xticks=np.arange(0, int(self.nx * self.dz / 1000), 2),
        #         yticks=np.arange(0., self.t.numpy().max(), .5),
        #         fontsize=8,
        #         cbar_width="10%",
        #         cbar_height="50%",
        #         cbar_loc='lower left',
        #         cbar_mode="corner",
        #         path=os.path.join(segy_save_file_name, f'{pic_name}.png'))
        return
    def train(self, 
              MaxIter=10000, 
              rho=None,
              rho_air=None,
              real_Apres=None,
              real_Phase=None,
              alpha=0,
              alpha_ref=0.1,
              learning_rate=1e-4,
              log_interval=1,
              wandb=None,
              Vp_save_file_name='',
              Model_save_file_name='',
              segy_save_file_name='',
              Initial_save_file_name='',
              Vp_min=0, Vp_max=1):
        """
        Args:
            - MaxIter:              Maximum training iteration
            - learning_rate:        learning rate for Adam optimizer
            - resume_file_name:     resume training from a saved file
            - log_interval:         output log interval
            - alpha:                trade-off between data_loss and regularization loss
                                    default=0, i.e., no regularization
                                    if 'auto', trade-off will be zero before 90% of data loss is reduced,
                                        after that, trade-off will be applied to make regularization be 10% of data loss 
            - rho:               in km/s
                                    for the IRN-mode, rho is the label data in shape [num_vels, nz, nx, num_neurons_in_output_layers],
                                    for the FWI-mode, rho is the initial velocity in shape [num_vels, nz, nx],
                                    for the IFWI-mode, rho is not required or used.
            - wavelet:              for the IRN-mode, wavelet is not required,
                                    for the FWI/IFWI-mode, wavelet needs to be provided as a tensor,
                                        and will be used in the forward propagation process.
            - shots:                for the IRN-mode, shots is not required,
                                    for the FWI/IFWI-mode, shots will be used as label data in shape [num_vels, ns, nt, nx].
            - option:               default=0, is not required for the IRN-mode, 
                                        and will be used in segmented forward propagation 
                                        (i.e., trunaced RNN) under the FWI/IFWI-modes.
        """
        if wandb is not None:
            self.run = wandb.init(project=self.netOpt + " Project",
                                  reinit=True,
                                  config={"learning_rate": learning_rate})
        
        self.rho = rho
        self.rho_air = rho_air
        if self.netOpt in ['IRN', 'IMT']:
            self.clip = 0.25
            self.vel_net.train()
            self.params = self.vel_net.parameters()

        else:
            # self.clip = 100
            # self.rho = torch.nn.Parameter(copy.deepcopy(rho))
            #
            self.clip = 100
            self.rho = torch.nn.Parameter(copy.deepcopy(rho))
            self.params = [self.rho]
            # self.params = self.rho

            # self.rnn.register_parameter("rho", self.rho)
            # self.params = self.rnn.parameters()
            # self.rnn.train()
            # optimizer = torch.optim.Adam(lr=learning_rate, params=[self.rho])
            optimizer = torch.optim.Adam(lr=learning_rate, params=self.params)
        ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                                       patience=10, min_lr=1e-05)
        
        best_loss_model=0
        best_loss_epoch = 0
        resume_from_epoch = 0
        train_loss_history = []
        # if isinstance(resume_file_name, str):
        #     resume_from_epoch, best_loss, best_loss_epoch, best_loss_model, \
        #         train_loss_history, optimizer = self.load_state(resume_file_name, False, optimizer)
        
        trade_off = 0 if alpha =='auto' else alpha
        trade_ref = 0 if alpha =='auto' else alpha_ref
        start_time = perf_counter()
        index = 0
        Loss_record = 0
        Loss = []

        for epoch in range(resume_from_epoch, MaxIter):

            rhopred, loss, loss1 = self.train_one_epoch(optimizer, self.rho, real_Apres, real_Phase, trade_off, trade_ref)
            if self.netOpt == 'IMT':
                Loss_record += loss1.item()
                Loss += loss


            # if alpha =='auto' and loss[1] / train_loss_history[-1][1] <= 0.1:
            #     trade_off = 0.01 * loss[1] / loss[2]
            # if wandb is not None:
            #     self.run.log({'Total Loss': loss[0], 'Data Loss': loss[1], 'Regularization Loss': loss[2]})

            if self.netOpt == 'IRN':
                if (epoch + 1) % log_interval == 0:
                    current_time = perf_counter()
                    elapsed_time = current_time - start_time
                    print(
                        "Epoch: {0:5d}, Loss: {1:.4e}, elapsed_time: {2:.4e}".format(epoch + 1, loss, elapsed_time))
                if loss < self.min_loss:
                    self.min_loss = loss
                    best_loss_epoch = epoch + 1
                    best_loss_model = copy.deepcopy(self.vel_net.state_dict())
                    torch.save(best_loss_model, os.path.join(Initial_save_file_name, 'imt_pretrain_marmousi.pth'))

                if epoch == MaxIter-1:
                    self.predict(Vp_save_file_name=Vp_save_file_name, Model_save_file_name=Model_save_file_name,
                                 segy_save_file_name=segy_save_file_name, Vp_max=Vp_max, Vp_min=Vp_min,
                                 best_loss_epoch=best_loss_epoch, Initial_save_file_name=Initial_save_file_name,
                                 real_Apres=real_Apres, real_Phase=real_Phase)
            else:

                train_loss_history.append(loss)
                if (epoch + 1) % log_interval == 0:
                    current_time = perf_counter()
                    elapsed_time = current_time - start_time
                    ReduceLROnPlateau.step(Loss_record / log_interval)
                    learning_rate = optimizer.param_groups[0]['lr']
                    result1 = [xx / log_interval for xx in Loss]
                    print("Epoch: {0:5d}, Loss: {1[0]:.4e}, ApresLoss: {1[1]:.4e}, PhaseLoss: {1[2]:.4e}, RefLoss: {1[3]:.4e}, RegLoss: {1[4]:.4e},  "
                          "elapsed_time: {2:.4e}".format(epoch+1, result1, elapsed_time))
                    print('Learning rate:%.5f' % float(learning_rate))
                    Loss_record = 0
                    Loss = []
                if loss[0] < self.min_loss:
                    self.min_loss = loss[0]
                    best_loss_epoch = epoch + 1
                    best_loss_model = copy.deepcopy(
                        self.vel_net.state_dict() if self.netOpt in ['IRN', 'IMT'] else self.rho)
                    torch.save(best_loss_model, os.path.join(Model_save_file_name, 'GS2DInvNet.pth'))
                    # self.save_state(epoch,
                    #                 best_loss,
                    #                 best_loss_epoch,
                    #                 best_loss_model,
                    #                 train_loss_history,
                    #                 optimizer,
                    #                 log_interval,
                    #                 save_file_name)
                if (epoch + 1) % 50 == 0:
                    self.min_loss= float('inf')
                    self.predict(Vp_save_file_name=Vp_save_file_name, Model_save_file_name=Model_save_file_name,
                                 segy_save_file_name=segy_save_file_name, Vp_max=Vp_max, Vp_min=Vp_min,
                                 best_loss_epoch=best_loss_epoch, real_Apres=real_Apres, real_Phase=real_Phase)
        if self.netOpt == 'IMT':
            self.Loss_plot(train_loss_history, floder=Vp_save_file_name)
        # if wandb is not None:
        #     self.run.finish()
        return train_loss_history, rhopred

    def train_one_epoch(self, optimizer, rho=None, real_Apres=None, real_Phase=None, trade_off=0, trade_ref=0):
        """
        This function performs the training for one-epoch, 
            including all possible segmented wavelet/shots for Truncated RNN.
        """
        if self.netOpt == 'IRN':
            optimizer.zero_grad()
            rhopred = self.forward_process(None)
            loss = self.loss_function(rhopred, rho.float())
            loss.backward()
            optimizer.step()
            return rhopred, loss, loss
        else:

            loss_Reg = 0
            loss_ref = 0

            optimizer.zero_grad()
            rhopred, Apres, Phase = self.forward_process(rho)

            loss_Apres = torch.sqrt(self.loss_function(Apres, real_Apres)) / torch.std(real_Apres)
            loss_Phase = torch.sqrt(self.loss_function(Phase, real_Phase)) / torch.std(real_Phase)

            # if self.reg_op == "TV":
            #     loss_Reg = (rhograd**2 + 1e-6).sqrt().mean() * trade_off  # Total-Variation regularization
            if self.reg_op == "L1":
                loss_Reg = torch.norm(rhopred, p=1) * trade_off
            if self.ref_op == "Half_space":
                loss_ref = torch.sqrt(self.loss_function(rhopred, rho.float())) * trade_ref

            has1_nan = torch.isnan(rhopred).any()
            has2_nan = torch.isnan(Phase).any()
            # has_nan = torch.isnan(loss_Reg).any()
            # has_nan = torch.isnan(loss_ref).any()


            loss = loss_Apres + loss_Phase
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.params, self.clip)
            optimizer.step()


            loss_total = loss.detach().cpu().item()
            loss_ApresAll = loss_Apres.detach().cpu().item()
            loss_PhaseAll = loss_Phase.detach().cpu().item()
            loss_RegAll = loss_Reg.detach().cpu().item()
            loss_RefAll = loss_ref.detach().cpu().item()
            return rhopred.detach(), [loss_total, loss_ApresAll, loss_PhaseAll, loss_RefAll, loss_RegAll], loss
        
    def forward_process(self, rho=None):
        """
        This function performs the forward process of IRN/FWI/IFWI, with segmented/full-time-steps wavelet.
        (all inputs need to be tensor except option)
        Args:
            - rho:       unit in km/s;
                            the velocity model to be inverted (required by rnn2D): [num_vels, nz, nx],
                            for FWI, it is the tensor of the initial model and requires_grad_(True),
                            for IRN and  IFWI, this is not needed (num_vels can only be 1 in the case of IFWI).
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

        if self.netOpt == "IRN":
            # In IRN-mode, rho is scaled
            rho, coords = self.vel_net(self.coords)  # output rho shape [num_vels, nz, nx, 1]

            rho = rho.squeeze(dim=-1)
            # rho = rho * self.std + self.mean
            return rho
        elif self.netOpt == "MT":

            # rho = rho.squeeze(dim=-1)
            Apres, Phase, _ = self.rnn.mt2d(torch.cat((self.rho_air, rho), dim=0), mode=self.mode)
            # Apres, Phase, _ = self.rnn.mt2d(rho, mode=self.mode)

            return rho, Apres, Phase
        else:
            rho, coords = self.vel_net(self.coords)  # output rho shape [num_vels，nz, nx, 1]
            vgrad = 0
            if self.reg_op == "TV": 
                vgrad = self.gradient(rho, coords)
                # vlapl = self.laplace(rho, coords)

            rho = rho.squeeze(dim=-1)
            # rho = rho * self.std + self.mean
            Apres, Phase, _= self.rnn.mt2d(rho, mode=self.mode)

            return rho, vgrad, Apres, Phase
    
    def gradient(self, y, x, grad_outputs=None):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        return grad
    
    def divergence(self, y, x):
        div = 0.
        for i in range(y.shape[-1]):
            div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
        return div

    def laplace(self, y, x):
        grad = self.gradient(y, x)
        return self.divergence(grad, x)

    def load_state(self, resume_file_name=None, best=False, optimizer=None):
        checkpoint = torch.load(resume_file_name)
        resume_from_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_epoch = checkpoint['best_loss_epoch']
        best_loss_model = checkpoint['best_loss_model']
        train_loss_history = checkpoint['train_loss']
        if best:
            print("Loading the best loss model at Epoch {}".format(best_loss_epoch))
        if self.netOpt in ['IRN', 'IFWI']:
            self.vel_net.load_state_dict(best_loss_model if best else checkpoint['state_dict'])
        else:
            self.rho.data = best_loss_model if best else checkpoint['state_dict']
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return resume_from_epoch, best_loss, best_loss_epoch, best_loss_model, train_loss_history, optimizer

    def save_state(self, 
                   epoch, 
                   best_loss, 
                   best_loss_epoch, 
                   best_loss_model, 
                   train_loss_history, 
                   optimizer,
                   log_interval=1,
                   file_name=""):
        state = {'epoch': epoch + 1,
                 'best_loss': best_loss,
                 'best_loss_epoch': best_loss_epoch,
                 'best_loss_model': best_loss_model,
                 'train_loss': train_loss_history,
                 'state_dict': self.vel_net.state_dict() if self.netOpt in ['IRN', 'IFWI'] else self.rho,
                 'optimizer': optimizer.state_dict(),
                 }
        torch.save(state, file_name + "checkpoint-{}.pth".format(best_loss_epoch))
        # if os.path.exists(file_name + "checkpoint-{}.pth".format(epoch + 1 - log_interval)) and epoch % 10*log_interval != 0:
        #     os.remove(file_name + "checkpoint-{}.pth".format(epoch + 1 - log_interval))

    def Loss_plot(self, Loss_append, floder):
        Loss = np.array(Loss_append)
        plt.figure(figsize=(10, 5))
        plt.plot(Loss)
        plt.legend(['Total loss', 'Apres loss', 'Phase loss', 'Ref loss'], loc='upper right', fontsize=16)
        # plt.legend(['Total loss', 'Apres loss', 'Phase loss', 'Ref loss'], loc='upper right', fontsize=16)

        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        # plt.yscale('log')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid()
        image_path_Loss = os.path.join(floder, f'Loss.png')
        plt.savefig(image_path_Loss)
        return



