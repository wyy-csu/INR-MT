import numpy as np
import matplotlib
matplotlib.use('Agg') # 在 import pyplot 之前调用
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable




def plot_Apres_phase_record(res_true, res_pred, phase_true, phase_pred, path, epoch, fre, ry, figsize=[10, 10]):


    Apres_misfit = torch.abs(res_true - res_pred)

    # phase_true = phase_true * 180 / math.pi
    # phase_pred = phase_pred * 180 / math.pi
    phase_misfit = torch.abs(phase_true - phase_pred)

    y_ticks = fre.cpu().detach().numpy()
    x_ticks = ry.cpu().detach().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(figsize[0], figsize[1]))

    im00 = axes[0, 0].imshow(res_true.cpu().detach().numpy(), aspect='auto',
                             vmin=res_true.min(),
                             vmax=res_true.max(), cmap='jet')

    axes[0, 0].set_title('Apres_true', fontsize=12)
    # axes[0, 0].set_xticks(x_ticks)
    # axes[0, 0].set_yticks(y_ticks)
    fig.colorbar(im00, ax=axes[0, 0])  # 添加色标

    im01 = axes[0, 1].imshow(res_pred.cpu().detach().numpy(), aspect='auto',
                             vmin=res_true.min(),
                             vmax=res_true.max(), cmap='jet')

    axes[0, 1].set_title('Apres_pred', fontsize=12)
    axes[0, 1].set_xticks(x_ticks)
    axes[0, 1].set_yticks(y_ticks)
    fig.colorbar(im01, ax=axes[0, 1])  # 添加色标

    im02 = axes[0, 2].imshow(Apres_misfit.cpu().detach().numpy(), aspect='auto', cmap='jet')
    axes[0, 2].set_title('Apres_misfit', fontsize=12)
    # axes[0, 2].set_xticks(x_ticks)
    # axes[0, 2].set_yticks(y_ticks)
    fig.colorbar(im02, ax=axes[0, 2])  # 添加色标



    im10 = axes[1, 0].imshow(phase_true.cpu().detach().numpy(), aspect='auto',
                             vmin=phase_true.min(),
                             vmax=phase_true.max(), cmap='jet')

    axes[1, 0].set_title('Phase_true', fontsize=12)
    # axes[1, 0].set_xticks(x_ticks)
    # axes[1, 0].set_yticks(y_ticks)
    fig.colorbar(im10, ax=axes[1, 0])  # 添加色标

    im11 = axes[1, 1].imshow(phase_pred.cpu().detach().numpy(), aspect='auto',
                             vmin=phase_true.min(),
                             vmax=phase_true.max(), cmap='jet')

    axes[1, 1].set_title('Phase_pred', fontsize=12)
    # axes[1, 1].set_xticks(x_ticks)
    # axes[1, 1].set_yticks(y_ticks)
    fig.colorbar(im11, ax=axes[1, 1])  # 添加色标
    im12 = axes[1, 2].imshow(phase_misfit.cpu().detach().numpy(), aspect='auto', cmap='jet')
    axes[1, 2].set_title('Phase_misfit', fontsize=12)
    # axes[1, 2].set_xticks(x_ticks)
    # axes[1, 2].set_yticks(y_ticks)
    fig.colorbar(im12, ax=axes[1, 2])  # 添加色标

    plt.tight_layout()
    # plt.show()

    #
    image_path = os.path.join(path,'Apres_phase')
    os.makedirs(image_path, exist_ok=True)

    image_path = os.path.join(image_path, f'joint_epoch_{epoch}.png')

    plt.savefig(image_path)


def plot_Apres_phase_1D(res_true, res_pred, phase_true, phase_pred, path, epoch, fre, file_name=None, x=0, figsize=[10, 10]):

    # phase_true = phase_true * 180 / math.pi
    # phase_pred = phase_pred * 180 / math.pi
    x_ticks = np.log10(fre.cpu().detach().numpy())

    fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]))
    axes[0].plot(x_ticks, res_true[:, x].cpu().detach().numpy(), label='True')
    axes[0].plot(x_ticks, res_pred[:, x].cpu().detach().numpy(), label='Pred')
    axes[0].set_xlabel('Frequrency(log)')
    # axes[0].set_xticks(x_ticks)
    axes[0].set_title('Apparent_resistivity', fontsize=12)
    axes[0].legend()

    axes[1].plot(x_ticks, phase_true[:, x].cpu().detach().numpy(), label='True')
    axes[1].plot(x_ticks, phase_pred[:, x].cpu().detach().numpy(), label='Pred')
    axes[1].set_title('Phase', fontsize=12)
    axes[1].set_xlabel('Frequrency(log)')
    # axes[1].set_xticks(x_ticks)
    axes[1].legend()

    plt.tight_layout()

    image_path = os.path.join(path,'Apres_phase_1D')
    os.makedirs(image_path, exist_ok=True)
    if file_name:
        image_path = os.path.join(image_path, file_name)
    else:
        image_path = os.path.join(image_path, f'joint_epoch_{epoch}.png')

    plt.savefig(image_path)


def plot_Apres_phase_1D_TEM(res_true_TE, res_pred_TE, phase_true_TE, phase_pred_TE,
                            res_true_TM, res_pred_TM, phase_true_TM, phase_pred_TM,
                            res_err_TE, res_err_TM, phse_err_TE, phase_err_TM,
                            path, epoch,
                            fre, file_name=None, x=0):

    x_ticks = np.log10(fre)

    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)

    plt.errorbar(
        x_ticks,        # X轴位置
        res_true_TE[:, x],        # Y轴中心点
        yerr=res_err_TE[:, x], # 垂直误差值
        fmt='o',      # 数据点的格式：'o' 代表圆点，也可以用 's' (方块), '^' (三角形) 等
        capsize=5,    # 误差棒两端横线的长度 (cap)
        color='red', # 线的颜色
        ecolor='black', # 误差棒的颜色 (如果想和数据点区分)
        elinewidth=1.5, # 误差棒线的宽度
        label='real-TE'
    )

    plt.errorbar(
        x_ticks,        # X轴位置
        res_true_TM[:, x],        # Y轴中心点
        yerr=res_err_TM[:, x], # 垂直误差值
        fmt='s',      # 数据点的格式：'o' 代表圆点，也可以用 's' (方块), '^' (三角形) 等
        capsize=5,    # 误差棒两端横线的长度 (cap)
        color='blue', # 线的颜色
        ecolor='black', # 误差棒的颜色 (如果想和数据点区分)
        elinewidth=1.5, # 误差棒线的宽度
        label='real-TM'
    )


    plt.plot(x_ticks, res_pred_TE[:, x], label='Pred-TE', color='red')
    plt.plot(x_ticks, res_pred_TM[:, x], label='Pred-TM', color='blue')


    plt.xlabel(r'$\log_{10}$ Frequency(Hz)', fontsize=15)
    plt.ylabel(r'$\log_{10}$ Apparent Resistivity ($\Omega \cdot m$)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Apparent Resistivity', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(2, 1, 2)

    plt.errorbar(
        x_ticks,  # X轴位置
        phase_true_TE[:, x],  # Y轴中心点
        yerr=phse_err_TE[:, x],  # 垂直误差值
        fmt='o',  # 数据点的格式：'o' 代表圆点，也可以用 's' (方块), '^' (三角形) 等
        capsize=5,  # 误差棒两端横线的长度 (cap)
        color='red',  # 线的颜色
        ecolor='black',  # 误差棒的颜色 (如果想和数据点区分)
        elinewidth=1.5,  # 误差棒线的宽度
        label='real-TE'
    )

    plt.errorbar(
        x_ticks,  # X轴位置
        phase_true_TM[:, x],  # Y轴中心点
        yerr=phase_err_TM[:, x],  # 垂直误差值
        fmt='s',  # 数据点的格式：'o' 代表圆点，也可以用 's' (方块), '^' (三角形) 等
        capsize=5,  # 误差棒两端横线的长度 (cap)
        color='blue',  # 线的颜色
        ecolor='black',  # 误差棒的颜色 (如果想和数据点区分)
        elinewidth=1.5,  # 误差棒线的宽度
        label='real-TM'
    )

    plt.plot(x_ticks, phase_pred_TE[:, x], label='Pred-TE', color='red')
    plt.plot(x_ticks, phase_pred_TM[:, x], label='Pred-TM', color='blue')

    plt.xlabel(r'$\log_{10}$ Frequency(Hz)', fontsize=15)
    plt.ylabel(r'Phase ($^\circ$)', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Phase', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    image_path = os.path.join(path,'Apres_phase_1D')
    os.makedirs(image_path, exist_ok=True)
    if file_name:
        image_path = os.path.join(image_path, file_name)
    else:
        image_path = os.path.join(image_path, f'joint_epoch_{epoch}.png')

    plt.savefig(image_path, dpi=300)
    plt.close(fig)


def plot_Apres_phase_2D(res_true, res_pred, phase_true, phase_pred, path, epoch, fre, ry, Apres_name=None, Phase_name=None, figsize=[10, 10]):

    fre = torch.log10(fre)
    ticks = np.linspace(ry[0].cpu().detach().numpy(), ry[-1].cpu().detach().numpy(), 4)

    ### Apres_p ###
    plt.figure(figsize=(15, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(res_true.cpu().detach().numpy(), cmap='jet_r',
               # vmin=np.min(res_true.cpu().detach().numpy()),
               # vmax=np.max(res_true.cpu().detach().numpy()),
               vmin=0, vmax=2,
               extent=[ry[0].cpu(), ry[-1].cpu(), fre[-1].cpu(), fre[0].cpu()])

    plt.xlabel('Distance (km)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.title('Ground Truth', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.yscale('log')
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    vmin = np.min(res_true.cpu().numpy())
    vmax= np.max(res_true.cpu().numpy())

    plt.subplot(1, 3, 2)
    plt.imshow(res_pred.cpu().detach().numpy(), cmap='jet_r',
               # vmin=np.min(res_true.cpu().detach().numpy()),
               # vmax=np.max(res_true.cpu().detach().numpy()),
               vmin=0, vmax=2,
               extent=[ry[0].cpu(), ry[-1].cpu(), fre[-1].cpu(), fre[0].cpu()])
    plt.xlabel('Distance (km)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.title('Prediction', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.yscale('log')
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    ax.set_aspect('auto')

    plt.subplot(1, 3, 3)
    plt.imshow((res_pred.cpu().detach().numpy() - res_true.cpu().detach().numpy()), cmap='jet_r', extent=[ry[0].cpu(), ry[-1].cpu(), fre[-1].cpu(), fre[0].cpu()])

    plt.xlabel('Distance (km)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.title('Relative Difference',fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.colorbar()

    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    ax.set_aspect('auto')

    plt.tight_layout()
    image_path = os.path.join(path,'Apres')
    os.makedirs(image_path, exist_ok=True)
    if Apres_name:
        image_path = os.path.join(image_path, Apres_name)
    else:
        image_path = os.path.join(image_path, f'epoch_{epoch}_Apres.png')

    plt.savefig(image_path)
    plt.close()


    plt.figure(figsize=(15, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(phase_true.cpu().detach().numpy(), cmap='jet_r',
               # vmin=np.min(phase_true.cpu().detach().numpy()),
               # vmax=np.max(phase_true.cpu().detach().numpy()),
               vmin=20, vmax=70,
               extent=[ry[0].cpu(), ry[-1].cpu(), fre[-1].cpu(), fre[0].cpu()])

    plt.xlabel('Distance (km)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.title('Ground Truth', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.yscale('log')
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    ax.set_aspect('auto')

    plt.subplot(1, 3, 2)
    plt.imshow(phase_pred.cpu().detach().numpy(), cmap='jet_r',
               # vmin=np.min(phase_true.cpu().detach().numpy()),
               # vmax=np.max(phase_true.cpu().detach().numpy()),
               vmin=20, vmax=70,
               extent=[ry[0].cpu(), ry[-1].cpu(), fre[-1].cpu(), fre[0].cpu()])

    plt.xlabel('Distance (km)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.title('Prediction', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.yscale('log')
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    ax.set_aspect('auto')


    plt.subplot(1, 3, 3)
    plt.imshow((phase_pred.cpu().detach().numpy() - phase_true.cpu().detach().numpy()), cmap='jet_r',
               extent=[ry[0].cpu(), ry[-1].cpu(), fre[-1].cpu(), fre[0].cpu()])
    plt.xlabel('Distance (km)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.title('Relative Difference', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.yscale('log')
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    ax.set_aspect('auto')


    plt.tight_layout()
    image_path = os.path.join(path,'Phase')
    os.makedirs(image_path, exist_ok=True)

    if Phase_name:
        image_path = os.path.join(image_path, Phase_name)
    else:
        image_path = os.path.join(image_path, f'epoch_{epoch}_Phase.png')

    plt.savefig(image_path)
    plt.close()



def plot_Apres_phase_2D(res_true, res_pred, phase_true, phase_pred, path, epoch, fre, ry, Apres_name=None, Phase_name=None, figsize=[10, 10]):

    fre = torch.log10(fre)
    ticks = np.linspace(ry[0].cpu().detach().numpy(), ry[-1].cpu().detach().numpy(), 4)

    ### Apres_p ###
    plt.figure(figsize=(15, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(res_true.cpu().detach().numpy(), cmap='jet_r',
               vmin=np.min(res_true.cpu().detach().numpy()),
               vmax=np.max(res_true.cpu().detach().numpy()),
               # vmin=0, vmax=2,
               extent=[ry[0].cpu(), ry[-1].cpu(), fre[-1].cpu(), fre[0].cpu()])

    plt.xlabel('Distance (km)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.title('Ground Truth', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    vmin = np.min(res_true.cpu().numpy())
    vmax= np.max(res_true.cpu().numpy())

    plt.subplot(1, 3, 2)
    plt.imshow(res_pred.cpu().detach().numpy(), cmap='jet_r',
               vmin=np.min(res_true.cpu().detach().numpy()),
               vmax=np.max(res_true.cpu().detach().numpy()),
               # vmin=0, vmax=2,
               extent=[ry[0].cpu(), ry[-1].cpu(), fre[-1].cpu(), fre[0].cpu()])
    plt.xlabel('Distance (km)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.title('Prediction', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.yscale('log')
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    ax.set_aspect('auto')

    plt.subplot(1, 3, 3)
    # plt.imshow((res_pred.cpu().detach().numpy() - res_true.cpu().detach().numpy()) / np.max(res_true.T.cpu().detach().numpy()), cmap='jet',
    #            vmin=-0.1,
    #            vmax=0.1, extent=[ry[0], ry[-1], fre[-1], fre[0]])
    plt.imshow((res_pred.cpu().detach().numpy() - res_true.cpu().detach().numpy()), cmap='jet_r', extent=[ry[0].cpu(), ry[-1].cpu(), fre[-1].cpu(), fre[0].cpu()])

    plt.xlabel('Distance (km)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.title('Relative Difference',fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.colorbar()

    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    ax.set_aspect('auto')

    plt.tight_layout()
    image_path = os.path.join(path,'Apres')
    os.makedirs(image_path, exist_ok=True)
    if Apres_name:
        image_path = os.path.join(image_path, Apres_name)
    else:
        image_path = os.path.join(image_path, f'epoch_{epoch}_Apres.png')

    plt.savefig(image_path)
    plt.close()


    plt.figure(figsize=(15, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(phase_true.cpu().detach().numpy(), cmap='jet_r',
               vmin=np.min(phase_true.cpu().detach().numpy()),
               vmax=np.max(phase_true.cpu().detach().numpy()),
               # vmin=20, vmax=70,
               extent=[ry[0].cpu(), ry[-1].cpu(), fre[-1].cpu(), fre[0].cpu()])

    plt.xlabel('Distance (km)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.title('Ground Truth', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.yscale('log')
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    ax.set_aspect('auto')

    plt.subplot(1, 3, 2)
    plt.imshow(phase_pred.cpu().detach().numpy(), cmap='jet_r',
               vmin=np.min(phase_true.cpu().detach().numpy()),
               vmax=np.max(phase_true.cpu().detach().numpy()),
               # vmin=20, vmax=70,
               extent=[ry[0].cpu(), ry[-1].cpu(), fre[-1].cpu(), fre[0].cpu()])

    plt.xlabel('Distance (km)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.title('Prediction', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.yscale('log')
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    ax.set_aspect('auto')


    plt.subplot(1, 3, 3)
    plt.imshow((phase_pred.cpu().detach().numpy() - phase_true.cpu().detach().numpy()), cmap='jet_r',
               extent=[ry[0].cpu(), ry[-1].cpu(), fre[-1].cpu(), fre[0].cpu()])
    plt.xlabel('Distance (km)', fontsize=10)
    plt.ylabel('Frequency (Hz)', fontsize=10)
    plt.title('Relative Difference', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.yscale('log')
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(ticks)
    ax.invert_yaxis()
    ax.set_aspect('auto')


    plt.tight_layout()
    image_path = os.path.join(path,'Phase')
    os.makedirs(image_path, exist_ok=True)

    if Phase_name:
        image_path = os.path.join(image_path, Phase_name)
    else:
        image_path = os.path.join(image_path, f'epoch_{epoch}_Phase.png')

    plt.savefig(image_path)
    plt.close()

