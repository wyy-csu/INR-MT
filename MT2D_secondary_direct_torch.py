'''
2-D MT forward modeling code using finite difference method (FDM).

second filed methods (different with total field method)
'''

# import ray
# ray.init(num_cpus=20, num_gpus=0)
import numpy as np
import math
# from scipy.linalg import lu
import scipy.io as scio
import scipy.sparse as scipa 
import scipy.sparse.linalg as scilg
import cmath as cm
import torch
import torch_spsolve
from utils import *
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class MT2DFD(object):
    
    def __init__(self, nza, zn, yn, freq, ry,n_add=5):
        '''
        zn: np.array, size(nz+1,); position of z nodes, begin from 0, Down is the positive direction
        yn: np.array, size(ny+1,); position of y nodes;
        freq: np.array, size(n,);  
        ry: observation system
        sig: conductivity of domain, size(nz,ny);
        nza: number of air layer
        n_add: add n times points for 1D field computation
        '''
        self.nza = nza
        self.miu = 4.0e-7*torch.pi
        self.II = cm.sqrt(-1)
        self.zn = zn
        self.nz = len(zn)
        self.dz = zn[1:] - zn[:-1]
        self.yn = yn
        self.ny = len(yn)
        self.dy = yn[1:] - yn[:-1]
        self.freq = freq 
        self.nf = len(freq)
        self.ry = ry # observation system
        self.nry = len(ry)
        # self.sig = sig
        self.n_add = n_add


        # self.BC_u = BC_u
        
    def mt2d(self,sig, mode="TETM"):
        #2-D Magnetotellurics(MT) forward modeling solver.
        self.sig = 1/ (10 ** sig)
        self.sig_back = torch.ones_like(self.sig).float() * self.sig[:,0:1]
        self.sig_diff = self.sig - self.sig_back
        dy = self.dy
        dz = self.dz
        if sig.shape != (self.nz-1,self.ny-1):
            raise ValueError("bad size of sigma, must be (nz-1,ny-1)")
        # sigma of background (here use sigma in left boundary as background)

        yn = self.yn
        ry = self.ry
        nza = self.nza
        n_add = self.n_add
        
        Zxy = torch.zeros((self.nf,self.nry),dtype=torch.complex128, device=device)
        Zyx = torch.zeros((self.nf,self.nry),dtype=torch.complex128, device=device)
        rhoxy = torch.zeros((self.nf,self.nry),dtype=torch.float32, device=device)
        phsxy = torch.zeros((self.nf,self.nry),dtype=torch.float32, device=device)
        rhoyx = torch.zeros((self.nf,self.nry),dtype=torch.float32, device=device)
        phsyx = torch.zeros((self.nf,self.nry),dtype=torch.float32, device=device)
        
        #loop over frequencies.
        if mode == "TE":

            for kf in range(0,self.nf):
                # print(f"TE: calculation the frequency point: {1.0/self.freq[kf]}s")
                ex = self.mt2dte(self.freq[kf], dy, dz, self.sig, self.sig_diff, n_add)

                hys, hzs = self.mt2dhyhz(self.freq[kf], dy, dz, self.sig, ex)

                exs = ex[nza, :]
                # interprolation in observation staiton
                exr = torch_interp(ry, yn, exs)
                hyr = torch_interp(ry, yn, hys)

                Zxy[kf, :], rhoxy[kf, :], phsxy[kf, :] = self.mt2dzxy(self.freq[kf], exr, hyr)
            #zyx = 0
            return rhoxy, phsxy, Zxy
        elif mode == "TM":
            # exrf = np.zeros((self.nf,self.nry),dtype=complex)
            # hyrf = np.zeros((self.nf,self.nry),dtype=complex)   
            # # no air layer     
            dz = self.dz[nza:]
            sig = self.sig[nza:,:]
            sig_diff = self.sig_diff[nza:, :]
            for kf in range(0,self.nf):
                # print(f"TM: calculation the frequency point: {1.0/self.freq[kf]}s")
                hx = self.mt2dtm(self.freq[kf],dy,dz,sig,sig_diff,n_add)
                eys,ezs = self.mt2deyez(self.freq[kf],dy,dz,sig,hx)

                hxs = hx[0,:]
                hxr = torch_interp(ry,yn,hxs)
                eyr = torch_interp(ry,yn,eys)

                Zyx[kf,:],rhoyx[kf,:],phsyx[kf,:] = self.mt2dzyx(self.freq[kf], hxr, eyr)

            return rhoyx,phsyx,Zyx
        elif mode == "TETM":
            # TE
            for kf in range(0,self.nf):
                # print(f"TE: calculation the frequency point: {1.0/self.freq[kf]}s")
                ex = self.mt2dte(self.freq[kf],dy,dz,self.sig,self.sig_diff,n_add)
                hys,hzs = self.mt2dhyhz(self.freq[kf],dy,dz,self.sig,ex)

                exs = ex[nza,:]
                exr = torch_interp(ry,yn,exs)
                hyr = torch_interp(ry,yn,hys)

                Zxy[kf,:],rhoxy[kf,:],phsxy[kf,:] = self.mt2dzxy(self.freq[kf],exr,hyr)


            # TM
            # no air layer
            dz = self.dz[nza:]
            sig = self.sig[nza:,:]  
            sig_diff = self.sig_diff[nza:,:]           
            for kf in range(0,self.nf):
                # print(f"TM: calculation the frequency point: {1.0/self.freq[kf]}s")
                hx = self.mt2dtm(self.freq[kf],dy,dz,sig,sig_diff,nza)
                eys,ezs = self.mt2deyez(self.freq[kf],dy,dz,sig,hx)

                hxs = hx[0,:]
                hxr = torch_interp(ry,yn,hxs)
                eyr = torch_interp(ry,yn,eys)

                Zyx[kf,:],rhoyx[kf,:],phsyx[kf,:] = self.mt2dzyx(self.freq[kf],hxr,eyr)
                #exrf[kf] = exr
                #hyrf[kf] = hyr
            return rhoxy,phsxy, rhoyx,phsyx ,Zxy,Zyx

    def mt2dte(self, freq, dy, dz, sig, sig_diff, n_add):
        '''
        compute secondary electrical filed

        # n_add: add n times points for 1D field computation
        '''
        omega = 2.0 * math.pi * freq
        ny = self.ny-1
        nz = self.nz-1
        #1.compute the system mat
        # 展平为2维方便矩阵计算
        dy0, dz0 = torch.meshgrid(dy, dz, indexing='xy')
        dyc = (dy0[0:nz-1, 0:ny-1]+dy0[0:nz-1, 1:ny])/2.0
        dzc = (dz0[0:nz-1, 0:ny-1]+dz0[1:nz, 0:ny-1])/2.0
        w1 = dy0[0:nz-1, 0:ny-1]*dz0[0:nz-1, 0:ny-1] # notice: for index, ny==-1,nz==-1
        w2 = dy0[0:nz-1, 1:ny] * dz0[0:nz-1, 0:ny-1]
        w3 = dy0[0:nz-1, 0:ny-1]*dz0[1:nz,0:ny-1]
        w4 = dy0[0:nz-1,1:ny] * dz0[1:nz,0:ny-1]
        area = (w1+w2+w3+w4)/4.0
        sigc = (sig[0:nz-1,0:ny-1]*w1 + sig[0:nz-1,1:ny]*w2 + sig[1:nz,:ny-1]*w3 + sig[1:nz,1:ny]*w4)/(area*4.0)
        val = dzc/dy0[0:nz-1,0:ny-1] + dzc/dy0[0:nz-1,1:ny] + dyc/dz0[0:nz-1,0:ny-1] +dyc/dz0[1:nz,0:ny-1]
        mtx1 = self.II * omega * self.miu * sigc*area - val
        mtx1_1 = mtx1.T.reshape(-1, 1).to(torch.complex128) # flatten in column, because outter 'for' loop is in y;
        ##the first lower and upper diagonal terms

        mtx20 = dyc[1:nz-1, 0:ny-1]/dz0[1:nz-1, 0:ny-1]
        mtx2 = torch.cat((mtx20, torch.zeros((1, ny-1), dtype=torch.float32, device=device)), 0)
        mtx2_1 = mtx2.T.reshape(-1, 1)[:-1, 0] # the last element is not needed.

        ##the secend lower and upper diagonal terms
        mtx3 = dzc[0:nz-1,1:ny-1]/dy0[0:nz-1,1:ny-1]
        mtx3_1 = mtx3.T.reshape(-1, 1).squeeze(-1).to(device)
        k2 = nz
        ## build dense mtxA
        n = mtx1_1.size(0)

        indices = torch.arange(n).to(torch.int64).to(device)
        indices_off1 = torch.arange(1, n).to(torch.int64).to(device)
        indices_off2 = torch.arange(n - 1).to(torch.int64).to(device)
        indices_off_k2 = torch.arange(k2 - 1, n).to(torch.int64).to(device)
        indices_off_1_k2 = torch.arange(0, n - k2 + 1).to(torch.int64).to(device)
        indices_main = torch.stack((indices, indices))

        values_main = mtx1_1[:, 0]

        indices_off1 = torch.stack((indices_off1, indices_off1 - 1))
        values_off1 = mtx2_1[:n - 1]

        indices_off2 = torch.stack((indices_off2, indices_off2 + 1))
        values_off2 = mtx2_1[:n - 1]

        indices_off_k2 = torch.stack((indices_off_k2, indices_off_k2 - (k2 - 1)))
        values_off_k2 = mtx3_1[:n - (k2 - 1)]

        indices_off_1_k2 = torch.stack((indices_off_1_k2, indices_off_1_k2 + (k2 - 1)))
        values_off_1_k2 = mtx3_1[:n - (k2 - 1)]

        indices_all = torch.cat((indices_main, indices_off1, indices_off2, indices_off_k2, indices_off_1_k2), dim=1)
        values_all = torch.cat((values_main, values_off1, values_off2, values_off_k2, values_off_1_k2))

        mtxA = torch.sparse_coo_tensor(indices_all, values_all, size=(n, n))


        #2.2compute the primary field
        ex1d, _ = self.mt1dte(freq, dz, (sig-sig_diff)[:, 0], n_add)

        # ex1d_1 = ex1d.reshape(-1, 1) * torch.ones((nz+1, ny+1), device=device)
        #
        # #2.compute right hand side
        # # for secondary filed, Bcs are zero
        # sigc_diff = (sig_diff[0:nz-1, 0:ny-1]*w1 + sig_diff[0:nz-1, 1:ny]*w2 + sig_diff[1:nz, :ny-1]*w3 + sig_diff[1:nz,1:ny]*w4)/(area*4.0)
        # coef = self.II*omega*self.miu*sigc_diff*area
        # rhs = -1 * coef * ex1d_1[1:nz, 1:ny]
        #
        #
        # # rhs = - rhs
        # rhs_1 = rhs.T.reshape((ny-1)*(nz-1), 1)
        # ex1d = ex1d.to(torch.complex128)
        ex1d = ex1d.reshape(-1, 1) * torch.ones((nz + 1, ny + 1), device=device, dtype=torch.complex128)
        ex1d_clone = ex1d.clone()

        sigc_diff = (sig_diff[0:nz - 1, 0:ny - 1] * w1 + sig_diff[0:nz - 1, 1:ny] * w2 + sig_diff[1:nz,:ny - 1] * w3 + sig_diff[1:nz,1:ny] * w4) / (area * 4.0)
        coef = self.II * omega * self.miu * sigc_diff * area

        ex1d_part = ex1d_clone[1:nz, 1:ny].clone()
        rhs = -1 * coef * ex1d_part
        rhs = rhs.T.reshape((ny - 1) * (nz - 1), 1)

        ex, _ = self.equation_solve(mtxA, rhs)

        ex2d = ex.reshape(ny - 1, nz - 1).T

        # ex0d = torch.zeros_like(ex1d, dtype=torch.complex128)

        # boundary_mask = torch.ones_like(ex0d, dtype=torch.bool)
        # boundary_mask[1:nz, 1:ny] = False

        # ex0d[boundary_mask] = ex1d[boundary_mask].to(torch.complex128)
        # ex0d[~boundary_mask] = (ex1d[~boundary_mask] + ex2d.reshape(-1)).to(torch.complex128)
        #3.solve the system equation:mtxA * ex = rhs by LU factorization.
        # p,l,u = lu(mtxA)
        # p = np.mat(p)
        # l = np.mat(l)
        # u = np.mat(u)
        # ex = u.I*(l.I*p*rhs)
        # time0 = default_timer()
        # get secondary field
        # X = torch.linalg.solve(mtxA.to_dense(), rhs).T.reshape(nz-1,ny-1)

        # lu, pivots = torch.linalg.lu_factor(mtxA)
        # ex = torch.linalg.lu_solve(lu, pivots, rhs).T.reshape(nz-1,ny-1)

        # ex, _ = self.equation_solve(mtxA, rhs_1)
        #
        # # time1 = default_timer()
        # # print(f"time using in solve TE equation: {time1-time0}s")
        #
        # ex2d = ex.reshape(ny-1, nz-1).T
        # # total field
        ex0d = ex1d_clone
        # ex0d: total filed
        # ex1d: primary field
        # ex2d: secondary field
        ex0d[1:nz, 1:ny] = ex1d[1:nz, 1:ny] + ex2d
        return ex0d
        
    def mt2dtm(self,freq,dy,dz,sig,sig_diff,n_add):
        '''
        compute secondary magnetic field
        '''
        omega = 2.0*math.pi*freq
        ny = len(dy)
        nz = len(dz)

        #1.compute the system mat
        ##the diagnal termsgg
        # mtx1 = np.zeros((ny-1)*(nz-1),dtype=complex)
        dy0,dz0 = torch.meshgrid(dy,dz, indexing='xy')
        dyc = (dy0[0:nz-1,0:ny-1]+dy0[0:nz-1,1:ny])/2.0
        dzc = (dz0[0:nz-1,0:ny-1]+dz0[1:nz,0:ny-1])/2.0
        w1 = 2 * dz0[0:nz-1, 0:ny-1] # dz1
        w2 = 2 * dz0[1:nz,   0:ny-1] # dz2
        w3 = 2 * dy0[0:nz-1, 0:ny-1] # dy1
        w4 = 2 * dy0[0:nz-1, 1:ny]   # dy2
        area = (w1+w2+w3+w4)/4.0
        A = (1.0/sig[0:nz-1,0:ny-1] * dy0[0:nz-1,0:ny-1] + 1.0/sig[0:nz-1,1:ny]*dy0[0:nz-1,1:ny])/w1 # (dy1 * rho_11 + dy2 * rho_12) / (2*dz1)
        B = (1.0/sig[1:nz  ,0:ny-1] * dy0[0:nz-1,0:ny-1] + 1.0/sig[1:nz  ,1:ny]*dy0[0:nz-1,1:ny])/w2 # (dy1 * rho_21 + dy2 * rho_22) / (2*dz2)
        C = (1.0/sig[0:nz-1,0:ny-1] * dz0[0:nz-1,0:ny-1] + 1.0/sig[1:nz,0:ny-1]*dz0[1:nz,0:ny-1])/w3 # (dz1 * rho_11 * dz2 * rho_21) / (2*dy1)
        D = (1.0/sig[0:nz-1,1:ny  ] * dz0[0:nz-1,0:ny-1] + 1.0/sig[1:nz,1:ny  ]*dz0[1:nz,0:ny-1])/w4 # (dz1 * rho_12 + dz2 * rho_22) / (2*dy2)
        mtx1 = self.II * omega * self.miu * dyc * dzc - A - B - C - D
        # mtx1 = mtx1.flatten('F') # flatten in column, because outter 'for' loop is in y;
        mtx1_1 = mtx1.T.reshape(-1, 1).to(torch.complex128)
        ##the first lower and upper diagonal terms 
        mtx20 = B[0:nz-2,0:ny-1]#1.0/sig[1:nz-1,0:ny-1] * dy[1:nz-1,0:ny-1]  + 1.0/sig[1:nz-1, 1:ny]*dy[1:nz-1,1:ny]/(2 * dz[1:nz-1,0:ny-1])
        # mtx2[-1:,:] = 0.0 # the last line is zero
        mtx2 = torch.cat((mtx20,torch.zeros((1,ny-1), dtype=torch.float32, device=device)),0)
        # mtx2 = mtx2.flatten('F')[:-1] # the last element is not needed.
        mtx2_1 = mtx2.T.reshape(-1, 1)[:-1, 0]
            
        ##the secend lower and upper diagonal terms
        
        mtx3 = D[0:nz-1,0:ny-2]#1.0/sig[0:nz-1,1:ny  ] * dz[0:nz-1,0:ny-1] + 1.0/sig[1:nz,1:ny] * dz[1:nz,0:ny-1]/(2 * dy[0:nz-1,1:ny])
        # mtx3 = mtx3.flatten('F')
        mtx3_1 = mtx3.T.reshape(-1, 1).squeeze(-1).to(device)
        k2 =  nz         
        # mtxA = scipa.diags(mtx1,format='csc')+\
        #     scipa.diags(mtx2,-1,format='csc')+scipa.diags(mtx2,1,format='csc')+\
        #         scipa.diags(mtx3,1-k2,format='csc')+scipa.diags(mtx3,k2-1,format='csc')
        #

        n = mtx1_1.size(0)

        indices = torch.arange(n).to(torch.int64).to(device)
        indices_off1 = torch.arange(1, n).to(torch.int64).to(device)
        indices_off2 = torch.arange(n - 1).to(torch.int64).to(device)
        indices_off_k2 = torch.arange(k2 - 1, n).to(torch.int64).to(device)
        indices_off_1_k2 = torch.arange(0, n - k2 + 1).to(torch.int64).to(device)
        indices_main = torch.stack((indices, indices))

        values_main = mtx1_1[:, 0]

        indices_off1 = torch.stack((indices_off1, indices_off1 - 1))
        values_off1 = mtx2_1[:n - 1]

        indices_off2 = torch.stack((indices_off2, indices_off2 + 1))
        values_off2 = mtx2_1[:n - 1]

        indices_off_k2 = torch.stack((indices_off_k2, indices_off_k2 - (k2 - 1)))
        values_off_k2 = mtx3_1[:n - (k2 - 1)]

        indices_off_1_k2 = torch.stack((indices_off_1_k2, indices_off_1_k2 + (k2 - 1)))
        values_off_1_k2 = mtx3_1[:n - (k2 - 1)]

        indices_all = torch.cat((indices_main, indices_off1, indices_off2, indices_off_k2, indices_off_1_k2), dim=1)
        values_all = torch.cat((values_main, values_off1, values_off2, values_off_k2, values_off_1_k2))

        mtxA = torch.sparse_coo_tensor(indices_all, values_all, size=(n, n))


        #2.compute right hand side        
        ey1d, hx1d = self.mt1dtm(freq,dz,(sig-sig_diff)[:,0],n_add)
        ey1d = ey1d.reshape(-1,1)*torch.ones((nz+1,ny+1), device=device, dtype=torch.complex128)
        hx1d = hx1d.reshape(-1,1)*torch.ones((nz+1,ny+1), device=device, dtype=torch.complex128)
        hx1d_clone = hx1d.clone()
        dy0,dz0 = torch.meshgrid(dy,dz, indexing='xy')
        A1 = dy0[0:nz-1,0:ny-1]*dz0[0:nz-1,0:ny-1] # notice: for index, ny==-1,nz==-1
        A2 = dy0[0:nz-1,1:ny]  *dz0[0:nz-1,0:ny-1]
        A3 = dy0[0:nz-1,0:ny-1]*dz0[1:nz,0:ny-1]
        A4 = dy0[0:nz-1,1:ny]  *dz0[1:nz,0:ny-1]
        area = (A1+A2+A3+A4)/4.0

        #2.compute right hand side
        # for secondary filed, Bcs are zero
        sig_scale = sig_diff/sig
        sigc_diff = (sig_scale[0:nz-1,0:ny-1]*A1 + sig_scale[0:nz-1,1:ny]*A2 + sig_scale[1:nz,0:ny-1]*A3 + sig_scale[1:nz,1:ny]*A4)/(area*4.0)
        # sig_scale = 1.0/sig
        # sigc = (sig_scale[0:nz-1,0:ny-1]*A1 + sig_scale[0:nz-1,1:ny]*A2 + sig_scale[1:nz,:ny-1]*A3 + sig_scale[1:nz,1:ny]*A4)/(area*4.0)
        coef = self.II*omega*self.miu*sigc_diff* area
        hx1d_part = hx1d_clone[1:nz, 1:ny].clone()
        rhs  = coef * hx1d_part

        # derivative term
        # ey_t = (ey1d[1:nz  ,1:ny]+ey1d[0:nz-1,1:ny])/2.0
        # ey_b = (ey1d[1:nz  ,1:ny]+ey1d[2:nz+1,1:ny])/2.0
        # ey_c = (ey_t + ey_b)/2.0
        sigc_t = (sig_scale[0:nz-1,0:ny-1] * dy0[0:nz-1,0:ny-1] + sig_scale[0:nz-1,1:ny]*dy0[0:nz-1,1:ny])/(dy0[0:nz-1,0:ny-1]+dy0[0:nz-1,1:ny]) # (dy1 * sig_11 + dy2 * sig_12) /(dy1+dy2)
        sigc_b = (sig_scale[1:nz  ,0:ny-1] * dy0[1:nz  ,0:ny-1] + sig_scale[1:nz  ,1:ny]*dy0[1:nz  ,1:ny])/(dy0[1:nz  ,0:ny-1]+dy0[1:nz  ,1:ny]) # (dy1 * sig_21 + dy2 * sig_22) /(dy1+dy2)
        ey_d = (sigc_b - sigc_t)/((dz0[0:nz-1,0:ny-1]+dz0[1:nz,0:ny-1])/2.0)*area*ey1d[1:nz,1:ny]
        rhs = -1 * (rhs - ey_d)

        # rhs = - rhs
        # rhs = rhs.reshape((ny-1)*(nz-1),1,order='F')
        rhs = rhs.T.reshape((ny - 1) * (nz - 1), 1)
        #3.solve the system equation:mtxA * ex = rhs by LU factorization.
        # p,l,u = lu(mtxA)
        # p = np.mat(p)
        # l = np.mat(l)
        # u = np.mat(u)
        # hx = u.I*(l.I*p*rhs)
        # time0 = default_timer()
        hx, _ = self.equation_solve(mtxA, rhs)
        # time1 = default_timer()
        # print(f"time using in solve TM equation: {time1-time0}s")
        # hx2d = hx.reshape(nz-1,ny-1,order='F')
        hx2d = hx.reshape(ny - 1, nz - 1).T
        hx0d = hx1d
        # hx0d: total filed
        # hx1d: primary field
        # hx2d: secondary field
        hx0d[1:nz,1:ny] = hx1d[1:nz,1:ny] + hx2d
        return hx0d

    def mt2dhyhz(self,freq,dy,dz,sig,ex):
        #Interpolater of H-field for 2-D Magnetotellurics(MT) TE mode solver.
        omega = 2.0*math.pi*freq
        ny = len(dy)
        #1.compute Hy
        hys = torch.zeros((ny+1),dtype=torch.complex128, device=device)
        #1.1compute Hy at the top left corner
        kk = self.nza 
        delz = dz[kk]
        sigc = sig[kk,0]
        c0 = -1.0/(self.II*omega*self.miu*delz) + (3.0/8.0)*sigc*delz
        c1 = 1.0/(self.II*omega*self.miu*delz) + (1.0/8.0)*sigc*delz
        hys[0] = c0*ex[kk,0] + c1*ex[kk+1,0]
        #1.2compute Hy at the top right corner
        sigc = sig[kk,ny-1]
        c0 = -1.0/(self.II*omega*self.miu*delz) + (3.0/8.0)*sigc*delz
        c1 = 1.0/(self.II*omega*self.miu*delz) + (1.0/8.0)*sigc*delz
        hys[ny] = c0*ex[kk,ny] + c1*ex[kk+1,ny]
        #1.3compute the Hy at other nodes
        dyj = dy[0:ny-1]+dy[1:ny]
        sigc = (sig[kk,0:ny-1]*dy[0:ny-1]+sig[kk,1:ny]*dy[1:ny])/dyj
        cc = delz/(4.0*self.II*omega*self.miu*dyj) # should devided by 8.0?
        c0 = -1.0/(self.II*omega*self.miu*delz) + (3.0/8.0)*sigc*delz - cc*3.0*(1.0/dy[1:ny]+1.0/dy[0:ny-1])
        c1 = 1.0/(self.II*omega*self.miu*delz) + (1.0/8.0)*sigc*delz - cc*1.0*(1.0/dy[1:ny]+1.0/dy[0:ny-1])
        c0l = 3.0*cc/dy[0:ny-1]
        c0r = 3.0*cc/dy[1:ny]
        c1l = 1.0*cc/dy[0:ny-1]
        c1r = 1.0*cc/dy[1:ny]
        hys[1:ny] = c0l*ex[kk,0:ny-1] + c0*ex[kk,1:ny] + c0r*ex[kk,2:ny+1] + \
                    c1l*ex[kk+1,0:ny-1] + c1*ex[kk+1,1:ny] + c1r*ex[kk+1,2:ny+1]
        #2.compute Hz
        hzs = torch.zeros((ny+1), dtype=torch.complex128, device=device)
        #2.1compute Hz at the topleft and top right corner
        hzs[0] = -1.0/(self.II*omega*self.miu)*(ex[kk,1]-ex[kk,0])/dy[0]
        hzs[ny] = -1.0/(self.II*omega*self.miu)*(ex[kk,ny]-ex[kk,ny-1])/dy[ny-1]
        #2.2compute Hz at other nodes
        # for kj in range(1,ny):
        hzs[1:ny] = -1.0/(self.II*omega*self.miu)*(ex[kk,2:ny+1]-ex[kk,0:ny-1])/(dy[0:ny-1]+dy[1:ny])

        return hys,hzs
    
    def mt2deyez(self,freq,dy,dz,sig,hx):
        #Interpolater of H-field for 2-D Magnetotellurics(MT) TE mode solver.
        omega = 2.0*math.pi*freq
        # ny = np.size(dy)
        ny = len(dy)
        #1.compute Hy
        eys = torch.zeros((ny + 1), dtype=torch.complex128, device=device)
        #1.1compute Hy at the top left corner
        # kk = self.nza 
        kk = 0 # no air layer
        delz = dz[kk]
        sigc = sig[kk,0]
        temp_beta = self.II * omega * self.miu * delz
        temp_1 = sigc * delz
        c0 = -1.0/temp_1 + (3.0/8.0)*temp_beta
        c1 = 1.0/temp_1 + (1.0/8.0)*temp_beta
        eys[0] = c0*hx[kk,0] + c1*hx[kk+1,0]
        #1.2compute Hy at the top right corner
        sigc = sig[kk,ny-1]
        temp_1 = sigc * delz
        c0 = -1.0/temp_1 + (3.0/8.0)*temp_beta
        c1 = 1.0/temp_1+ (1.0/8.0)*temp_beta
        eys[ny] = c0*hx[kk,ny] + c1*hx[kk+1,ny]
        #1.3compute the Hy at other nodes
        # for kj in range(1,ny):
        dyj = (dy[0:ny-1]+dy[1:ny])/2.0
        tao = 1.0/sig[kk,0:ny]
        taoc = (tao[0:ny-1]*dy[0:ny-1] + tao[1:ny]*dy[1:ny])/(2*dyj)
        temp_1 = self.II*omega*self.miu*delz
        temp_2 = taoc/delz
        temp_3 = delz/dyj
        temp_4 = tao/dy
        c0 =  (3.0/8.0)*temp_1 - temp_2
        c1 =  (1.0/8.0)*temp_1 + temp_2 - (1.0/8.0)*temp_3*(temp_4[0:ny-1]+temp_4[1:ny])
        c1l = (1.0/8.0)*temp_3*temp_4[0:ny-1]
        c1r = (1.0/8.0)*temp_3*temp_4[1:ny]
        eys[1:ny] = c0*hx[kk, 1:ny] + c1l*hx[kk+1,0:ny-1]+c1*hx[kk+1,1:ny]+c1r*hx[kk+1,2:ny+1]
        #2.compute Hz
        # ezs = np.zeros((ny+1), dtype=complex)
        ezs = torch.zeros((ny + 1), dtype=torch.complex128, device=device)
        # to do

        # #2.1compute Hz at the topleft and top right corner
        # ezs[0] = -1.0/(self.II*omega*self.miu)*(hx[kk,1]-hx[kk,0])/dy[0]
        # ezs[ny] = -1.0/(self.II*omega*self.miu)*(hx[kk,ny]-hx[kk,ny-1])/dy[ny-1]
        # #2.2compute Hz at other nodes
        # for kj in range(1,ny):
        #     ezs[kj] = -1.0/(self.II*omega*self.miu)*(hx[kk,kj+1]-hx[kk,kj-1])/(dy[kj-1]+dy[kj])

        return eys, ezs
        
    def mt2dzxy(self, freq, exr, hyr):
        #compute the impedance, apparent resistivity and phase of TE mode 2-D Magnetotellurics(MT) forward modeling problem
        omega = 2.0*math.pi*freq
        #compute the outputs
        zxy = exr/hyr
        rhote = abs(zxy)**2/(omega*self.miu)
        # nzxy = np.size(zxy.imag)
        # phste = np.zeros(nzxy,dtype=float)
        # for i in range(0,nzxy):
        phste = -torch.arctan2(zxy.imag, zxy.real)*180.0/math.pi

        return zxy,rhote,phste

    def mt2dzyx(self,freq,hxr,eyr):
        #compute the impedance, apparent resistivity and phase of TE mode 2-D Magnetotellurics(MT) forward modeling problem
        omega = 2.0 * math.pi*freq
        #compute the outputs
        zyx = eyr/hxr
        rhotm = abs(zyx)**2/(omega*self.miu)
        # nzyx = np.size(zyx.imag)
        # phstm = np.zeros(nzyx,dtype=float)
        # for i in] range(0,nzyx):
        phstm = 180 - torch.arctan2(zyx.imag, zyx.real)*180.0/np.pi
        # phstm = torch.arctan2(zyx.imag, zyx.real) * 180.0 / np.pi
            # phstm[i] = cm.phase(zyx[i])*180.0/np.pi

        return zyx,rhotm,phstm
    
    def mt1dte(self,freq,dz0,sig0,n_add):
        # n: points of interpolation
        #extend model
        omega = 2.0*math.pi*freq
        dz = torch.cat([dz0[i] / n_add * torch.ones(n_add, device=device, dtype=torch.float32) for i in range(dz0.size(0))])
        sig = torch.cat([sig0[i] * torch.ones(n_add, device=device, dtype=torch.float32) for i in range(dz0.size(0))])

        nz = sig.size(0)

        sig = torch.cat((sig, sig[nz - 1].unsqueeze(0)))
        dz = torch.cat((dz, torch.sqrt(2.0 / (sig[nz] * omega * self.miu)).unsqueeze(0)))

        diagA = (self.II*omega*self.miu*(sig[0:nz]*dz[0:nz]+sig[1:nz+1]*dz[1:nz+1]) - 2.0/dz[0:nz] - 2.0/dz[1:nz+1]).to(torch.complex128)
        # for ki in range(0,nz-1):
        offdiagA=(2.0/dz[1:nz])
        ##system matix
        # mtxA = scipa.diags(diagA,format='csc')+scipa.diags(offdiagA,1,format='csc')+scipa.diags(offdiagA,-1,format='csc')
        n = diagA.size(0)

        indices_main = torch.stack((torch.arange(n, device=device, dtype=torch.int64), torch.arange(n, device=device, dtype=torch.int64)))

        indices_off1 = torch.stack((torch.arange(n - 1, device=device, dtype=torch.int64), torch.arange(1, n, device=device, dtype=torch.int64)))

        indices_off2 = torch.stack((torch.arange(1, n, device=device, dtype=torch.int64), torch.arange(n - 1, device=device, dtype=torch.int64)))


        indices_all = torch.cat((indices_main, indices_off1, indices_off2), dim=1)
        values_all = torch.cat((diagA, offdiagA, offdiagA))

        mtxA = torch.sparse_coo_tensor(indices_all, values_all, size=(n, n))

        #compute right hand sides
        ##using boundary conditions:ex[0]=1.0,ex[nz-1]=0.0
        rhs = torch.zeros((nz,1), device=device, dtype=torch.complex128)
        rhs[0] = torch.tensor(-2.0, dtype=torch.complex128, device=device) / dz[0]

        # ex,_ = self.equation_solve(mtxA,rhs)
        # lup = scilg.splu(mtxA)
        # ex0 = lup.solve(rhs)
        solver = torch_spsolve.TorchSparseOp(mtxA)
        solver.factorize()
        ex0 = solver.solve(rhs)

        ex = torch.cat((torch.tensor([1.0], device=device, dtype=torch.complex128),ex0.reshape(-1)), dim=0)
        hy0 =((ex[1:]-ex[:-1])/dz[:-1]/self.II/omega/self.miu).to(torch.complex128)
        hy = torch.cat((hy0, hy0[-1:]), dim=0)

        idx = torch.arange(sig0.size(0)+1)*n_add
        # ex_n = np.concatenate((ex[idx],ex[:-1]))
        # hy_n = np.concatenate((hy[idx],hy[:-1]))
        # return ex_n,hy_n
        return ex[idx], hy[idx]

    def mt1dtm(self,freq,dz0,sig0,n_add):
        #extend model
        omega = 2.0*math.pi*freq
        # dz = np.array([dz0[i]/n_add*np.ones(n_add) for i in range(np.size(dz0))]).flatten()
        # sig = np.array([sig0[i]*np.ones(n_add) for i in range(np.size(dz0))]).flatten()
        dz = torch.cat([dz0[i] / n_add * torch.ones(n_add, device=device, dtype=torch.float32) for i in range(dz0.size(0))])
        sig = torch.cat([sig0[i] * torch.ones(n_add, device=device, dtype=torch.float32) for i in range(dz0.size(0))])

        nz = sig.size(0)

        # sig = np.hstack((sig,sig[nz-1]))
        # dz = np.hstack((dz,np.array(np.sqrt(2.0/(sig[nz]*omega*self.miu)),dtype=float)))
        sig = torch.cat((sig, sig[nz - 1].unsqueeze(0)))
        dz = torch.cat((dz, torch.sqrt(2.0 / (sig[nz] * omega * self.miu)).unsqueeze(0)))

        
        diagA = (self.II*omega*self.miu*(dz[0:nz]+dz[1:nz+1]) - 2.0/(dz[0:nz]*sig[0:nz]) - 2.0/(dz[1:nz+1]*sig[1:nz+1])).to(torch.complex128)
       
        offdiagA=2.0/(dz[1:nz]*sig[1:nz])
        
        ##system matix
        # mtxA = scipa.diags(diagA,format='csc')+scipa.diags(offdiagA,-1,format='csc')+scipa.diags(offdiagA,1,format='csc')
        n = diagA.size(0)

        indices_main = torch.stack((torch.arange(n, device=device, dtype=torch.int64), torch.arange(n, device=device, dtype=torch.int64)))

        indices_off1 = torch.stack((torch.arange(n - 1, device=device, dtype=torch.int64), torch.arange(1, n, device=device, dtype=torch.int64)))

        indices_off2 = torch.stack((torch.arange(1, n, device=device, dtype=torch.int64), torch.arange(n - 1, device=device, dtype=torch.int64)))

        indices_all = torch.cat((indices_main, indices_off1, indices_off2), dim=1)
        values_all = torch.cat((diagA, offdiagA, offdiagA))

        mtxA = torch.sparse_coo_tensor(indices_all, values_all, size=(n, n))

        #compute right hand sides
        ##using boundary conditions:ex[0]=1.0,ex[nz-1]=0.0
        # BCs
        # rhs = np.zeros((nz,1))
        # rhs[0] = -2.0/(dz[0]*sig[0])
        rhs = torch.zeros((nz,1), device=device, dtype=torch.complex128)
        rhs[0] = torch.tensor(-2.0, dtype=torch.complex128, device=device) / (dz[0]*sig[0])
    
        # hy,_ = self.equation_solve(mtxA,rhs)
        # lup = scilg.splu(mtxA)
        # hx0 = lup.solve(rhs)
        solver = torch_spsolve.TorchSparseOp(mtxA)
        solver.factorize()
        hx0 = solver.solve(rhs)

        # hx = np.concatenate(([complex(1,0)],hx0.reshape(-1)))
        hx = torch.cat((torch.tensor([1.0], device=device, dtype=torch.complex128), hx0.reshape(-1)), dim=0)
        ey0 = (hx[1:]-hx[:-1])/dz[:-1]/sig[:-1]
        # ey = np.concatenate((ey0,ey0[-1:]))
        ey = torch.cat((ey0, ey0[-1:]), dim=0)
        # idx = np.arange(np.size(sig0)+1)*n_add
        idx = torch.arange(sig0.size(0) + 1) * n_add
        # ey_n = np.concatenate((ey[idx],ey[:-1]))
        # hx_n = np.concatenate((hx[idx],hx[:-1]))
        # return ey_n,hx_n
        return ey[idx], hx[idx]
    
    def equation_solve(self,mtxA,rhs):
        '''
        solve Ax=b
        mtxA: A 
        rhs : b

        return:
        x: size(n,1)
        '''
        # bicgstab solver
#         ilu = scilg.spilu(mtxA)
#         M = scilg.LinearOperator(ilu.shape, ilu.solve)
#         # M = spar.diags(1. / mtx1, offsets = 0, format = 'csc')       
#         ex, exitCode = scilg.bicgstab(mtxA, rhs,maxiter=5000, M = M)

#         return ex, exitCode
#         lup = scilg.splu(mtxA)
#         ex = lup.solve(rhs)

        solver = torch_spsolve.TorchSparseOp(mtxA)
        solver.factorize()
        ex = solver.solve(rhs)
        # ex = ex.unsqueeze(1)
        # mtxA = scipa.diags((mtxA.to_dense()).cpu().numpy(), format='csc')
        #
        # lup = scilg.splu(mtxA)
        # ex = lup.solve(rhs.cpu().numpy())

        return ex, 0
        
def save_model(model_name,zn, yn, freq, ry, sig_log, rhoxy, phsxy,zxy,rhoyx,phsyx,zyx):
    '''
    save data as electrical model and field 
    for field, save as matrix with size of (n_model, n_obs, n_freq)

    '''
    scio.savemat(model_name,{'zn':zn, 'yn':yn, 'freq':freq, 'obs':ry,'sig':sig_log,
                            'rhoxy':rhoxy, 'phsxy':phsxy,'zxy':zxy,
                            'rhoyx':rhoyx,'phsyx':phsyx,'zyx':zyx})


def func_remote(nza, zn, yn, freq, ry, sig, mode="TETM"):
    result = []

    model = MT2DFD(nza, zn, yn, freq, ry)
    result.append(model.mt2d(sig, mode))

    return result

import torch


def torch_interp(x, xp, fp):

    assert torch.all(xp[:-1] <= xp[1:]), "xp must be monotonically increasing"

    indices = torch.searchsorted(xp, x).to(device)

    indices = torch.clamp(indices, 0, len(xp) - 2)

    x0 = xp[indices]
    x1 = xp[indices + 1]
    y0 = fp[indices]
    y1 = fp[indices + 1]

    slope = (y1 - y0) / (x1 - x0)
    y = y0 + slope * (x - x0)

    return y


if __name__ == '__main__':
    SigModel = np.ones((40, 80)) * 3
    SigModel[10:20, 10:25] = 1
    SigModel[15:25, 55:70] = 1
    SigModel = torch.tensor(SigModel).to(device)

    dx = dz = 25

    nz, nx = SigModel.shape
    print("Resampled Model Shape: {}, Grid Interval: {}m".format((nz, nx), dz))

    freeSurface = True  # free surface option for forward modeling
    # freq = torch.logspace(0, 3, 31).to(device)
    freq = np.logspace(np.log10(1e-3), np.log10(1e4), 31)  # dominant frequency of wavelet in Hz
    freq = torch.tensor(freq).float().to(device)
    nza = 10
    size_b = 5
    zn, yn, zn0, yn0, ry = SigCor(input_model=SigModel.cpu().detach().numpy(), dz_k=dz, dx_k=dx, nza=nza, size_b=size_b)

    sig_padded = SigPad(input_tensor=SigModel, size_b=size_b, nza=nza, device=device, mode='constant').to(torch.float32)

    zn = torch.tensor(zn).float().to(device)
    yn = torch.tensor(yn).float().to(device)
    zn0 = torch.tensor(zn0).float().to(device)
    yn0 = torch.tensor(yn0).float().to(device)
    ry = torch.tensor(ry).float().to(device)

    forward_rnn = MT2DFD(nza, zn, yn, freq, ry)

    Apres_truth, Phase_truth, _ = forward_rnn.mt2d(sig=sig_padded.to(device), mode='TE')