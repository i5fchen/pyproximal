import sys
import lbfgs

from functools import partial
from scipy.optimize import minimize
from pylops.basicoperators import MatrixMult
from pyproximal.proximal import L2, Nonlinear
from pyproximal.optimization.primal import ADMM

from numpy.linalg import norm

import numpy as np
from mpi4py import MPI 

from mpi4py import rc
rc.initialize = True
rc.finalize = False

# Initialize MPI communication
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Initialize L-BFGS optimizer
opt = lbfgs.LBFGS()
opt.linesearch = 'default' 
opt.m = 10  

class x2tiledqlr_dense_mmm(Nonlinear):
    """
    Class to form a dense matrix X from tiled Q, L, and R matrices, 
    and calculate the gradient of the loss for each tile.
    """
    def setup(self, Op, d, size, ranks):
        """
        Op: Operator matrix related to down-going
        d: data related to up-going
        size: The optimal tile size determined based on the benchmark solution 
        ranks: Array of rank values for the tiles
        """
        self.Op, self.d = Op, d

        # Tile size determines the partitioning of the unknown matrix X
        self.tile_size = size 
        
        # Ranks represents the dimensions of the block matrix
        self.tile_dim = ranks.shape
        self.ranks = ranks

        # Prepare a dense matrix for the solution
        n_dense = Op.otherdims[0]
        self.x = np.empty_like(self.Op, dtype=Op.dtype, shape=(n_dense, n_dense))

    def fun(self, ixx):
        """
        Compute the loss function, which is based on matrix multiplication.
        ixx: Input vector to reshape and compute matrix operations.
        Its first half is real and the second half is for imaginary of the complex X
        """
        for i in range(self.tile_dim[0]):
            for j in range(i+1):

                nk = self.ranks[i, j]
                num = nk * self.tile_size
                num *= 2

                if i == j:
                    # Diagonal block handling
                    pp = ixx[:num]
                    pp = pp[:num // 2] + pp[num // 2:] * 1j
                    pp = pp.reshape(self.tile_size, nk)
                    ixx = ixx[num:]

                    if nk == self.tile_size:
                        ret = 0.5 * (pp + pp.T)  # Symmetry for full-rank tile 
                    else:
                        ret = pp.dot(pp.T) # Symmetry with Q.dot(Q.T)

                    self.x[i * self.tile_size:(i + 1) * self.tile_size, 
                           j * self.tile_size:(j + 1) * self.tile_size] = ret

                else:
                    # Off-diagonal block handling
                    pp = ixx[:num]
                    pp = pp[:num // 2] + pp[num // 2:] * 1j
                    pp = pp.reshape(self.tile_size, nk)
                    ixx = ixx[num:]

                    qq = ixx[:num]
                    qq = qq[:num // 2] + qq[num // 2:] * 1j
                    qq = qq.reshape(nk, self.tile_size)
                    ixx = ixx[num:]

                    ret = pp.dot(qq)
                    
                    # Fill the symmetric matrix blocks
                    self.x[i * self.tile_size:(i + 1) * self.tile_size, 
                           j * self.tile_size:(j + 1) * self.tile_size] = ret
                    self.x[j * self.tile_size:(j + 1) * self.tile_size, 
                           i * self.tile_size:(i + 1) * self.tile_size] = ret.T

        # Compute residual and loss
        self.res = self.Op.dot(self.x) - self.d
        loss = 0.5 * norm(self.res.ravel()) ** 2
        return loss

    def grad(self, ixx):
        """
        Compute the gradient of the loss function with respect to ixx.
        ixx: Input vector for gradient computation
        """
        ata = self.Op.T.conj().dot(self.res)
        ata = ata + ata.T  # Symmetric part of the gradient
        ggg = []

        for i in range(self.tile_dim[0]):
            for j in range(i + 1):
               
                iata = ata[i * self.tile_size:(i + 1) * self.tile_size, 
                           j * self.tile_size:(j + 1) * self.tile_size]
                nk = self.ranks[i, j]
                num = nk * self.tile_size * 2

                if i == j:
                    if nk == self.tile_size:
                        gg = iata + iata.T
                        gg *= 0.5  
                    else:
                        pp = ixx[:num]
                        ixx = ixx[num:]
                        pp = pp[:num // 2] + pp[num // 2:] * 1j
                        pp = pp.reshape(self.tile_size, nk)
                        gg = iata.dot(pp.conj())

                    gg = gg.ravel()
                    ggg.append(np.concatenate((gg.real, gg.imag)))

                else:
                   
                    pp = ixx[:num]
                    pp = pp[:num // 2] + pp[num // 2:] * 1j
                    pp = pp.reshape(self.tile_size, nk)
                    ggr = pp.T.conj().dot(iata).ravel()
                    ixx = ixx[num:]

                    qq = ixx[:num]
                    qq = qq[:num // 2] + qq[num // 2:] * 1j
                    qq = qq.reshape(nk, self.tile_size)
                    ixx = ixx[num:]
                    ggl = iata.dot(qq.conj().T).ravel()
                    ggg.append(np.concatenate((ggl.real, ggl.imag)))
                    ggg.append(np.concatenate((ggr.real, ggr.imag)))

        ggg = np.concatenate(ggg)
        return ggg

    def fungrad(self, x):
        """
        Return both the function value and the gradient for optimization.
        x: Input parameter vector
        """
        return self.fun(x), self.grad(x)

    def _fungradprox(self, x, g, tau):
        """
        Proximal gradient update for optimization.
        x: Current value of x
        g: Gradient at current x
        tau: Hyperparameter for ADMM
        """
        f, g_ = self.fungrad(x)
        f = f + 1. / (2 * tau) * ((x - self.y) ** 2).sum()
        g[:] = g_ + 1. / tau * (x - self.y)
        return f

    def optimize(self):
        """
        Perform optimization using L-BFGS algorithm.
        """
        fg = partial(self._fungradprox, tau=self.tau)
        sol = lbfgs.fmin_lbfgs(fg, x0=self.x0, max_iterations=10)
        return sol


# Scaling factor for all frequencies
scale = 0.04  # Largest amplitude from the dominant frequency of operator side

# Frequency index based on MPI rank
idx_fname = rank + 10  # Frequency index, chosen empirically to dump uninformative frequencies

# Load tiling plan (ranks and tile sizes)
ranks = np.load("./input/tiling_plan_eps1en2_%d.npz" % idx_fname)
ivv, itile = ranks['arr_0'], ranks['arr_1']
ivv = ivv.astype('int32')
itile = itile.astype('int32')

# Load pre-calculated Hilbert curve for receiver geometry
idx_r = np.load('./input/hilbert_index_rec.npy', allow_pickle=1)

ntile = ivv.shape[0]  # Number of tiles

# Load down- and up-going matrices and normalize by scale
ia = np.load('./input/Freq%d.npy' % idx_fname)[::, idx_r] / scale
ib = np.load('./input/PUPFreq%d.npy' % idx_fname)[::jj, idx_r] / scale

ia = ia.astype('complex64')
ib = ib.astype('complex64')

ns, nr = ia.shape
Op = MatrixMult(ia, otherdims=(nr,), dtype=np.complex64)

# Regularization parameters and ADMM iterations
lam = [10, 1, 0.1]
admm_niter = [5, 10, 25]

# Load initial guess for the solution
x0 = np.load("./input/init_from_benchX_4_x2qlr_%d_lam_0.10_perc100_itile%d.npy" % (idx_fname, itile), allow_pickle=1)

verbose = False
tau_init = float(sys.argv[-1])  # rho = 1.0 / tau

for ilam_idx, ilam in enumerate(lam):
    # Set up the Frobenious proximal operator
    prox_F = L2(sigma=ilam)

    # Initialize the problem setup
    fnl = x2tiledqlr_dense_mmm(x0=x0, warm=True)
    fnl.setup(Op, ib, itile, ivv)
    
    # Run the ADMM optimization
    if idx_fname == 90:
        verbose = True  

    # Run ADMM optimization
    x0, z0, primal_res, dual_res, pfloss = ADMM(
        fnl, prox_F,
        tau=tau_init,
        x0=x0,
        niter=admm_niter[ilam_idx], show=verbose)

    # Save the results to output files
    np.save('./output/Xqlr_%d_lam_%f_eps1en2' % (idx_fname, ilam), x0)
    np.save('./output/primal_res_%d_tau_%f_lam_%f_constrho' % (idx_fname, tau_init, ilam), primal_res)
    np.save('./output/dual_res_%d_tau_%f_lam_%f_constrho' % (idx_fname, tau_init, ilam), dual_res)
    np.save('./output/pfloss_%d_tau_%f_lam_%f_constrho' % (idx_fname, tau_init, ilam), pfloss)
