import numpy as np
from mpi4py import MPI
from vmad.lib import fastpm, unary, linalg, mpi
from vmad import autooperator, operator
from vmad.core.symbol import Literal
from vmad.core.stdlib import eval
from vmad.core.stdlib.operators import binary
import time
import gc
from collections import OrderedDict

@operator
class ReLU:
    aout={'y' : 'RealField'}
    ain={'x': 'RealField'}

    def apl(node, x):
        y = x.copy()
        y[y<=0] = 0.
        return dict(y=y)

    def vjp(node, _y, x):
        _x = _y.copy()
        mask = x <= 0
        _x[mask] = 0
        return dict(_x=_x)
        
@operator
class masking:
    #mask the field for training-validation-test split
    aout={'y' : 'RealField'}
    ain={'x': 'RealField'}

    def apl(node, x, mask):
        y = x.copy()[mask]
        return dict(y=y)

    def vjp(node, _y, x, mask):
        _x = np.zeros_like(x)
        _x[mask] = _y
        return dict(_x=_x)
        
@operator
class complex_mul(binary):
    def apl(node, x1, x2):
        return dict(y = x1 * x2)

    def vjp(node, _y, x1, x2):
        return dict(_x1 = _y * np.conj(x2),
                    _x2 = _y * np.conj(x1))

@operator
class compensate2factor:
    #To get correct gradients in Fourier space
    aout={'y' : 'ComplexField'}
    ain={'x': 'ComplexField'}

    def apl(node, x):
        y = x.copy()
        return dict(y=y)

    def vjp(node, _y):
        _x = _y.pm.create(type='complex')
        for i, a, b in zip(_x.slabs.i, _x.slabs, _y.slabs):
            # modes that are self conjugates do not gain a factor
            mask = np.ones(a.shape, '?')
            mask &= ((_x.Nmesh[-1] - i[-1]) % _x.Nmesh[-1] == i[-1])
            a[~mask] = b[~mask] + np.conj(b[~mask])
            a[mask] = b[mask]
        return dict(_x=_x)
        

@autooperator('param->X1')
def Displacement(param, X, pm, Nstep):

    #Lagrangian displacement

    #normalization constant for overdensity 
    fac = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(X), op=MPI.SUM)

    for i in range(Nstep):

        #move the particles across different MPI ranks
        layout = fastpm.decompose(X, pm)
        xl = fastpm.exchange(X, layout)
        delta = fac * fastpm.paint(xl, 1.0, None, pm)

        #take parameters
        alpha = linalg.take(param, 5*i, axis=0)
        gamma = linalg.take(param, 5*i+1, axis=0)
        kl = linalg.take(param, 5*i+2, axis=0)
        ks = linalg.take(param, 5*i+3, axis=0)
        n = linalg.take(param, 5*i+4, axis=0)
        
        #delta**gamma
        gamma = mpi.allbcast(gamma, comm=pm.comm)
        gamma = linalg.broadcast_to(gamma, eval(delta, lambda x : x.shape)) 
        delta = (delta+1e-8) ** gamma

        #Fourier transform
        deltak = fastpm.r2c(delta)

        #Green's operator in Fourier space
        Filter = Literal(pm.create(type='complex', value=1).apply(lambda k, v: k.normp(2, zeromode=1e-8) ** 0.5))
        kl = mpi.allbcast(kl, comm=pm.comm)
        kl = linalg.broadcast_to(kl, eval(Filter, lambda x : x.shape)) 
        ks = mpi.allbcast(ks, comm=pm.comm)
        ks = linalg.broadcast_to(ks, eval(Filter, lambda x : x.shape)) 
        n = mpi.allbcast(n, comm=pm.comm)
        n = linalg.broadcast_to(n, eval(Filter, lambda x : x.shape)) 
        
        Filter = - unary.exp(-Filter**2/ks**2) * unary.exp(-kl**2/Filter**2) * Filter**n
        Filter = compensate2factor(Filter) 

        p = complex_mul(deltak, Filter)

        #gradient of potential
        r1 = []
        for d in range(pm.ndim):
            dx1_c = fastpm.apply_transfer(p, fastpm.fourier_space_neg_gradient(d, pm, order=1))
            dx1_r = fastpm.c2r(dx1_c)
            dx1l = fastpm.readout(dx1_r, xl, None)
            dx1 = fastpm.gather(dx1l, layout)
            r1.append(dx1)

        #displacement
        S = linalg.stack(r1, axis=-1)
        alpha = mpi.allbcast(alpha, comm=pm.comm)
        alpha = linalg.broadcast_to(alpha, eval(S, lambda x : x.shape)) 
        S = S * alpha

        X = X+S
        
    return X


@autooperator('param->F')
def LDL(param, X, pm, Nstep, baryon=True):

    fac = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(X), op=MPI.SUM)

    X = Displacement(param, X, pm, Nstep)

    #paint particle overdensity field
    layout = fastpm.decompose(X, pm)
    Xl = fastpm.exchange(X, layout)
    delta = fac * fastpm.paint(Xl, 1., None, pm)

    if baryon:
        #take parameters
        gamma = linalg.take(param, 5*Nstep, axis=0)
        bias0 = linalg.take(param, 5*Nstep+1, axis=0)
        bias1 = linalg.take(param, 5*Nstep+2, axis=0)
    
        gamma = mpi.allbcast(gamma, comm=pm.comm)
        gamma = linalg.broadcast_to(gamma, eval(delta, lambda x : x.shape))
        bias0 = mpi.allbcast(bias0, comm=pm.comm)
        bias0 = linalg.broadcast_to(bias0, eval(delta, lambda x : x.shape))
        bias1 = mpi.allbcast(bias1, comm=pm.comm)
        bias1 = linalg.broadcast_to(bias1, eval(delta, lambda x : x.shape))
    
        #Field transformation
        F = ReLU(bias0 * (delta+1e-8) ** gamma + bias1)
    else:
        F = delta
    
    return F
    

def smoothing(n):
    def kernel(k, v):
        kk = sum(ki ** 2 for ki in k)
        kk = kk ** 0.5
        mask = kk == 0
        kk[mask] = 1
        b = v * (kk**(-n)+1.)
        b[mask] = v[mask]
        return b
    return kernel

@autooperator('param->residue')
def smoothed_residue(param, X, pm, Nstep, target, n, baryon=True):

    F = LDL(param, X, pm, Nstep, baryon=baryon)

    #residue field
    residue = F - target 
    
    #smooth the field
    Filter = pm.create(type='complex', value=1).apply(smoothing(n=n))
    residuek = fastpm.r2c(residue)
    residuek = residuek * Filter
    residue = fastpm.c2r(residuek)

    return residue


@autooperator('residue->loss')
def lossfunc(residue, mask, comm=MPI.COMM_WORLD, L1=True):
    
    #If L1=False, use L2 loss

    residue = unary.absolute(residue)
    loss = masking(residue, mask)
    Npixel = np.sum(mask)
    if L1:
        loss = linalg.sum(loss)
    else:
        loss = linalg.sum(loss**2)
    loss = mpi.allreduce(loss, comm=comm)
    Npixel = mpi.allreduce(Npixel, comm=comm)
    loss = loss / Npixel
    return loss



def loss_and_gradient(param, residue_model, loss_train_model, loss_validate_model=None):

    #Calculate the loss function and the gradient, given parameters and vmad models.
    
    #calculate residue field and prepare the tape for back propogation
    residue, residue_tape = residue_model.compute('residue', init=dict(param=param), return_tape=True)
    vjpvout = residue_tape.get_vjp_vout()
    vjp = residue_tape.get_vjp()
    
    #calculate the loss and the gradient w.r.t. residue field for training data
    loss_train, _residue = loss_train_model.compute_with_vjp(init=dict(residue=residue), v=dict(_loss=1.0))
    loss_train = loss_train[0]
    _residue = _residue[0]
    
    #gradient w.r.t. param
    gradient = vjp.compute(vjpvout, init=OrderedDict(dict(_residue=_residue)))
    gradient = gradient[0]

    #loss for validation data
    if loss_validate_model:
        loss_validate = loss_validate_model.compute('loss', init=dict(residue=residue))
    else:
        loss_validate = None

    gc.collect()
    return loss_train, gradient, loss_validate 


class MemoizeJac:
    """ Apapted from https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
        Decorator that caches the return values of a function returning `(fun, grad, fun_validate)`
        each time it is called. """

    def __init__(self, fun, save=None, verbose=True, comm=MPI.COMM_WORLD):
        self.fun = fun
        self.jac = None
        self._value = None
        self.x = None

        self.best_value = np.inf
        self.best_x = None
        self.save = save
        self.verbose = verbose
        self.comm = comm

    def _compute_if_needed(self, x, *args):
        if not np.all(x == self.x) or self._value is None or self.jac is None:
            self.x = np.asarray(x).copy()
            t = time.time()
            fg = self.fun(x, *args)
            self.jac = fg[1]
            self._value = fg[0]
            
            if fg[2] is not None:
                if self.best_value > fg[2]:
                    self.best_value = fg[2]
                    self.best_x = np.asarray(x).copy()
                    if self.save:
                        np.savetxt(self.save, self.best_x)
            elif self.best_value > fg[0]:
                self.best_value = fg[0]
                self.best_x = np.asarray(x).copy()
                if self.save:
                    np.savetxt(self.save, self.best_x)

            if self.verbose:
                if self.comm.rank == 0:
                    if fg[2] is not None:
                        print('Training Loss:', self._value, 'Validation Loss:', fg[2], 'Best Validation Loss:', self.best_value, 'Time:', time.time()-t)
                    else:
                        print('Training Loss:', self._value, 'Best Training Loss:', self.best_value, 'Time:', time.time()-t)
                    print('Parameter:', self.x)
                    print('Gradient:', self.jac)
                    print()

    def __call__(self, x, *args):
        """ returns the the function value """
        self._compute_if_needed(x, *args)
        return self._value

    def derivative(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.jac
