import numpy as np
from model import Displacement, LDL, smoothed_residue, lossfunc, loss_and_gradient, MemoizeJac 
from readTNG import load_TNG_map, load_TNG_data
from scipy.optimize import minimize
import time
import argparse
from nbodykit.lab import BigFileMesh, FieldMesh
from pmesh.pm import ParticleMesh
from bigfile import File
from mpi4py import MPI

parser = argparse.ArgumentParser()

parser.add_argument('--target', type=str, default='Mstar', choices=['dm', 'Mstar', 'nHI'], help='The target field.')

parser.add_argument('--FastPMpath', type=str, help='The path to load in FastPM particles. If not provided, TNGDark will be the input.')

parser.add_argument('--improveFastPM', type=str, help='The path to load in LDL parameters for improving FastPM matter distribution.')

parser.add_argument('--TNGpath', type=str, default='/global/cscratch1/sd/biwei/TNG300-1', help='The path to read TNG data.')

parser.add_argument('--TNGDarkpath', type=str, default='/global/cscratch1/sd/biwei/TNG300-3-Dark', help='The path to read TNG Dark data.')

parser.add_argument('--snapNum', type=int, default=99, help='The snapshot number of TNG and TNG Dark.')

parser.add_argument('--Nmesh', type=int, default=625, help='Mesh resolution of LDL.')

parser.add_argument('--Nstep', type=int, default=2, help='Number of displacement layers in LDL.')

parser.add_argument('--n', type=float, default=1., help='The hyperparameter n in the smoothing kernel. n determines the relative weight between the large scale and the small scale in the loss function.')

parser.add_argument('--save', type=str, default='/global/cscratch1/sd/biwei/LDL/', help='Where to save the optimized parameters.')

parser.add_argument('--restore', type=str, help='Path to load in LDL parameters.')

parser.add_argument('--evaluateOnly', action='store_true', help='The path to load in pretrained LDL parameters.')

args = parser.parse_args()

#Particle Mesh
pm = ParticleMesh(Nmesh=[args.Nmesh]*3, BoxSize=205, resampler='cic')
comm = pm.comm

#load input data
if args.FastPMpath:
    X = File(args.FastPMpath)['Position']
    start = comm.rank * X.size // comm.size
    end = (comm.rank+1) * X.size // comm.size
    X = np.array(X[start:end]).astype(np.float32)
    #improve FastPM matter distribution using LDL with given parameters
    if args.improveFastPM:
        param = np.loadtxt(args.improveFastPM)
        assert len(param) % 5 == 0
        Nstep = len(param) // 5
        model = Displacement.build(X=X, pm=pm, Nstep=Nstep)
        X = model.compute('X1', init=dict(param=param))
        del model
else:
    X = []
    for mdi in range(3):
        X.append(load_TNG_data(TNG_basepath=args.TNGDarkpath, snapNum=args.snapNum, partType='dm', field='Coordinates', mdi=mdi))
    X = np.array(X).T
    
if not args.evaluateOnly:
    #target map
    if args.target == 'dm':
        targetmap = load_TNG_map(TNG_basepath=args.TNGDarkpath, snapNum=args.snapNum, field=args.target, pm=pm)
    else:
        targetmap = load_TNG_map(TNG_basepath=args.TNGpath, snapNum=args.snapNum, field=args.target, pm=pm)

    #split among training, validation and test set
    index = pm.mesh_coordinates()
    Nmesh_test = int(0.44 * args.Nmesh)
    Nmesh_validate = args.Nmesh - Nmesh_test
    select_test = ((index[:,0]<Nmesh_test) & (index[:,1]<Nmesh_test) & (index[:,2]<Nmesh_test)).reshape(targetmap.shape)
    select_validate = ((index[:,0]>=Nmesh_test) & (index[:,1]<Nmesh_validate) & (index[:,2]<Nmesh_validate)).reshape(targetmap.shape)
    
    mask_train = np.ones_like(targetmap, dtype='?')
    mask_validate = np.zeros_like(targetmap, dtype='?')
    mask_test = np.zeros_like(targetmap, dtype='?')
    mask_train[select_test] = False
    mask_train[select_validate] = False
    mask_validate[select_validate] = True
    mask_test[select_test] = True
    
    #build LDL model
    if args.target == 'dm':
        baryon = False
        L1 = False  #L2 works better for dm
    else:
        baryon = True 
        L1 = True
    verbose = True

    residue_model = smoothed_residue.build(X=X, pm=pm, Nstep=args.Nstep, target=targetmap, n=args.n, baryon=baryon)
    loss_train_model = lossfunc.build(mask=mask_train, comm=comm, L1=L1)
    loss_validate_model = lossfunc.build(mask=mask_validate, comm=comm, L1=L1)
    save = args.save + '/' + args.target + '_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_bestparam.txt' % (args.Nmesh, args.Nstep, args.n)
    loss = MemoizeJac(loss_and_gradient, save=save, verbose=verbose, comm=comm)
    
    #initial guess
    if args.restore:
        x0 = np.loadtxt(args.restore)
        if args.target == 'dm':
            assert len(x0) == 5 * args.Nstep
        else:
            assert len(x0) == 5 * args.Nstep + 3
    else:
        x0 = [0.01, 0.5, 0.5, 5., 0.] * args.Nstep
        if baryon:
            x0 += [1., targetmap.csum() / comm.allreduce(len(X), op=MPI.SUM), 0.]
        x0 = np.array(x0)
    
    bounds = [(None, None), (1e-2,3), (0.03,2*np.pi*args.Nmesh/205.), (0.03,2*np.pi*args.Nmesh/205.), (-4,4)] * args.Nstep
    if baryon:
        bounds += [(1e-3,None), (0., None), (None, None)]
    
    #train
    res = minimize(loss, x0=x0, method='L-BFGS-B', args=(residue_model, loss_train_model, loss_validate_model), jac=loss.derivative, bounds=bounds, options={'maxiter': 2000, 'disp': False})
    if comm.rank == 0:
        print('Finished optimization.')
        print('Best validate loss:', loss.best_value)
        print('Best param:', loss.best_x)
    param = loss.best_x

else:
    param = np.loadtxt(args.restore)
    if args.target == 'dm':
        assert len(param) == 5 * args.Nstep
        baryon = False
    else:
        assert len(param) == 5 * args.Nstep + 3
        baryon = True

#evaluate
model = LDL.build(X=X, pm=pm, Nstep=args.Nstep, baryon=baryon)
LDLmap = model.compute('F', init=dict(param=param))

save = args.save + '/' + args.target + '_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_map' % (args.Nmesh, args.Nstep, args.n)
FieldMesh(LDLmap).save(save)
