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

parser.add_argument('--target', type=str, default='Mstar', choices=['dm', 'Mstar', 'nHI', 'kSZ', 'tSZ_ne', 'tSZ_T', 'Xray_ne', 'Xray_T'], help='The target field.')

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

parser.add_argument('--restore_ne', type=str, help='Path to load in the LDL parameters for ne. Only applicable when fitting the temperature of tSZ and Xray.')

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
        assert args.FastPMpath
        targetmap = load_TNG_map(TNG_basepath=args.TNGDarkpath, snapNum=args.snapNum, field=args.target, pm=pm)
    elif args.target in ['Mstar', 'nHI']:
        targetmap = load_TNG_map(TNG_basepath=args.TNGpath, snapNum=args.snapNum, field=args.target, pm=pm)
    elif args.target == 'kSZ':
        targetmap = load_TNG_map(TNG_basepath=args.TNGpath, snapNum=args.snapNum, field='ne', pm=pm)
        targetmap *= 1e5 #The values are too small. Multiply the field with 1e5.
    elif args.target in ['tSZ_ne', 'Xray_ne']:
        targetmap_ne = load_TNG_map(TNG_basepath=args.TNGpath, snapNum=args.snapNum, field='ne', pm=pm)
        targetmap = load_TNG_map(TNG_basepath=args.TNGpath, snapNum=args.snapNum, field='nT', pm=pm)
        select = targetmap_ne <= 0
        targetmap_T = targetmap / targetmap_ne
        targetmap_T[select] = 0
        targetmap_ne *= 1e5 #The values are too small. Multiply the field with 1e5.
        targetmap_T *= 1e-5 #The values are too large. Multiply the field with 1e-5.
        if args.target == 'Xray_ne':
            targetmap = targetmap_ne**2 * targetmap_T**0.5 
        bias = targetmap_ne.csum() / comm.allreduce(len(X), op=MPI.SUM) 
        del targetmap_ne
    elif args.target in ['tSZ_T', 'Xray_T']:
        targetmap_ne = load_TNG_map(TNG_basepath=args.TNGpath, snapNum=args.snapNum, field='ne', pm=pm)
        targetmap_ne *= 1e5 #The values are too small. Multiply the field with 1e5.
        targetmap = load_TNG_map(TNG_basepath=args.TNGpath, snapNum=args.snapNum, field='nT', pm=pm)
        select = targetmap_ne <= 0
        map_T = targetmap / targetmap_ne
        map_T[select] = 0
        bias = map_T.csum() / comm.allreduce(len(X), op=MPI.SUM) 
        if args.target == 'Xray_T':
            targetmap = targetmap_ne**1.5 * targetmap**0.5 
        del targetmap_ne, map_T

        param = np.loadtxt(args.restore_ne)
        assert len(param) % 5 == 3
        Nstep = len(param) // 5
        model = LDL.build(X=X, pm=pm, Nstep=Nstep, baryon=True)
        map_ne = model.compute('F', init=dict(param=param))
        del model

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

    if args.target == 'tSZ_ne':
        residue_model = smoothed_residue.build(X=X, pm=pm, Nstep=args.Nstep, target=targetmap, n=args.n, baryon=baryon, index=1, field2=targetmap_T)
    elif args.target == 'tSZ_T':
        residue_model = smoothed_residue.build(X=X, pm=pm, Nstep=args.Nstep, target=targetmap, n=args.n, baryon=baryon, index=1, field2=map_ne)
    elif args.target == 'Xray_ne':
        residue_model = smoothed_residue.build(X=X, pm=pm, Nstep=args.Nstep, target=targetmap, n=args.n, baryon=baryon, index=2, field2=targetmap_T**0.5)
    elif args.target == 'Xray_T':
        residue_model = smoothed_residue.build(X=X, pm=pm, Nstep=args.Nstep, target=targetmap, n=args.n, baryon=baryon, index=0.5, field2=map_ne**2)
    else:
        residue_model = smoothed_residue.build(X=X, pm=pm, Nstep=args.Nstep, target=targetmap, n=args.n, baryon=baryon, index=1, field2=None)
    save = args.save + '/' + args.target + '_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_bestparam.txt' % (args.Nmesh, args.Nstep, args.n)
    loss_train_model = lossfunc.build(mask=mask_train, comm=comm, L1=L1)
    loss_validate_model = lossfunc.build(mask=mask_validate, comm=comm, L1=L1)
    loss = MemoizeJac(loss_and_gradient, save=save, verbose=verbose, comm=comm)
    
    #initial guess
    if args.restore:
        x0 = np.loadtxt(args.restore)
        if args.target == 'dm':
            assert len(x0) == 5 * args.Nstep
        else:
            assert len(x0) == 5 * args.Nstep + 3
    else:
        x0 = [0.001, 0.5, 1., 8., 0.] * args.Nstep
        if baryon:
            if args.target in ['tSZ_ne', 'tSZ_T', 'Xray_ne', 'Xray_T']:
                x0 += [1., bias, 0.]
            else:
                x0 += [1., targetmap.csum() / comm.allreduce(len(X), op=MPI.SUM), 0.]
        x0 = np.array(x0)
    
    bounds = [(None, None), (0.05,2), (0.03,2*np.pi*args.Nmesh/205.), (0.03,2*np.pi*args.Nmesh/205.), (-4.5,4.5)] * args.Nstep
    if baryon:
        bounds += [(0.1,None), (0., None), (None, None)]
    
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

if args.target in ['dm', 'Mstar', 'nHI']:
    save = args.save + '/' + args.target + '_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_map' % (args.Nmesh, args.Nstep, args.n)
    FieldMesh(LDLmap).save(save)

elif args.target == 'kSZ':
    LDLmap *= 1e-5 #The learned LDL map is actually 1e5*map_ne.
    save = args.save + '/ne_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_map' % (args.Nmesh, args.Nstep, args.n)
    FieldMesh(LDLmap).save(save)
    if args.FastPMpath:
        X = File(args.FastPMpath)['Position']
        X = np.array(X[start:end]).astype(np.float32)
        Vz = File(args.FastPMpath)['Velocity']
        Vz = np.array(Vz[start:end])[:,2].astype(np.float32)
    else:
        from readTNG import scalefactor
        a = scalefactor(args.TNGDarkpath, args.snapNum)
        Vz = load_TNG_data(TNG_basepath=args.TNGDarkpath, snapNum=args.snapNum, partType='dm', field='Velocities', mdi=2) * a**0.5
    layout = pm.decompose(X)
    X = layout.exchange(X)
    Vz = layout.exchange(Vz)
    map_vz = pm.create(type="real")
    map_vz.paint(X, mass=Vz, layout=None, hold=False)
    map_delta = pm.create(type="real")
    map_delta.paint(X, mass=1., layout=None, hold=False)
    select = map_delta > 0
    map_vz[select] = map_vz[select] / map_delta[select]
    map_vz[~select] = 0
    
    LDLmap = LDLmap * map_vz
    save = args.save + '/' + args.target + '_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_map' % (args.Nmesh, args.Nstep, args.n)
    FieldMesh(LDLmap).save(save)

elif args.target in ['tSZ_T', 'Xray_T']:
    param = np.loadtxt(args.restore_ne)
    assert len(param) % 5 == 3
    Nstep = len(param) // 5
    model = LDL.build(X=X, pm=pm, Nstep=Nstep, baryon=True)
    map_ne = model.compute('F', init=dict(param=param))
    del model

    if args.target == 'tSZ_T':
        LDLmap = LDLmap * map_ne #the 1e5 factor in ne map and 1e-5 factor in T map cancels.
        save = args.save + '/tSZ_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_map' % (args.Nmesh, args.Nstep, args.n)
    elif args.target == 'Xray_T':
        LDLmap = (LDLmap*1e5)**0.5 * (map_ne*1e-5)**2
        save = args.save + '/Xray_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_map' % (args.Nmesh, args.Nstep, args.n)
    FieldMesh(LDLmap).save(save)
