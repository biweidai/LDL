from fastpm.core import StateVector, Solver, leapfrog 
from fastpm.background import MatterDominated
from nbodykit.cosmology import Planck15
from nbodykit.lab import ArrayCatalog
from pmesh.pm import ParticleMesh
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('output_redshift', nargs='+', type=float, help='The redshift of output snapshots.')

parser.add_argument('--Nmesh', type=int, default=625, help='The resolution of FastPM.')

parser.add_argument('--Nstep', type=int, default=10, help='Number of time steps of FastPM.')

parser.add_argument('--save', type=str, default='/global/cscratch1/sd/biwei/LDL', help='Path to save the snapshots.')

args = parser.parse_args()

#particle mesh
pm = ParticleMesh(BoxSize=205, Nmesh=[args.Nmesh]*3)
#use a finer mesh for generating IC
pm_IC = ParticleMesh(BoxSize=205, Nmesh=[2*args.Nmesh]*3)

#linear power spectrum
powerspectrum = np.loadtxt('input_spectrum_PLANCK15.txt')
logk = powerspectrum[:,0]
logpower = powerspectrum[:,1]
norm = 1.45426 #match sigma8

def Power(k):
    mask = k == 0
    k[mask] = 1.
    delta = np.interp(np.log10(k), logk, logpower)
    power = norm * (10 ** delta) * (2*np.pi**2) / k**3
    power[mask] = 0.
    k[mask] = 0.
    return power

#time steps in scale factor
stages = np.linspace(0.1, 1., args.Nstep, endpoint=True)

a_output = 1. / (np.array(args.output_redshift) + 1.)

#IC 2LPT
t = time.time()
solver_IC = Solver(pm_IC, Planck15, B=2)
wn = solver_IC.whitenoise(2695896)
dlin = solver_IC.linear(wn, Power)

Q = pm.generate_uniform_particle_grid(shift=0)
solver = Solver(pm, Planck15, B=2)
state = solver.lpt(dlin, Q, stages[0], order=2)

if pm.comm.rank == 0:
    print('Finish generating initial conditions with 2LPT. Time:', time.time()-t)

X0 = state.X
V0 = state.V
a0 = np.array(stages[0])


def monitor(action, ai, ac, af, state, event):
    if not state.synchronized: return

    global a0, X0, V0
    
    for a in a_output:
        if a > a0 and a <= af:
            #interpolate particle positions and velocities
            pt = MatterDominated(state.cosmology.Om0, a=[a0, a, af], a_normalize=state.solver.a_linear)
            fac1 = (pt.Gp(af) - pt.Gp(a)) / (pt.Gp(af) - pt.Gp(a0))
            fac2 = (pt.Gp(a) - pt.Gp(a0)) / (pt.Gp(af) - pt.Gp(a0))
            X = fac1 * X0 + fac2 * state.X
            fac1 = (pt.Gf(af) - pt.Gf(a)) / (pt.Gf(af) - pt.Gf(a0))
            fac2 = (pt.Gf(a) - pt.Gf(a0)) / (pt.Gf(af) - pt.Gf(a0))
            V = fac1 * V0 + fac2 * state.V

            #save the snapshot
            cat = ArrayCatalog({'Position' : X, 'Velocity' : V}, BoxSize=state.pm.BoxSize)
            cat.save(args.save + '/FastPM_Nmesh%d_Nstep%d_z%.2f' % (args.Nmesh, args.Nstep, 1./a-1.), ('Position', 'Velocity'))
            del X, V
            if state.pm.comm.rank == 0:
                print('Finish writing snapshot at redshift %.2f' % (1./a-1.), 'Time:', time.time()-t)

    a0 = np.array(af)
    X0 = state.X
    V0 = state.V

    if state.pm.comm.rank == 0:
        print('Finish redshift %.2f' % (1./af-1.), 'Time:', time.time()-t)

#run FastPM
state = solver.nbody(state, leapfrog(stages), monitor=monitor)
