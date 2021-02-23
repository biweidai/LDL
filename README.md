# Lagrangian Deep Learning

This repository is the official implementation of [Lagrangian Deep Learning](https://arxiv.org/abs/2010.02926). 

## Requirements

The code is built on the following packages:

[vmad](https://github.com/rainwoodman/vmad)  
[nbodykit](https://github.com/bccp/nbodykit)  
[fastpm-python](https://github.com/rainwoodman/fastpm-python)  

## Training

To train LDL (at redshift 0) with FastPM input, run the following commands:

```trainFastPM
#Run FastPM
python fastpmSim.py 1.0 0.5 0.0 --Nmesh=625 --Nstep=10 --save=FASTPM_PATH

#Calibrate FastPM matter distribution against TNGDark
python LDL.py --target='dm' --FastPMpath=FASTPM_Z0_PATH --TNGDarkpath=TNGDARK_PATH --snapNum=99 --Nmesh=625 --Nstep=1 --n=0 --save=IMPROVE_FASTPM_PATH 

#Train LDL on stellar mass / kSZ / HI
python LDL.py --target=TARGET --FastPMpath=FASTPM_Z0_PATH --improveFastPM=IMPROVE_FASTPM_FILE --TNGpath=TNG_PATH --snapNum=99 --Nmesh=625 --Nstep=2 --n=1 --save=SAVE_PATH   

#Train LDL on tSZ (or Xray)
python LDL.py --target=tSZ_ne --FastPMpath=FASTPM_Z0_PATH --improveFastPM=IMPROVE_FASTPM_FILE --TNGpath=TNG_PATH --snapNum=99 --Nmesh=625 --Nstep=2 --n=1 --save=SAVE_PATH   
python LDL.py --target=tSZ_T --FastPMpath=FASTPM_Z0_PATH --improveFastPM=IMPROVE_FASTPM_FILE --TNGpath=TNG_PATH --snapNum=99 --Nmesh=625 --Nstep=2 --n=1 --save=SAVE_PATH --restore_ne=PARAM_NE_PATH 
```

To train LDL (at redshift 0) with TNGDark input, run the following commands:

```trainTNG
#Train LDL on stellar mass / kSZ / HI
python LDL.py --target=TARGET --TNGpath=TNG_PATH --TNGDarkpath=TNGDARK_PATH --snapNum=99 --Nmesh=625 --Nstep=2 --n=1 --save=SAVE_PATH

#Train LDL on tSZ (or Xray)
python LDL.py --target=tSZ_ne --TNGpath=TNG_PATH --TNGDarkpath=TNGDARK_PATH --snapNum=99 --Nmesh=625 --Nstep=2 --n=1 --save=SAVE_PATH   
python LDL.py --target=tSZ_T --TNGpath=TNG_PATH --TNGDarkpath=TNGDARK_PATH --snapNum=99 --Nmesh=625 --Nstep=2 --n=1 --save=SAVE_PATH --restore_ne=PARAM_NE_PATH 
```

Note 1: For the first time training, the code needs to read TNG data from HDF5 files for painting the target map. This part does not support MPI (only rank 0 reads the files, and the other ranks wait), and is quite slow (depending on the data type and redshift). To paint the target map it also needs to load in all the particles, which takes lots of memory (more than the training code). The target map will be saved, and for the next time the map will be directly loaded without the need of reading the particles and painting the map. In other words, the first time training takes more time and memory because it needs to read the hydro particles and paint the target map.
Note 2: The training of LDL depends on the starting point x0, since the likelihood is not convex, and the BFGS optimizer only finds the local minima. If the results are not satisfying, try changing the values of x0 and train again.

## Evaluation

The LDL maps will be generated automatically after the training is finished. To generate LDL maps with pretrained parameters, run LDL.py with --restore and --evaluateOnly arguments.

## Pretrained models

The best fit parameters for reproducing the results in the [paper](https://arxiv.org/abs/2010.02926) will be uploaded soon. 
