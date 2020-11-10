# Lagrangian Deep Learning

This repository is the official implementation of [Lagrangian Deep Learning](https://arxiv.org/abs/2010.02926). 

## Requirements

The code is built on the following packages:

[vmad](https://github.com/rainwoodman/vmad)
[nbodykit](https://github.com/bccp/nbodykit)
[fastpm-python](https://github.com/rainwoodman/fastpm-python)

## Training

To train LDL (at redshift 0) with FastPM input, run the following commands:

```train
#Run FastPM
python fastpmSim.py 1.0 0.5 0.0 --Nmesh=625 --Nstep=10 --save=FASTPM_PATH

#Calibrate FastPM matter distribution against TNGDark
python LDL.py --target='dm' --FastPMpath=FASTPM_Z0_PATH --TNGDarkpath=TNGDARK_PATH --snapNum=99 --Nmesh=625 --Nstep=1 --n=0 --save=IMPROVE_FASTPM_PATH 

#Train LDL
python LDL.py --target=TARGET --FastPMpath=FASTPM_Z0_PATH --improveFastPM=IMPROVE_FASTPM_FILE --TNGpath=TNG_PATH --snapNum=99 --Nmesh=625 --Nstep=2 --n=1 --save=SAVE_PATH   
```

To train LDL (at redshift 0) with TNGDark input, run the following commands:

```train
python LDL.py --target=TARGET --TNGpath=TNG_PATH --TNGDarkpath=TNGDARK_PATH --snapNum=99 --Nmesh=625 --Nstep=2 --n=1 --save=SAVE_PATH
```

##Evaluation

The LDL maps will be generated automatically after the training is finished. To generate LDL maps with pretrained parameters, run LDL.py with --restore and --evaluateOnly arguments.

