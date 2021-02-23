""" Part of this code is adapted from https://github.com/illustristng/illustris_python


Copyright (c) 2017, illustris & illustris_python developers All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the authors and should not be interpreted as representing official policies, either expressed or implied, of the FreeBSD Project.
"""

import numpy as np
import h5py
import six
from mpi4py import MPI
import os
from nbodykit.lab import BigFileMesh, FieldMesh

def partTypeNum(partType):
    """ Mapping between common names and numeric particle types. """
    if str(partType).isdigit():
        return int(partType)

    if str(partType).lower() in ['gas','cells']:
        return 0
    if str(partType).lower() in ['dm','darkmatter']:
        return 1
    if str(partType).lower() in ['tracer','tracers','tracermc','trmc']:
        return 3
    if str(partType).lower() in ['star','stars','stellar']:
        return 4 # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ['wind']:
        return 4 # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ['bh','bhs','blackhole','blackholes']:
        return 5

    raise Exception("Unknown particle type name.")


def snapPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = basePath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath = snapPath + 'snap_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    return filePath


def getNumPart(header):
    """ Calculate number of particles of all types given a snapshot header. """
    nTypes = 6

    nPart = np.zeros(nTypes, dtype=np.int64)
    for j in range(nTypes):
        nPart[j] = header['NumPart_Total'][j] | (header['NumPart_Total_HighWord'][j] << 32)

    return nPart


def loadSubset(basePath, snapNum, partType, fields=None, subset=None, mdi=None, writefile=None, sq=True, float32=True):
    """ Load a subset of fields for all particles/cells of a given partType.
        If offset and length specified, load only that subset of the partType.
        If mdi is specified, must be a list of integers of the same length as fields,
        giving for each field the multi-dimensional index (on the second dimension) to load.
          For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
          of y-Coordinates only, together with Masses.
        If writefile is not None, write the data to writefile.
        If sq is True, return a numpy array instead of a dict if len(fields)==1.
        If float32 is True, load any float64 datatype arrays directly as float32 (save memory). """
    result = {}

    ptNum = partTypeNum(partType)
    gName = "PartType" + str(ptNum)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(snapPath(basePath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())
        nPart = getNumPart(header)

        # decide global read size, starting file chunk, and starting file chunk offset
        if subset:
            offsetsThisType = subset['offsetType'][ptNum] - subset['snapOffsets'][ptNum, :]

            fileNum = np.max(np.where(offsetsThisType >= 0))
            fileOff = offsetsThisType[fileNum]
            numToRead = subset['lenType'][ptNum]
        else:
            fileNum = 0
            fileOff = 0
            numToRead = nPart[ptNum]

        result['count'] = numToRead

        if not numToRead:
            # print('warning: no particles of requested type, empty return.')
            return result

        # find a chunk with this particle type
        i = 1
        while gName not in f:
            f = h5py.File(snapPath(basePath, snapNum, i), 'r')
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        if len(fields) > 1:
            print('Cannot write several fields to file.')
            writefile = None

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Particle type ["+str(ptNum)+"] does not have field ["+field+"]")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception("Read error: mdi requested on non-2D field ["+field+"]")
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and float32: dtype = np.float32
            if not writefile:
                result[field] = np.zeros(shape, dtype=dtype)

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    if writefile:
        fw = open(writefile, 'wb')

    while numToRead:
        f = h5py.File(snapPath(basePath, snapNum, fileNum), 'r')

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff  = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff

        #print('['+str(fileNum).rjust(3)+'] off='+str(fileOff)+' read ['+str(numToReadLocal)+\
        #      '] of ['+str(numTypeLocal)+'] remaining = '+str(numToRead-numToReadLocal))

        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            if mdi is None or mdi[i] is None:
                if writefile:
                    (f[gName][field][fileOff:fileOff+numToReadLocal]).astype(dtype).tofile(fw)
                else:
                    result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            else:
                if writefile:
                    (f[gName][field][fileOff:fileOff+numToReadLocal, mdi[i]]).astype(dtype).tofile(fw)
                else:
                    result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal, mdi[i]]

        wOffset   += numToReadLocal
        numToRead -= numToReadLocal
        fileNum   += 1
        fileOff    = 0  # start at beginning of all file chunks other than the first

        f.close()

    # verify we read the correct number
    if origNumToRead != wOffset:
        raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(origNumToRead)+"]")

    if writefile:
        fw.close()
        return

    # only a single field? then return the array instead of a single item dict
    if sq and len(fields) == 1:
        return result[fields[0]]

    return result



def load_TNG_data(TNG_basepath, snapNum, partType, field, mdi=None, comm=MPI.COMM_WORLD):

    #The returned data will be distributed across different MPI ranks
    #mdi (0 or 1 or 2) needs to be provided for reading vector fields (e.g., coordinate / velocity).

    filename = TNG_basepath + '/snapdir_' + str(snapNum).zfill(3) + '/' + partType + '_' + field
    if mdi is not None:
        if mdi == 0:
            filename = filename + '_x'
        elif mdi == 1:
            filename = filename + '_y'
        elif mdi == 2:
            filename = filename + '_z'
        else:
            raise ValueError
        mdi = [mdi]

    if comm.rank == 0:
        if not os.path.exists(filename):
            loadSubset(TNG_basepath, snapNum, partType, fields=field, mdi=mdi, writefile=filename)   #TODO: loadSubset with MPI

    comm.Barrier()

    f = open(filename, 'rb')
    Np = int(os.path.getsize(filename) / 4)
    start = comm.rank * Np // comm.size
    end = (comm.rank + 1) * Np // comm.size
    f.seek(4 * start, os.SEEK_SET)
    particle = np.fromfile(f, dtype=np.float32, count=end-start)
    f.close()

    if field == 'Coordinates':
        particle /= 1000.
        particle %= 205.

    return particle


def n_cm3(mass, X, a, Nmesh):
    h = 0.6774
    XH = 0.76
    mp = 1.6726219e-27
    Msun10 = 1.989e40
    BoxSize = 205.
    Mpc_cm = 3.085678e24
    return Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass*XH/mp*X


def scalefactor(TNG_basepath, snapNum):
    with h5py.File(snapPath(TNG_basepath, snapNum), 'r') as f:
        header = dict(f['Header'].attrs.items())
        z = header['Redshift']
        a = 1. / (1. + z)
    return a


def load_TNG_map(TNG_basepath, snapNum, field, pm):

    assert field in ['dm', 'Mstar', 'ne', 'nT', 'nHI', 'neVz', 'MstarVz']

    #Try directly loading map. If it does not exist, paint the map and save it for future use.
    address = TNG_basepath + '/snapdir_' + str(snapNum).zfill(3) + '/' + field + 'map_Nmesh' + str(pm.Nmesh[0])
    try:
        TNGmap = BigFileMesh(address, dataset='Field').to_real_field()
    except:

        if field == 'dm':
            partType = 'dm'
        elif field in ['Mstar', 'MstarVz']:
            partType = 'stars'
        elif field in ['ne', 'nT', 'nHI', 'neVz']:
            partType = 'gas'

        if field == 'Mstar':
            mass = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='Masses')

        elif field == 'ne':
            gasmass = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='Masses')
            Xe = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='ElectronAbundance')
            a = scalefactor(TNG_basepath, snapNum) 
            mass = n_cm3(gasmass, Xe, a, pm.Nmesh[0])
            del gasmass, Xe

        elif field == 'nT':

            def temperature(Xe, u):
                XH = 0.76
                kb = 1.38064852e-23
                mp = 1.6726219e-27
                ufac = 1e6 #u: (km/s)^2 -> (m/s)^2
                mu = 4./(1.+3.*XH+4.*XH*Xe) * mp
                T = 2./3. * ufac * u / kb * mu
                return T

            Xe = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='ElectronAbundance')
            u = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='InternalEnergy')
            T = temperature(Xe, u)
            del u
            gasmass = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='Masses')
            a = scalefactor(TNG_basepath, snapNum) 
            ne = n_cm3(gasmass, Xe, a, pm.Nmesh[0])
            del gasmass, Xe
            mass = ne * T
            del ne, T

        elif field == 'nHI':
            gasmass = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='Masses')
            XHI = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='NeutralHydrogenAbundance')
            a = scalefactor(TNG_basepath, snapNum) 
            mass = n_cm3(gasmass, XHI, a, pm.Nmesh[0])
            del gasmass, XHI

        elif field == 'neVz':
            gasmass = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='Masses')
            Xe = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='ElectronAbundance')
            a = scalefactor(TNG_basepath, snapNum) 
            ne = n_cm3(gasmass, Xe, a, pm.Nmesh[0])
            vz = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='Velocities', mdi=2) * a**0.5
            mass = ne * vz
            del ne, vz

        elif field == 'MstarVz':
            mass = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='Masses')
            a = scalefactor(TNG_basepath, snapNum) 
            vz = load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='Velocities', mdi=2) * a**0.5
            mass = mass * vz
            del vz

        pos = []
        for mdi in range(3):
            pos.append(load_TNG_data(TNG_basepath=TNG_basepath, snapNum=snapNum, partType=partType, field='Coordinates', mdi=mdi))
        pos = np.array(pos).T

        layout = pm.decompose(pos)
        pos1 = layout.exchange(pos)
        if field == 'dm':
            mass1 = 1.0 * pm.Nmesh.prod() / pm.comm.allreduce(len(pos), op=MPI.SUM)
        else:
            mass1 = layout.exchange(mass)
            del mass
        del pos

        TNGmap = pm.create(type="real")
        TNGmap.paint(pos1, mass=mass1, layout=None, hold=False)
        del pos1, mass1
        
        FieldMesh(TNGmap).save(address)

    return TNGmap
