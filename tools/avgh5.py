#!/usr/bin/python 

import sys
import numpy as np
import h5py as h5

argv = sys.argv
if len(argv) != 2:
    print "usage: %s <file.h5>" % argv[0]
    exit(1)

f = h5.File(argv[1], "r")

vx = f['u']
(nx, ny, nz, nu) = vx.shape

vx = vx.value
vx = vx.reshape(nx, ny, nz)
vx =  np.sum(vx, (0,1)) / (nx * ny)

zz = np.arange(nz) - nz/2

vx = vx.reshape(nz, 1)
zz = zz.reshape(nz, 1)
np.savetxt(sys.stdout, np.concatenate((zz, vx), axis=1));

f.close()
