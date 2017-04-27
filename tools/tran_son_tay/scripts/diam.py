import numpy as np
import glob
from numpy import linalg as la
from argparse import ArgumentParser
from plyfile import PlyData
from math import floor


def read_ply_file(ply):
    vertex = ply['vertex']
    x, y, z = (vertex[t] for t in ('x', 'y', 'z'))
    return x, y, z


def get_rot(data):
    dim, n = data.shape
    I = np.zeros((dim, dim))
    for i in range(0, n):
        ri = data[:,i]
        I += np.sum(ri**2)*np.identity(dim) - np.kron(ri, ri).reshape(dim, dim)
    e, v = la.eig(I)
    # idx = np.argsort(e)
    return v[:, idx]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ply')
    args = parser.parse_args()

    listing = glob.glob(args.ply+"/rbcs-*.ply")
    n = len(listing)
    d = np.zeros((2, n))
    dm = 0; df = 0; si = 0
    ch = int(floor(n/10))
    for i in range(0,n):
        fullpath = listing[i]
        x, y, z = read_ply_file(PlyData.read(fullpath))
        xyz = np.array([x,y,z])

        # subtract COM
        com = xyz.mean(axis=1); xyz -= com[:, np.newaxis]

        # rotate data
        rot = get_rot(xyz)
        xyz_new = np.matrix(la.inv(rot))*np.matrix(xyz)

        xyz_new[0, :].sort(); xyz_new[1, :].sort()
        d[0, i] = np.mean(xyz_new[0, -6:-1]) - np.mean(xyz_new[0, 0:5])
        d[1, i] = np.mean(xyz_new[1, -6:-1]) - np.mean(xyz_new[1, 0:5])

        # check whether we're in a steady state
        if ch > 0 and (i+1) % ch == 0:
            cdm = np.mean(d[0, i-ch+1:i])
            if not(df) and abs(dm-cdm) < 0.02*dm:
                print "Steady state reached after", i, "steps out of", n
                df = 1; si = i
            else:
                dm = cdm

    m1 = np.mean(d[0,si:n])
    s1 = np.std( d[0,si:n])
    m2 = np.mean(d[1,si:n])
    s2 = np.std( d[1,si:n])

    with open('diam.txt', 'w') as f:
        f.write('# a au t tu\n')
        f.write('%g %g %g %g\n' % (m1, s1, m2, s2))
