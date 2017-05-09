#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
import numpy as np

from argparse import ArgumentParser
from efit import fit_ellipsoid_ply
from glob import glob
from matplotlib.pyplot import plot, savefig, close, legend, quiver
from numpy.linalg import norm
from os import makedirs
from os.path import exists
from plyfile import PlyData
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit, leastsq


ar = np.array;


def fit1(r, v, ab):
    def func(f, r, v): return norm(f*ar([ab, -1./ab])*r[:,[1, 0]] - v)
    f0 = 0.2
    return leastsq(func, f0, args=(r, v))[0]


def test3():
    def func(par, x, y): return norm(par[0] + par[1]*x + par[2]*x*x - y, axis=1)
    xdata = ar([[0.0,0.0], [1.0,1.0], [2.0,2.0]])
    ydata = ar([[2.0,2.0], [3.0,3.0], [6.0,6.0]])
    a0 = 1; b0 = 1; c0 = 1
    return leastsq(func, [a0, b0, c0], args=(xdata, ydata))[0]



def vfit(x, y, vx, vy, q):
    """
    Mean squarer fit of Keller-Skalak frequency (`fr').
    q = a/b, [vx, vy] ~ [fr*a/b*y, -fr*b/a*x]
    """
    sm = np.sum
    svxy, svyx = sm(vx*y), sm(vy*x)
    sxx,   syy = sm( x*x), sm( y*y)
    q2 = q**2
    q4 = q**4
    fr = (q*(q2*svxy-svyx))/(q4*syy+sxx)
    return fr


if __name__ == '__main__':
    sh = 5
    A = 0; B = 2; C = 1
    f = '/Users/kulina/workspace/mounts/falcon_scratch/RBC/tanktreading/simulations/run_27/ply/rbcs-00500.ply'
    center, rot, radii, chi2, xyz, uvw = fit_ellipsoid_ply(f, 'c', 'e')

    xyz -= np.mean(xyz, axis=0)
    uvw -= np.mean(uvw, axis=0)

    xyz = np.dot(xyz, rot)
    tmp = np.dot(xyz+uvw, rot)
    uvw = tmp-xyz

    r = xyz[:, [0,2]]; v = uvw[:, [0,2]]
    ab = radii[A]/radii[B]
    ttf = fit1(r, v, ab)
    ve = ttf*ar([ab, -1./ab])*r[:,[1, 0]]
    print ttf/sh, norm(ve-v)
    print vfit(r[:,0], r[:,1], v[:,0], v[:,1], ab)/sh

    quiver(r[:,0], r[:,1], v [:,0], v [:,1], color='r')
    quiver(r[:,0], r[:,1], ve[:,0], ve[:,1], color='b')
    savefig('t.pdf'); close()
