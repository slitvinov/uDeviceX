#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
import numpy as np

from argparse import ArgumentParser
from efit import fit_ellipsoid_ply
from glob import glob
from matplotlib.pyplot import plot, savefig, close, legend
from numpy.linalg import norm
from os import makedirs
from os.path import exists
from plyfile import PlyData
from sklearn.decomposition import PCA
from scipy.optimize import leastsq


verbose = 0


def wrap(x,left,right):
    if x < left:   x = wrap(x+(right-left), left, right)
    if x >= right: x = wrap(x-(right-left), left, right)
    return x


def save_txt(filename, arr_in):
    arr = ()
    for a in arr_in:
        arr = arr + (a.reshape((len(a), 1)),)
    np.savetxt(filename, np.concatenate(arr, axis=1), fmt='%.6e', delimiter=' ')


def read_ply(fname):
    ply = PlyData.read(fname)
    vertex = ply['vertex']
    x, y, z = (vertex[p] for p in ('x', 'y', 'z'))
    return x, y, z


def get_angle_btw_vectors(v1, v2):
    res = np.dot(v1, v2) / (norm(v1)*norm(v2))
    return np.degrees(np.arccos(res))


# find the current marker angle
def get_om(fname, idx, th):
    x, y, z = read_ply(fname)
    x -= np.mean(x); y -= np.mean(y); z -= np.mean(z)
    om = th - np.degrees(np.arctan2(z[idx], x[idx]))
    om = wrap(om, -180, 180)
    return om


def get_fr_sk(xyz, uvw, rot, ab):
    xyz -= np.mean(xyz, axis=0)
    uvw -= np.mean(uvw, axis=0)

    xyz = np.dot(xyz, rot)
    uvw = np.dot(xyz+uvw, rot) - xyz

    r = xyz[:,[0,2]]; v = uvw[:,[0,2]]

    def func(f, r, v): return norm(f*np.array([ab, -1./ab])*r[:,[1, 0]] - v)
    f0 = 0.2
    return leastsq(func, f0, args=(r, v))[0]


def get_fr(x, y):
    fr = 0; fru = 100

    # find the points where the sign changes -- these are the peaks
    peakind = []
    for i in range(len(y)-1):
        if y[i] > 0 and y[i+1] < 0:
            peakind.append(i)

    pers = x[peakind[1:]]-x[peakind[0:-1]]
    cfr  = np.mean(1/pers)
    cfru = np.std(1/pers)

    if fru > cfru: fr = cfr; fru = cfru

    # plot
    plot(x, y, 'b-o', label='theta')
    plot(x[peakind], y[peakind], 'ro', label='peaks')
    legend()
    savefig('peaks.png')
    close()

    return fr, fru/fr


def process_data(plydir, dt, ntspd, sh):
    ed = 'e' # directory for ellipsoid dumps
    if not exists(ed): makedirs(ed)
    cd = 'c' # directory for COM dumps
    if not exists(cd): makedirs(cd)

    # initialization
    files = glob(plydir+"/rbcs-*.ply"); files.sort()
    n = len(files)
    th = np.zeros(n)  # angle with the projection on Ox
    om = np.zeros(n)  # angle of the marker with the current RBC axis
    el = np.zeros(n)  # ellipticity
    fr = np.zeros(n)  # tanktreading frequency
    a  = np.zeros(n); b  = np.zeros(n); c  = np.zeros(n)
    a_ = np.zeros(n); b_ = np.zeros(n); c_ = np.zeros(n)
    ch = int(np.floor(n/20))
    steady = False; si = int(0.75*n); ave = 0
    A = 0; B = 2; C = 1  # x, y, z corresponding to a, b, c

    # main loop
    for i in range(n):
        fname = files[i]
        center, rot, radii, chi2, xyz, uvw = fit_ellipsoid_ply(fname,
            '%s/%05d' % (cd, i), '%s/%05d' % (ed, i))

        if i == 0:
            mi = np.argmax(xyz[:,A])  # the rightmost point will be a marker
            # a0 = radii[A]; b0 = radii[B]; c0 = radii[C]
            a0 = np.max(xyz[:,A]) - np.min(xyz[:,A])
            b0 = np.max(xyz[:,B]) - np.min(xyz[:,B])
            c0 = np.max(xyz[:,C]) - np.min(xyz[:,C])

        a[i] = 2*radii[A]/a0
        b[i] = 2*radii[B]/b0
        c[i] = 2*radii[C]/c0
        a_[i] = (np.max(xyz[:,A]) - np.min(xyz[:,A]))/a0
        b_[i] = (np.max(xyz[:,B]) - np.min(xyz[:,B]))/b0
        c_[i] = (np.max(xyz[:,C]) - np.min(xyz[:,C]))/c0
        th[i] = get_angle_btw_vectors(rot[:,A], np.array([1,0,0]))
        om[i] = get_om(fname, mi, th[i])
        el[i] = chi2
        fr[i] = get_fr_sk(xyz, uvw, rot, a[i]/b[i])

        # check whether we're in a steady state
        if ch > 0 and (i+1) % ch == 0:
            cur = np.mean(a[i-ch+1:i])
            if not steady and np.abs(ave-cur) < 0.02*ave:
                steady = True; si = i
                if verbose: print 'Steady state reached after %d/%d steps' % (i, n)
            else: ave = cur

        if verbose and i % 100 == 0: print 'Computed %d/%d steps' % (i, n)

    # plot
    t = dt*ntspd*np.arange(n)  # DPD time
    save_txt('result.txt', (t, th, om, a, b, c, el, a_, b_, c_))

    plot(t, th, 'r-', label='theta')
    plot(t, om, 'b-', label='omega')
    plot([t[si], t[si]], [-180, 180], 'k--')
    legend()
    savefig('angle.png')
    close()

    plot(t, a, 'r-', label='a')
    plot(t, b, 'g-', label='b')
    plot(t, c, 'b-', label='c')
    plot([t[si], t[si]], [0, 2], 'k--')
    legend()
    savefig('diam.png')
    close()

    plot(t, el, 'r-', label='ellipticity')
    plot([t[si], t[si]], [0, 100], 'k--')
    legend()
    savefig('ellipticity.png')
    close()

    plot(t, fr, 'r-', label='TTF')
    plot([t[si], t[si]], [0, 1], 'k--')
    legend()
    savefig('ttf.png')
    close()

    # compute means and stds
    a,  au  = np.mean( a[si:]), np.std( a[si:])
    b,  bu  = np.mean( b[si:]), np.std( b[si:])
    c,  cu  = np.mean( c[si:]), np.std( c[si:])
    el, elu = np.mean(el[si:]), np.std(el[si:])
    th, thu = np.mean(th[si:]), np.std(th[si:])
    a_, au_ = np.mean(a_[si:]), np.std(a_[si:])
    b_, bu_ = np.mean(b_[si:]), np.std(b_[si:])
    c_, cu_ = np.mean(c_[si:]), np.std(c_[si:])
    fr, fru = np.mean(fr[si:]), np.std(fr[si:])
    fr /= sh; fru /= sh
    # fr, fru = get_fr(t[si:], om[si:]); fr *= 2.*np.pi/sh; fru /= sh

    with open('post.txt', 'w') as f:
        f.write('# fr\tfru\ta\tau\tb\tbu\tc\tcu\tth\tthu\tel\telu\ta_\tau_\tc_\tcu_\n')
        f.write('  %.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n' %
                (  fr,   fru,  a,    au,   b,    bu,   c,    cu,   th,   thu,  el,   elu,  a_,   au_,  b_,   bu_,  c_,   cu_))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ply', default='ply')
    parser.add_argument('--sh',  default='1')
    parser.add_argument('--st',  default='1')
    parser.add_argument('--dt',  default='1')
    args = parser.parse_args()
    plydir = args.ply
    sh     = float(args.sh)
    ntspd  = int(args.st)  # number of t steps per dump
    dt     = float(args.dt)

    process_data(plydir, dt, ntspd, sh)
