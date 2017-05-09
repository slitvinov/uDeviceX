#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
import numpy as np

from argparse import ArgumentParser
from efit import fit_ellipsoid_ply
from glob import glob
from matplotlib.pyplot import plot, savefig, close, legend, quiver, axis
from numpy.linalg import norm
from os import makedirs
from os.path import exists
from plyfile import PlyData
from sklearn.decomposition import PCA
from scipy.optimize import leastsq
from time import time


verbose = 0
A = 0; B = 2; C = 1
X = 0; Y = 1; Z = 2

'''
1. All vectors are column vectors
2. Ellipsoid half-axes are ordered such that b <= c <= a
'''


def wrap(x, l, r):
    if x <  l: x = wrap(x+(r-l), l, r)
    if x >= r: x = wrap(x-(r-l), l, r)
    return x


def save_txt(f, a_in):
    a_out = ()
    for a in a_in:
        a_out = a_out + (a.reshape((a.size, 1)),)
    np.savetxt(f, np.concatenate(a_out, axis=1), fmt='%.6e', delimiter=' ')


def plot_all(si, t, th, om, a, b, c, el, fr):
    plot(t, th, 'r-', label='theta')
    plot(t, om, 'b-', label='omega')
    plot([t[si], t[si]], [-180, 180], 'k--')
    legend()
    savefig('angle.pdf')
    close()

    plot(t, a, 'r-', label='a')
    plot(t, b, 'g-', label='b')
    plot(t, c, 'b-', label='c')
    plot([t[si], t[si]], [0, 2], 'k--')
    legend()
    savefig('diam.pdf')
    close()

    plot(t, el, 'r-', label='ellipticity')
    plot([t[si], t[si]], [0, 100], 'k--')
    legend()
    savefig('ellipticity.pdf')
    close()

    plot(t, fr, 'r-', label='TTF')
    plot([t[si], t[si]], [0, 1], 'k--')
    legend()
    savefig('ttf.pdf')
    close()


def get_angle_btw_vectors(v1, v2):
    res = np.dot(v1, v2) / (norm(v1)*norm(v2))
    return np.degrees(np.arccos(res))


# find the current marker angle
def get_om(xyz, idx, th):
    xyz -= np.mean(xyz, axis=0)
    om = th - np.degrees(np.arctan2(xyz[idx, Z], xyz[idx, X]))
    om = wrap(om, -180, 180)
    return om


def fit_sk1(r, v, ab):
    def func(f, r, v): return norm(f*np.array([ab, -1./ab])*r[:, [Y, X]] - v)
    f0 = 0.2
    return leastsq(func, f0, args=(r, v))[0]


def fit_sk2(r, v, ab):
    """
    Mean squarer fit of Keller-Skalak frequency (`fr').
    ab = a/b, [vx, vy] ~ [fr*a/b*y, -fr*b/a*x]
    """
    sm = np.sum
    svxy, svyx = sm(v[:, X]*r[:, Y]), sm(v[:, Y]*r[:, X])
    sxx,   syy = sm(r[:, X]*r[:, X]), sm(r[:, Y]*r[:, Y])
    ab2 = ab**2
    ab4 = ab**4
    fr = (ab*(ab2*svxy-svyx))/(ab4*syy+sxx)
    return fr


def get_fr_sk(xyz, uvw, rot, ab, it):
    # subtract mean
    xyz -= np.mean(xyz, axis=0)
    uvw -= np.mean(uvw, axis=0)

    # rotate
    xyz = np.dot(xyz, rot)
    uvw = np.dot(xyz+uvw, rot) - xyz

    # fit
    r = xyz[:, [A, B]]; v = uvw[:, [A, B]]
    f = fit_sk2(r, v, ab)

    # # plot
    # ve = f*np.array([ab, -1./ab])*r[:,[Y, X]]
    # quiver(r[:, X], r[:, Y], v [:, X], v [:, Y], color='r')
    # quiver(r[:, X], r[:, Y], ve[:, X], ve[:, Y], color='b')
    # axis('equal')
    # savefig('fit.pdf'); close()

    return f


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
    savefig('peaks.pdf')
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
    ch = int(0.05*n); steady = False; si = int(0.5*n); ave = 0

    tstart = time()
    for i in range(n):
        fname = files[i]
        center, rot, radii, chi2, xyz, uvw = fit_ellipsoid_ply(fname,
            '%s/%05d' % (cd, i), '%s/%05d' % (ed, i))

        if i == 0:
            mi = np.argmax(xyz[:,A])  # the rightmost point will be a marker
            a0 = np.max(xyz[:, A]) - np.min(xyz[:, A])
            b0 = np.max(xyz[:, B]) - np.min(xyz[:, B])
            c0 = np.max(xyz[:, C]) - np.min(xyz[:, C])

        a[i] = 2*radii[A]/a0
        b[i] = 2*radii[B]/b0
        c[i] = 2*radii[C]/c0
        a_[i] = (np.max(xyz[:, A]) - np.min(xyz[:, A]))/a0
        b_[i] = (np.max(xyz[:, B]) - np.min(xyz[:, B]))/b0
        c_[i] = (np.max(xyz[:, C]) - np.min(xyz[:, C]))/c0
        th[i] = get_angle_btw_vectors(rot[:, A], np.array([1, 0, 0]))
        om[i] = get_om(xyz, mi, th[i])
        el[i] = chi2
        fr[i] = get_fr_sk(xyz, uvw, rot, radii[A]/radii[B], i)

        # check whether we're in a steady state
        if ch > 0 and i >= si and i % ch == 0:
            cur = np.mean(a[i-ch+1:i])
            if not steady and np.abs(ave-cur) < 0.02*ave:
                steady = True; si = i
                if verbose: print 'Steady state reached after %d/%d steps' % (i, n)
            else: ave = cur

        if verbose and i % 100 == 0: print 'Computed %d/%d steps' % (i, n)

    print 'Elapsed time: %.1f sec' % (time()-tstart)

    t = dt*ntspd*np.arange(n)  # DPD time
    save_txt('result.txt', (t, th, om, a, b, c, el, a_, b_, c_, fr))
    plot_all(si, t, th, om, a, b, c, el, fr)

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
    parser.add_argument('--sh',  default=1)
    parser.add_argument('--st',  default=1)
    parser.add_argument('--dt',  default=1)
    args = parser.parse_args()
    plydir = args.ply
    sh     = float(args.sh)
    ntspd  = int(args.st)  # number of t steps per dump
    dt     = float(args.dt)

    process_data(plydir, dt, ntspd, sh)
