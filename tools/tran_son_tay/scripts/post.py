#!/usr/bin/env python

from argparse import ArgumentParser
from glob import glob
import numpy as np
from numpy.linalg import norm
from plyfile import PlyData
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import savefig
import efit as ef
import os


def wrap(x,left,right):
    if x < left:   x = wrap(x+(right-left), left, right)
    if x >= right: x = wrap(x-(right-left), left, right)
    return x


def save_res(filename, arr_in):
    arr = ()
    for a in arr_in:
        arr = arr + (a.reshape((len(a),1)),)
    np.savetxt(filename, np.concatenate(arr,axis=1), fmt='%.6e', delimiter=' ')


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

    plt.plot(x, y, 'b-o', label='theta')
    plt.plot(x[peakind], y[peakind], 'ro', label='peaks')
    plt.legend()
    plt.savefig('peaks.png')
    plt.close()

    return fr, fru/fr


def process_data(plydir, dt, ntspd, sh):
    ed = 'e' # directory for ellipsoid dumps
    if not os.path.exists(ed): os.makedirs(ed)
    cd = 'c' # directory for COM dumps
    if not os.path.exists(cd): os.makedirs(cd)

    # initialization
    files = glob(plydir+"/rbcs-*.ply"); files.sort()
    n = len(files)
    th = np.zeros(n)  # angle with the projection on Ox
    om = np.zeros(n)  # angle of the marker with the current RBC axis
    el = np.zeros(n)
    a  = np.zeros(n)
    b  = np.zeros(n)
    c  = np.zeros(n)
    a_ = np.zeros(n)
    c_ = np.zeros(n)
    ch = int(np.floor(n/20))
    steady = 0; si = 0.75*n; ave = 0

    # main loop
    for i in range(n):
        fname = files[i]
        center, rot, radii, chi2, xyz = ef.fit_ellipsoid_ply(fname,
            '%s/%05d' % (cd, i), '%s/%05d' % (ed, i))

        if i == 0:
            mi = np.argmax(xyz[:,0])  # the rightmost point will be a marker
            a0 = radii[0]; b0 = radii[1]; c0 = radii[2]
            a_0 = np.max(xyz[:,0]) - np.min(xyz[:,0])
            c_0  np.max(xyz[:,2]) - np.min(xyz[:,2])

        a[i] = radii[0]/a0; b[i] = radii[1]/b0; c[i] = radii[2]/c0
        th[i] = get_angle_btw_vectors(rot[:,0], np.array([1,0,0]))
        om[i] = get_om(fname, mi, th[i])
        el[i] = chi2
        a_[i] = (np.max(xyz[:,0]) - np.min(xyz[:,0]))/a_0
        c_[i] = (np.max(xyz[:,2]) - np.min(xyz[:,2]))/c_0

        # check whether we're in a steady state
        if ch > 0 and (i+1) % ch == 0:
            cur = np.mean(a[i-ch+1:i])
            if not steady and abs(ave-cur) < 0.02*ave:
                print "Steady state reached after", i, "steps out of", n
                steady = 1; si = i
            else: ave = cur

        if i % 100 == 0: print 'Computed up to %d/%d' % (i, n)

    t = dt*ntspd*np.arange(n)  # DPD time
    save_res('result.txt', (t, th, om, a, b, c, el, a_, c_))
    plt.plot(t, th, 'r-', label='theta')
    plt.plot(t, om, 'b-', label='omega')
    plt.plot([t[si], t[si]], [-180, 180], 'k--')
    plt.legend()
    savefig('angle.png')
    plt.close()
    plt.plot(t, a, 'r-', label='a')
    plt.plot(t, b, 'g-', label='b')
    plt.plot(t, c, 'b-', label='c')
    plt.plot([t[si], t[si]], [0, 2], 'k--')
    plt.legend()
    savefig('diam.png')
    plt.close()
    plt.plot(t, el, 'r-', label='ellipticity')
    plt.plot([t[si], t[si]], [0, 100], 'k--')
    plt.legend()
    savefig('ellipticity.png')
    plt.close()

    # compute means and stds
    a,  au  = np.mean( a[si:]), np.std( a[si:])
    b,  bu  = np.mean( b[si:]), np.std( b[si:])
    c,  cu  = np.mean( c[si:]), np.std( c[si:])
    el, elu = np.mean(el[si:]), np.std(el[si:])
    th, thu = np.mean(th[si:]), np.std(th[si:])
    a_, au_ = np.mean(a_[si:]), np.std(a_[si:])
    c_, cu_ = np.mean(c_[si:]), np.std(c_[si:])
    fr, fru = get_fr(t[si:], om[si:]); fr *= 2.*np.pi/sh; fru /= sh

    with open('post.txt', 'w') as f:
        f.write('# fr\tfru\ta\tau\tb\tbu\tc\tcu\tth\tthu\tel\telu\ta_\tau_\tc_\tcu_\n')
        f.write('  %.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n' %
                (  fr,   fru,  a,    au,   b,    bu,   c,    cu,   th,   thu,  el,   elu,  a_,   au_,  c_,   cu_))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ply')
    parser.add_argument('--sh')
    parser.add_argument('--st')
    parser.add_argument('--dt')
    args = parser.parse_args()
    plydir = args.ply
    sh     = float(args.sh)
    ntspd  = int(args.st)  # number of t steps per dump
    dt     = float(args.dt)

    process_data(plydir, dt, ntspd, sh)
