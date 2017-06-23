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
2. Ellipsoid half-axes are ordered such that eb <= ec <= ea
'''


def wrap(x, l, r):
    if x <  l: x = wrap(x+(r-l), l, r)
    if x >= r: x = wrap(x-(r-l), l, r)
    return x


def save_txt(f, h, a_in):
    a_out = ()
    for a in a_in:
        a_out = a_out + (a.reshape((a.size, 1)),)
    np.savetxt(f, np.concatenate(a_out, axis=1), fmt='%.6e', delimiter=' ', header=h)


def plot_all(si, t, fr, ea, eb, ec, pa, pb, pc, th, om, el):
    plot(t, th, 'r-', label='theta')
    plot(t, om, 'b-', label='omega')
    plot([t[si], t[si]], [-180, 180], 'k--')
    legend()
    savefig('angle.pdf')
    close()

    plot(t, pa, 'r-', label='pa')
    plot(t, pb, 'g-', label='pb')
    plot(t, pc, 'b-', label='pc')
    plot(t, ea, 'r--', label='ea')
    plot(t, eb, 'g--', label='eb')
    plot(t, ec, 'b--', label='ec')
    plot([t[si], t[si]], [0, 3], 'k--')
    legend()
    savefig('diam.pdf')
    close()

    plot(t, el, 'r-', label='ellipticity')
    plot([t[si], t[si]], [0, 0.25], 'k--')
    legend()
    savefig('ellipticity.pdf')
    close()

    plot(t, fr, 'r-', label='TTF')
    plot([t[si], t[si]], [0, 0.5], 'k--')
    legend()
    savefig('ttf.pdf')
    close()


def print_all(si, sh, t, fr, ea, eb, ec, pa, pb, pc, th, om, el):
    p = {}
    p['sh'] = sh
    p['ea'], p['eau'] = np.mean(ea[si:]), np.std(ea[si:])
    p['eb'], p['ebu'] = np.mean(eb[si:]), np.std(eb[si:])
    p['ec'], p['ecu'] = np.mean(ec[si:]), np.std(ec[si:])
    p['el'], p['elu'] = np.mean(el[si:]), np.std(el[si:])
    p['th'], p['thu'] = np.mean(th[si:]), np.std(th[si:])
    p['pa'], p['pau'] = np.mean(pa[si:]), np.std(pa[si:])
    p['pb'], p['pbu'] = np.mean(pb[si:]), np.std(pb[si:])
    p['pc'], p['pcu'] = np.mean(pc[si:]), np.std(pc[si:])
    p['fr'], p['fru'] = np.mean(fr[si:]), np.std(fr[si:])
    # fr, fru = get_fr(t[si:], om[si:]); fr *= 2.*np.pi/sh; fru /= sh
    # print fr, fru

    with open('post.txt', 'w') as f:
        for key, value in sorted(p.iteritems()):
            f.write('%s\t%.16f\n' % (key, value))


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
    ab = ea/eb, [vx, vy] ~ [fr*ea/eb*y, -fr*eb/ea*x]
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
    idx = np.logical_and(
            np.abs(xyz[:, C]) < 0.1*np.max(xyz[:, C]),
            np.abs(xyz[:, A]) < 0.3*np.max(xyz[:, A]) )
    xyz = xyz[idx, :]; uvw = uvw[idx, :]
    r = xyz[:, [A, B]]; v = uvw[:, [A, B]]
    f = fit_sk2(r, v, ab)

    if (it % 100 == 0):
        # plot
        ve = f*np.array([ab, -1./ab])*r[:,[Y, X]]
        quiver(r[:, X], r[:, Y], ve[:, X], ve[:, Y], color='b')
        quiver(r[:, X], r[:, Y], v [:, X], v [:, Y], color='r')
        axis('equal')
        savefig('fit_%d.pdf' % it); close()

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

    # initialization
    files = glob(plydir+"/rbcs-*.ply"); files.sort()
	files = files[::10]
    n = len(files)
    th = np.zeros(n)  # angle with the projection on Ox
    om = np.zeros(n)  # angle of the marker with the current RBC axis
    el = np.zeros(n)  # ellipticity
    fr = np.zeros(n)  # tanktreading frequency
    ea = np.zeros(n); eb = np.zeros(n); ec = np.zeros(n)
    pa = np.zeros(n); pb = np.zeros(n); pc = np.zeros(n)
    ch = int(0.05*n); steady = False; si = int(0.5*n); ave = 0

    tstart = time()
    for i in range(n):
        fname = files[i]
        center, rot, radii, chi2, xyz, uvw = fit_ellipsoid_ply(fname, '%s/%05d' % (ed, i))

        if i == 0:
            mi = np.argmax(xyz[:,A])  # the rightmost point will be a marker
            a0 = np.max(xyz[:, A]) - np.min(xyz[:, A])
            b0 = np.max(xyz[:, B]) - np.min(xyz[:, B])
            c0 = np.max(xyz[:, C]) - np.min(xyz[:, C])

        ea[i] = 2*radii[A]/a0
        eb[i] = 1.       # 2*radii[B]/b0
        ec[i] = 1./ea[i] # 2*radii[C]/c0
        pa[i] = (np.max(xyz[:, A]) - np.min(xyz[:, A]))/a0
        pb[i] = (np.max(xyz[:, B]) - np.min(xyz[:, B]))/b0
        pc[i] = (np.max(xyz[:, C]) - np.min(xyz[:, C]))/c0
        th[i] = get_angle_btw_vectors(rot[:, A], np.array([1, 0, 0]))
        om[i] = get_om(xyz, mi, th[i])
        el[i] = chi2 / xyz.shape[0]
        fr[i] = get_fr_sk(xyz, uvw, rot, (ea[i]*a0)/(eb[i]*b0), i) / sh

        # check whether we're in a steady state
        if ch > 0 and i >= si and i % ch == 0:
            cur = np.mean(ea[i-ch+1:i])
            if not steady and np.abs(ave-cur) < 0.02*ave:
                steady = True; si = i
                if verbose: print 'Steady state reached after %d/%d steps' % (i, n)
            else: ave = cur

        if verbose and i % 100 == 0: print 'Computed %d/%d steps' % (i, n)

    print 'Elapsed time: %.1f sec' % (time()-tstart)

    t = dt*ntspd*np.arange(n)  # DPD time
    save_txt('result.txt', '#t\tfr\tea\teb\tec\tpa\tpb\tpc\tth\tom\tel',
                            (t, fr, ea, eb, ec, pa, pb, pc, th, om, el))
    plot_all(si, t, fr, ea, eb, ec, pa, pb, pc, th, om, el)
    print_all(si, sh, t, fr, ea, eb, ec, pa, pb, pc, th, om, el)


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
