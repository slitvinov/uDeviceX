#!/usr/bin/env python

from argparse import ArgumentParser
from glob import glob
from math import floor, atan, atan2
import numpy as np
import numpy.linalg as la
from plyfile import PlyData
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import savefig
from efit import ellipsoid_fit


def wrapTo(x,left,right):
    if x < left:
        x = wrapTo(x+(right-left), left, right)
        if x >= right: x = wrapTo(x-(right-left), left, right)
    return x


def read_ply_file(ply):
    vertex = ply['vertex']
    x,y,z = (vertex[p] for p in ('x', 'y', 'z'))
    return x,y,z


def save_res(filename, arr_in):
    arr = ()
    for a in arr_in:
        arr = arr + (a.reshape((len(a),1)),)
    np.savetxt(filename, np.concatenate(arr,axis=1), fmt='%.6e', delimiter=' ')


def read_data(plydir, dt, ntspd):
    listing = glob(plydir+"/rbcs-*.ply"); listing.sort()
    nfiles = len(listing)

    t = dt*ntspd*np.arange(nfiles)  # DPD t
    start = nfiles-int(floor(0.5*nfiles))  # Length of signal
    print start
    end = min(start+2e6, nfiles)  # Length of signal
    t = t[start:end]
    ls = end-start

    # find marker
    fullpath = listing[0]
    x, y, z = read_ply_file(PlyData.read(fullpath))
    midx = np.argmax(x)  # the rightmost point will be a marker
    a0 = np.max(x)-np.min(x)
    c0 = np.max(y)-np.min(y)

    th = np.zeros((ls,))  # angle with the projection on Ox
    om = np.zeros((ls,))  # angle of the marker with the current RBC axis
    el = np.zeros((ls,))
    a  = np.zeros((ls,))
    c  = np.zeros((ls,))

    for i in range(ls):
        fullpath = listing[start+i]
        x, y, z = read_ply_file(PlyData.read(fullpath))
        x -= np.mean(x); y -= np.mean(y); z -= np.mean(z)

        th[i] = get_th(x, y, z)
        om[i] = get_om(x, y, z, midx, th[i])
        el[i] = get_el(x, y, z)
        a[i]  = (np.max(x)-np.min(x))/a0
        c[i]  = (np.max(y)-np.min(y))/c0

        if i % 100 == 0: print 'Computed up to', i, '/', ls

    save_res('result.txt', (t,th,om,a,c))
    plt.plot(t, om, 'b-'); plt.plot(t, th, 'r-')
    savefig('angle.png')
    plt.close()
    plt.plot(t, a, 'b-'); plt.plot(t, c, 'r-')
    savefig('diam.png')
    plt.close()

    return t, th, om, a, c, el


# find the distance from the ellipse
def get_el(x, y, z):
    xyz = np.array([x, y, z]).T
    center, radii, evecs, v, chi2 = ellipsoid_fit(xyz)
    print chi2
    if (not np.isnan(chi2)): plot_el(center, evecs, radii)
    return chi2


def plot_el(center, evecs, radii):
    a = radii[0]; b = radii[1]; c = radii[2]

    ndump = 100
    uu = np.linspace(0, 2*np.pi, ndump)
    vv = np.linspace(0,   np.pi, ndump)
    [uu, vv] = np.meshgrid(uu, vv); n = uu.size
    uu = uu.reshape(n); vv = vv.reshape(n)
    xx = np.zeros(n); yy = np.zeros(n); zz = np.zeros(n)

    for i in range(n):
        u = uu[i]; v = vv[i]
        x = a*np.cos(u)*np.sin(v) # u: [0, pi]
        y = b*np.sin(u)*np.sin(v) # v: [0, pi]
        z = c*np.cos(v)

        r = np.array([x, y, z]).reshape((3, 1))
        r = np.asarray(np.matrix(evecs) * np.matrix(r))
        r = r + center

        xx[i] = r[0]; yy[i] = r[1]; zz[i] = r[2]

    print 'writing: e.3d'
    with open('e.3d', 'w') as f:
        f.write('x y z sc\n')
        sc = np.zeros(n) # fake scalar
        for i in range(n):
            f.write('%g %g %g %g\n' % (xx[i], yy[i], zz[i], sc[i]));


# find the current axis
def get_th(x,y,z):
    pca = PCA(n_components=2)
    pca.fit(np.array([x,z]).T)
    pc = pca.components_
    res = atan(pc[0,1]/pc[0,0])
    res = wrapTo(np.rad2deg(res), -90, 90)
    return res


# find the current marker angle
def get_om(x,y,z,id,th):
    res = atan2(z[id],x[id])
    res = wrapTo(np.rad2deg(res), -180, 180)
    res = wrapTo(th-res, -180, 180)
    return res


def get_fr(x, y):
    from scipy import signal

    best = 0, 100

    # find the points where the sign changes -- these are the peaks
    peakind = []
    for i in range(len(y)-1):
        if y[i] > 0 and y[i+1] < 0:
            peakind.append(i)

    pers = x[peakind[1:]]-x[peakind[0:-1]]
    freq = np.mean(1/pers)
    frequ = np.std(1/pers)

    if best[1] > frequ: best = freq, frequ

    plt.plot(x, y, 'b-o')
    plt.plot(x[peakind], y[peakind], 'ro')
    plt.savefig('peaks.png')
    plt.close()

    return best[0], best[1]/best[0]


def get_an(th):
    return np.mean(th), np.std(th)


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

    t, th, om, a, c, el = read_data(plydir, dt, ntspd)
    a, au = np.mean(a), np.std(a)
    c, cu = np.mean(c), np.std(c)
    el, elu = np.mean(el), np.std(el)
    th, thu = get_an(th)
    fr, fru = get_fr(t, om)
    ish = 1./sh; fr *= 2.*np.pi*ish; fru *= ish

    with open('post.txt', 'w') as f:
        f.write('# fr fru a au c cu th thu el elu\n')
        f.write('%g %g %g %g %g %g %g %g %g %g\n' % (fr, fru, a, au, c, cu, th, thu, el, elu))
