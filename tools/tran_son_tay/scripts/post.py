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


def wrap_to(x,left,right):
    if x < left:   x = wrap_to(x+(right-left), left, right)
    if x >= right: x = wrap_to(x-(right-left), left, right)
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
    return ply, x, y, z


def read_data(plydir, dt, ntspd):
    listing = glob(plydir+"/rbcs-*.ply"); listing.sort()
    nfiles = len(listing)

    t = dt*ntspd*np.arange(nfiles)  # DPD t
    start = nfiles-int(np.floor(0.5*nfiles))  # Length of signal
    end = min(start+2e6, nfiles)  # Length of signal
    t = t[start:end]
    ls = end-start

    # find marker
    fullpath = listing[0]
    _, x, y, z = read_ply(fullpath)
    midx = np.argmax(x)  # the rightmost point will be a marker
    a0 = np.max(x)-np.min(x)
    c0 = np.max(y)-np.min(y)

    th = np.zeros(ls)  # angle with the projection on Ox
    om = np.zeros(ls)  # angle of the marker with the current RBC axis
    el = np.zeros(ls)
    a  = np.zeros(ls)
    b  = np.zeros(ls)
    c  = np.zeros(ls)

    ed = 'e' # directory for ellipsoid dumps
    if not os.path.exists(ed): os.makedirs(ed)

    ec = 'c' # directory for COM dumps
    if not os.path.exists(ec): os.makedirs(ec)

    for i in range(ls):
        fullpath = listing[start+i]
        ply, x, y, z = read_ply(fullpath)
        # x -= np.mean(x); y -= np.mean(y); z -= np.mean(z)

        # th[i] = get_th(x, y, z)
        # om[i] = get_om(x, y, z, midx, th[i])
        # el[i] = get_el(x, y, z)
        # a[i]  = (np.max(x)-np.min(x))/a0
        # c[i]  = (np.max(y)-np.min(y))/c0

        xyz = np.array([x, y, z]).T
        center, radii, rot, v, chi2 = ef.ellipsoid_fit(xyz)
        idx = np.argsort(-radii); radii = radii[idx]; rot = rot[:, idx]
        a[i] = radii[0]; b[i] = radii[1]; c[i] = radii[2]
        th[i] = get_angle_btw_vectors(rot[:,0], np.array([1,0,0]))
        print a[i], b[i], c[i], th[i]
        if not any(np.isnan(radii)):
            # dump ellipsoid
            ef.ellipsoid_dump_ply('%s/%05d.ply' % (ed, start+i), center, rot, radii)

            # dump ply
            ply['vertex']['x'] = x - center[0]
            ply['vertex']['y'] = y - center[1]
            ply['vertex']['z'] = z - center[2]
            ply.write('%s/%05d.ply' % (ec, start+i))

        if i % 100 == 0: print 'Computed up to %d/%d' % (i, ls)

    save_res('result.txt', (t,th,om,a,c))
    plt.plot(t, om, 'b-'); plt.plot(t, th, 'r-')
    savefig('angle.png')
    plt.close()
    plt.plot(t, a, 'b-'); plt.plot(t, c, 'r-')
    savefig('diam.png')
    plt.close()

    return t, th, om, a, c, el


def get_angle_btw_vectors(v1, v2):
    res = np.dot(v1, v2) / (norm(v1)*norm(v2))
    return np.degrees(np.arccos(res))


# find the distance from the ellipse
def get_el(x, y, z):
    xyz = np.array([x, y, z]).T
    center, radii, rot, v, chi2 = ellipsoid_fit(xyz)
    ellipsoid_dump('e.3d', center, rot, radii)
    return chi2


# find the current axis
def get_th(x, y, z):
    pca = PCA(n_components=2)
    pca.fit(np.array([x,z]).T)
    pc = pca.components_
    res = np.arctan(pc[0,1]/pc[0,0])
    res = wrap_to(np.degrees(res), -90, 90)
    return res


# find the current marker angle
def get_om(x, y, z, id, th):
    res = np.arctan2(z[id],x[id])
    res = wrap_to(np.degrees(res), -180, 180)
    res = wrap_to(th-res, -180, 180)
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
