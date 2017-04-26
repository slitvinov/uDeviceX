import numpy as np
import glob
from argparse import ArgumentParser
from plyfile import PlyData
from math import floor, atan, atan2
from sklearn.decomposition import PCA
import scipy.fftpack as scfft
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import savefig


def read_data(plydir, dt, ntspd):
    listing = glob.glob(plydir+"/rbcs-*.ply"); listing.sort()
    nfiles = len(listing)

    t = dt*ntspd*np.arange(nfiles)  # DPD t
    start = nfiles-int(floor(0.5*nfiles))  # Length of signal
    end = min(start+2e6, nfiles)  # Length of signal
    t = t[start:end]
    ls = end-start

    # find marker
    fullpath = listing[0]
    x,y,z = read_ply_file(PlyData.read(fullpath))
    midx = np.argmax(x)  # the rightmost point will be a marker
    a0 = np.max(x)-np.min(x)
    c0 = np.max(y)-np.min(y)

    th = np.zeros((ls,))  # angle with the projection on Ox
    om = np.zeros((ls,))  # angle of the marker with the current RBC axis
    diam_a = np.zeros((ls,))
    diam_c = np.zeros((ls,))

    for i in range(ls):
        fullpath = listing[start+i]
        x,y,z = read_ply_file(PlyData.read(fullpath))
        x = x-np.mean(x); y = y-np.mean(y); z = z-np.mean(z)

        th[i] = get_th(x, y, z)
        om[i] = get_om(x, y, z, midx, th[i])
        diam_a[i] = (np.max(x)-np.min(x))/a0
        diam_c[i] = (np.max(y)-np.min(y))/c0

        if i % 100 == 0: print 'Computed up to', i, '/', ls

    save_res('Tran-Son-Tay_result.txt', (t,th,om,diam_a,diam_c))
    plt.plot(t, om, 'b-'); plt.plot(t, th, 'r-')
    savefig('Tran-Son-Tay_angle.png')
    plt.close()
    plt.plot(t, diam_a, 'b-'); plt.plot(t, diam_c, 'r-')
    savefig('Tran-Son-Tay_diam.png')
    plt.close()

    return t, th, om, diam_a, diam_c


def wrapTo(x,left,right):
    if x < left:
        x = wrapTo(x+(right-left), left, right)
        if x >= right: x = wrapTo(x-(right-left), left, right)
    return x


def read_ply_file(ply):
    vertex = ply['vertex']
    x,y,z = (vertex[p] for p in ('x', 'y', 'z'))
    return x,y,z


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


def save_res(filename, arr_in):
    arr = ()
    for a in arr_in:
        arr = arr + (a.reshape((len(a),1)),)
    np.savetxt(filename, np.concatenate(arr,axis=1), fmt='%.6e', delimiter=' ')


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
    plt.savefig('Tran-Son-Tay_peaks.png')
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

    t, th, om, a, c = read_data(plydir, dt, ntspd)
    a, au = np.mean(a), np.std(a)
    c, cu = np.mean(c), np.std(c)
    th, thu = get_an(th)
    fr, fru = get_fr(t, om)
    ish = 1./sh; fr *= 2.*np.pi*ish; fru *= ish

    with open('post.txt', 'w') as f:
        f.write('# fr fru a au c cu th thu\n')
        f.write('%g %g %g %g %g %g %g %g\n' % (fr, fru, a, au, c, cu, th, thu))
