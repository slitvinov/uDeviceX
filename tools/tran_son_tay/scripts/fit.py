#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
import numpy as np

from matplotlib.pyplot import plot, savefig, close, legend, quiver, axis
from numpy.linalg import norm
from scipy.optimize import curve_fit, least_squares
from sys import argv

# gamma_dot acth_s std_acth_s c_s std_c_s f_tsc std_f
exp_data = np.array([
    [28.6 , 5.5, 0.5, 3.2, 0.2, 6.22, 0.31],
    [42.9 , 5.9, 0.7, 3.0, 0.5, 9.42, 0.57],
    [57.1 , 6.5, 0.8, 2.5, 0.3, 12.7, 0.57],
    [114.3, 7.3, 0.4, 2.3, 0.5, 24.6, 1.07],
    [171.4, 7.6, 0.4, 2.1, 0.3, 37.4, 1.45]])
R0 = 4
for i in range(1, 5): exp_data[:, i] /= R0
for i in range(5, 6): exp_data[:, i] /= exp_data[:, 0]


def fd(x, a, t):
    return a+(1-a)*np.exp(-t*x)


def fds(x, a, t, tsc):
    return fd(x/tsc, a, t)


def fit_diam(x, y, a0, t0):
    try:
        popt, _ = curve_fit(fd, x, y, p0=(a0, t0))
        a, t = popt
        r = norm(y - fd(x, a, t))
    except:
        a, t, r = a0, t0, -1
    return a, t, r


def fit_tsc(xe, ye, a1, t1, a2, t2):
    def f(tsc):
        y1 = fds(xe, a1, t1, tsc)
        y2 = fds(xe, a2, t2, tsc)
        y = np.array([y1, y2]).T
        return norm(y - ye)
    tsc0 = 40
    try:
        tsc = (least_squares(f, tsc0, bounds=(0, 100)).x)[0]
    except:
        tsc = tsc0
    return tsc


def loglike(sim_data, a1, t1, r1, a2, t2, r2, tsc):
    xe = exp_data[:, 0]
    n = xe.shape[0]
    ye = exp_data[:, [1, 3, 5]]
    se = exp_data[:, [2, 4, 6]]
    ys = np.zeros((n, 3))
    y1 = fds(xe, a1, t1, tsc)
    y2 = fds(xe, a2, t2, tsc)
    ys = np.array([y1, y2]).T
    ss = np.tile(np.array([r1, r2]), (n, 1))
    ys[:, 2] = sim_data[:, 5]
    ss[:, 2] = sim_data[:, 6]
    par = np.loadtxt('params.txt')
    sm = par[-1]

    ye = ye.reshape((3*n,1))
    se = se.reshape((3*n,1))
    ys = ys.reshape((3*n,1))
    ss = ss.reshape((3*n,1))

    ll = np.sum((ys - ye)**2/(sm**2+se**2+ss**2)) # inside exp
    ll += n*log(2*np.pi)
    ll *= -0.5
    return ll


if __name__ == '__main__':
    sim_data = np.loadtxt(argv[1])
    n = sim_data.shape[0]

    for i in range(0, n/4):
        p  = sim_data[4*i, 0]
        x  = sim_data[4*i:4*(i+1), 1]
        y1 = sim_data[4*i:4*(i+1), 2] # a
        y2 = sim_data[4*i:4*(i+1), 3] # c
        fr = np.mean(sim_data[4*i:4*(i+1), 4])
        a1, t1, r1 = fit_diam(x, y1, 2, 0.02)
        a2, t2, r2 = fit_diam(x, y2, 0, 0.02)

        xe = exp_data[:, 0]
        ye = exp_data[:, [1, 3]]
        tsc = fit_tsc(xe, ye, a1, t1, a2, t2)

        xp = np.linspace(0, 1.1*np.max(exp_data[:, 0]), num=200)
        plot(xp, fds(xp, a1, t1, tsc), 'b-', label='simulation fit')
        plot(xp, fds(xp, a2, t2, tsc), 'b-')
        plot(tsc*x, y1, 'bo', label='simulation')
        plot(tsc*x, y2, 'bo')
        plot(xe, exp_data[:, 1], 'r-o', label='experiment')
        plot(xe, exp_data[:, 3], 'r-o')
        legend()
        savefig('fit_%d.pdf' % p)
        close()

        # ll = loglike(sim_data, a1, t1, r1, a2, t2, r2, tsc)
        # print ll
        # print 'loglike:', ll

        # print 'gc\t%.16f' % p
        # print 'a1\t%.16f' % a1
        # print 't1\t%.16f' % t1
        # print 'r1\t%.16f' % r1
        # print 'a2\t%.16f' % a2
        # print 't2\t%.16f' % t2
        # print 'r2\t%.16f' % r2
        # print 'fr\t%.16f' % fr
        # print 'tsc\t%.16f' % tsc
        print '%d\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f' % (p, a1, t1, a2, t2, fr, tsc)
