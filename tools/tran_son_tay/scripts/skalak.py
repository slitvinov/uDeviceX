#!/usr/bin/env python

from math import sqrt
from sys import argv
from scipy.integrate import odeint
from scipy.optimize import bisect

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.special as special

plt = False


def fr_an(ca1, ca2, ca3, eta_m):
    gamma = 1 # 1/s
    eta_i = 10e-3 # Pa*s
    eta_0 = 3*eta_i # Pa*s
    mu_m = 0 # Pa, the bigger, the bigger the oscillations of theta
    a1 = ca1*2.60e-6 # m
    a2 = ca2*0.86e-6 # m
    a3 = ca3*2.60e-6 # m
    e = 50e-9 # m


    inva = 1./((a1*a2*a3)**(1./3.))
    alpha1 = a1*inva
    alpha2 = a2*inva
    alpha3 = a3*inva
    def g(s): return 1. / ((alpha1**2 + s)**1.5 * (alpha2**2 + s)**1.5 * sqrt(alpha3**2 + s))
    g3 = integrate.quad(g, 0, np.inf)[0]

    r2 = a2/a1
    z1 = 0.5 * (1./r2 - r2)
    z2 = g3 * (alpha1**2 + alpha2**2)
    f1 = (1./r2 - r2)**2
    f2 = 4 * z1**2 * (1 - 2./z2)
    f3 = -4 * z1/z2

    V = 4./3. * np.pi * a1*a2*a3 # m^3
    Sigma = 4 * np.pi * ( ( (a1*a2)**1.6 + (a1*a3)**1.6 + (a2*a3)**1.6 )/ 3 )**(1./1.6) # m^2
    Omega = Sigma*e # m^3

    def f(y, t):
        omega, theta = y      # unpack current values of y

        rhs_omega = -( f3 / (f2 - eta_i/eta_0*(1 + eta_m/eta_i * Omega/V) * f1) ) * (
            np.cos(2*theta) - 0.5*f1/f3 * mu_m/(eta_0*gamma) * omega/V * np.sin(2.*omega) )

        rhs_theta = -0.5 - (2.*a1*a2) / (a1**2 + a2**2) * rhs_omega + 0.5 * (
            a1**2 - a2**2) / (a1**2 + a2**2) * np.cos(2.*theta)

        derivs = [gamma*rhs_omega,      # list of dy/dt=f functions
                  gamma*rhs_theta]
        return derivs

    # Initial values
    theta0 = 0.0
    omega0 = 0.0
    y0 = [theta0, omega0]

    # Make time array for solution
    tStop = 100./gamma
    tInc = 0.01/gamma
    t = np.arange(0., tStop, tInc)

    # Call the ODE solver
    psoln = odeint(f, y0, t)
    psoln = np.rad2deg(psoln)%360

    peakind = []
    for i in range(1,len(t)):
        if psoln[i-1, 0] < psoln[i, 0]: peakind.append(i)
    per = t[peakind[1:]] - t[peakind[:-1]]
    fr = 2.*np.pi*np.mean(1./per) / gamma
    an = np.mean(psoln[len(t)//2:,1])

    if plt:
        # Plot results
        fig = plt.figure(1, figsize=(8,8))

        # Plot theta as a function of time
        ax1 = fig.add_subplot(211)
        ax1.plot(t, psoln[:,0])
        ax1.set_xlabel('time')
        ax1.set_ylabel('theta')

        # Plot omega as a function of time
        ax2 = fig.add_subplot(212)
        ax2.plot(t, psoln[:,1])
        ax2.set_xlabel('time')
        ax2.set_ylabel('omega')

        # plt.savefig('angles.png')
        # plt.tight_layout()
        plt.show()

    return fr, an


if __name__ == '__main__':
    ca1 = float(argv[1])
    ca2 = float(argv[2])
    ca3 = float(argv[3])
    fr0 = float(argv[4])
    an0 = float(argv[5])
    eta_m = 0 # Pa*s, the bigger, the bigger shear rate is needed fot TT

    def fit_sk1(fr0):
        def func(eta_m):
            fr, _ = fr_an(ca1, ca2, ca3, eta_m)
            return fr-fr0
        return bisect(func, 0, 0.5)

    eta_m = fit_sk1(fr0)
    fr, an = fr_an(ca1, ca2, ca3, eta_m)
    print '%.6f %.6f %.6f' % (fr, an, eta_m)

    def fit_sk2(an0):
        def func(eta_m):
            _, an = fr_an(ca1, ca2, ca3, eta_m)
            return an-an0
        return bisect(func, 0, 0.5)

    eta_m = fit_sk2(an0)
    fr, an = fr_an(ca1, ca2, ca3, eta_m)
    print '%.6f %.6f %.6f' % (fr, an, eta_m)
