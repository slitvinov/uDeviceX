#!/usr/bin/python

import numpy as np

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

# test
phi = np.linspace(0, np.pi)
x, y = np.cos(phi), np.sin(phi)
a, b  = 1.0, 2.0
fr = 42.0
vx, vy = fr*a/b*y, -fr*b/a*x
print vfit(x, y, vx, vy, a/b)
