#!/usr/bin/env python

import sys
import efit as ef

ip = sys.argv[1]
center, rot, radii, chi2, _ = ef.fit_ellipsoid_ply(ip, 'o.ply', 'e.ply')
print 'center = \n', center
print 'radii  = \n', radii
print 'rot    = \n', rot
print 'chi2   = \n', chi2
