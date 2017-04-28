import efit as ef
import numpy as np

th = np.pi/4;
center = [0, 0, 0];
rot = [[np.cos(th), -np.sin(th), 0],
       [np.sin(th),  np.cos(th), 0],
       [0          , 0         , 1]]
radii = [1, 0.5, 0.5];
ef.ellipsoid_dump_ply('e.ply', center, rot, radii)
