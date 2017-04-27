#!/usr/bin/env octave-qf

fn = '../test_data/rbc.ply'

fd = fopen(fn); nl  = @() fgetl(fd); % next line
nl(); nl();
l = nl(); nv = sscanf(l, 'element vertex %d');
nl(); nl(); nl(); nl(); nl(); nl();
l = nl(); nf = sscanf(l, 'element face %d');
nl(); nl();

nvar  = 6; % x, y, z, u, v, w
D = fread(fd, [nvar     , nv], 'float32');
D = D(1:3, :);
xx = D(1, :);  yy = D(2, :);  zz = D(3, :);

[center, radii, evecs] = ellipsoid_fit(D')
