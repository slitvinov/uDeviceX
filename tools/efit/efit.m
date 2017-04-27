#!/usr/bin/env octave-qf

fn = '../test_data/rbc.ply'

X = 1; Y = 2; Z = 3;

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
fclose(fd);

[center, radii, evecs] = ellipsoid_fit(D');
a = radii(1); b = radii(2); c = radii(3);

ndump = 100;
uu = linspace(0, 2*pi, 100);
vv = linspace(0,   pi, 100);
[uu, vv] = meshgrid (uu, vv);

n = numel(uu);
xx = zeros(1, n); yy = zeros(1, n); zz = zeros(1, n); 

for i = 1:numel(uu)
  u = uu(i); v = vv(i);
  x = a*cos(u)*sin(v); % u: [0, pi]
  y = b*sin(u)*sin(v); % v: [0, pi]
  z = c*cos(v);

  r = [x, y, z]';
  r = evecs * r;
  r = r + center;
  
  xx(i) = r(X); yy(i) = r(Y); zz(i) = r(Z);
end


disp("writing: e.3d");
fd = fopen("e.3d", "w");
fdisp(fd, "x y z sc");
sc = zeros(1, n); % fake scalar
dlmwrite(fd, [xx; yy; zz; sc]', ' ');

fclose(fd);
