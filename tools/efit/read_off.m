#!/usr/bin/env octave-qf

1;
function [D, F]  = read_ply(fn)
  nvar  = 6; % x, y, z, u, v, w
  nv_pf = 3; % number of vertices per face (3 for triangle)

  fd = fopen(fn); nl  = @() fgetl(fd); % next line
  nl(); nl();
  l = nl(); nv = sscanf(l, 'element vertex %d');
  nl(); nl(); nl(); nl(); nl(); nl();
  l = nl(); nf = sscanf(l, 'element face %d');
  nl(); nl();

  D = fread(fd, [nvar     , nv], 'float32');
  D = D(1:3, :);

  F = fread(fd, [nv_pf + 1, nf], 'int32');
  F = F(2:end, :);
  
  fclose(fd);
endfunction
