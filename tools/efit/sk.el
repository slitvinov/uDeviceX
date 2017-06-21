#!/usr/bin/env octave-qf

1;
function varargout = fscn(f, fmt) # simpler fscanf
  l = fgets(f);
  [varargout{1:nargout}] = strread(l, fmt);
endfunction

function e = dbl(e); e = double(e); endfunction

function read_header(f)
  global nv nf ne
  fscn(f, "%s") # skip OFF
  [nv, nf, ne] = fscn(f, "%d %d %d\n");
  nv = dbl(nv); nf = dbl(nf); ne = dbl(ne);
endfunction

function read_data(f)
  global nv nf D F
  ndim = 3; nfp = 3;
  D = dlmread(f, ' ', [0, 0,   nv - 1, ndim - 1]);
  F = dlmread(f, ' ', [0, 0,   nf - 1, nfp     ]);
endfunction

function read(fn)
  f = fopen(fn);
  read_header(f);
  read_data(f);
  fclose(f);
endfunction

function write_header(f)
  
endfunction

function write_data(f)
  
endfunction

function write(fn)
  f = fopen(fn, "w");
  write_header(f);
  write_data(f);
  fclose(f);
endfunction

fn = argv(){1};
read(fn);

fn = argv(){2};
write(fn);


# TEST: sk.t0
# sk.el > sk.out.txt
#
