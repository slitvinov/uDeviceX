#!/usr/bin/env octave-qf

1;
global PLY_FMT
PLY_FMT = "bin";

function ini()
  global PLY_FMT fmt
  if eq(PLY_FMT, "ascii"); fmt = "ascii"; else fmt = "binary_little_endian"; endif
endfunction

function varargout = fscn(f, fmt) # simpler fscanf
  l = fgets(f);
  [varargout{1:nargout}] = strread(l, fmt);
endfunction

function e = dbl(e); e = double(e); endfunction

function read_header(f)
  global nv nf ne
  fscn(f, "%s"); # skip OFF
  [nv, nf, ne] = fscn(f, "%d %d %d\n");
  nv = dbl(nv); nf = dbl(nf); ne = dbl(ne);
endfunction

function read_data(f)
  global nv nf xx yy zz
  global ff1 ff2 ff3
  
  ndim = 3; nfp = 3;
  X = 1; Y = 2; Z = 3;
  D = dlmread(f, ' ', [0, 0,   nv - 1, ndim - 1]); D = D';
  xx = D(X, :); yy = D(Y, :); zz = D(Z, :);
  
  F = dlmread(f, ' ', [0, 0,   nf - 1, nfp     ]); F = F';
  i = 2; ff1 = F(i++, :); ff2 = F(i++, :); ff3 = F(i++, :);
endfunction

function read(fn)
  f = fopen(fn);
  read_header(f);
  read_data(f);
  fclose(f);
endfunction

function write_header(f)
  global nv nf fmt
  w = @(fmt, varargin) fprintf(f, fmt, [varargin{:}]);
  w("ply\n")
  w("format %s 1.0\n", fmt);
  w("element vertex %d\n", nv);
  w("property float x\n");
  w("property float y\n");
  w("property float z\n");
  w("property float u\n");
  w("property float v\n");
  w("property float w\n");
  w("element face %d\n", nf);
  w("property list int int vertex_index\n");
  w("end_header\n");
endfunction

function r = eq(a, b); r = strcmp(a, b); endfunction

function write_ascii(f, D)
  dlmwrite(f, D', ' ')
endfunction

function write_bin(f, D, type); fwrite(f, D, type); endfunction

function dwrite(f, D, type) # data write
  global PLY_FMT
  if eq(PLY_FMT, "ascii"); write_ascii(f, D); else write_bin(f, D, type); endif
endfunction

function write_vert(f)
  global xx yy zz
  global vvx vvy vvz
  D = vertcat(xx, yy, zz, vvx, vvy, vvz);
  dwrite(f, D, 'float32');
endfunction

function write_face(f)
  global ff1 ff2 ff3
  s = size(ff1); nvp0 = 3;
  nvp = 3 * ones(s);
  F = vertcat(nvp, ff1, ff2, ff3);
  dwrite(f, F, 'int32');
endfunction

function write(fn)
  f = fopen(fn, "w");
  write_header(f);
  write_vert(f);
  write_face(f);  
  fclose(f);
endfunction

function vel_ini()
  global xx yy zz vvx vvy vvz
  s = size(xx);
  vvx = vvy = vvz = zeros(s);
endfunction

function vel_sk(ax, ay, fr)
  global xx yy   vvx vvy
  vvx = -ax/ay * yy;
  vvy =  ay/ax * xx;
endfunction

function swap() # yz -> zy
  global  yy zz
  global vvy vvz
  t =  yy;  yy =  zz;  zz = t;
  t = vvy; vvy = vvz; vvz = t;
endfunction

function def(ax, ay, az) # deform
  global xx yy zz
  xx = ax * xx;
  yy = ay * yy;
  zz = az * zz;
endfunction

function rot(t)
  global xx yy
  global vvx vvy
  
endfunction

ini();
fi = argv(){1};
fo = argv(){2};

read(fi);
vel_ini();
def(ax=3, ay=1, az=1);
vel_sk(ax, ay, fr=42);
# rot(theta=1)

swap();
write(fo);

# TEST: sk.t0
# sk.el test_data/sph.498.off sk.out.ply
#
