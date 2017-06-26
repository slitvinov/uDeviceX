namespace rbc {
#define fst(t) ( (t).x )
#define scn(t) ( (t).y )
#define thr(t) ( (t).z )
#define __dfi__ __device__ __forceinline__

texture<float2, 1, cudaReadModeElementType> texV;
texture<int, 1, cudaReadModeElementType> texAdjV;
texture<int, 1, cudaReadModeElementType> texAdjV2;
texture<int4, cudaTextureType1D> texTriangles4;
__constant__ float A[4][4];

__device__ float3 fangle(float3 a, float3 b, float3 c, float area, float volume) {
  double Ak, A0, n_2, cA, cV,
	r, xx, b_wlc, kp, b_pow, ka0, kv0, x0, l0, lmax,
	kbToverp;

  float3 ab = b - a, ac = c - a, bc = c - b;
  float3 nn = cross(ab, ac); /* normal */

  Ak = 0.5 * sqrt(dot(nn, nn));

  A0 = RBCtotArea / (2.0 * RBCnv - 4.);
  n_2 = 1.0 / Ak;
  ka0 = RBCka / RBCtotArea;
  cA =
      -0.25 * (ka0 * (area - RBCtotArea) * n_2) -
      RBCkd * (Ak - A0) / (4. * A0 * Ak);

  kv0 = RBCkv / (6.0 * RBCtotVolume);
  cV = kv0 * (volume - RBCtotVolume);
  float3 FA = cA * cross(nn, bc);
  float3 FV = cV * cross( c,  b);

  r = sqrt(dot(ab, ab));
  r = r < 0.0001 ? 0.0001 : r;
  l0 = sqrt(A0 * 4.0 / sqrt(3.0));
  lmax = l0 / RBCx0;
  xx = r / lmax;

  kbToverp = RBCkbT / RBCp;
  b_wlc =
      kbToverp * (0.25 / ((1.0 - xx) * (1.0 - xx)) - 0.25 + xx) /
      r;

  x0 = RBCx0;
  kp =
      (RBCkbT * x0 * (4 * x0 * x0 - 9 * x0 + 6) * l0 * l0) /
      (4 * RBCp * (x0 - 1) * (x0 - 1));
  b_pow = -kp / pow(r, RBCmpow) / r;

  return make_float3(
		     FA.x + FV.x + (b_wlc + b_pow) * ab.x,
		     FA.y + FV.y + (b_wlc + b_pow) * ab.y,
		     FA.z + FV.z + (b_wlc + b_pow) * ab.z
		     );
}

__dfi__ float3 fvisc(float3 r1, float3 r2, float3 u1, float3 u2) {
  float3 du = u2 - u1, dr = r1 - r2;
  double gC = RBCgammaC, gT = RBCgammaT;

  return gT                             * du +
	 gC * dot(du, dr) / dot(dr, dr) * dr;
}

template <int update>
__dfi__ float3 _fdihedral(float3 v1, float3 v2, float3 v3,
					     float3 v4) {
  double overIksiI, overIdzeI, cosTheta, IsinThetaI2, sinTheta_1,
    beta, b11, b12, phi, sint0kb, cost0kb;

  float3 ksi = cross(v1 - v2, v1 - v3), dze = cross(v3 - v4, v2 - v4);
  overIksiI = rsqrt(dot(ksi, ksi));
  overIdzeI = rsqrt(dot(dze, dze));

  cosTheta = dot(ksi, dze) * overIksiI * overIdzeI;
  IsinThetaI2 = 1.0 - cosTheta * cosTheta;

  sinTheta_1 = copysignf
    (rsqrt(max(IsinThetaI2, 1.0e-6)),
     dot(ksi - dze, v4 - v1)); // ">" because the normals look inside

  phi = RBCphi / 180.0 * M_PI;
  sint0kb = sin(phi) * RBCkb;
  cost0kb = cos(phi) * RBCkb;
  beta = cost0kb - cosTheta * sint0kb * sinTheta_1;

  b11 = -beta *  cosTheta * overIksiI * overIksiI;
  b12 =  beta * overIksiI * overIdzeI;

  if (update == 1) {
    return b11 * cross(ksi, v3 - v2)  + b12 * cross(dze, v3 - v2);
  } else if (update == 2) {
    double b22 = -beta * cosTheta * overIdzeI * overIdzeI;
    return  b11 *  cross(ksi, v1 - v3) +
	    b12 * (cross(ksi, v3 - v4) + cross(dze, v1 - v3)) +
	    b22 *  cross(dze, v3 - v4) ;
  } else
    return make_float3(0, 0, 0);
}

__device__ void tt2r(float2 t1, float2 t2, /**/ float3 *r) {
  r->x = fst(t1); r->y = scn(t1); r->z = fst(t2);
}

__device__ void ttt2ru(float2 t1, float2 t2, float2 t3, /**/ float3 *r, float3 *u) {
  r->x = fst(t1); r->y = scn(t1); r->z = fst(t2);
  u->x = scn(t2); u->y = fst(t3); u->z = scn(t3);
}

template <int nv>
__device__ float3 adj_tris(float2 t0, float2 t1, float *av) {
  int md = 7; /* was degreemax */
  int i1, i2, i3, ne, cid, lid, off;
  bool valid;
  float2 t2;
  float3 r1, u1, r2, u2, r3; /* no `u3' */
  float3 f; /* force */
  float area, volume;

  i1 = (threadIdx.x + blockDim.x * blockIdx.x) / md;
  ne = (threadIdx.x + blockDim.x * blockIdx.x) % md; /* neighbor id */
  cid = i1 / nv; /* cell  id */
  lid = i1 % nv; /* local id */

  t2 = tex1Dfetch(texV, i1 * 3 + 2);
  ttt2ru(t0, t1, t2, /**/ &r1, &u1);

  i2 = tex1Dfetch(texAdjV,   ne            + md * lid);
  i3 = tex1Dfetch(texAdjV, ((ne + 1) % md) + md * lid);

  valid = i2 != -1;
  if (i3 == -1 && valid) i3 = tex1Dfetch(texAdjV, 0 + md * lid);

  if (valid) {
    off = cid * nv * 3; /* offset */
    t0 = tex1Dfetch(texV, off + i2 * 3 + 0);
    t1 = tex1Dfetch(texV, off + i2 * 3 + 1);
    t2 = tex1Dfetch(texV, off + i2 * 3 + 2);
    ttt2ru(t0, t1, t2, /**/ &r2, &u2);

    t0 = tex1Dfetch(texV, off + i3 * 3 + 0);
    t1 = tex1Dfetch(texV, off + i3 * 3 + 1);
    tt2r(t0, t1, /**/ &r3);

    area = av[2*cid]; volume = av[2*cid + 1];
    f  = fangle(r1, r2, r3, area, volume);
    f +=  fvisc(r1, r2,     u1, u2);
    f +=   frnd(r1, r2,     i1, i2);
    return f;
  }
  return make_float3(-1.0e10, -1.0e10, -1.0e10);
}

template <int nvertices>
__device__ float3 adj_dihedrals(float2 tmp0, float2 tmp1) {
  int degreemax = 7;
  int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;
  int lid = pid % nvertices;
  int offset = (pid / nvertices) * nvertices * 3;
  int neighid = (threadIdx.x + blockDim.x * blockIdx.x) % degreemax;

  float3 v0 = make_float3(tmp0.x, tmp0.y, tmp1.x);

  /*
	 v4
       /   \
     v1 --> v2 --> v3
       \   /
	 V
	 v0

   dihedrals: 0124, 0123
  */

  int idv1, idv2, idv3, idv4;
  idv1 = tex1Dfetch(texAdjV, neighid + degreemax * lid);
  bool valid = idv1 != -1;

  idv2 = tex1Dfetch(texAdjV, ((neighid + 1) % degreemax) + degreemax * lid);

  if (idv2 == -1 && valid) {
    idv2 = tex1Dfetch(texAdjV, 0 + degreemax * lid);
    idv3 = tex1Dfetch(texAdjV, 1 + degreemax * lid);
  } else {
    idv3 =
	tex1Dfetch(texAdjV, ((neighid + 2) % degreemax) + degreemax * lid);
    if (idv3 == -1 && valid) idv3 = tex1Dfetch(texAdjV, 0 + degreemax * lid);
  }

  idv4 = tex1Dfetch(texAdjV2, neighid + degreemax * lid);

  if (valid) {
    float2 tmp0 = tex1Dfetch(texV, offset + idv1 * 3 + 0);
    float2 tmp1 = tex1Dfetch(texV, offset + idv1 * 3 + 1);
    float2 tmp2 = tex1Dfetch(texV, offset + idv2 * 3 + 0);
    float2 tmp3 = tex1Dfetch(texV, offset + idv2 * 3 + 1);
    float2 tmp4 = tex1Dfetch(texV, offset + idv3 * 3 + 0);
    float2 tmp5 = tex1Dfetch(texV, offset + idv3 * 3 + 1);
    float2 tmp6 = tex1Dfetch(texV, offset + idv4 * 3 + 0);
    float2 tmp7 = tex1Dfetch(texV, offset + idv4 * 3 + 1);

    float3 v1 = make_float3(tmp0.x, tmp0.y, tmp1.x);
    float3 v2 = make_float3(tmp2.x, tmp2.y, tmp3.x);
    float3 v3 = make_float3(tmp4.x, tmp4.y, tmp5.x);
    float3 v4 = make_float3(tmp6.x, tmp6.y, tmp7.x);

    return _fdihedral<1>(v0, v2, v1, v4) + _fdihedral<2>(v1, v0, v2, v3);
  }
  return make_float3(-1.0e10, -1.0e10, -1.0e10);
}

template <int nvertices>
__global__ void force(int nc, float *__restrict__ av,
			    float *acc) {
  int degreemax = 7;
  int pid = (threadIdx.x + blockDim.x * blockIdx.x) / degreemax;

  if (pid < nc * nvertices) {
    float2 tmp0 = tex1Dfetch(texV, pid * 3 + 0);
    float2 tmp1 = tex1Dfetch(texV, pid * 3 + 1);

    /* all triangles and dihedrals adjusting to `pid` */
    float3 f = adj_tris<nvertices>(tmp0, tmp1, av);
    f += adj_dihedrals<nvertices>(tmp0, tmp1);

    if (f.x > -1.0e9) {
      atomicAdd(&acc[3 * pid + 0], f.x);
      atomicAdd(&acc[3 * pid + 1], f.y);
      atomicAdd(&acc[3 * pid + 2], f.z);
    }
  }
}

__global__ void addKernel(float* axayaz, float* __restrict__ addfrc, int n) {
  uint pid = threadIdx.x + blockIdx.x * blockDim.x;
  if (pid < n) axayaz[3*pid + 0] += addfrc[pid];
}

__dfi__ float3 tex2vec(int id) {
  float2 tmp0 = tex1Dfetch(texV, id + 0);
  float2 tmp1 = tex1Dfetch(texV, id + 1);
  return make_float3(tmp0.x, tmp0.y, tmp1.x);
}

__dfi__ float2 warpReduceSum(float2 val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val.x += __shfl_down(val.x, offset);
    val.y += __shfl_down(val.y, offset);
  }
  return val;
}

__global__ void areaAndVolumeKernel(float *totA_V) {
#define sq(a) (a)*(a)
#define abscross2(a, b) \
  (sq((a).y*(b).z - (a).z*(b).y) +  \
   sq((a).z*(b).x - (a).x*(b).z) +  \
   sq((a).x*(b).y - (a).y*(b).x))
#define abscross(a, b) sqrt(abscross2(a, b)) /* |a x b| */

  float2 a_v = make_float2(0.0, 0.0);
  int cid = blockIdx.y;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < RBCnt;
       i += blockDim.x * gridDim.x) {
    int4 ids = tex1Dfetch(texTriangles4, i);

    float3 v0(tex2vec(3 * (ids.x + cid * RBCnv)));
    float3 v1(tex2vec(3 * (ids.y + cid * RBCnv)));
    float3 v2(tex2vec(3 * (ids.z + cid * RBCnv)));

    a_v.x += 0.5 * abscross(v1 - v0, v2 - v0);
    a_v.y += 0.1666666667 *
      ((v0.x*v1.y-v0.y*v1.x)*v2.z +
       (v0.z*v1.x-v0.x*v1.z)*v2.y +
       (v0.y*v1.z-v0.z*v1.y)*v2.x);
  }
  a_v = warpReduceSum(a_v);
  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicAdd(&totA_V[2 * cid + 0], a_v.x);
    atomicAdd(&totA_V[2 * cid + 1], a_v.y);
  }
#undef sq
#undef abscross2
#undef abscross
}

__global__ void transformKernel(float *xyzuvw, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float x = xyzuvw[6 * i + 0];
  float y = xyzuvw[6 * i + 1];
  float z = xyzuvw[6 * i + 2];

  xyzuvw[6 * i + 0] = A[0][0] * x + A[0][1] * y + A[0][2] * z + A[0][3];
  xyzuvw[6 * i + 1] = A[1][0] * x + A[1][1] * y + A[1][2] * z + A[1][3];
  xyzuvw[6 * i + 2] = A[2][0] * x + A[2][1] * y + A[2][2] * z + A[2][3];
}

#undef fst
#undef scn
#undef thr
} /* namespace rbc */
