namespace rbc {
__device__ curandState rrnd[MAX_RND_NUM];

__device__ float rnd(int i1, int i2) {
    int id = i1*RBCnv + i2;
    curandState *state = &rrnd[id];
    return curand_normal(state);
}

__global__ void rnd_ini() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nv = RBCnv;
  int i1 = tid % nv, i2 = tid / nv;
  unsigned long long seed = 1, sequence = i1*nv+i2, offset = 0;
  curandState *state;

  if (i1 >= nv) return;
  if (i2 >= i1) return;

  state = &rrnd[i1*nv+i2];
  curand_init(seed, sequence, offset, state);
  state = &rrnd[i2*nv+i1];
  curand_init(seed, sequence, offset, state);
}

__dfi__ void frnd0(double dx, double dy, double dz, double W[], /**/ double f[]) {
  /* see (3.25) in Fedosov, D. A. Multiscale modeling of blood flow
     and soft matter. Brown Uni, 2010 */
  enum {XX, YY, ZZ,   XY, XZ, YZ};
  enum {X, Y, Z};
  enum {nd = 3};   /* [n]umber of [d]imensions */
  double trW; /* trace */
  double gC = RBCgammaC, gT = RBCgammaT, kbT = RBCkbT;
  double k1, k2; /* aux scalar */
  double wex, wey, wez; /* dot(W, e) */
  k1 = rsqrt(dx*dx + dy*dy + dz*dz);
  double ex = k1*dx, ey = k1*dy, ez = k1*dz;

  trW = W[XX] + W[YY] + W[ZZ];
  W[XX] -= trW/nd; W[YY] -= trW/nd; W[ZZ] -= trW/nd;

  wex = W[XX]*ex + W[XY]*ey + W[XZ]*ez;
  wey = W[XY]*ex + W[YY]*ey + W[YZ]*ez;
  wez = W[XZ]*ex + W[YZ]*ey + W[ZZ]*ez;

  k1 = 2 * sqrt(kbT * gT) / sqrt(dt);
  f[X] = k1*wex; f[Y] = k1*wey; f[Z] = k1*wez;
  k2 = sqrt(2 * kbT * (3 * gC - gT)) / 3 /sqrt(dt);
  f[X] += k2*trW*ex; f[Y] += k2*trW*ey; f[Z] += k2*trW*ez;
}

__dfi__ float3 frnd(float3 r1, float3 r2, int i1, int i2) {
  enum {X, Y, Z};
  enum {nd = 3};   /* [n]umber of [d]imensions */
  double f[nd], W[nd*(nd+1)/2]; /* symmetric part of Wiener processes increments */
  double dx = r1.x - r2.x, dy = r1.y - r2.y, dz = r1.z - r2.z;
  int c;
  for (c = 0; c < nd*(nd+1)/2; c++) W[c] = rnd(i1, i2);
  frnd0(dx, dy, dz, W, /**/ f);
  return make_float3(f[X], f[Y], f[Z]);
}
} /* namespace rbc */
