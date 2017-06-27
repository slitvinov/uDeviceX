/* maximum degree */
#define md ( RBCmd )

void reg_idx(int i, int j, float l0, /**/ int *idx, float *ll) {
  i *= md;
  int k;
  for (;;) {
    k = idx[i++];
    if      (k  == j) break;
    else if (k != -1) {
      idx[i] = j; ll[i] = l0; break;
    }
  }
}

void reg(const std::vector<Particle>& pp, int i1, int i2, /**/ int *idx, float *ll) { /* register edge */
  enum {X, Y, Z};
  float dx, dy, dz, l0;
  const float *r1 = pp[i1].r;
  const float *r2 = pp[i2].r;

  dx = r1[X] - r2[X];
  dy = r1[Y] - r2[Y];
  dz = r1[Z] - r2[Z];
  l0 = sqrt(dx*dx + dy*dy + dz*dz);
  reg_idx(i1,   i2, l0, /**/ idx, ll);
}

void sfree_ini0(const std::vector<Particle>& pp, const std::vector<int3>& tris, /**/ int *idx, float *ll) {
  int i, i1, i2, i3, n;
  n = tris.size();
  for (i = 0; i < n; i++) {
    int3 t = tris[i];
    i1 = t.x; i2 = t.y; i3 = t.z;
    reg(pp, i1, i2, /**/ idx, ll); reg(pp, i2, i1, /**/ idx, ll);
    reg(pp, i1, i3, /**/ idx, ll); reg(pp, i3, i1, /**/ idx, ll);
    reg(pp, i2, i3, /**/ idx, ll); reg(pp, i2, i3, /**/ idx, ll);
  }
}

void sfree_ini(const std::vector<Particle>& pp, const std::vector<int3>& tris) {
  int idx[MAX_VERT*md];  /* edge search structure */
  float ll[MAX_VERT*md]; /* array of lengths for edges */
  int i, n;
  n = pp.size();
  for (i = 1; i < md*n; i++) idx[i] = -1;
  sfree_ini0(pp, tris, /**/ idx, ll);

  int offset = 0;
  CC(cudaMemcpyToSymbol(k_rbc::idx, idx, sizeof(idx[0])*n*md, offset, H2D));
  CC(cudaMemcpyToSymbol(k_rbc::ll,   ll, sizeof( ll[0])*n*md, offset, H2D));
}

#undef md
