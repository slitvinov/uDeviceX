namespace odstr { namespace sub { namespace dev {

#define SAFE
#ifdef SAFE
static const float eps = 1e-6f;
static __device__ void rescue(int L, float *x) {
    if (*x < -L/2) *x = -L/2;
    if (*x >= L/2) *x =  L/2 - eps;
}
#else
static __device__ void rescue(int L, float *x) {}
#endif


static __device__ void fid2shift(int id, /**/ int s[3]) {
    enum {X, Y, Z};
    s[X] = XS * ((id     + 1) % 3 - 1);
    s[Y] = YS * ((id / 3 + 1) % 3 - 1);
    s[Z] = ZS * ((id / 9 + 1) % 3 - 1);
}

static __device__ void shiftPart(const int s[3], Part *p) {
    enum {X, Y, Z};
    float *r = p->r;
    
    r[X] += s[X];
    r[Y] += s[Y];
    r[Z] += s[Z];

    rescue(XS, r + X);
    rescue(YS, r + Y);
    rescue(ZS, r + Z);
}

static __device__ void shift_1p(int i, const int strt[], /*io*/ Part *p) {
    /* i: particle index */
    enum {X, Y, Z};
    int fid;     /* fragment id */
    int shift[3];

    fid  = k_common::fid(strt, i);
    fid2shift(fid, /**/ shift);
    shiftPart(shift, p);
}

__global__ void shift(const int n, const int strt[], /*io*/ float2 *pp) {
    int ws, dw;  /* warp start and shift (lane) */
    Part p; /* [p]article and its [l]ocation in memory */
    Lo l;

    warpco(&ws, &dw); /* warp coordinates */
    if (ws >= n) return;
    pp2Lo(pp, n, ws, /**/ &l);
    readPart(l, /**/ &p);   /* collective */
    if (!endLo(&l, dw))
        shift_1p(ws + dw, strt, /*io*/ &p);
    writePart(&p, /**/ l); /* collective */
}

}}} // namespace
