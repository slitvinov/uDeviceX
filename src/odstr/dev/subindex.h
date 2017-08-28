namespace odstr { namespace sub { namespace dev {

struct Ce { /* coordinates of a cell */
    int ix, iy, iz;
    int id; /* linear index */
};

__device__ void Pa2Ce(const Part *p, /**/ Ce *c) {
    /* particle to cell coordinates */
    r2c(p->r, /**/ &c->ix, &c->iy, &c->iz, &c->id);
}

__device__ void regCe(Ce *c, int i, /*io*/ int *counts, /**/ uchar4 *subids) {
    /* a particle `i` will lives in `c'. [Reg]ister it. */
    int subindex;
    subindex = atomicAdd(counts + c->id, 1);
    subids[i] = make_uchar4(c->ix, c->iy, c->iz, subindex);
}

__device__ void checkPav(const Part *p) { /* check particle velocity */
    enum {X, Y, Z};
    const float *v = p->v;
    check_vel(v[X], XS);
    check_vel(v[Y], YS);
    check_vel(v[Z], ZS);
}

__device__ void subindex0(int i, const int strt[], const Part *p, /*io*/ int *counts, /**/ uchar4 *subids) {
    /* i: particle index */
    enum {X, Y, Z};
    Ce c; /* cell coordinates */

    Pa2Ce(p, /**/ &c); /* to cell coordinates */
    checkPav(p); /* check velocity */
    regCe(&c, i, /*io*/ counts, subids); /* register */
}

__global__ void subindex(const int n, const int strt[], float2 *pp, /*io*/ int *counts, /**/ uchar4 *subids) {
    enum {X, Y, Z};
    int ws, dw;  /* warp start and shift (lane) */
    Part p; /* [p]article and its [l]ocation in memory */
    Lo l;

    warpco(&ws, &dw); /* warp coordinates */
    if (ws >= n) return;
    pp2Lo(pp, n, ws, /**/ &l);
    readPart(l, /**/ &p);   /* collective */
    if (!endLo(&l, dw))
        subindex0(ws + dw, strt, /*io*/ &p, counts, /**/ subids);
}

}}} /* namespace */
