static void setup_edg0(const float *rr, AdjMap m, /**/ Edg *edg) {
    int i0, i1, i2;
    const float *r0, *r1, *r2;
    float r01[3], r12[3], r20[3];
    float a, b, c, A; /* edges and area */

    i0 = m.i0; i1 = m.i1; i2 = m.i2;

    r0 = &rr[3*i0]; r1 = &rr[3*i1]; r2 = &rr[3*i2];

    diff(r0, r1, /**/ r01);
    diff(r1, r2, /**/ r12);
    diff(r2, r0, /**/ r20);

    a = vabs(r01); b = vabs(r12); c = vabs(r20);
    A = area_heron(a, b, c);

    edg->a = a; edg->A = A;
}

static void setup_edg1(int md, int nv, Adj *adj, const float *rr, /**/
                       Edg *edg, float *ptotArea) {
    int valid, i;
    AdjMap m;
    float totArea;

    totArea = 0;
    for (i = 0; i < md*nv; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        setup_edg0(rr, m, /**/ &edg[i]);
        totArea += edg[i].A;
    }
    totArea /= 3; /* seen every face three times */

    msg_print("totArea: %g", totArea);
    *ptotArea = totArea;
}

static void setup_edg(int md, int nv, Adj *adj, /**/ Edg *dev, float *totArea) {
    const float *rr;
    Edg *hst;
    const char *path = "rbc.stress.free";
    OffRead *cell;
    UC(off_read(path, &cell));
    rr = off_get_vert(cell);
    if (nv != off_get_nv(cell))
        ERR("nv=%d != off_get_nv(cell)=%d", nv, off_get_nv(cell));

    EMALLOC(md*nv, &hst);
    UC(setup_edg1(md, nv, adj, rr, /**/ hst, totArea));
    cH2D(dev, hst, md*nv);
    EFREE(hst);
    off_fin(cell);
}

static void setup_anti(int md, int nv, Adj *adj, /**/ int *dev) {
    int n;
    int *hst;
    n = md*nv;
    EMALLOC(n, &hst);
    adj_get_anti(md, nv, adj, /**/ hst);
    cH2D(dev, hst, n);
    EFREE(hst);
}

static void setup0(int md, int nv, Adj *adj, /**/
                   int *anti, Edg *edg, float *totArea) {
    if (RBC_STRESS_FREE) UC(setup_edg(md,  nv, adj, /**/ edg, totArea));
    if (RBC_RND)         UC(setup_anti(md, nv, adj, /**/ anti));
}

static void setup(int md, int nt, int nv, const int4 *tt, /**/ RbcQuants *q) {
    Adj *adj;
    UC(adj_ini(md, nt, nv, tt, /**/ &adj));
    UC(adj_view_ini(adj, /**/ &q->adj_v));
    UC(setup0(md, nv, adj, /**/ q->shape.anti, q->shape.edg, &q->shape.totArea));
    UC(adj_fin(adj));
}
