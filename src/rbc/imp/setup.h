static void edg_sfree1(Adj *adj, /**/ float *a_dev, float *A_dev) {
    /* stress free ini */
    const char *path = "rbc.stress.free";
    int n;
    const float *rr;
    float *a_hst, *A_hst;
    OffRead *cell;
    RbcShape *shape;
    UC(off_read_off(path, &cell));
    rr = off_get_vert(cell);
    UC(rbc_shape_ini(adj, rr, /**/ &shape));
    n = adj_get_max(adj);

    rbc_shape_edg(shape, &a_hst);
    rbc_shape_area(shape, &A_hst);

    cH2D(a_dev, a_hst, n);
    cH2D(A_dev, A_hst, n);

    UC(rbc_shape_fin(shape));
    UC(off_fin(cell));
}

static void anti(Adj *adj, /**/ int *dev) {
    int n;
    int *hst;
    n = adj_get_max(adj);
    EMALLOC(n, &hst);
    adj_get_anti(adj, /**/ hst);
    cH2D(dev, hst, n);
    EFREE(hst);
}

static void setup0(Adj *adj, /**/ Shape *shape) {
    if (RBC_STRESS_FREE) UC(edg_sfree1(adj, /**/  shape->a,   shape->A));
    if (RBC_RND)         UC(anti(adj, /**/ shape->anti));
}

static void setup(int md, int nt, int nv, const int4 *tt, /**/ RbcQuants *q) {
    Adj *adj;
    UC(adj_ini(md, nt, nv, tt, /**/ &adj));
    UC(adj_view_ini(adj, /**/ &q->adj_v));
    UC(setup0(adj, /**/ &q->shape));
    UC(adj_fin(adj));
}
