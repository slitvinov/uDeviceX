static void pp2rr(const Particle *pp, const int n, float *rr) {
    for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c)
    rr[3*i + c] = pp[i].r[c];
}

static void gen_from_strt(int maxp, const Coords *coords, const int id, int *ns, int *nps, int *n, float *rr0_hst, Solid *ss_hst) {
    Particle *pp;
    EMALLOC(maxp, &pp);
    restart_read_pp(coords, "rig", RESTART_TEMPL, pp, nps);
    pp2rr(pp, *nps, rr0_hst);
    EFREE(pp);

    restart_read_ss(coords, "rig", id, ss_hst, ns);
    *n = *ns * (*nps);
}

void rig_strt_quants(const Coords *coords, const int id, RigQuants *q) {
    gen_from_strt(q->maxp, coords, id, /**/ &q->ns, &q->nps, &q->n, q->rr0_hst, q->ss_hst);
    gen_pp_hst(q->ns, q->rr0_hst, q->nps, /**/ q->ss_hst, q->pp_hst);
    gen_ipp_hst(q->ss_hst, q->ns, q->nv, q->hvv, /**/ q->i_pp_hst);
    cpy_H2D(q);
}

static void rr2pp(const float *rr, const int n, Particle *pp) {
    for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) {
        pp[i].r[c] = rr[3*i + c];
        pp[i].v[c] = 0;
    }
}

static void strt_dump_templ0(const Coords *coords, const int nps, const float *rr0_hst) {
    Particle *pp;
    EMALLOC(nps, &pp);
    rr2pp(rr0_hst, nps, pp);

    restart_write_pp(coords, "rig", RESTART_TEMPL, pp, nps);
    
    EFREE(pp);
}

void rig_strt_dump_templ(const Coords *coords, const RigQuants *q) {
    strt_dump_templ0(coords, q->nps, q->rr0_hst);
}

void rig_strt_dump(const Coords *coords, const int id, const RigQuants *q) {
    restart_write_ss(coords, "rig", id, q->ss_hst, q->ns);
}
