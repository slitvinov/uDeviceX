static void get_num_capacity(int maxns, /**/ int numc[NBAGS]) {
    // TODO save memory here?
    for (int i = 0; i < NBAGS; ++i)
        numc[i] = maxns;
}

void ini(int maxns, int nv, Pack *p) {
    int numc[NBAGS];
    get_num_capacity(maxns, /**/ numc);

    UC(ini_map(NBAGS, numc, /**/ &p->map));

    /* one datum is here a full mesh, so bsize is nv * sizeof(Particle) */
    UC(ini(PINNED, DEV_ONLY, nv * sizeof(Particle), numc, /**/ &p->hipp, &p->dipp));
    
    UC(ini(PINNED, DEV_ONLY, sizeof(Solid), numc, /**/ &p->hss, &p->dss));
}

void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c) {
    UC(ini(comm, /*io*/ tg, /**/ &c->ipp));
    UC(ini(comm, /*io*/ tg, /**/ &c->ss));
}

void ini(int maxns, int nv, Unpack *u) {
    int numc[NBAGS];
    get_num_capacity(maxns, /**/ numc);

    /* one datum is here a full mesh, so bsize is nv * sizeof(Particle) */
    UC(ini(HST_ONLY, NONE, nv * sizeof(Particle), numc, /**/ &u->hipp, NULL));

    UC(ini(HST_ONLY, NONE, sizeof(Solid), numc, /**/ &u->hss, NULL));
}