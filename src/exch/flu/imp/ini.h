void eflu_pack_ini(bool colors, int3 L, int maxd, EFluPack **pack) {
    int i, nc, cap[NBAGS], ncs[NBAGS];
    size_t sz;
    EFluPack *p;

    EMALLOC(1, pack);
    p = *pack;

    p->L = L;
    frag_hst::estimates(L, NFRAGS, maxd, /**/ cap);
    cap[BULK] = 0;
    memcpy(p->cap.d, cap, NFRAGS * sizeof(int));
    
    for (i = 0; i < NFRAGS; ++i) {
        ncs[i] = nc = frag_hst::ncell(L, i) + 1;
        sz = nc * sizeof(int);
        CC(d::Malloc((void**) &p->bcc.d[i], sz));
        CC(d::Malloc((void**) &p->bss.d[i], sz));
        CC(d::Malloc((void**) &p->fss.d[i], sz));
        
        sz = cap[i] * sizeof(int);
        CC(d::Malloc((void**) &p->bii.d[i], sz));
    }
    ncs[BULK] = 0;
    
    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ &p->hpp, &p->dpp));
    if (colors)
        UC(comm_bags_ini(PINNED_DEV, NONE,  sizeof(int), cap, /**/ &p->hcc, &p->dcc));

    UC(comm_bags_ini(PINNED_HST, NONE, sizeof(int), ncs, /**/ &p->hfss, NULL));

    memcpy(p->hfss.counts, ncs, sizeof(ncs));
    
    sz = 26 * sizeof(int);
    CC(d::Malloc((void**) &p->counts_dev, sz));

    p->opt.colors = colors;
}

void eflu_comm_ini(bool colors, MPI_Comm cart, /**/ EFluComm **com) {
    EFluComm *c;
    EMALLOC(1, com);
    c = *com;
    
    UC(comm_ini(cart, /**/ &c->pp));
    UC(comm_ini(cart, /**/ &c->fss));
    if (colors)
        UC(comm_ini(cart, /**/ &c->cc));

    c->opt.colors = colors;
}

void eflu_unpack_ini(bool colors, int3 L, int maxd, EFluUnpack **unpack) {
    int i, cap[NBAGS], ncs[NBAGS];
    EFluUnpack *u;

    EMALLOC(1, unpack);
    u = *unpack;

    u->L = L;
    frag_hst::estimates(L, NFRAGS, maxd, /**/ cap);
    cap[BULK] = 0;
    
    for (i = 0; i < NFRAGS; ++i)
        ncs[i] = frag_hst::ncell(L, i) + 1;
    ncs[BULK] = 0;
    
    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(Particle), cap, /**/ &u->hpp, &u->dpp));
    if (colors)
        UC(comm_bags_ini(PINNED_DEV, NONE,  sizeof(int), cap, /**/ &u->hcc, &u->dcc));

    UC(comm_bags_ini(PINNED_DEV, NONE, sizeof(int), ncs, /**/ &u->hfss, &u->dfss));

    u->opt.colors = colors;
}

