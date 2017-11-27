void fin(Pack *p) {
    fin_map(NBAGS, /**/ &p->map);
    fin(PINNED, DEV_ONLY, /**/ &p->hpp, &p->dpp);
    CC(d::Free(p->minext));
    CC(d::Free(p->maxext));

    if (rbc_ids) {
        fin_host_map(NBAGS, /**/ &p->hmap);
        fin(HST_ONLY, HST_ONLY, /**/ &p->hii, NULL);
    }
}

void fin(Comm *c) {
    fin(&c->pp);
    if (rbc_ids)
        fin(&c->ii);
}

void fin(Unpack *u) {
    fin(HST_ONLY, NONE, /**/ &u->hpp, NULL);
    if (rbc_ids)
        fin(HST_ONLY, NONE, /**/ &u->hii, NULL);
}