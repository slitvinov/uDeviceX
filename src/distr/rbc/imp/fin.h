void fin(Pack *p) {
    fin_map(NBAGS, /**/ &p->map);
    fin(PINNED, DEV_ONLY, /**/ &p->hpp, &p->dpp);
    CC(d::Free(p->minext));
    CC(d::Free(p->maxext));
}

void fin(Comm *c) {
    fin(&c->pp);
}

void fin(Unpack *u) {
    fin(HST_ONLY, NONE, &u->hpp, NULL);
}
