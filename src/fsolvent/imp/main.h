static void zip(const int n, const float *pp, /**/ float4 *zipped_pp, ushort4 *zipped_rr) {
    assert(sizeof(Particle) == 6 * sizeof(float)); /* :TODO: implicit dependency */
    KL(dev::zip, (k_cnf(n)), (n, pp, zipped_pp, zipped_rr));
}


void prepare(int n, const Cloud *c, /**/ BulkData *b) {
    if (n == 0) return;
    zip(n, c->pp, /**/ b->zipped_pp, b->zipped_rr);
    if (multi_solvent)
        b->colors = c->cc;
}

void bulk_forces(int n, const BulkData *b, const int *start, const int *count, /**/ Force *ff) {
    if (multi_solvent)
        flocal_color(b->zipped_pp, b->zipped_rr, b->colors, n, start, count, b->rnd, /**/ ff);
    else
        flocal(b->zipped_pp, b->zipped_rr, n, start, count, b->rnd, /**/ ff);
}
