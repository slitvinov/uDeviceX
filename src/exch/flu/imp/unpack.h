static void upload_data(int n, const hBags *h, /**/ dBags *d) {
    int i, c;
    size_t sz;
    for (i = 0; i < n; ++i) {
        c = h->counts[i];
        sz = c * h->bsize;
        d::MemcpyAsync(d->data[i], h->data[i], sz, H2D);
    }
}

/* upload recved data on the device */
void unpack(Unpack *u) {
    upload_data(NFRAGS, &u->hfss, /**/ &u->dfss);
    upload_data(NFRAGS, &u->hpp, /**/ &u->dpp);
    if (multi_solvent)
        upload_data(NFRAGS, &u->hcc, /**/ &u->dcc);
}