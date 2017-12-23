int26 get_counts(Unpack *u) {
    int26 cc;
    memcpy(cc.d, u->hpp.counts, NFRAGS * sizeof(int));
    return cc;
}

static void upload(int nfrags, const hBags h, /**/ dBags d) {
    int i, c;
    size_t sz;
    data_t *src, *dst;

    for (i = 0; i < nfrags; ++i) {
        c = h.counts[i];
        if (c) {
            sz  = h.bsize * c;
            dst = d.data[i];
            src = h.data[i];
            CC(d::MemcpyAsync(dst, src, sz, H2D));
        }
    }    
}

static void shift_pp(int nfrags, const int counts[], /**/ dBags d) {
    int i, n;
    Particle *pp;

    for (i = 0; i < nfrags; ++i) {
        n = counts[i];
        if (n) {
            pp = (Particle *) d.data[i];
            KL(dev::shift_one_frag, (k_cnf(n)), (n, i, /**/ pp));
        }
    }    
}

Pap26 upload_shift(Unpack *u) {
    upload(NFRAGS, u->hpp, /**/ u->dpp);
    shift_pp(NFRAGS, u->hpp.counts, /**/ u->dpp);    
    Pap26 wrap;
    bag2Sarray(u->dpp, &wrap);
    return wrap;
}

static void unpack_ff(int nfrags, Fop26 ff, Map map, int nw, /**/ FoWrap *ww) {
    int i, stride;
    stride = nfrags + 1;
    const FoWrap *w;
    PackHelper ph;
    
    for (i = 0; i < nw; ++i) {
        w = ww + i;
        ph.starts  = map.starts  + i * stride;
        ph.offsets = map.offsets + i * stride;
        memcpy(ph.indices, map.ids, nfrags * sizeof(int*));

        KL(dev::unpack_ff, (14 * 16, 128), (ff, ph, /**/ w->ff));
    }
}

void unpack_ff(const UnpackF *u, const Pack *p, int nw, /**/ FoWrap *ww) {
    Fop26 wrap;
    upload(NFRAGS, u->hff, /**/ u->dff);
    bag2Sarray(u->dff, &wrap);
    unpack_ff(NFRAGS, wrap, p->map, nw, /**/ ww);
}