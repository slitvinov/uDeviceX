namespace dpdr {
namespace sub {

static void alloc_Bbuf_frag(const int i, const int est, const int nfragcells, /**/ Bbufs *b) {
    Dalloc(&b->cum.d[i], nfragcells + 1);
    Dalloc(&b->pp.d[i], est);

    Palloc(&b->cumhst.d[i], nfragcells + 1);
    Link(&b->cumdev.d[i], b->cumhst.d[i]);

    Palloc(&b->pphst.d[i], est);
    Link(&b->ppdev.d[i], b->pphst.d[i]);
}

static void free_Bbuf_frag(const int i, /**/ Bbufs *b) {
    CC(d::Free(b->cum.d[i]));
    CC(d::Free(b->pp.d[i]));
        
    CC(d::FreeHost(b->cumhst.d[i]));
    CC(d::FreeHost(b->pphst.d[i]));
}

static void alloc_Sbuf_frag(const int i, const int est, const int nfragcells, /**/ Sbufs *b) {
    alloc_Bbuf_frag(i, est, nfragcells, /**/ b);
    Dalloc(&b->str.d[i], nfragcells + 1);
    Dalloc(&b->cnt.d[i], nfragcells + 1);
    Dalloc(&b->ii.d[i], est);
};

static void free_Sbuf_frag(const int i, /**/ Sbufs *b) {
    free_Bbuf_frag(i, /**/ b);
    CC(d::Free(b->str.d[i]));
    CC(d::Free(b->cnt.d[i]));
    CC(d::Free(b->ii.d[i]));
}

void alloc_Sbufs(const int26 estimates, const int26 nfragcells, /**/ Sbufs *b) {
    for (int i = 0; i < 26; ++i)
        alloc_Sbuf_frag(i, estimates.d[i], nfragcells.d[i], /**/ b);
}

void free_Sbufs(/**/ Sbufs *b) {
    for (int i = 0; i < 26; ++i)
        free_Sbuf_frag(i, /**/ b);
}

static void alloc_Rbuf_frag(const int i, const int est, const int nfragcells, /**/ Rbufs *b) {
    alloc_Bbuf_frag(i, est, nfragcells, /**/ b);
};

static void free_Rbuf_frag(const int i, /**/ Rbufs *b) {
    free_Bbuf_frag(i, /**/ b);
}

void alloc_Rbufs(const int26 estimates, const int26 nfragcells, /**/ Rbufs *b) {
    for (int i = 0; i < 26; ++i)
        alloc_Rbuf_frag(i, estimates.d[i], nfragcells.d[i], /**/ b);
}

void free_Rbufs(/**/ Rbufs *b) {
    for (int i = 0; i < 26; ++i)
        free_Rbuf_frag(i, /**/ b);
}

static void alloc_Ibuf_frag(const int i, const int est, /**/ Ibuf *b) {
    CC(d::Malloc((void **) &b->ii.d[i], est * sizeof(int)));

    CC(cudaHostAlloc(&b->iihst.d[i], est * sizeof(int), cudaHostAllocMapped));
    CC(cudaHostGetDevicePointer(&b->iidev.d[i], b->iihst.d[i], 0));
}

static void free_Ibuf_frag(const int i, /**/ Ibuf *b) {
    CC(d::Free(b->ii.d[i]));
    CC(cudaFreeHost(b->iihst.d[i]));
}

void alloc_SIbuf(const int26 estimates, /**/ SIbuf *b) {
    for (int i = 0; i < 26; ++i)
        alloc_Ibuf_frag(i, estimates.d[i], /**/ b);
}

void free_SIbuf(/**/ SIbuf *b) {
    for (int i = 0; i < 26; ++i)
        free_Ibuf_frag(i, /**/ b);
}

void alloc_RIbuf(const int26 estimates, /**/ RIbuf *b) {
    for (int i = 0; i < 26; ++i)
        alloc_Ibuf_frag(i, estimates.d[i], /**/ b);
}

void free_RIbuf(/**/ RIbuf *b) {
    for (int i = 0; i < 26; ++i)
        free_Ibuf_frag(i, /**/ b);
}

} // sub
} // dpdr
