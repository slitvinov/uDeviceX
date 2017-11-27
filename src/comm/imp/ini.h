/* pinned allocation */

static void alloc_counts(int n, /**/ int **hc) {
    UC(emalloc(n * sizeof(int), (void**) hc));
}

/* generic allocation */
static void alloc_pair(int i, AllocMod mod, /**/ hBags *hb, dBags *db) {
    size_t n = hb->capacity[i] * hb->bsize;
    
    switch (mod) {
    case HST_ONLY:
        UC(emalloc(n, (void**) &hb->data[i]));
        break;
    case DEV_ONLY:
        CC(d::Malloc((void**) &db->data[i], n));
        break;
    case PINNED:
        CC(d::alloc_pinned(&hb->data[i], n));
        CC(d::HostGetDevicePointer(&db->data[i], hb->data[i], 0));
        break;
    case PINNED_HST:
        CC(d::alloc_pinned(&hb->data[i], n));
        break;
    case PINNED_DEV:
        CC(d::alloc_pinned(&hb->data[i], n));
        CC(d::Malloc((void **) &db->data[i], n));
        break;
    case NONE:
    default:
        break;
    }
}

int ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS], /**/ hBags *hb, dBags *db) {
    hb->bsize = bsize;
    memcpy(hb->capacity, capacity, NBAGS * sizeof(int));

    /* fragments */
    for (int i = 0; i < NFRAGS; ++i)
        UC(alloc_pair(i, fmod, /**/ hb, db));

    /* bulk */
    UC(alloc_pair(frag_bulk, bmod, /**/ hb, db));

    UC(alloc_counts(NBAGS, /**/ &hb->counts));
    return 0;
}

/* stamp allocation */

int ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Stamp *s) {
    int i, c, crd_rnk[3];
    
    for (i = 0; i < NFRAGS; ++i) {
        for (c = 0; c < 3; ++c)
            crd_rnk[c] = m::coords[c] + frag_i2d(i,c);
        MC(m::Cart_rank(comm, crd_rnk, s->ranks + i));
        s->tags[i] = frag_anti(i);
    }
    s->bt = get_tag(tg);
    MC(m::Comm_dup(comm, &s->cart));
    return 0;
}