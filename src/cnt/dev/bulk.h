__global__ void bulk(int n, const float2 *pp, 
                     float seed, int objid0, /**/ float *ff) {
    Map m; /* see map/ */
    float x, y, z;

    float fx, fy, fz, rnd;
    forces::Pa a, b;
    int gid, aid, zplane;
    float2 dst0, dst1, dst2;
    float xforce, yforce, zforce;
    int i, slot;

    int objid;
    int bid;
    int sentry;
    float2 stmp0, stmp1, stmp2;
    int mapstatus;

    gid = threadIdx.x + blockDim.x * blockIdx.x;
    aid = gid / 3;
    zplane = gid % 3;

    if (aid >= n) return;

    dst0 = __ldg(pp + 3 * aid + 0);
    dst1 = __ldg(pp + 3 * aid + 1);
    dst2 = __ldg(pp + 3 * aid + 2);
    x = dst0.x;
    y = dst0.y;
    z = dst1.x;
    mapstatus = r2map(zplane, n, x, y, z, /**/ &m);
    
    if (mapstatus == EMPTY) return;
    xforce = yforce = zforce = 0;
    for (i = 0; !endp(m, i); ++i) {
        slot = m2id(m, i);
        get(slot, &objid, &bid);
        if (objid0 < objid || objid0 == objid && aid <= bid)
            continue;

        sentry = 3 * bid;
        stmp0 = __ldg(c::PP[objid] + sentry);
        stmp1 = __ldg(c::PP[objid] + sentry + 1);
        stmp2 = __ldg(c::PP[objid] + sentry + 2);

        rnd = rnd::mean0var1ii(seed, aid, bid);
        forces::f2k2p(dst0,   dst1,  dst2, SOLID_KIND, /**/ &a);
        forces::f2k2p(stmp0, stmp1, stmp2, SOLID_KIND, /**/ &b);
        pair(a, b, rnd, /**/ &fx, &fy, &fz);
        xforce += fx;
        yforce += fy;
        zforce += fz;
        atomicAdd(c::FF[objid] + sentry,     -fx);
        atomicAdd(c::FF[objid] + sentry + 1, -fy);
        atomicAdd(c::FF[objid] + sentry + 2, -fz);
    }

    atomicAdd(ff + 3 * aid + 0, xforce);
    atomicAdd(ff + 3 * aid + 1, yforce);
    atomicAdd(ff + 3 * aid + 2, zforce);
}
