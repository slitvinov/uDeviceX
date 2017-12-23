static __device__ int r2map(int zplane, int n, float x, float y, float z, const int *starts, /**/ Map *m) {
    /* coordinate [r] to map */
    int cnt0, cnt1, cnt2, org0;
    int org1, org2;
    int xcenter, xstart, xcount;
    int ycenter, zcenter, zmy;
    bool zvalid;
    int count0, count1, count2;
    int cid0, cid1, cid2;

    xcenter = min(XS - 1, max(0, XOFFSET + (int)floorf(x)));
    xstart = max(0, xcenter - 1);
    xcount = min(XS, xcenter + 2) - xstart;
    if (xcenter - 1 >= XS || xcenter + 2 <= 0) return EMPTY;
    ycenter = min(YS - 1, max(0, YOFFSET + (int)floorf(y)));
    zcenter = min(ZS - 1, max(0, ZOFFSET + (int)floorf(z)));
    zmy = zcenter - 1 + zplane;
    zvalid = zmy >= 0 && zmy < ZS;
    count0 = count1 = count2 = 0;
    if (zvalid && ycenter - 1 >= 0 && ycenter - 1 < YS) {
        cid0 = xstart + XS * (ycenter - 1 + YS * zmy);
        org0 = starts[cid0];
        count0 = starts[cid0 + xcount] - org0;
    }
    if (zvalid && ycenter >= 0 && ycenter < YS) {
        cid1 = xstart + XS * (ycenter + YS * zmy);
        org1 = starts[cid1];
        count1 = starts[cid1 + xcount] - org1;
    }

    if (zvalid && ycenter + 1 >= 0 && ycenter + 1 < YS) {
        cid2 = xstart + XS * (ycenter + 1 + YS * zmy);
        org2 = starts[cid2];
        count2 = starts[cid2 + xcount] - org2;
    }

    cnt0 = count0;
    cnt1 = count0 + count1;
    cnt2 = cnt1 + count2;

    org1 -= cnt0;
    org2 -= cnt1;

    m->org0 = org0; m->org1 = org1; m->org2 = org2;
    m->cnt0 = cnt0; m->cnt1 = cnt1; m->cnt2 = cnt2;
    return FULL;
}