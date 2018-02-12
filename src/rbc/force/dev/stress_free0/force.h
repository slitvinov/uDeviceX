__device__ real3 tri(RbcParams_v par, real3 r1, real3 r2, real3 r3, Shape0 shape0, real area, real volume) {
    real a0, A0, totArea, totVolume;
    A0 = shape0.A0;
    a0 = shape0.a0;
    totArea = shape0.totArea;
    totVolume = shape0.totVolume;
    return tri0(par, r1, r2, r3,   a0, A0, totArea, totVolume,   area, volume);
}

__device__ real3 dih(RbcParams_v par, real3 r0, real3 r1, real3 r2, real3 r3, real3 r4) {
    real3 f1, f2;
    f1 = dih0<1>(par, r0, r2, r1, r4);
    f2 = dih0<2>(par, r1, r0, r2, r3);
    add(&f1, /**/ &f2);
    return f2;
}
