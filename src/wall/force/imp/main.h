void force(Wvel_v wv, Coords c, Cloud cloud, int n, RNDunif *rnd, Wa wa, /**/ Force *ff) {
    KL(dev::force,
       (k_cnf(3*n)),
       (wv, c, cloud, n, rnd_get(rnd), wa, /**/ (float*)ff));
}
