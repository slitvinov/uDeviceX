static void get_start(const flu::LFrag lfrags[26], /**/ int start[27]) {
    /* generate padded start */
    int i;
    start[0] = 0;
    for (i = 0; i < 26; ++i) start[i + 1] = start[i] + 16 * ((lfrags[i].n + 15) / 16);
}

void interactions(const flu::LFrag26 lfrags, const flu::RFrag26 rfrags, const flu::RndFrag26 rrnd, /**/ float *ff) {
    int27 start;
    int n; /* number of threads */
    get_start(lfrags.d, /**/ start.d);
    n = start.d[26];
    KL(dev::force, (k_cnf(n)), (start, lfrags, rfrags, rrnd, /**/ ff));
}