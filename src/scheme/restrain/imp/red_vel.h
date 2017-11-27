static void report0() {
    enum {X, Y, Z};
    int n;
    float v[3];
    stats(/**/ &n, v);
    MSG("restrain RED: n = %d [% .3e % .3e % .3e]", n, v[X], v[Y], v[Z]);
}

static void report(int it) {
    bool cond;
    int freq;
    freq = RESTRAIN_REPORT_FREQ;
    cond = freq > 0 && it % freq == 0;
    if (cond) report0();
}

void main(const int *cc, NN nn, long it, /**/ QQ qq) {
    color::vel(cc, RED_COLOR, nn.o, /**/ qq.o);
    report(it);
}