void ini(Sdf **pq) {
    Sdf *q;
    UC(emalloc(sizeof(Sdf), (void**)&q));
    UC(array3d_ini(&q->arr, XTE, YTE, ZTE));
    *pq = q;
}

void fin(Sdf *q) {
    UC(array3d_fin(q->arr));
    q->tex.destroy();
    UC(efree(q));
}

void bounce(Wvel_v wv, Coords c, const Sdf *q, int n, /**/ Particle *pp) {
    UC(bounce_back(wv, c, q->tex, n, /**/ pp));
}