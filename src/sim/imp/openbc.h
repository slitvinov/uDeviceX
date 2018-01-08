void apply_inflow(Inflow *i, Flu *f) {
    UC(create_pp(i, &f->q.n, f->q.pp));
}

void mark_outflow(const Flu *f, Outflow *o) {
    UC(filter_particles(f->q.n, f->q.pp, /**/ o));
    UC(download_ndead(o));
}

void mark_outflowden(const Flu *f, const DContMap *m, /**/ DCont *d) {
    const int *ss, *cc;
    int n;
    n = f->q.n;
    ss = f->q.cells.starts;
    cc = f->q.cells.counts;

    UC(reset(n, /**/ d));
    UC(filter_particles(m, ss, cc, /**/ d));
    UC(download_ndead(/**/ d));
}
