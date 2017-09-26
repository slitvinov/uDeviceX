void clear_vel() {
    scheme::clear_vel(o::q.pp, o::q.n);
    if (solids) scheme::clear_vel(s::q.pp, s::q.n);
    if (rbcs  ) scheme::clear_vel(r::q.pp, r::q.n);
}

void update_solid() {
    if (!s::q.n) return;

    rig::update(s::ff, s::q.rr0, s::q.n, s::q.ns, /**/ s::q.pp, s::q.ss);
    rig::update_mesh(s::q.ss, s::q.ns, s::q.nv, s::q.dvv, /**/ s::q.i_pp);
    // for dump
    cD2H(s::q.ss_dmp, s::q.ss, s::q.ns);
    rig::reinit_ft(s::q.ns, /**/ s::q.ss);
}

void update_solvent(long it) {
    scheme::restrain(o::qc.ii, o::q.n, it, /**/ o::q.pp);
    scheme::move(dpd_mass, o::q.pp, o::ff, o::q.n);
}

void update_rbc(long it) {
    bool cond;
    cond = multi_solvent && color_freq && it % color_freq == 0;
    if (cond) {MSG("recolor"); gen_colors();};
    scheme::move(rbc_mass, r::q.pp, r::ff, r::q.n);
}

void bounce() {
    sdf::bounce(&w::qsdf, o::q.n, /**/ o::q.pp);
    // if (rbcs) sdf::bounce(&w::qsdf, r::q.n, /**/ r::q.pp);
}

/* single node only for now */
void bounce_rbc() {
    // TODO
    // build_tcells_dev(s::q.m_dev, s::t.i_pp, nsbb, /**/ s::t.tcs_dev, s::t.tcc_dev, s::t.tci_dev, /*w*/ &s::ws);
    // mbounce::bounce_dev(o::ff, s::q.m_dev, s::t.i_pp, s::t.tcs_dev, s::t.tcc_dev, s::t.tci_dev, o::q.n, nsbb*s::q.m_dev.nt, /**/ o::q.pp, &bb::tm);
    // mbounce::collect_rbc_dev(s::q.m_dev.nt, nsbb, &bb::tm, /**/ s::t.ss);
}
