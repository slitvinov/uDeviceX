void clear_vel(Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    scheme_move_clear_vel(flu->q.n, flu->q.pp);
    if (s->opt.rig) scheme_move_clear_vel(rig->q.n, rig->q.pp);
    if (s->opt.rbc) scheme_move_clear_vel(rbc->q.n, rbc->q.pp);
}

void update_solid(float dt, Rig *s) {
    if (!s->q.n) return;
    RigQuants *q = &s->q;
    
    rig_update(s->pininfo, dt, q->n, s->ff, q->rr0, q->ns, /**/ q->pp, q->ss);
    rig_update_mesh(dt, q->ns, q->ss, q->nv, q->dvv, /**/ q->i_pp);
    // for dump
    cD2H(q->ss_dmp, q->ss, q->ns);
    rig_reinit_ft(q->ns, /**/ q->ss);
}

void bounce_solid(float dt, int3 L, BounceBack *bb, Rig *s, Flu *flu) {
    int n, nm, nt, nv, *ss, *cc, nmhalo, counts[NFRAGS];
    int4 *tt;
    Particle *pp, *i_pp;
    MeshInfo mi;
    
    RigQuants *qs = &s->q;
    BBexch     *e = &bb->e; 
    
    nm = qs->ns;
    nt = qs->nt;
    nv = qs->nv;
    tt = qs->dtt;
    i_pp = qs->i_pp;

    n  = flu->q.n;
    pp = flu->q.pp;
    cc = flu->q.cells.counts;
    ss = flu->q.cells.starts;

    mi.nv = nv;
    mi.nt = nt;
    mi.tt = tt;
    
    /* send meshes to frags */

    emesh_build_map(nm, nv, i_pp, /**/ e->p);
    emesh_pack(nv, i_pp, /**/ e->p);
    emesh_download(e->p);

    UC(emesh_post_send(e->p, e->c));
    UC(emesh_post_recv(e->c, e->u));

    emesh_wait_send(e->c);
    emesh_wait_recv(e->c, e->u);

    /* unpack at the end of current mesh buffer */
    emesh_unpack(nv, e->u, /**/ &nmhalo, i_pp + nm * nv);
    
    /* perform bounce back */
    
    meshbb_reini(n, /**/ bb->d);
    if (nm + nmhalo)
        CC(d::MemsetAsync(bb->mm, 0, nt * (nm + nmhalo) * sizeof(Momentum)));

    meshbb_find_collisions(dt, nm + nmhalo, mi, i_pp, L, ss, cc, pp, flu->ff, /**/ bb->d);
    meshbb_select_collisions(n, /**/ bb->d);
    meshbb_bounce(dt, flu->mass, n, bb->d, flu->ff, mi, i_pp, /**/ pp, bb->mm);

    /* send momentum back */

    emesh_get_num_frag_mesh(e->u, /**/ counts);
    
    emesh_packM(nt, counts, bb->mm + nm * nt, /**/ e->pm);
    emesh_downloadM(counts, /**/ e->pm);

    UC(emesh_post_recv(e->cm, e->um));
    UC(emesh_post_send(e->pm, e->cm));
    emesh_wait_recv(e->cm, e->um);
    emesh_wait_send(e->cm);

    emesh_upload(e->um);
    emesh_unpack_mom(nt, e->p, e->um, /**/ bb->mm);
    
    /* gather bb momentum */
    meshbb_collect_rig_momentum(dt, nm, mi, i_pp, bb->mm, /**/ qs->ss);

    /* for dump */
    cD2H(qs->ss_dmp_bb, qs->ss, nm);
}


void update_solvent(float dt, /**/ Flu *f) {
    scheme_move_apply(dt, f->mass, f->q.n, f->ff, f->q.pp);
}

void update_rbc(float dt, long it, Rbc *r, Sim *s) {
    bool cond;
    cond = s->opt.flucolors && s->opt.recolor_freq && it % s->opt.recolor_freq == 0;
    if (cond) {
        /* TODO: does not belong here*/
        msg_print("recolor");
        UC(gen_colors(r, &s->colorer, /**/ &s->flu));
    } 
    scheme_move_apply(dt, r->mass, r->q.n, r->ff, r->q.pp);
}

void restrain(long it, Sim *s) {
    SchemeQQ qq;
    
    qq.o = s->flu.q.pp;
    qq.r = s->rbc.q.pp;
    qq.on = s->flu.q.n;
    qq.rn = s->rbc.q.n;
    
    scheme_restrain_apply(s->cart, s->flu.q.cc, it, /**/ s->restrain, qq);
}

void bounce_wall(float dt, bool rbc, const Coords *c, Wall *w, /**/ Flu *f, Rbc *r) {
    sdf_bounce(dt, w->velstep, c, w->sdf, f->q.n, /**/ f->q.pp);
    if (rbc) sdf_bounce(dt, w->velstep, c, w->sdf, r->q.n, /**/ r->q.pp);
}
