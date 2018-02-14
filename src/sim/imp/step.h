static void step(Time *time, BForce *bforce, bool wall0, int ts, int it, Sim *s) {
    float dt;
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wall = &s->wall;
    const Opt *opt = &s->opt;
    dt = time_dt(time);

    if (walls && !s->equilibrating)
        UC(wvel_get_view(dt, it - ts, wall->vel, /**/ &wall->vview));

    UC(check_sizes(s));
    UC(check_pos_soft(s));

    UC(distribute_flu(s));
    if (s->solids0) UC(distribute_rig(/**/ rig));
    if (rbcs)       UC(distribute_rbc(/**/ rbc));

    UC(check_sizes(s));
    UC(forces(dt, wall0, s));
    UC(check_forces(dt, s));

    dump_diag(time, it, s);
    dump_diag_after(time, it, s->solids0, s);
    UC(body_force(it, bforce, s));

    UC(restrain(it, /**/ s));
    UC(update_solvent(dt, s->moveparams, /**/ flu));
    if (s->solids0) update_solid(dt, /**/ rig);
    if (rbcs)       update_rbc(dt, s->moveparams, it, rbc, s);

    UC(check_vel(dt, s));

    if (opt->vcon && !s->equilibrating) {
        sample(s->coords, it, flu, /**/ &s->vcon);
        adjust(it, /**/ &s->vcon, bforce);
        log(it, &s->vcon);
    }

    if (wall0) bounce_wall(dt, s->coords, wall, /**/ flu, rbc);

    if (opt->sbounce && s->solids0) bounce_solid(dt, s->L, /**/ &s->bb, rig, flu);

    UC(check_pos_soft(s));
    UC(check_vel(dt, s));

    if (! s->equilibrating) {
        if (opt->inflow)     apply_inflow(s->kBT, dt, s->inflow, /**/ flu);
        if (opt->outflow)    mark_outflow(flu, /**/ s->outflow);
        if (opt->denoutflow) mark_outflowden(flu, s->mapoutflow, /**/ s->denoutflow);
    }
}
