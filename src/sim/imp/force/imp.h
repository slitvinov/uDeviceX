void forces(float dt0, bool wall0, Sim *s) {
    Flu *flu = &s->flu;
    Rbc *rbc = &s->rbc;
    Rig *rig = &s->rig;
    Wall *wall = &s->wall;

    UC(clear_forces(flu->ff, flu->q.n));
    if (s->solids0) UC(clear_forces(rig->ff, rig->q.n));
    if (rbcs)       UC(clear_forces(rbc->ff, rbc->q.n));

    UC(forces_dpd(flu));
    if (wall0 && wall->q.n) forces_wall(wall, s);
    if (rbcs) forces_rbc(dt0, rbc);

    UC(forces_objects(s));
    
    dSync();
}
