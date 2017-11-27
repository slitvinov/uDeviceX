static void check_size(long n, long max) {
    if (n < 0 || n >= max)
        ERR("wrong size: %ld / %ld", n, max);
}

void step(scheme::force::Param *fpar, bool wall0, int it) {
    UC(check_size(r::q.nc, MAX_CELL_NUM));
    UC(check_size(r::q.n , MAX_PART_NUM));
    UC(check_size(o::q.n , MAX_PART_NUM));
    
    UC(distribute_flu());
    if (solids0) UC(distribute_rig());
    if (rbcs)    UC(distribute_rbc());

    UC(check_size(r::q.nc, MAX_CELL_NUM));
    UC(check_size(r::q.n , MAX_PART_NUM));
    UC(check_size(o::q.n , MAX_PART_NUM));

    forces(wall0);

    dump_diag0(it);
    dump_diag_after(it, wall0, solids0);
    body_force(*fpar);

    restrain(it);
    update_solvent(it);
    if (solids0) update_solid();
    if (rbcs)    update_rbc(it);

    if (VCON && wall0) {
        sample(it, o::q.n, o::q.pp, o::q.cells.starts, o::q.cells.counts, /**/ &o::vcont);
        adjust(it, /**/ &o::vcont, fpar);
        log(it, &o::vcont);
    }

    if (wall0) bounce_wall();

    if (sbounce_back && solids0) bounce_solid(it);

    recolor_flux();
}