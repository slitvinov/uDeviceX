static bool same(const char *a, const char *b) {
    return 0 == strcmp(a, b);
}

static void ini_flu_exch(MPI_Comm comm, /**/ FluExch *e) {
    using namespace exch::flu;
    int maxd = HSAFETY_FACTOR * numberdensity;
    
    UC(ini(maxd, /**/ &e->p));
    UC(ini(comm, /**/ &e->c));
    UC(ini(maxd, /**/ &e->u));
}

static void ini_obj_exch(MPI_Comm comm, /**/ ObjExch *e) {
    using namespace exch::obj;
    int maxpsolid = MAX_PSOLID_NUM;
    
    UC(ini(MAX_OBJ_TYPES, MAX_OBJ_DENSITY, maxpsolid, &e->p));
    UC(ini(comm, /**/ &e->c));
    UC(ini(MAX_OBJ_DENSITY, maxpsolid, /**/ &e->u));
    UC(ini(MAX_OBJ_DENSITY, maxpsolid, /**/ &e->pf));
    UC(ini(MAX_OBJ_DENSITY, maxpsolid, /**/ &e->uf));
}

static void ini_mesh_exch(int nv, int max_m, MPI_Comm comm, /**/ Mexch *e) {
    using namespace exch::mesh;
    UC(ini(nv, max_m, /**/ &e->p));
    UC(ini(comm, /**/ &e->c));
    UC(ini(nv, max_m, /**/ &e->u));
}

static void ini_bb_exch(int nt, int nv, int max_m, MPI_Comm comm, /**/ BBexch *e) {
    UC(ini_mesh_exch(nv, max_m, comm, /**/ e));

    using namespace exch::mesh;
    UC(ini(nt, max_m, /**/ &e->pm));
    UC(ini(comm, /**/ &e->cm));
    UC(ini(nt, max_m, /**/ &e->um));
}

static void ini_flu_distr(MPI_Comm comm, /**/ FluDistr *d) {
    using namespace distr::flu;
    float maxdensity = ODSTR_FACTOR * numberdensity;
    UC(ini(maxdensity, /**/ &d->p));
    UC(ini(comm, /**/ &d->c));
    UC(ini(maxdensity, /**/ &d->u));
}

static void ini_rbc_distr(int nv, MPI_Comm comm, /**/ RbcDistr *d) {
    using namespace distr::rbc;
    UC(ini(MAX_CELL_NUM, nv, /**/ &d->p));
    UC(ini(comm, /**/ &d->c));
    UC(ini(MAX_CELL_NUM, nv, /**/ &d->u));
}

static void ini_rig_distr(int nv, MPI_Comm comm, /**/ RigDistr *d) {
    using namespace distr::rig;
    UC(ini(MAX_SOLIDS, nv, /**/ &d->p));
    UC(ini(comm, /**/ &d->c));
    UC(ini(MAX_SOLIDS, nv, /**/ &d->u));
}

static void ini_vcont(MPI_Comm comm, /**/ PidVCont **c) {
    int3 L = {XS, YS, ZS};
    float3 V = {VCON_VX, VCON_VY, VCON_VZ};
    UC(vcont_ini(comm, L, V, VCON_FACTOR, /**/ c));
}

static void ini_outflow(Coords coords, const Config *cfg, Outflow **o) {
    UC(ini(MAX_PART_NUM, /**/ o));
    const char *type;
    UC(conf_lookup_string(cfg, "outflow.type", &type));
    
    if      (same(type, "circle")) {
        float3 center;
        float R;
        UC(conf_lookup_float(cfg, "outflow.R", &R));
        UC(conf_lookup_float3(cfg, "outflow.center", &center));
        
        ini_params_circle(coords, center, R, /**/ *o);
    }
    else if (same(type, "plate")) {
        int dir;
        float r0;
        UC(conf_lookup_int(cfg, "outflow.direction", &dir));
        UC(conf_lookup_float(cfg, "outflow.position", &r0));
        ini_params_plate(coords, dir, r0, /**/ *o);
    }
    else {
        ERR("Unrecognized type <%s>", type);
    }
}

static void ini_denoutflow(Coords coords, DCont **d, DContMap **m) {
    UC(den_ini(MAX_PART_NUM, /**/ d));
    UC(den_ini_map(coords, /**/ m));
}

static void ini_inflow(Coords coords, const Config *cfg, Inflow **i) {
    const char *type;
    UC(conf_lookup_string(cfg, "outflow.type", &type));
    
    /* number of cells */
    int2 nc = make_int2(YS, ZS/2);
    ini(nc, /**/ i);

    if      (same(type, "circle")) {
        float R, H, U;
        int poiseuille;
        float3 center;

        UC(conf_lookup_float(cfg, "inflow.R", &R));
        UC(conf_lookup_float(cfg, "inflow.H", &H));
        UC(conf_lookup_float(cfg, "inflow.U", &U));
        UC(conf_lookup_bool(cfg, "inflow.poiseuille", &poiseuille));
        UC(conf_lookup_float3(cfg, "inflow.center", &center));        
        UC(ini_params_circle(coords, center, R, H, U, poiseuille, /**/ *i));
    }
    else if (same(type, "plate")) {
        int upois, vpois, dir;
        float3 origin, u;
        float L1, L2;

        UC(conf_lookup_float(cfg, "inflow.L1", &L1));
        UC(conf_lookup_float(cfg, "inflow.L2", &L2));
        UC(conf_lookup_int(cfg, "inflow.direction", &dir));
        UC(conf_lookup_bool(cfg, "inflow.upoiseuille", &upois));
        UC(conf_lookup_bool(cfg, "inflow.vpoiseuille", &vpois));
        UC(conf_lookup_float3(cfg, "inflow.origin", &origin));
        UC(conf_lookup_float3(cfg, "inflow.u", &u));
        
        ini_params_plate(coords, origin, dir, L1, L2, u, upois, vpois, /**/ *i);
    }
    else {
        ERR("unknown inflow type <%s>", type);
    }
    
    UC(ini_velocity(*i));
}

static void ini_colorer(int nv, MPI_Comm comm, /**/ Colorer *c) {
    UC(ini_mesh_exch(nv, MAX_CELL_NUM, comm, &c->e));
    Dalloc(&c->pp, MAX_PART_NUM);
    Dalloc(&c->minext, MAX_CELL_NUM);
    Dalloc(&c->maxext, MAX_CELL_NUM);
}

static void ini_flu(MPI_Comm cart, /**/ Flu *f) {

    flu::ini(&f->q);
    ini(MAX_PART_NUM, /**/ &f->bulkdata);
    ini(cart, /**/ &f->halodata);
    
    UC(ini_flu_distr(cart, /**/ &f->d));
    UC(ini_flu_exch(cart, /**/ &f->e));
    
    UC(Dalloc(&f->ff, MAX_PART_NUM));
    UC(emalloc(MAX_PART_NUM * sizeof(Force), /**/ (void**) &f->ff_hst));
}

static void ini_rbc(MPI_Comm cart, /**/ Rbc *r) {
    Dalloc(&r->ff, MAX_CELL_NUM * RBCnv);
    UC(rbc::main::ini(&r->q));

    UC(ini_rbc_distr(r->q.nv, cart, /**/ &r->d));
    if (rbc_com_dumps) UC(rbc::com::ini(MAX_CELL_NUM, /**/ &r->com));
    if (RBC_STRETCH)   UC(rbc::stretch::ini("rbc.stretch", r->q.nv, /**/ &r->stretch));
}

static void ini_rig(MPI_Comm cart, /**/ Rig *s) {
    rig::ini(&s->q);
    scan::alloc_work(XS * YS * ZS, /**/ &s->ws);
    UC(emalloc(sizeof(&s->ff_hst)*MAX_PART_NUM, (void**) &s->ff_hst));
    Dalloc(&s->ff, MAX_PART_NUM);

    UC(ini_rig_distr(s->q.nv, cart, /**/ &s->d));    
}

static void ini_bounce_back(MPI_Comm cart, Rig *s, /**/ BounceBack *bb) {
    meshbb::ini(MAX_PART_NUM, /**/ &bb->d);
    Dalloc(&bb->mm, MAX_PART_NUM);

    UC(ini_bb_exch(s->q.nt, s->q.nv, MAX_CELL_NUM, cart, /**/ &bb->e));
}

static void ini_wall(Wall *w) {
    ini(&w->sdf);
    wall::alloc_quants(&w->q);
    wall::alloc_ticket(&w->t);
    Wvel *vw = &w->vel;

#if defined(WVEL_HS)
    WvelHS p;
    p.u = WVEL_PAR_U;
    p.h = WVEL_PAR_H;
#else
#if   defined(WVEL_SIN)
    WvelShearSin p;
    p.log_freq = WVEL_LOG_FREQ;
    p.w = WVEL_PAR_W;
    p.half = 1;
#else
    WvelShear p;
    p.half = 0;
#endif
    p.gdot = WVEL_PAR_A;
    p.vdir = 0;

    p.gdir = 0;
    if (WVEL_PAR_Y)
        p.gdir = 1;
    else if (WVEL_PAR_Z)
        p.gdir = 2;
#endif
    ini(p, vw);
}

static void ini_objinter(MPI_Comm cart, /**/ ObjInter *o) {
    int rank;
    MC(m::Comm_rank(cart, &rank));
    UC(ini_obj_exch(cart, &o->e));
    if (contactforces) cnt::ini(rank, /**/ &o->cnt);
    if (fsiforces)     fsi::ini(rank, /**/ &o->fsi);
}

static void read_opt(const Config *c, Opt *o) {
    int b;
    UC(conf_lookup_bool(c, "outflow.active", &b));
    o->outflow = b;
    UC(conf_lookup_bool(c, "inflow.active", &b));
    o->inflow = b;
}

void sim_ini(int argc, char **argv, MPI_Comm cart, /**/ Sim **sim) {
    Sim *s;
    UC(emalloc(sizeof(Sim), (void**) sim));
    s = *sim;

    MC(m::Comm_dup(cart, &s->cart));
    
    Config *cfg = s->cfg;    
    datatype::ini();

    UC(conf_ini(&cfg)); s->cfg = cfg;
    UC(conf_read(argc, argv, /**/ cfg));

    UC(read_opt(s->cfg, &s->opt));
    
    UC(coords_ini(s->cart, /**/ &s->coords));
    
    UC(emalloc(3 * MAX_PART_NUM * sizeof(Particle), (void**) &s->pp_dump));
    
    if (rbcs) UC(ini_rbc(s->cart, /**/ &s->rbc));

    if (VCON)    UC(ini_vcont(s->cart, /**/ &s->vcont));
    if (s->opt.outflow) UC(ini_outflow(s->coords, s->cfg, /**/ &s->outflow));
    if (s->opt.inflow)  UC(ini_inflow (s->coords, s->cfg, /**/ &s->inflow ));
    if (OUTFLOW_DEN) UC(ini_denoutflow(s->coords, /**/ &s->denoutflow, &s->mapoutflow));
    
    if (rbcs || solids)
        UC(ini_objinter(s->cart, /**/ &s->objinter));        
    
    UC(bop::ini(s->cart, &s->dumpt));

    if (walls) ini_wall(&s->wall);
    
    UC(ini_flu(s->cart, /**/ &s->flu));
   
    if (multi_solvent && rbcs)
        UC(ini_colorer(s->rbc.q.nv, s->cart, /**/ &s->colorer));
    
    if (solids) {
        UC(ini_rig(s->cart, /**/ &s->rig));

        if (sbounce_back)
            UC(ini_bounce_back(s->cart, &s->rig, /**/ &s->bb));
    }

    MC(MPI_Barrier(s->cart));
}
