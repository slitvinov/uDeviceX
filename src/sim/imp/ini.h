static void ini_obj_exch(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Objexch *e) {
    using namespace exch::obj;
    ini(MAX_OBJ_TYPES, MAX_OBJ_DENSITY, &e->p);
    ini(comm, /*io*/ tg, /**/ &e->c);
    ini(MAX_OBJ_DENSITY, &e->u);
    ini(MAX_OBJ_DENSITY, &e->pf);
    ini(MAX_OBJ_DENSITY, &e->uf);    
}

static void ini_bb_exch(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ BBexch *e) {
    using namespace exch::mesh;
    ini(MAX_VERT_NUM, MAX_CELL_NUM, &e->p);
    ini(m::cart, /*io*/ &tag_gen, /**/ &e->c);
    ini(MAX_VERT_NUM, MAX_CELL_NUM, &e->u);

    ini(MAX_FACE_NUM, MAX_CELL_NUM, &e->pm);
    ini(comm, /*io*/ tg, /**/ &e->cm);
    ini(MAX_FACE_NUM, MAX_CELL_NUM, &e->um);
}

void ini() {
    basetags::ini(&tag_gen);
    datatype::ini();
    if (rbcs) {
        Dalloc(&r::ff, MAX_PART_NUM);
        rbc::alloc_quants(&r::q);

        distr::rbc::ini(r::q.nv, /**/ &r::d.p);
        distr::rbc::ini(m::cart, /**/ &tag_gen, /**/ &r::d.c);
        distr::rbc::ini(r::q.nv, /**/ &r::d.u);
    }
    
    if (fsiforces) fsi::ini();

    bbhalo::ini(&tag_gen);
    cnt::ini();

    if (rbcs || solids)
        ini_obj_exch(m::cart, &tag_gen, &rs::e);
    
    bop::ini(&dumpt);

    if (walls) {
        sdf::alloc_quants(&w::qsdf);
        wall::alloc_quants(&w::q);
        wall::alloc_ticket(&w::t);
    }

    flu::alloc_quants(&o::q);
    flu::alloc_ticketZ(&o::tz);

    float maxdensity = ODSTR_FACTOR * numberdensity;
    distr::flu::ini(maxdensity, /**/ &o::d.p);
    distr::flu::ini(m::cart, /**/ &tag_gen, /**/ &o::d.c);
    distr::flu::ini(maxdensity, /**/ &o::d.u);

    dpdr::ini_ticketcom(m::cart, &tag_gen, &o::h.tc);
    dpdr::ini_ticketrnd(o::h.tc, /**/ &o::h.trnd);
    dpdr::alloc_ticketSh(/**/ &o::h.ts);
    dpdr::alloc_ticketRh(/**/ &o::h.tr);

    Dalloc(&o::ff, MAX_PART_NUM);

    if (global_ids) {
        flu::alloc_quantsI(&o::qi);
    }
    
    if (multi_solvent) {
        flu::alloc_quantsI(&o::qc);

        dpdr::ini_ticketIcom(/*io*/ &tag_gen, /**/ &o::h.tic);
        dpdr::alloc_ticketSIh(/**/ &o::h.tsi);
        dpdr::alloc_ticketRIh(/**/ &o::h.tri);
    }

    if (multi_solvent && rbcs) {
        exch::mesh::ini(MAX_VERT_NUM, MAX_CELL_NUM, &mc::e.p);
        exch::mesh::ini(m::cart, /*io*/ &tag_gen, /**/ &mc::e.c);
        exch::mesh::ini(MAX_VERT_NUM, MAX_CELL_NUM, &mc::e.u);
        Dalloc(&mc::pp, MAX_PART_NUM);
    }
    
    if (solids) {
        mrescue::ini(MAX_PART_NUM);
        rig::alloc_quants(&s::q);
        rig::alloc_ticket(&s::t);
        scan::alloc_work(XS*YS*ZS, /**/ &s::ws);
        s::ff_hst = (Force*)malloc(sizeof(&s::ff_hst)*MAX_PART_NUM);
        Dalloc(&s::ff, MAX_PART_NUM);

        tcells::alloc_quants(MAX_SOLIDS, &bb::qtc);
        mbounce::alloc_ticketM(&bb::tm);

        ini(MAX_PART_NUM, /**/ &bb::bbd);
        Dalloc(&bb::mm, MAX_PART_NUM);

        int nv = s::q.nv;
        distr::rig::ini(nv, &s::d.p);
        distr::rig::ini(m::cart, /*io*/ &tag_gen, /**/ &s::d.c);
        distr::rig::ini(nv, &s::d.u);

        if (sbounce_back)
            ini_bb_exch(m::cart, /*io*/ &tag_gen, /**/ &s::e);
    }

    MC(MPI_Barrier(m::cart));
}
