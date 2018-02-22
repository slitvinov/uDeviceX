static void gen0(const Coords *c, MPI_Comm comm, RigGenInfo rgi, int nsolid, int rcount, int idmax, int root, float *coms, /**/
                 int *ns, int *nps, float *rr0, Solid *ss, Particle *r_pp) {
    Solid model;
    // share model to everyone
    int npsolid = 0, rank;
    int3 L = subdomain(c);
    
    MC(m::Comm_rank(comm, &rank));
    if (rank == root) {
        npsolid = rcount;
        if (!npsolid) ERR("No particles remaining in root node.\n");
        for (int d = 0; d < 3; ++d)
            model.com[d] = coms[idmax*3 + d];
        ini_props(rgi.pi, npsolid, r_pp, rgi.mass, model.com, rgi.nt, rgi.tt, rgi.vv, /**/ rr0, &model);
        if (empty_solid_particles)
            empty_solid(rgi.nt, rgi.tt, rgi.vv, /* io */ rr0, &npsolid);
    }

    MC(m::Bcast(&npsolid,       1,   MPI_INT, root, comm) );
    MC(m::Bcast(rr0,  3 * npsolid, MPI_FLOAT, root, comm) );
    MC(m::Bcast(&model, 1, datatype::solid,   root, comm) );

    // filter coms to keep only the ones in my domain
    int id = 0;
    for (int j = 0; j < nsolid; ++j) {
        const float *com = coms + 3*j;

        if (-L.x/2 <= com[X] && com[X] < L.x/2 &&
            -L.y/2 <= com[Y] && com[Y] < L.y/2 &&
            -L.z/2 <= com[Z] && com[Z] < L.z/2 ) {
            ss[id] = model;

            for (int d = 0; d < 3; ++d)
            ss[id].com[d] = com[d];

            ++id;
        }
    }

    *ns = nsolid = id;
    *nps = npsolid;

    set_rig_ids(comm, nsolid, /**/ ss);
}

static void gen1(const Coords *coords, MPI_Comm comm, RigGenInfo rgi, int nsolid, float *coms, /**/
                 int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp,
                 /*w*/ int *tags, int *rcounts) {
    int root, idmax;
    elect(comm, rcounts, nsolid, /**/ &root, &idmax);
    MC(m::Bcast(&idmax, 1, MPI_INT, root, comm));

    int rcount = 0;
    kill(idmax, tags, /**/ s_n, s_pp, &rcount, r_pp);
    DBG("after kill: %d", rcount);

    share(coords, comm, root, /**/ r_pp, &rcount);
    DBG("after share: %d", rcount);

    gen0(coords, comm, rgi, nsolid, rcount, idmax, root, coms, /**/ ns, nps, rr0, ss, r_pp);
}

static void gen2(const Coords *coords, MPI_Comm comm, RigGenInfo rgi, int nsolid, float *coms, /**/
                 int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp,
                 /*w*/ int *tags, int *rcounts) {
    int3 L = subdomain(coords);
    count_pp_inside(rgi.pi, L, s_pp, *s_n, coms, nsolid, rgi.tt, rgi.vv, rgi.nt, /**/ tags, rcounts);
    gen1(coords, comm, rgi, nsolid, coms, /**/ ns, nps, rr0, ss, s_n, s_pp, r_pp, /*w*/ tags, rcounts);
}

static void gen3(const Coords *coords, MPI_Comm comm, RigGenInfo rgi, int nsolid, float *coms, /**/
                 int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp) {
    int *tags = new int[*s_n];
    int *rcounts = new int[nsolid];
    gen2(coords, comm, rgi, nsolid, coms, /**/ ns, nps, rr0, ss, s_n, s_pp, r_pp, /*w*/ tags, rcounts);
    delete[] rcounts;
    delete[] tags;
}

static void gen4(const Coords *coords, MPI_Comm comm, const char *fname, RigGenInfo rgi, /**/
                 int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp,
                 /*w*/ float *coms) {
    float3 minbb, maxbb;
    int nsolid = read_coms(fname, /**/ coms);
    if (nsolid == 0) ERR("No solid provided.\n");
    mesh_get_bbox(rgi.vv, rgi.nv, /**/ &minbb, &maxbb);
    nsolid = duplicate_PBC(coords, rgi.pi, minbb, maxbb, nsolid, /**/ coms);
    make_local(coords, nsolid, /**/ coms);
    gen3(coords, comm, rgi, nsolid, coms, /**/ ns, nps, rr0, ss, s_n, s_pp, r_pp);
}

static void gen(const Coords *coords, MPI_Comm comm, const char *fname, RigGenInfo rgi, /**/
                int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp) {
    float *coms = new float[MAX_SOLIDS * 3 * 10];
    gen4(coords, comm, fname, rgi, /**/ ns, nps, rr0, ss, s_n, s_pp, r_pp, /*w*/ coms);
    delete[] coms;
}

void gen_rig_from_solvent(const Coords *coords, MPI_Comm comm, RigGenInfo rgi,
                          /* io */ Particle *opp, int *on,
                          /* o */ int *ns, int *nps, int *n, float *rr0_hst, Solid *ss_hst, Particle *pp_hst) {
    // generate models
    msg_print("start rigid gen");
    gen(coords, comm, "rigs-ic.txt", rgi, /**/ ns, nps, rr0_hst, ss_hst, on, opp, pp_hst);
    msg_print("done rigid gen");

    *n = *ns * (*nps);
}
