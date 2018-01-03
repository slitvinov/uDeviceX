void body_force(Coords c, float mass, BForce_v view, int n, const Particle *pp, /**/ Force *ff) {
    int type;
    BForceParam_v p;
    type = view.type;
    p    = view.p;

    switch (type) {
    case BODY_FORCE_V_NONE:
        break;
    case BODY_FORCE_V_CSTE:
        KL(force, (k_cnf(n)), (c, p.cste, mass, n, pp, /**/ ff));
        break;
    case BODY_FORCE_V_DP:
        KL(force, (k_cnf(n)), (c, p.dp, mass, n, pp, /**/ ff));
        break;
    case BODY_FORCE_V_SHEAR:
        KL(force, (k_cnf(n)), (c, p.shear, mass, n, pp, /**/ ff));
        break;
    case BODY_FORCE_V_ROL:
        KL(force, (k_cnf(n)), (c, p.rol, mass, n, pp, /**/ ff));
        break;
    case BODY_FORCE_V_RAD:
        KL(force, (k_cnf(n)), (c, p.rad, mass, n, pp, /**/ ff));
        break;
    default:
        ERR("wrong type <%d>", type);
        break;
    };
}

static void get_view(BForce_cste bf, BForce_v *v) {
    v->type = BODY_FORCE_V_CSTE;
    v->p.cste.a = bf.a;
}

static void get_view(BForce_dp bf, BForce_v *v) {
    v->type = BODY_FORCE_V_DP;
    v->p.dp.a = bf.a;
}

static void get_view(BForce_shear bf, BForce_v *v) {
    v->type = BODY_FORCE_V_SHEAR;
    v->p.shear.a = bf.a;
}

static void get_view(BForce_rol bf, BForce_v *v) {
    v->type = BODY_FORCE_V_ROL;
    v->p.rol.a = bf.a;
}

static void get_view(BForce_rad bf, BForce_v *v) {
    v->type = BODY_FORCE_V_RAD;
    v->p.rad.a = bf.a;
}

void get_view(long it, BForce bforce, /**/ BForce_v *view) {
    int type;
    BForceParam p;
    type = bforce.type;
    p    = bforce.p;    

    switch (type) {
    case BODY_FORCE_NONE:
        view->type = BODY_FORCE_V_NONE;
        break;
    case BODY_FORCE_CSTE:
        get_view(p.cste, /**/ view);
        break;
    case BODY_FORCE_DP:
        get_view(p.dp, /**/ view);
        break;
    case BODY_FORCE_SHEAR:
        get_view(p.shear, /**/ view);
        break;
    case BODY_FORCE_ROL:
        get_view(p.rol, /**/ view);
        break;
    case BODY_FORCE_RAD:
        get_view(p.rad, /**/ view);
        break;
    default:
        ERR("wrong type");
        break;
    };    
}

void adjust(float3 f, /**/ BForce *bforce) {
    int type;
    BForceParam *p;
    type = bforce->type;
    p    = &bforce->p;

    switch (type) {
    case BODY_FORCE_NONE:
        break;
    case BODY_FORCE_CSTE:
        p->cste.a = f;
        break;
    case BODY_FORCE_RAD:
        /* do not control radial and z directions */
        p->rad.a = f.x;
        break;
    case BODY_FORCE_DP:
    case BODY_FORCE_SHEAR:
    case BODY_FORCE_ROL:
    default:
        ERR("not implemented");
        break;
    };
}