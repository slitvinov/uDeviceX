void body_force(Coords c, float mass, FParam fpar, int n, const Particle *pp, /**/ Force *ff) {
    int type;
    FParam_d p;
    type = fpar.type;
    p    = fpar.dev;

    switch (type) {
    case TYPE_NONE:
        break;
    case TYPE_CSTE:
        KL(force, (k_cnf(n)), (c, p.cste, mass, n, pp, /**/ ff));
        break;
    case TYPE_DP:
        KL(force, (k_cnf(n)), (c, p.dp, mass, n, pp, /**/ ff));
        break;
    case TYPE_SHEAR:
        KL(force, (k_cnf(n)), (c, p.shear, mass, n, pp, /**/ ff));
        break;
    case TYPE_ROL:
        KL(force, (k_cnf(n)), (c, p.rol, mass, n, pp, /**/ ff));
        break;
    case TYPE_RAD:
        KL(force, (k_cnf(n)), (c, p.rad, mass, n, pp, /**/ ff));
        break;
    default:
        ERR("wrong type");
        break;
    };
}

void adjust(float3 f, /**/ FParam *fpar) {
    int type;
    FParam_d *p;
    type = fpar->type;
    p    = &fpar->dev;

    switch (type) {
    case TYPE_NONE:
        break;
    case TYPE_CSTE:
        p->cste.a = f;
        break;
    case TYPE_RAD:
        /* do not control radial and z directions */
        p->rad.a = f.x;
        break;
    case TYPE_DP:
    case TYPE_SHEAR:
    case TYPE_ROL:
    default:
        ERR("not implemented");
        break;
    };
}