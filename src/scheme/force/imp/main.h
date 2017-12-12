void body_force(Coords c, float mass, FParam fpar, int n, const Particle *pp, /**/ Force *ff) {
    int type;
    FParam_v p;
    type = fpar.type;
    p    = fpar.dev;

    switch (type) {
    case TYPE_V_NONE:
        break;
    case TYPE_V_CSTE:
        KL(force, (k_cnf(n)), (c, p.cste, mass, n, pp, /**/ ff));
        break;
    case TYPE_V_DP:
        KL(force, (k_cnf(n)), (c, p.dp, mass, n, pp, /**/ ff));
        break;
    case TYPE_V_SHEAR:
        KL(force, (k_cnf(n)), (c, p.shear, mass, n, pp, /**/ ff));
        break;
    case TYPE_V_ROL:
        KL(force, (k_cnf(n)), (c, p.rol, mass, n, pp, /**/ ff));
        break;
    case TYPE_V_RAD:
        KL(force, (k_cnf(n)), (c, p.rad, mass, n, pp, /**/ ff));
        break;
    default:
        ERR("wrong type");
        break;
    };
}

void adjust(float3 f, /**/ FParam *fpar) {
    int type;
    FParam_v *p;
    type = fpar->type;
    p    = &fpar->dev;

    switch (type) {
    case TYPE_V_NONE:
        break;
    case TYPE_V_CSTE:
        p->cste.a = f;
        break;
    case TYPE_V_RAD:
        /* do not control radial and z directions */
        p->rad.a = f.x;
        break;
    case TYPE_V_DP:
    case TYPE_V_SHEAR:
    case TYPE_V_ROL:
    default:
        ERR("not implemented");
        break;
    };
}
