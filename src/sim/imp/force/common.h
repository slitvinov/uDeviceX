void body_force(float driving_force0) {
    scheme::force(1,          o::q.pp, o::ff, o::q.n, driving_force0);
    if (pushsolid && solids0)
        scheme::force(solid_mass, s::q.pp, s::ff, s::q.n, driving_force0);
    if (pushrbc && rbcs)
        scheme::force(rbc_mass,   r::q.pp, r::ff, r::q.n, driving_force0);
}

void forces_rbc() {
    if (rbcs)
        rbc::forces(r::q, r::tt, /**/ r::ff);
}

void clear_forces(Force* ff, int n) {
    if (n) DzeroA(ff, n);
}

void forces_wall() {
    using namespace wall;
    hforces::Cloud co, cs, cr;
    ini_cloud(o::q.pp, &co);
    ini_cloud(s::q.pp, &cs);
    ini_cloud(r::q.pp, &cr);
    if (multi_solvent) ini_cloud_color(o::qc.ii, &co);
    
    if (o::q.n)           color::pair(w::qsdf, w::q, w::t, SOLVENT_KIND, co, o::q.n, /**/ o::ff);
    if (solids0 && s::q.n) grey::pair(w::qsdf, w::q, w::t, SOLID_KIND  , cs, s::q.n, /**/ s::ff);
    if (rbcs && r::q.n)    grey::pair(w::qsdf, w::q, w::t, SOLID_KIND  , cr, r::q.n, /**/ r::ff);
}

void forces_cnt(std::vector<PaWrap> *pwr, std::vector<FoWrap> *fwr) {
    cnt::bind(*pwr, *fwr);
    cnt::bulk(*pwr, *fwr);
}

void forces_fsi(fsi::SolventWrap *w_s, std::vector<PaWrap> *pwr, std::vector<FoWrap> *fwr) {
    fsi::bind(*w_s);
    fsi::bulk(*pwr, *fwr);
}
