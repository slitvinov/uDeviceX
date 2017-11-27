void scalar(MPI_Comm cart, float *D, const char *name) {
    char path[BUFSIZ];
    sprintf(path, DUMP_BASE "/h5/%s.h5", name);
    UC(h5::write(cart, path, &D, &name, 1, XS, YS, ZS));
    if (!m::rank) xmf::write(path, &name, 1, XS, YS, ZS);
}
