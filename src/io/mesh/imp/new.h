#define PATTERN "%s/%s/%05d"

static void ini(MPI_Comm comm, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    int i;
    MeshWrite *q;
    EMALLOC(1, &q);

    q->nv = nv; q->nt = nt;
    cpy(q->directory, directory);
    UC(mkdir(comm, DUMP_BASE, directory));    
    EMALLOC(nt, &q->tt);
    for (i = 0; i < nt; i++)
        q->tt[i] = tt[i];
    *pq = q;
}

void mesh_write_ini(MPI_Comm comm, const int4 *tt, int nv, int nt, const char *directory, /**/ MeshWrite **pq) {
    UC(ini(comm, tt, nv, nt, directory, /**/ pq));
}

void mesh_write_ini_off(MPI_Comm comm, MeshRead *cell, const char *directory, /**/ MeshWrite **pq) {
    int nv, nt;
    const int4 *tt;
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    tt = mesh_read_get_tri(cell);
    UC(ini(comm, tt, nv, nt, directory, /**/ pq));
}

void mesh_write_fin(MeshWrite *q) {
    EFREE(q->tt);
    EFREE(q);
}

void mesh_write_particles(MeshWrite *q, MPI_Comm comm, const Coords *coords, int nc, const Particle *pp, int id) {
    const char *fmt = "%s/%s/%05d.ply";
    char path[FILENAME_MAX];
    if (sprintf(path, fmt, DUMP_BASE, q->directory, id) < 0)
        ERR("sprintf failed");
    UC(mesh_write(comm, coords, pp, q->tt, nc, q->nv, q->nt, path));
}
