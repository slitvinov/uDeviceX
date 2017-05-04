namespace mesh
{
    int inside_1p(const float *r, const float *vv, const int *tt, const int nt)
    {
        int c = 0;

        const float origin[3] = {0, 0, 0};
        
        for (int i = 0; i < nt; ++i)
        {
            const int *t = tt + 3 * i;

            if (k_mesh::in_tetrahedron(r, vv + 3*t[0], vv + 3*t[1], vv + 3*t[2], origin)) ++c;
        }
        
        return c%2;
    }

    void inside_hst(const Particle *pp, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
    {
        for (int i = 0; i < n; ++i)
        inout[i] = inside_1p(pp[i].r, vv, tt, nt);
    }

    void inside_dev(const Particle *pp, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
    {
        k_mesh::inside <<< k_cnf(n) >>>(pp, n, vv, tt, nt, /**/ inout);
    }

    /* bbox: minx, maxx, miny, maxy, minz, maxz */
    void bbox(const float *vv, const int nt, /**/ float *bbox)
    {
        if (nt == 0) return;
        
        bbox[0] = bbox[1] = vv[0];
        bbox[2] = bbox[3] = vv[1];
        bbox[4] = bbox[5] = vv[2];

        auto higher = [](float a, float b) {return a > b ? a : b;};
        auto lower  = [](float a, float b) {return a > b ? b : a;};

        for (int i = 1; i < nt; ++i)
        for (int d = 0; d < 3; ++d)
        {
            bbox[2*d + 0] =  lower(bbox[2*d + 0], vv[3*i + d]);
            bbox[2*d + 1] = higher(bbox[2*d + 1], vv[3*i + d]); 
        }
    }
}
