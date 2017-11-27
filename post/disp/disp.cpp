#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "bop_common.h"
#include "bop_serial.h"

#include "macros.h"
#include "pp_id.h"

int X, Y, Z;

enum {EMPTY=0, OCCUPIED=1};
void empty_tags(const int bufsize, int *tags) {
    for (int i = 0; i < bufsize; ++i) tags[i] = EMPTY;
}

void compute_tags(const int *ii, const int n, int *tags) {
    for (int j = 0; j < n; ++j) {
        const int i = ii[j];
        tags[i] = OCCUPIED;
    }
}

void disp0(const float *rp, const float *rc, float *dr) {
    const int dL[3] = {X, Y, Z};
    for (int c = 0; c < 3; ++c) {
        dr[c] = rc[c] - rp[c];
        const float sign = dr[c] > 0 ? 1.f : -1.f;
        dr[c] -= fabs(dr[c]) > dL[c]/2 ? sign * dL[c] : 0.f;        
    }
}

void disp(const float *rrp, const float *rrc, const int buffsize, /**/ float *ddr) {
    for (int i = 0; i < buffsize; ++i) {
        const float *rp = rrp + 3*i;
        const float *rc = rrc + 3*i;
        float *dr       = ddr + 3*i;
        disp0(rp, rc, /**/ dr);
    }
}

void outname(const char *inrr, char *out) {
    const int l = strlen(inrr);
    memcpy(out, inrr, l * sizeof(char));
    const int strt = l - 4;
    const char newext[] = ".disp";
    memcpy(out + strt, newext, sizeof(newext));
}

void dump(const Type type, const char *fout, const float *rr, const float *ddr, const int *tags, const int buffsize, /*w*/ float *work) {
    /* fill work buffer */
    int j = 0;
    for (int i = 0; i < buffsize; ++i)
    if (tags[i] == OCCUPIED) {
        for (int c = 0; c < 3; ++c) {
            work[6 * j + 0 + c] = rr[3 * i + c];
            work[6 * j + 3 + c] = ddr[3 * i + c];
        }
        ++j;
    }

    /* dump */
    BopData d;
    d.n = j;
    d.nvars = 6;
    d.type = type;
    strcpy(d.vars, "x y z dx dy dz");

    bop_alloc(&d);
    
    memcpy(d.data, work, d.n * d.nvars * sizeof(float));

    bop_write_header(fout, &d);
    bop_write_values(fout, &d);
    
    bop_free(&d);
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: po.disp <X> <Y> <Z> <rr-*.bop> -- <ii-*.bop>\n");
        exit(1);
    }
    int iarg = 1;
    X = atoi(argv[iarg++]);
    Y = atoi(argv[iarg++]);
    Z = atoi(argv[iarg++]);

    const int sep = separator(argc, argv);
    const int nin = sep - iarg;

    if (nin < 2) ERR("Need more than one file\n");
    
    char **ffpp = argv + iarg;
    char **ffii = ffpp + nin + 1;
    
    BopData dpp, dii;
    const int *ii;
    const float *pp;
    
    read_data(ffpp[0], &dpp, ffii[0], &dii);

    pp = (const float *) dpp.data;
    ii = (const int *) dii.data;
    
    const int buffsize = max_index(ii, dii.n) + 1;

    float *rrc = new float[3*buffsize]; /* current  positions     */
    float *rrp = new float[3*buffsize]; /* previous positions     */
    float *ddr = new float[3*buffsize]; /* displacements          */
    float *rrw = new float[6*buffsize]; /* work                   */
    int  *tags = new int[buffsize];     /* tags: particle with this id or not? */

    memset(rrp, 0, 3*buffsize*sizeof(float));
    pp2rr_sorted(ii, pp, dpp.n, dpp.nvars, /**/ rrp);

    empty_tags(buffsize, /**/ tags);
    compute_tags(ii, dii.n, /**/ tags);

    bop_free(&dpp); bop_free(&dii);
    
    for (int i = 0; i < nin-1; ++i) {
        char fout[1024] = {0};
        outname(ffpp[i], /**/ fout);
        printf("%s -- %s -> %s\n", ffpp[i], ffii[i], fout);

        read_data(ffpp[i+1], &dpp, ffii[i+1], &dii);
        pp = (const float *) dpp.data;
        ii = (const int *) dii.data;

        pp2rr_sorted(ii, pp, dpp.n, dpp.nvars, /**/ rrc);

        disp(rrp, rrc, buffsize, /**/ ddr);

        dump(dpp.type, fout, rrp, ddr, tags, buffsize, /*w*/ rrw);
        compute_tags(ii, dii.n, /**/ tags);
        
        bop_free(&dpp);  bop_free(&dii);

        {   /* swap */
            float *const tmp = rrp;
            rrp = rrc; rrc = tmp;
        }
    }

    delete[] rrp; delete[] rrc;
    delete[] ddr; delete[] rrw;
    delete[] tags;
    
    return 0;
}

/*
  
  # nTEST: small.t0
  # make install -j
  # po.disp 4 4 4 data/s-??.bop -- data/id-??.bop
  # cp data/s-00.disp.values disp.out.txt

 */