/* fragment */
enum FragType { BULK = 0, FACE = 1, EDGE = 2, CORNER = 3 };

struct SFrag { /* "send" fragment */
    float *xdst;
    int *ii;
    int ndst;
}

struct Frag {
    float *xdst;
    int *ii;
    int ndst;

    float2 *xsrc;
    int nsrc, *cellstarts, dx, dy, dz, xcells, ycells, zcells;
    FragType type;
};

struct Rnd {
    float seed;
    int mask;
};
