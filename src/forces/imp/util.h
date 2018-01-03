static __device__ bool seteq(int a, int b,   int x, int y) {
    /* true if sets {a, b} and {x, y} are equal */
    return (a == x && b == y) || (a == y && b == x);
}
static __device__ void copy_color(int ka, int kb, int k, /**/ int *ca, int *cb) {
    /* k: master kind: copy color from this kind */
    if      (ka == k) *cb = *ca;
    else if (kb == k) *ca = *cb;
    else assert(0);
}

static __device__ float cap(float x, float lo, float hi) {
    if      (x > hi) return hi;
    else if (x < lo) return lo;
    else             return x;
}

static const float EPS = 1e-6;
enum {NORM_OK, NORM_BIG, NORM_SMALL};
static __device__ int norm(/*io*/ float *px, float *py, float *pz, /**/ float *pr, float *pinvr) {
    /* normalize r = [x, y, z], sets |r| and 1/|r| if not big */
    float x, y, z, invr, r;
    float r2;
    x = *px; y = *py; z = *pz;

    r2 = x*x + y*y + z*z;
    if      (r2 >= 1 )   return NORM_BIG;
    else if (r2 < EPS) {
        *pr = *px = *py = *pz = 0;
        return NORM_SMALL;
    } else {
        invr = rsqrtf(r2);
        r = r2 * invr;
        x *= invr; y *= invr; z *= invr;
        *px = x; *py = y; *pz = z; *pr = r; *pinvr = invr;
        return NORM_OK;
    }
}