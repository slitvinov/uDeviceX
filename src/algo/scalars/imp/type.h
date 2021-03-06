struct Scalars {
    int type;
    int n;
    union {
        const float  *ff;
        const double *dd;
        const Vectors *vec;
    } D;
};

static double float_get (const Scalars*, int i);
static double double_get(const Scalars*, int i);
static double zero_get(const Scalars*, int i);

static double vecx_get(const Scalars*, int i);
static double vecy_get(const Scalars*, int i);
static double vecz_get(const Scalars*, int i);

enum {FLOAT, DOUBLE, VECX, VECY, VECZ, ZERO};
static double (*get[])(const Scalars*, int i) =
{ float_get,  double_get, vecx_get, vecy_get, vecz_get, zero_get};
