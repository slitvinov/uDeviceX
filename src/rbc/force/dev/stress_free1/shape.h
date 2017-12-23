struct Shape0 {
    float a; /* edge equilibrium lengths */
    float A; /* local area */
    float totArea;
};

/* extract edge specific shape info */
static void __device__ edg_shape(Shape shape, int i, /**/ Shape0 *shape0) {
    assert ( i < RBCnv * RBCmd);
    Edg edg;
    edg = shape.edg[i];
    shape0->a = edg.a;
    shape0->A = edg.A;
    shape0->totArea = shape.totArea;
}