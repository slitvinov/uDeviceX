struct Map {
    const int *cc;
    int color;
};
static __device__ int goodp(const Map m, int i) {
    return m.cc[i] == m.color;
}