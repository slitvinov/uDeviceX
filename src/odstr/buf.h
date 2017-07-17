/* pinned buffers (array of pinned buffers) */ 
template <typename T, int N=27>
struct Pbufs {
    T **dev;   /* data on device  */
    T *dp[N];  /* device pointers */
    T *hst[N]; /* data on host    */
};

template <typename T, int N>
void alloc_pinned(const int i, const int sz, /**/ Pbufs<T, N> *b) {
    if (sz){
        CC(cudaHostAlloc(&b->hst[i], sizeof(T) * sz, cudaHostAllocMapped));
        CC(cudaHostGetDevicePointer(&b->dp[i], b->hst[i], 0));
    } else {
        b->hst[i] = NULL;
    }
}

template <typename T, int N>
void alloc_dev(/**/ Pbufs<T, N> *b) {
    CC(cudaMalloc(&b->dev, SZ_PTR_ARR(b->dp)));
    CC(cudaMemcpy(b->dev, b->dp, sizeof(b->dp), H2D));
}

template <typename T, int N>
void dealloc(Pbufs<T, N> *b) {
    for (int i = 0; i < N; ++i) {
        if (b->dp[i] != NULL) CC(cudaFreeHost(b->hst[i]));
    }
    CC(cudaFree(b->dev));
}

struct Send {
    int **iidx; /* indices */

    Pbufs<float2> pp;
    Pbufs<int> ii; /* global ids */
    
    int *size_dev, *strt;
    int size[27];
    PinnedHostBuffer4<int>* size_pin;

    int    *iidx_[27];
};

struct Recv {
    Pbufs<float2> pp;
    Pbufs<int> ii; /* global ids */
    
    int *strt;
    int tags[27];
    int    size[27];
};
