const float gamma_dot     = _gamma_dot;
const float hydrostatic_a = _hydrostatic_a / rc;
const float kBT           = _kBT / (rc * rc);
const int   numberdensity = _numberdensity * (rc * rc * rc);

/* maximum particle number per one processor for static allocation */
#define MAX_PART_NUM 5000000

/* maximum number of faces per one RBC */
#define MAX_FACE_NUM 5000000

/* maximum number of random states per one RBC */
#define MAX_RND_NUM 4000000

/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less
   than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )

/* a common kernel execution configuration */
#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(cudaDeviceSynchronize())
#define D2D cudaMemcpyDeviceToDevice
#define D2H cudaMemcpyDeviceToHost
#define H2D cudaMemcpyHostToDevice
#define H2H cudaMemcpyHostToHost

/* [c]cuda [c]heck */
#define CC(ans)							\
  do { cudaAssert((ans), __FILE__, __LINE__); } while (0)
inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file,
	    line);
    abort();
  }
}

/* [m]pi [c]heck */
#define MC(ans)							\
  do { mpiAssert((ans), __FILE__, __LINE__); } while (0)
inline void mpiAssert(int code, const char *file, int line) {
  if (code != MPI_SUCCESS) {
    char error_string[2048];
    int length_of_error_string = sizeof(error_string);
    MPI_Error_string(code, error_string, &length_of_error_string);
    printf("mpiAssert: %s %d %s\n", file, line, error_string);
    MPI_Abort(MPI_COMM_WORLD, code);
  }
}

// AoS is the currency for dpd simulations (because of the spatial locality).
// AoS - SoA conversion might be performed within the hpc kernels.
struct Particle {
  float r[3], v[3];
  static bool initialized;
  static MPI_Datatype mytype;
  static MPI_Datatype datatype() {
    if (!initialized) {
      MC(MPI_Type_contiguous(6, MPI_FLOAT, &mytype));
      MC(MPI_Type_commit(&mytype));
      initialized = true;
    }
    return mytype;
  }
};

template <typename T>
void mpDeviceMalloc(T **D) { /* a "[m]ax [p]article number" device
			       allocation (takes a pointer to
			       pointer!) */
  CC(cudaMalloc(D, sizeof(T) * MAX_PART_NUM));
}

struct Force {
  float a[3];
};

struct ParticlesWrap {
  const Particle *p;
  Force *f;
  int n;
  ParticlesWrap() : p(NULL), f(NULL), n(0) {}
  ParticlesWrap(const Particle *const p, const int n, Force *f)
    : p(p), n(n), f(f) {}
};

struct SolventWrap : ParticlesWrap {
  const int *cellsstart, *cellscount;
  SolventWrap() : cellsstart(NULL), cellscount(NULL), ParticlesWrap() {}
  SolventWrap(const Particle *const p, const int n, Force *f,
	      const int *const cellsstart, const int *const cellscount)
    : ParticlesWrap(p, n, f),
      cellsstart(cellsstart),
      cellscount(cellscount) {}
};

/* container for the gpu particles during the simulation */
template <typename T> struct DeviceBuffer {
  /* `C': capacity; `S': size; `D' : data*/
  int C, S; T *D;

  explicit DeviceBuffer(int n = 0) : C(0), S(0), D(NULL) { resize(n); }
  ~DeviceBuffer() {
    if (D != NULL) CC(cudaFree(D));
    D = NULL;
  }

  void resize(int n) {
    S = n;
    if (C >= n) return;
    if (D != NULL) CC(cudaFree(D));
    int conservative_estimate = (int)ceil(1.1 * n);
    C = 128 * ((conservative_estimate + 129) / 128);
    CC(cudaMalloc(&D, sizeof(T) * C));
  }

  void preserve_resize(int n) {
    T *old = D;
    int oldS = S;

    S = n;
    if (C >= n) return;
    int conservative_estimate = (int)ceil(1.1 * n);
    C = 128 * ((conservative_estimate + 129) / 128);
    CC(cudaMalloc(&D, sizeof(T) * C));
    if (old != NULL) {
      CC(cudaMemcpy(D, old, sizeof(T) * oldS, D2D));
      CC(cudaFree(old));
    }
  }
};

template <typename T> struct PinnedHostBuffer {
private:
  int capacity;
public:
  /* `S': size; `D' is for data; `DP' device pointer */
  int S;  T *D, *DP;

  explicit PinnedHostBuffer(int n = 0)
    : capacity(0), S(0), D(NULL), DP(NULL) {
    resize(n);
  }

  ~PinnedHostBuffer() {
    if (D != NULL) CC(cudaFreeHost(D));
    D = NULL;
  }

  void resize(const int n) {
    S = n;
    if (capacity >= n) return;
    if (D != NULL) CC(cudaFreeHost(D));
    const int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);

    CC(cudaHostAlloc(&D, sizeof(T) * capacity, cudaHostAllocMapped));

    CC(cudaHostGetDevicePointer(&DP, D, 0));
  }

  void preserve_resize(const int n) {
    T *old = D;
    const int oldS = S;
    S = n;
    if (capacity >= n) return;
    const int conservative_estimate = (int)ceil(1.1 * n);
    capacity = 128 * ((conservative_estimate + 129) / 128);
    D = NULL;
    CC(cudaHostAlloc(&D, sizeof(T) * capacity, cudaHostAllocMapped));
    if (old != NULL) {
      CC(cudaMemcpy(D, old, sizeof(T) * oldS, H2H));
      CC(cudaFreeHost(old));
    }
    CC(cudaHostGetDevicePointer(&DP, D, 0));
  }
};

/* container for the cell lists, which contains only two integer
   vectors of size ncells.  the start[cell-id] array gives the entry in
   the particle array associated to first particle belonging to cell-id
   count[cell-id] tells how many particles are inside cell-id.  building
   the cell lists involve a reordering of the particle array (!) */
struct CellLists {
  const int ncells, LX, LY, LZ;
  int *start, *count;
  CellLists(const int LX, const int LY, const int LZ)
    : ncells(LX * LY * LZ + 1), LX(LX), LY(LY), LZ(LZ) {
    CC(cudaMalloc(&start, sizeof(int) * ncells));
    CC(cudaMalloc(&count, sizeof(int) * ncells));
  }

  void build(Particle *const p, const int n,
	     int *const order = NULL, const Particle *const src = NULL);

  ~CellLists() {
    CC(cudaFree(start));
    CC(cudaFree(count));
  }
};

void diagnostics(Particle *_particles, int n, int idstep);
