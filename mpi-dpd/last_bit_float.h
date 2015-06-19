#include <bitset>

namespace last_bit_float {
    // return a last bit of a float
    // http://stackoverflow.com/a/1723938
    __host__ __device__ bool get(const float f) {
	unsigned char const *c = reinterpret_cast<unsigned char const*>(&f);
	return c[0] & 1;
    }

    // set a last bit of a float to `bit'
    __host__ __device__ void set(float& f, const bool bit) {
	unsigned char *c = reinterpret_cast<unsigned char *>(&f);
	if (bit) c[0] |= 1;
	else     c[0] &= ~1;
    }

    /* Last bit preserver If you do last_bit_float::Preserver zp(z);
       the last bit of `z' will be restored if variable `zp' goes out
       of the scope (see mpi-dpd/main_last_bit_float.cpp) */
    class Preserver {
	float& _f;
	const bool bit;
    public:
	__host__ __device__ explicit Preserver(float& f): _f(f), bit(get(f)) { }
	__host__ __device__ ~Preserver() {
	   set(_f, bit);
	}
    };
}
