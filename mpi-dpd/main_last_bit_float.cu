/* Set, get and preserve a last bit of a float
   Compile:

   (nvcc -std=c++11     last_bit_float.cu -c -dc -o last_bit_float.o ; \
    nvcc -std=c++11     main_last_bit_float.cu -c -o main_last_bit_float.o; \
    nvcc -dlink main_last_bit_float.o last_bit_float.o -o gpuCode.o; \
    g++ gpuCode.o last_bit_float.o main_last_bit_float.o  -o main_last_bit_float -Wl,-rpath=/usr/local/cuda-6.5/bin/..//lib64 -L/usr/local/cuda-6.5/bin/..//lib64 -lcudart )

*/
#include "last_bit_float.h"
#include <iostream>
#include <cassert>

int main() {

    for (auto  zin : {4.0f, -4.0f, 1e8f, -1e8f, 0.0f}) {
	float z   = zin;
	last_bit_float::set(z, true);
	assert(last_bit_float::get(z));
	last_bit_float::set(z, false);
	assert(!last_bit_float::get(z));
	assert(z - zin == 0.0f);
    }

    float z = -1.0e8;
    last_bit_float::set(z, true);
    {
	last_bit_float::Preserver zp(z);
	last_bit_float::set(z, false);
    }
    std::cout << last_bit_float::get(z) << '\n';
}
