/* Set, get and preserve a last bit of a float
   Compile:
   nvcc -std=c++11 main_last_bit_float.cu -o main_last_bit_float
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
