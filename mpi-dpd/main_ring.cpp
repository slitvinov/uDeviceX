/*
   A test function for Ring class
   To compile:
   g++ main_ring.cpp -o main_ring
 */

#include <iostream>
#include "ring.h"

int main() {
    Ring r(10);
    r.shift(-1);
    r.shift(-1);
    // after two shifts
    // it should look like
    //
    // 0 1 2 3 4 5 6 7 8 9
    // 8 9 0 1 2 3 4 5 6 7

    for (int i = 0; i<10; i++)
      std::cout << i << ' ';
    std::cout << '\n';
    for (int i = 0; i<10; i++)
      std::cout << r(i) << ' ';
    std::cout << '\n';
}
