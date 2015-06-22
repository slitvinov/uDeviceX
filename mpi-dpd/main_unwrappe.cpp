/*
   A test for unwrape.h
   To compile:
   g++ -std=c++11 main_unwrappe.cpp -o main_unwrappe
 */

#include <iostream>
#include <vector>
#include <iostream>
#include "unwrappe.h"

int main() {
  std::vector<float> r {0.1, 0.25, 0.50, 0.75, 1.0, 0.25, 0.50, 0.25, 0.9};

  for (int i = 0; i<r.size()-1; i++)
    std::cout << unwrappe(r[i+1], r[i], 1.05) << ' ';
  // <0.1> 0.25 0.5 0.75 1 1.3 0.5 0.25 -0.15

  std::cout << '\n';
}
