/* helper functions for DPD MSD calculations */
#include "msd-aux.h"
#include "common.h"
#include "last_bit_float.h"
#include <bitset>

void set_traced_particles(int n, Particle * particles) {
  for (int i = 0; i<n; i++) {
    /* 000: all other particles */
    last_bit_float::set(particles[i].u[0], false);
    last_bit_float::set(particles[i].u[1], false);
    last_bit_float::set(particles[i].u[2], false);
  }

  for (int i = 0; i<msd_calculations_module::n_tracers; i++) {
    /*
       001, 010, ..., 111: for traced particless
     */
    const std::bitset<3> bs (i+1);
    last_bit_float::set(particles[i].u[0], bs.test(0));
    last_bit_float::set(particles[i].u[1], bs.test(1));
    last_bit_float::set(particles[i].u[2], bs.test(2));
  }
}

std::vector<int> get_traced_list(int n, Particle * const particles) {
  std::vector<int> ilist(msd_calculations_module::n_tracers, 0);
  for (int i = 0; i<n; i++) {
    const bool bx = last_bit_float::get(particles[i].u[0]);
    const bool by = last_bit_float::get(particles[i].u[1]);
    const bool bz = last_bit_float::get(particles[i].u[2]);
    const bool traced = bx || by || bz;
    if (traced) {
      std::bitset<3> bs (0);
      bs.set(0, bx); bs.set(1, by); bs.set(2, bz);
      const unsigned long idx = bs.to_ulong() - 1ul;
      assert(idx < msd_calculations_module::n_tracers);

      // we should not see the same particles two times
      assert(ilist[idx] == 0);
      ilist[idx] = i;
    }
  }
  return ilist;
}
