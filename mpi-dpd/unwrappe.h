/* unwrappe trajectory: makes trajectory continues in periodic domain */
#include <algorithm>
#include <cmath>

float unwrappe(const float r_next, const float r_prev, const float L) {
  /* r_prev: previously saved coordinate
     r_next: current coordinate
     L     : a size of a periodic box

     Try `r_next', `r_next + L', `r_next - L' and use the one which
     gives the smallest distance from `r_prev' */

  const float d1 = std::fabs(r_next     - r_prev);
  const float d2 = std::fabs(r_next + L - r_prev);
  const float d3 = std::fabs(r_next - L - r_prev);

  if (d1 < d2 && d1 < d3)
    return r_next;

  if (d2 < d1 && d2 < d3)
    return r_next + L;

    return r_next - L;
}
