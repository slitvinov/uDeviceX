#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "utils/msg.h"
#include "mpi/wrapper.h"
#include "mpi/glb.h"
#include "glob/type.h"
#include "glob/ini.h"

/* local */
#include "lib/imp.h"

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    msg_ini(m::rank);
    Coords coords;
    coords_ini(m::cart, &coords);
    run(coords, "rbc.off", "rbcs-ic.txt");
    coords_fin(&coords);
    m::fin();
}
