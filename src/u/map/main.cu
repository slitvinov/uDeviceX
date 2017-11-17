#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "msg.h"

#include "mpi/glb.h"
#include "inc/dev.h"
#include "utils/cc.h"

#include "utils/kl.h"

#include "inc/def.h"
#include "inc/type.h"

#include "cloud/imp.h"
#include "hforces/imp.h"

#include "utils/map/dev.h"

namespace dev {
#include "dev.h"
}

void main0() {
    KL(dev::main, (1, 1), ());
    CC(d::PeekAtLastError());
    dSync();
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main0();
    m::fin();
}
