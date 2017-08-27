#include <assert.h>
#include <vector>
#include <mpi.h>
#include <stdint.h>
#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "l/m.h"
#include "l/off.h"
#include "scan/int.h"

#include "rnd/imp.h"
#include "rnd/dev.h"

#include "inc/def.h"
#include "msg.h"
#include "m.h"
#include "cc.h"
#include "mc.h"
#include "frag.h"

#include "kl.h"
#include "basetags.h"
#include "inc/type.h"
#include "inc/mpi.h"
#include "inc/dev.h"

#include "dual/type.h"
#include "dual/int.h"
#include "texo.h"
#include "te.h"

#include "inc/tmp/wrap.h"
#include "inc/tmp/pinned.h"
#include "io/field.h"
#include "io/rbc.h"
#include "bund.h"
#include "diag.h"

#include "dbg.h"

#include "restart.h"

#include "glb.h"

#include "k/read.h"
#include "k/write.h"
#include "k/common.h"
#include "k/index.h"

#include "clist/int.h"

#include "mcomm/type.h"
#include "mcomm/int.h"

#include "rbc/int.h"

#include "mdstr/buf.h"
#include "mdstr/tic.h"
#include "mdstr/int.h"
#include "rdstr/int.h"

#include "field.h"

#include "forces/imp.h"

#include "sdf/type.h"
#include "sdf/int.h"

#include "wall/int.h"

#include "flu/int.h"

#include "odstr/type.h"
#include "odstr/int.h"
#include "cnt/int.h"
#include "fsi/int.h"

#include "sdstr.decl.h"
#include "sdstr.impl.h"
#include "x/int.h"
#include "dpd/local.h"

namespace dpdx {
namespace dev {
#include "dpd/x/dev.h"
}
#include "dpd/x/imp.h"
}

#include "dpdr/type.h"
#include "dpdr/int.h"

#include "mesh/collision.h"
#include "mesh/bbox.h"

#include "solid.h"
#include "tcells/int.h"

#include "mbounce/imp.h"
#include "mrescue.h"

#include "bbhalo.decl.h"
#include "bbhalo.impl.h"

#include "dump/int.h"

#include "rig/int.h"

namespace sim {
namespace dev {
#ifdef FORWARD_EULER
  #include "sim/sch/euler.h"
#else
  #include "sim/sch/vv.h"
#endif

#include "sim/dev.h"
}
#include "sim/dec.h"
#include "sim/ini.h"
#include "sim/fin.h"
#include "sim/generic.h"
#include "sim/dump.h"
#include "sim/tag.h"
#include "sim/forces/dpd.h"
#include "sim/forces.h"

#if   defined(FORCE1)
  #include "sim/force1.h"
#elif defined(FORCE0)
  #include "sim/force0.h"
#else
  #error FORCE[01] is undefined
#endif


#define HST (true)
#define DEV (false)
#define DEVICE_SOLID
#ifdef DEVICE_SOLID
  #include "0dev/sim.impl.h"
#else
  #include "0hst/sim.impl.h"
#endif
#undef HST
#undef DEV

#if   defined(UPDATE1)
  #include "sim/update/release.h"
#elif defined(UPDATE_SAFE)
  #include "sim/update/safe.h"
#else
  #error UPDATE* is undefined
#endif

#if   defined(ODSTR1)
  #include "sim/odstr/release.h"
#elif defined(ODSTR0)
  #include "sim/odstr/none.h"
#elif defined(ODSTR_SAFE)
  namespace sub {
    #include "sim/odstr/release.h"
  }
  #include "sim/odstr/safe.h"
#else
  #error ODSTR* is undefined
#endif

#include "sim/step.h"
#include "sim/run.h"
#include "sim/imp.h"
}
