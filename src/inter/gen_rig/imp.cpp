#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <vector>
#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"
#include "inc/def.h"

#include "mpi/wrapper.h"
#include "mpi/type.h"
#include "utils/mc.h"

#include "coords/imp.h"

#include "utils/msg.h"
#include "utils/error.h"
#include "utils/imp.h"

#include "inc/type.h"

#include "math/linal/imp.h"
#include "mesh/props/imp.h"
#include "mesh/dist/imp.h"
#include "mesh/bbox/imp.h"
#include "d/ker.h"
#include "d/api.h"
#include "utils/cc.h"
#include "mesh/collision/imp.h"

#include "rigid/imp.h"
#include "imp.h"

#include "imp/ids.h"
#include "imp/ic.h"
#include "imp/share.h"
#include "imp/ini_props.h"
#include "imp/main.h"
