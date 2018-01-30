#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "coords/type.h"
#include "coords/ini.h"

#include "utils/error.h"
#include "utils/imp.h"

#include "d/api.h"
#include "utils/msg.h"

#include "inc/def.h"
#include "inc/type.h"
#include "inc/dev.h"

#include "utils/te.h"
#include "utils/texo.h"
#include "utils/cc.h"

#include "rbc/params/imp.h"
#include "rbc/type.h"
#include "rbc/imp.h"
#include "rbc/rnd/imp.h"
#include "rbc/force/imp.h"

#include "io/off/imp.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"

#include "imp.h"

#include "imp/main.h"
