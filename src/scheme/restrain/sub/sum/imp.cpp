#include <mpi.h>
#include <conf.h>
#include "inc/conf.h"

#include "utils/mc.h"
#include "mpi/wrapper.h"

#include "imp.h"

/* body */
namespace sum {
#include "imp/main.h"
}