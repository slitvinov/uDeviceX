$B/algo/minmax/imp.o: $S/algo/minmax/imp.h $B/conf.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/msg.h $S/utils/cc.h $S/utils/kl.h
$B/algo/scan/imp.o: $S/algo/scan/cpu/imp.h $S/algo/scan/cuda/imp.h $S/algo/scan/dev.h $S/algo/scan/imp.h $B/conf.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/msg.h $S/utils/cc.h $S/utils/kl.h
$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/utils/error.h $S/utils/halloc.h
$B/glb/imp.o: $B/conf.h $S/d/api.h $S/glb/get.h $S/glb/imp/dec.h $S/glb/imp/main.h $S/glb/imp/util.h $S/glb/set.h $S/glb/wvel/imp.h $S/inc/conf.h $S/mpi/glb.h
$B/glb/wvel/imp.o: $B/conf.h $S/glb/wvel/imp.h $S/glb/wvel/imp/flat.h $S/glb/wvel/imp/sin.h $S/inc/conf.h $S/msg.h
$B/io/off.o: $S/io/off.h $S/io/off/imp.h $S/utils/efopen.h $S/utils/error.h
$B/mesh/bbox.o: $S/algo/minmax/imp.h $S/inc/type.h $S/mesh/bbox.h
$B/mesh/collision.o: $B/conf.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/mesh/collision.h $S/msg.h $S/utils/cc.h $S/utils/error.h $S/utils/kl.h $S/utils/texo.dev.h $S/utils/texo.h
$B/mesh/dist.o: $S/inc/type.h $S/mesh/dist.h
$B/mesh/props.o: $S/inc/type.h $S/mesh/props.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/mpi/glb.h $S/msg.h
$B/u/mesh/main.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/io/off.h $S/mesh/collision.h $S/mpi/glb.h $S/msg.h $S/u/mesh/imp/main.h $S/utils/cc.h $S/utils/error.h $S/utils/kl.h $S/utils/texo.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc/common.h $S/utils/error.h
$B/utils/efopen.o: $S/utils/efopen.h $S/utils/error.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/halloc.o: $S/utils/error.h $S/utils/halloc.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/utils/error.h $S/utils/mc.h
$B/utils/os.o: $S/msg.h $S/utils/error.h $S/utils/os.h
